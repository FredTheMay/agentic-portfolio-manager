"""AWS infrastructure: Lambda for the scheduled cycle, EventBridge cron, DynamoDB, S3 + CloudFront.

Two properties this stack is built around.

**The schedule fires at least once, not exactly once.** EventBridge can deliver
a duplicate, and a Lambda retry after a timeout is indistinguishable from a
fresh invocation. The system already handles this three ways — content-hashed
mandate ids, broker-side client_order_id rejection, and put_item on the
mandate id here — because none of them is sufficient alone.

**Nothing in this stack can place a real trade.** There is no live broker
endpoint anywhere in the codebase, and the cycle function is granted no
secret that would let it reach one.
"""

from __future__ import annotations

import os

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_dynamodb as dynamodb,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_logs as logs,
    aws_s3 as s3,
    aws_s3_deployment as s3_deploy,
)
from constructs import Construct

#: Must match src/api/store.py's SNAPSHOT_KEY and RESEARCH_PREFIX. Not
#: imported directly — this app runs as `python3 app.py` from infra/, which
#: isn't on a path that resolves `src.*` without extra PYTHONPATH plumbing in
#: the deploy tooling.
SNAPSHOT_KEY = "SNAPSHOT#latest"
RESEARCH_PREFIX = "RESEARCH#"

#: Read from the deploying shell's environment, not committed anywhere —
#: `source .env` before `cdk deploy` if you want it set. Absent, CycleFunction
#: correctly falls back to LLM_PROVIDER=null rather than claiming a provider
#: it has no key for.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

#: numpy + scipy + pandas + scikit-learn alone run ~285MB unzipped, over the
#: 250MB limit for zip-based Lambdas — and both functions need the full set,
#: since ApiFunction's routes.py imports src.cfa.portfolio, which uses
#: sklearn. Container images support up to 10GB, so both functions build from
#: the same ../Dockerfile with a different `cmd` override instead. See
#: .dockerignore for what's excluded from the build context.
LAMBDA_IMAGE_DIR = "../"

#: 21:30 UTC on weekdays — after the US equity close, so the session's closing
#: prices exist. Running before the close would decide on stale marks.
CYCLE_SCHEDULE = events.Schedule.cron(
    minute="30", hour="21", week_day="MON-FRI", month="*", year="*"
)


class PortfolioManagerStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs: object) -> None:
        super().__init__(scope, construct_id, **kwargs)  # type: ignore[arg-type]

        # --- State -------------------------------------------------------
        # Single table, partition key `pk`. On-demand billing: the workload is
        # one write a day, and provisioning capacity for that is pure waste.
        table = dynamodb.Table(
            self,
            "StateTable",
            partition_key=dynamodb.Attribute(name="pk", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery=True,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # --- Scheduled decision cycle ------------------------------------
        cycle_function = lambda_.DockerImageFunction(
            self,
            "CycleFunction",
            code=lambda_.DockerImageCode.from_image_asset(
                LAMBDA_IMAGE_DIR, cmd=["src.api.handler.scheduled_cycle"]
            ),
            # Must match the host Docker builds natively for — a mismatch here
            # produces Runtime.InvalidEntrypoint at cold start, not a synth or
            # deploy-time error.
            architecture=lambda_.Architecture.ARM_64,
            # The optimizer runs SLSQP over a covariance matrix; 15s would be
            # tight and a timeout mid-cycle is the one failure that leaves the
            # book in an unknown state. At 1024MB, this same computation
            # (plus rendering every dashboard route and all 28 symbols'
            # research pages) measured over 300s and hit the timeout outright
            # — Lambda's CPU scales with memory, and 1024MB just isn't enough
            # of it for a real ~28-symbol backtest, not only a synthetic one.
            # With Gemini wired in, the fundamental agent alone makes one real
            # network call per equity symbol per cycle — roughly 23 x 21 =
            # ~480 calls, plus ~21 for macro — so the ceiling is raised to
            # Lambda's absolute maximum; actual billed duration is unaffected
            # by how high this is set.
            timeout=Duration.minutes(15),
            memory_size=3008,
            environment={
                "STATE_TABLE": table.table_name,
                # Default to the null provider: a deploy with no API key must
                # run the full pipeline rather than fail. Genuinely "gemini"
                # only when a key is actually wired in below — never claim a
                # provider that src/api/handler.py isn't using.
                "LLM_PROVIDER": "gemini" if GEMINI_API_KEY else "null",
                "EXECUTOR": "simulated_spread",
                **({"GEMINI_API_KEY": GEMINI_API_KEY} if GEMINI_API_KEY else {}),
            },
            log_retention=logs.RetentionDays.ONE_MONTH,
        )
        table.grant_read_write_data(cycle_function)

        events.Rule(
            self,
            "CycleSchedule",
            schedule=CYCLE_SCHEDULE,
            targets=[targets.LambdaFunction(cycle_function)],
            description="Runs one decision cycle after the US equity close",
        )

        # --- Read-only API ----------------------------------------------
        api_function = lambda_.DockerImageFunction(
            self,
            "ApiFunction",
            code=lambda_.DockerImageCode.from_image_asset(
                LAMBDA_IMAGE_DIR, cmd=["src.api.handler.api_handler"]
            ),
            architecture=lambda_.Architecture.ARM_64,
            # A cold container with nothing in the table yet does one full
            # backtest over the real 3-year, 28-symbol recording to answer its
            # own request — measured at ~88s on AWS at 3008MB, not the ~36s
            # seen locally, so this only has ~2s of margin against the
            # timeout below. That request also always 504s at the CloudFront
            # layer regardless (its 60s origin timeout is under AWS's own
            # hard cap without a support request) even though the Lambda
            # itself succeeds and persists a snapshot — priming the cache via
            # a direct `aws lambda invoke` of CycleFunction avoids exposing
            # that one request to CloudFront's timeout at all.
            timeout=Duration.seconds(90),
            memory_size=3008,
            environment={"STATE_TABLE": table.table_name},
            log_retention=logs.RetentionDays.ONE_MONTH,
        )
        table.grant_read_data(api_function)
        # The one exception to "read only": app_from_environment() persists a
        # snapshot (and every symbol's research page) after computing one
        # live, so the *next* request — on any container — reads instead of
        # recomputing, rather than every request paying the backtest cost
        # until the next scheduled cycle. Scoped to exactly the items this
        # path can produce — the single snapshot item, and one per-symbol
        # research item — not grant_write_data's whole-table access, so a bug
        # in a request handler still cannot touch a recorded cycle or
        # mandate — the property this table's whole design exists to keep.
        api_function.add_to_role_policy(
            iam.PolicyStatement(
                actions=["dynamodb:PutItem"],
                resources=[table.table_arn],
                conditions={
                    "ForAllValues:StringLike": {
                        "dynamodb:LeadingKeys": [SNAPSHOT_KEY, f"{RESEARCH_PREFIX}*"]
                    }
                },
            )
        )

        api_url = api_function.add_function_url(
            auth_type=lambda_.FunctionUrlAuthType.NONE,
            cors=lambda_.FunctionUrlCorsOptions(allowed_origins=["*"], allowed_methods=[lambda_.HttpMethod.GET]),
        )

        # --- Static dashboard -------------------------------------------
        bucket = s3.Bucket(
            self,
            "DashboardBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        distribution = cloudfront.Distribution(
            self,
            "DashboardDistribution",
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
            ),
            # The dashboard calls relative /api/* paths (same code path as the
            # Vite dev proxy), so CloudFront — not the browser — has to route
            # those to the API Lambda. Without this, /api/* falls through to
            # the S3 origin and 403s.
            additional_behaviors={
                "/api/*": cloudfront.BehaviorOptions(
                    origin=origins.FunctionUrlOrigin(api_url),
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_GET_HEAD_OPTIONS,
                    cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                ),
            },
            default_root_object="index.html",
            # A single-page app serves its own routes; a 404 from S3 is a deep
            # link, not a missing page.
            error_responses=[
                cloudfront.ErrorResponse(
                    http_status=404, response_http_status=200, response_page_path="/index.html"
                )
            ],
            comment="Educational paper-trading simulation. Not investment advice.",
        )

        s3_deploy.BucketDeployment(
            self,
            "DeployDashboard",
            sources=[s3_deploy.Source.asset("../web/dist")],
            destination_bucket=bucket,
            distribution=distribution,
            distribution_paths=["/*"],
        )

        self.api_url = api_url.url
        self.dashboard_url = distribution.distribution_domain_name

        # Plain attributes above never surfaced anywhere — cdk deploy printed
        # nothing and the console showed no Outputs tab. CfnOutput is what
        # actually makes these show up after a deploy.
        CfnOutput(self, "ApiUrl", value=self.api_url)
        CfnOutput(self, "DashboardUrl", value=f"https://{self.dashboard_url}")
