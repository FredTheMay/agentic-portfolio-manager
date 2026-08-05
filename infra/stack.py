"""AWS infrastructure (SPEC §10, M10).

    Lambda for the scheduled cycle, EventBridge cron, DynamoDB, S3 + CloudFront.

Two properties this stack is built around.

**The schedule fires at least once, not exactly once.** EventBridge can deliver
a duplicate, and a Lambda retry after a timeout is indistinguishable from a
fresh invocation. The system already handles this three ways — content-hashed
mandate ids (M4), broker-side client_order_id rejection (M8), and put_item on
the mandate id here — because none of them is sufficient alone.

**Nothing in this stack can place a real trade.** There is no live broker
endpoint anywhere in the codebase (SPEC §1), and the cycle function is granted
no secret that would let it reach one.
"""

from __future__ import annotations

from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    aws_cloudfront as cloudfront,
    aws_cloudfront_origins as origins,
    aws_dynamodb as dynamodb,
    aws_events as events,
    aws_events_targets as targets,
    aws_lambda as lambda_,
    aws_logs as logs,
    aws_s3 as s3,
    aws_s3_deployment as s3_deploy,
)
from constructs import Construct

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
        cycle_function = lambda_.Function(
            self,
            "CycleFunction",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="src.api.handler.scheduled_cycle",
            code=lambda_.Code.from_asset("../"),
            # The optimizer runs SLSQP over a covariance matrix; 15s would be
            # tight and a timeout mid-cycle is the one failure that leaves the
            # book in an unknown state.
            timeout=Duration.minutes(5),
            memory_size=1024,
            environment={
                "STATE_TABLE": table.table_name,
                # Default to the null provider: a deploy with no API key must
                # run the full pipeline rather than fail (SPEC §2.1).
                "LLM_PROVIDER": "null",
                "EXECUTOR": "simulated_spread",
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
        api_function = lambda_.Function(
            self,
            "ApiFunction",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="src.api.handler.api_handler",
            code=lambda_.Code.from_asset("../"),
            timeout=Duration.seconds(30),
            memory_size=1024,
            environment={"STATE_TABLE": table.table_name},
            log_retention=logs.RetentionDays.ONE_MONTH,
        )
        # Read only. The API must not be able to write state, so that a bug in
        # a request handler cannot alter a recorded decision.
        table.grant_read_data(api_function)

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
