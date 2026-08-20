#!/usr/bin/env bash
# Push the application to an EC2 instance.
#
#   HOST=ubuntu@1.2.3.4 KEY=~/.ssh/key.pem deploy/deploy.sh
#
# Ships the built front end, the source, and the recorded data cache. It does
# NOT ship .env: credentials are only needed to refresh the cache, and the API
# replays offline. Put them on the box by hand if you want the cron refresh.
set -euo pipefail

HOST="${HOST:?set HOST=ubuntu@<ip>}"
KEY="${KEY:?set KEY=path/to/key.pem}"
REMOTE=/opt/agentic-portfolio
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$HOST")

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

echo "==> building the front end"
(cd web && npm ci --silent && npm run build)

if [ ! -d data/cache ]; then
  echo "!! data/cache is empty — run 'make backfill' first, or the dashboard"
  echo "   will fall back to synthetic data and say so."
fi

echo "==> shipping to $HOST"
# --delete keeps the remote a mirror; without it a renamed asset lingers and
# the bundle silently grows forever.
rsync -az --delete -e "ssh -i $KEY" \
  --exclude '__pycache__' --exclude '*.pyc' \
  src/ "$HOST:/tmp/app-src/"
rsync -az --delete -e "ssh -i $KEY" config/ "$HOST:/tmp/app-config/"
rsync -az --delete -e "ssh -i $KEY" web/dist/ "$HOST:/tmp/app-web/"
rsync -az -e "ssh -i $KEY" pyproject.toml "$HOST:/tmp/pyproject.toml"
if [ -d data/cache ]; then
  rsync -az -e "ssh -i $KEY" data/cache/ "$HOST:/tmp/app-cache/"
fi
rsync -az -e "ssh -i $KEY" deploy/nginx.conf deploy/agentic-portfolio.service "$HOST:/tmp/"

echo "==> installing"
"${SSH[@]}" 'sudo bash -s' <<'REMOTE_SCRIPT'
set -euxo pipefail
REMOTE=/opt/agentic-portfolio

rsync -a --delete /tmp/app-src/    "$REMOTE/src/"
rsync -a --delete /tmp/app-config/ "$REMOTE/config/"
rsync -a --delete /tmp/app-web/    "$REMOTE/web/"
install -m 644 /tmp/pyproject.toml "$REMOTE/pyproject.toml"
mkdir -p "$REMOTE/data/cache"
[ -d /tmp/app-cache ] && rsync -a /tmp/app-cache/ "$REMOTE/data/cache/"

# The venv is rebuilt only when it does not exist; dependencies change rarely
# and a rebuild on every deploy would dominate deploy time.
if [ ! -d "$REMOTE/.venv" ]; then
  python3 -m venv "$REMOTE/.venv"
  "$REMOTE/.venv/bin/pip" install --upgrade pip wheel
  "$REMOTE/.venv/bin/pip" install numpy scipy pandas scikit-learn pydantic \
      fastapi httpx pyyaml uvicorn
fi

chown -R appuser:appuser "$REMOTE"

install -m 644 /tmp/nginx.conf /etc/nginx/sites-available/agentic-portfolio
ln -sf /etc/nginx/sites-available/agentic-portfolio /etc/nginx/sites-enabled/agentic-portfolio
rm -f /etc/nginx/sites-enabled/default
nginx -t

install -m 644 /tmp/agentic-portfolio.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now agentic-portfolio
systemctl restart agentic-portfolio
systemctl reload nginx

# The first request runs a backtest; wait for it before declaring success.
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    echo "API healthy"; break
  fi
  sleep 5
done
REMOTE_SCRIPT

echo "==> verifying"
"${SSH[@]}" 'curl -fsS localhost/api/status' | head -c 300
echo
echo "Deployed. Open http://${HOST#*@}/"
