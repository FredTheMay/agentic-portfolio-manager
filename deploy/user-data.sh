#!/usr/bin/env bash
# EC2 user-data: provision a fresh Ubuntu 22.04/24.04 instance.
#
# Installs the runtime and prepares the layout. It does NOT fetch the
# application — deploy/deploy.sh pushes that from your machine, so the instance
# never needs git credentials or API keys to build.
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3-venv python3-pip nginx rsync

# Unprivileged service account with no login shell.
id -u appuser >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin appuser

install -d -o appuser -g appuser /opt/agentic-portfolio
install -d -o appuser -g appuser /opt/agentic-portfolio/data
install -d -o appuser -g appuser /opt/agentic-portfolio/web

# Credentials file: root-owned, readable only by the service account.
touch /etc/agentic-portfolio.env
chown root:appuser /etc/agentic-portfolio.env
chmod 640 /etc/agentic-portfolio.env

rm -f /etc/nginx/sites-enabled/default
systemctl enable nginx

echo "Provisioned. Now run deploy/deploy.sh from your workstation."
