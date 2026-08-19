# EC2 deployment (M11)

Single small instance: **nginx** serves the built React bundle and reverse-proxies `/api` to
**uvicorn** on localhost. No load balancer, no container registry, no Kubernetes — the app is
one read-only process over a recorded cache, and anything more would be theatre.

## Why this shape

**uvicorn binds `127.0.0.1`, not `0.0.0.0`.** nginx is the only thing that can reach it, so
the API is not directly exposed regardless of how the security group is configured. That is a
property of the unit file rather than of a firewall rule someone can widen by accident.

**Credentials are not deployed.** They are needed only to *refresh* the cache; the API replays
offline. `deploy.sh` ships `data/cache` and never `.env`. If you want the nightly refresh, put
them in `/etc/agentic-portfolio.env` by hand (root-owned, `0640`, group `appuser`).

**The systemd unit is locked down** — `ProtectSystem=strict`, `NoNewPrivileges`, a single
writable path, and a memory cap. The service reads a cache and serves JSON; it needs nothing else.

## Instance

| | |
|---|---|
| Type | `t3.small` (2 GB) — `t3.micro` works but the startup backtest is tight at 1 GB |
| AMI | Ubuntu 22.04 or 24.04 LTS |
| Storage | 16 GB gp3 |
| Security group | inbound **80** from anywhere, **22** from your IP only |
| Cost | roughly $15/month on-demand, less on a Spot or a savings plan |

The `MemoryMax=1400M` in the unit is sized for `t3.small`. Raise it if you use a bigger box.

## Steps

**1. Launch** with `deploy/user-data.sh` pasted into *Advanced details → User data*. That installs
Python, nginx and rsync, and creates the `appuser` service account. It deliberately does **not**
fetch the application, so the instance never needs git credentials.

**2. Record data locally** (skip and the dashboard runs on synthetic data and says so):

```bash
make backfill
```

**3. Deploy from your workstation:**

```bash
HOST=ubuntu@<public-ip> KEY=~/.ssh/your-key.pem deploy/deploy.sh
```

It builds the front end, ships source + config + `web/dist` + `data/cache`, creates the venv on
first run, installs the nginx site and the systemd unit, and waits for `/api/health` before
reporting success.

**4. Open** `http://<public-ip>/`.

## Refreshing the data

The cache is a point-in-time recording, so it goes stale rather than wrong. To refresh:

```bash
make backfill && HOST=... KEY=... deploy/deploy.sh
```

Or, to refresh on the box, put credentials in `/etc/agentic-portfolio.env` and add a cron entry:

```
30 22 * * 1-5 cd /opt/agentic-portfolio && .venv/bin/python scripts/backfill.py && systemctl restart agentic-portfolio
```

22:30 UTC on weekdays — after the US close, so the session's closing prices exist.

## HTTPS

Not configured, because it needs a domain name. With one pointed at the instance:

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your.domain
```

Certbot rewrites the server block and sets up renewal. Until then the site is HTTP, which is
acceptable for a read-only educational dashboard carrying no user data and no login.

## Troubleshooting

```bash
sudo systemctl status agentic-portfolio
sudo journalctl -u agentic-portfolio -n 100 --no-pager
curl -s localhost:8000/api/status | head -c 300     # bypass nginx
sudo nginx -t && sudo tail -50 /var/log/nginx/error.log
```

**A 504 on first load** is usually the startup backtest still running — it takes a couple of
minutes on a small instance. `proxy_read_timeout` is set to 180s for that reason.

**`/api/status` reporting synthetic data** means `data/cache` did not arrive or is unreadable.
Check `ls /opt/agentic-portfolio/data/cache` and the ownership (`appuser`).
