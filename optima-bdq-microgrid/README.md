# Team OPTIMA - BDQ-Based Microgrid Battery Scheduling

Online scheduling of several distributed batteries (BESSs) in a campus microgrid with a
**Branching Dueling Q-Network (BDQ)**. Based on Shuai et al., *IEEE Trans. Smart Grid*, 2021.

Network outputs grow **linearly (N x n)** instead of exponentially (n^N): 25 vs 3,125 outputs for N = 5, n = 5.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python scripts/check_quality.py    # dependency audit, tests, compilation, diff check
python scripts/train.py             # ~3000 episodes, saves models/bdq_best.pt
python scripts/evaluate.py          # BDQ vs myopic vs price-threshold on test days
python scripts/scale_test.py        # N = 1, 3, 5, 7 -> outputs, time, improvement
streamlit run app/dashboard.py      # local demo only; not an authenticated deployment
```
Fast smoke run: `python scripts/train.py --episodes 200`.

`requirements.txt` contains application and command-line runtime dependencies.
`requirements-dev.txt` includes those dependencies plus pytest and pip-audit.
Use a project virtual environment rather than changing a shared Python
installation. `python scripts/check_quality.py` expects the development
requirements to be installed and runs the dependency audit, tests, compilation,
and Git whitespace check.

The requirements files intentionally keep compatible version ranges rather
than claiming to be an exact lock. For repeatable releases, generate and review
hash-checked, platform/Python-specific locks from these inputs with a resolver
such as pip-tools (for example, `pip-compile --generate-hashes` in a clean
environment for each supported Python/platform combination). Regenerate and
test those locks as part of dependency updates; do not generate a release lock
from an arbitrary developer's already-populated environment.

## Controlled security evaluation deployment

The project does not configure a hosting platform or campus identity provider
automatically. Choose the deployment-specific path below. Both templates keep
mandatory Nginx Basic Auth in front of every application route and WebSocket.
Basic Auth does not provide MFA; use unique temporary evaluator accounts and
any required campus VPN, private network, or identity-aware edge control.

### Render Web Service (Docker)

Render terminates HTTPS at its edge, redirects inbound HTTP to HTTPS, and
forwards requests to the web service over HTTP. Render requires the web-facing
container process to listen on `0.0.0.0:$PORT` (default `10000`). In this
configuration that process is Nginx; Streamlit remains bound exclusively to
`127.0.0.1:8501` inside the same container and is not published as a separate
port.

```text
Evaluator -- HTTPS --> Render TLS edge (HTTP redirected to HTTPS)
                            |
                            | Render-assigned PORT
                            v
                   Nginx Basic Auth (all paths)
                            |
                            | authenticated HTTP/WebSocket proxy
                            v
                   Streamlit 127.0.0.1:8501
```

Configure Render Dashboard > **New Web Service** as follows:

1. Select the repository and set **Root Directory** to
   `optima-bdq-microgrid` (the application directory in this repository).
2. Select **Docker** and set **Dockerfile Path** to
   `deploy/Dockerfile.render`. Do not use `deploy/Dockerfile`, which is the
   separate self-hosted Compose image.
3. Do not set or override `PORT`; Render provides it (normally `10000`). The
   entrypoint accepts Render's port and uses `10000` only as a local fallback.
4. Under **Environment**, set:

   | Key | Value |
   |---|---|
   | `BASIC_AUTH_FILE` | `/etc/secrets/optima-htpasswd` |

5. Under **Secret Files**, upload a file named exactly
   `optima-htpasswd`. Render mounts it in a Docker service at
   `/etc/secrets/optima-htpasswd`. Its contents must be Apache `htpasswd`
   bcrypt entries; do not enter a plaintext password into the file.
6. Do not configure a public health-check path that bypasses authentication.
   The Render entrypoint checks Streamlit's health endpoint locally before
   starting Nginx; external HTTP health checks remain protected by Basic Auth.

Generate the file locally using Apache `htpasswd` without printing its
contents, for example:

```bash
htpasswd -B -c /secure/path/optima-htpasswd evaluator-name
```

The command prompts for the password. Add other named accounts with
`htpasswd -B /secure/path/optima-htpasswd another-evaluator` (without `-c`).
Upload the resulting file via Render's Secret Files interface. Store the file
and generated passwords in an approved secret manager; never put them in Git,
build arguments, logs, or command-line arguments. Render documents that Docker
Secret Files are available at `/etc/secrets/<filename>` and recommends adding
the application user to group ID `1000` to read them; the Render Dockerfile
adds this supplementary group and otherwise runs as non-root UID `10001`.

The Render Nginx template has a single default/catch-all server and one
authenticated `location /`; it does not create an unauthenticated health,
static, or WebSocket route. It forwards WebSocket upgrade headers, replaces
forwarded host/IP/protocol values, and clears client-supplied authorization
and identity headers. `envsubst` substitutes only `${PORT}` and
`${BASIC_AUTH_FILE}`, preserving Nginx variables such as `$host`,
`$remote_addr`, `$http_upgrade`, and `$connection_upgrade`. The service uses
Render's HTTPS edge and does not require container TLS files. Render handles
HTTP-to-HTTPS redirection at the public edge; the Nginx response includes
HSTS and other security headers.

`deploy/render.env.example` records the required Render setting without
credentials. `deploy/Dockerfile.render` and the Render-specific entrypoint and
Nginx template are separate from the self-hosted Compose deployment. The
Docker build context excludes local CSVs, model checkpoints, results,
virtual environments, caches, and secret files.

### Render verification

After an authorized Render deploy, run from a test client on the authorized
network and again from an unauthorized network:

```bash
python -m pytest tests/test_deployment_security.py -q --tb=short
python -m py_compile scripts/verify_gateway.py
python scripts/verify_gateway.py
```

Provide these values to the verifier process through the test client's secret
manager, not source files or command-line arguments:

- `OPTIMA_GATEWAY_URL`: the HTTPS Render service URL (for example
  `https://<service>.onrender.com`).
- `OPTIMA_TEST_USERNAME` and `OPTIMA_TEST_PASSWORD`: temporary evaluator
  credentials.
- `OPTIMA_ORIGIN_URL`: a direct-origin probe target using port `8501`, tested
  from outside the Render service. Confirm the direct port is unreachable;
  the public Render URL alone proves only that the edge is reachable.

The verifier checks unauthenticated and authenticated `GET /` and
`/_stcore/health`, invalid credentials, unauthenticated and authenticated
WebSocket upgrades (401 and 101 respectively), TLS 1.2+, Render's cleartext
HTTP redirect/refusal, and direct-origin TCP reachability. It never prints
credentials. A successful run must be performed against the actual deployed
service from the relevant networks; static tests do not activate or prove
authentication. Do not change Render settings or deploy solely as part of a
local code review.

Rotate accounts by replacing the Secret File with newly generated bcrypt
entries and redeploying through Render. After evaluation, remove temporary
accounts and scanner/network access, rotate or revoke temporary credentials,
and retire the test service according to the organization's retention policy.

### Self-hosted Docker Compose (TLS at Nginx)

For deployments that terminate TLS in the container, use `compose.yaml` with
`deploy/Dockerfile` instead. This separate configuration requires `PUBLIC_HOST`
and external files for `BASIC_AUTH_FILE`, `TLS_CERT_FILE`, and `TLS_KEY_FILE`.
Copy `.env.example` to `.env`, provision the files outside source control, and
keep the default `GATEWAY_BIND_ADDRESS=127.0.0.1` unless an infrastructure
owner configures a private interface and allowlist. It publishes only the
gateway port; the existing Compose health check probes Streamlit locally.

```bash
docker compose --env-file .env config --quiet
docker compose --env-file .env up --build -d
docker compose --env-file .env ps
```

For this self-hosted deployment, use `scripts/verify_gateway.py` from an
authorized test client with `OPTIMA_GATEWAY_URL` set to the HTTPS gateway,
temporary `OPTIMA_TEST_USERNAME` and `OPTIMA_TEST_PASSWORD` supplied through
the test client's secret manager, and `OPTIMA_ORIGIN_URL` set to a direct
port-8501 probe from outside the deployment host. This verifies authenticated
HTTP/WebSocket access, TLS, invalid credentials, and origin reachability.

This self-hosted path requires a currently supported patched Docker Engine.
It is not the Render configuration and must not be used to pass container TLS
certificates to Render.

## Layout
| Path | Purpose |
|---|---|
| `configs/default.yaml` | All system, data and training parameters |
| `src/optima/data.py` | CSV loading/validation, synthetic generator, chronological day splits |
| `src/optima/dispatch.py` | Single-step economic dispatch LP (DG, grid, curtailment, shed) + heuristic fallback |
| `src/optima/env.py` | Gym-style simulator, N BESSs, SoC/power clipping, reward = -cost |
| `src/optima/bdq.py` | 3 LSTMs + FC trunk, state value V(s), one advantage branch per BESS |
| `src/optima/replay.py` | Prioritized replay |
| `src/optima/agent.py` | eps-greedy acting, shared double-Q TD target (Eq. 2-3), checkpointing |
| `src/optima/trainer.py` | Algorithm 1 training loop, validation-based model selection |
| `src/optima/baselines.py` | Myopic (mandatory) and price-threshold baselines |
| `src/optima/evaluate.py` | Rollouts, % improvement vs myopic, mean/min/max/std |
| `scripts/` | `train.py`, `evaluate.py`, `scale_test.py` |
| `app/dashboard.py` | Streamlit + Plotly: SoC, price vs action, daily cost, comparison |
| `tests/` | Data, dispatch balance, env limits, BDQ shapes / linear growth |

## Design notes
- Agent chooses only the battery levels; the LP dispatch sets DG, grid exchange and curtailment.
- Safety: actions are clipped to power and SoC limits; the test suite asserts zero SoC violations.
  If the LP fails, a rule-based fallback is used and counted (`fallbacks`).
- Myopic baseline = minimise current-step cost only (no value for stored energy).
  A terminal SoC value (mean train price) is applied identically to every policy.
- No forecasts are used: observations are current values plus H hours of history.

## Limitations (report honestly)
- Simplified energy-balance dispatch, no voltage/branch constraints; hourly steps; linear fuel cost.
- Synthetic data by default, so numbers will not match the paper (target: 3-5% over myopic with real data).
  The myopic baseline here is weak, so even the price-threshold rule beats it; always compare BDQ against
  price-threshold too and report any day where the agent is worse (`worse_days`).
- The operator keeps final authority; the dashboard is decision support only.

## 36-hour plan
1. H0-6: data pipeline, env, dispatch, baselines, tests. 2. H6-18: BDQ + training, tune n, reward scale.
3. H18-26: evaluation + scale test. 4. H26-32: dashboard, pre-trained model for a safe demo. 5. H32-36: slides, rehearsal.
