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

No hosting platform or campus identity provider is configured in this
repository. The following self-contained Docker Compose deployment prepares a
TLS-only Nginx Basic Auth gateway. It is a deployment template, not an active
service. It packages the Streamlit source and `.streamlit/config.toml` while
excluding local datasets, checkpoints, results, virtual environments, and
deployment secrets.

```text
Authorized evaluator/scanner
          |
          | HTTPS + named Basic Auth account
          v
Campus edge / approved network policy (if used)
          |
          v
Nginx TLS + Basic Auth :8443  [only published listener]
          |
          | same container/network namespace
          v
Streamlit 127.0.0.1:8501     [not published]
```

Nginx applies authentication to the complete application location, including
health and Streamlit WebSocket upgrade requests. It forwards WebSocket upgrade
headers only after Basic Auth succeeds, strips the Authorization and common
identity headers before proxying, and overwrites forwarded host/protocol/IP
headers rather than trusting client-supplied values. There is no HTTP listener.
The deployment uses a loopback-only host binding by default. Basic Auth does
not provide MFA; use named, unique, temporary evaluator accounts, an approved
VPN/private network, and any required campus identity-aware edge gateway.

### Prepare external configuration and secrets

1. Use a disposable host and synthetic data only. The image copies
   `configs/default.yaml` with `csv_path: null`; it does not copy `data/` or
   `models/`.
2. Copy `.env.example` to `.env`. Set `PUBLIC_HOST` to the DNS name covered by
   the TLS certificate. Set `BASIC_AUTH_FILE`, `TLS_CERT_FILE`, and
   `TLS_KEY_FILE` to protected files outside source control. `.env` and
   `deploy/secrets/` are ignored by Git. The example contains paths and
   placeholders only, never passwords or tokens.
3. Generate one named account per evaluator using Apache `htpasswd` with
   bcrypt (`htpasswd -B -c <protected-file> <unique-user>`); it prompts for
   the password without putting it in the command line. Add later users
   without `-c`. Keep the resulting file in the platform secret manager or a
   host-only directory readable by container UID 10001, not in the repository.
   Never log or share the plaintext password.
4. Provision a valid certificate chain and matching private key from the
   approved campus PKI or certificate manager. Do not use a self-signed
   certificate for a remote evaluation. Keep the private key readable only by
   the deployment service account/container UID 10001.
5. In `.env`, leave `GATEWAY_BIND_ADDRESS=127.0.0.1` when a trusted edge
   gateway runs on the same host. If a separate approved gateway must connect,
   bind only to a private interface and have the infrastructure owner restrict
   the gateway port to that gateway's source address. Do not bind to `0.0.0.0`
   as a shortcut. Never publish or forward port 8501.

Docker Compose requires all three secret files and `PUBLIC_HOST`; missing
values/files or an invalid TLS configuration prevent startup. No credentials
are accepted through environment variables. The container runs as non-root,
with a read-only filesystem, dropped Linux capabilities, an internal Docker
network, and only the gateway port published. Compose file-backed secrets are
read-only mounts, not a substitute for the cloud/platform secret manager.

### Start and validate the deployment

From the project directory, after provisioning the files:

```bash
docker compose --env-file .env config --quiet
docker compose --env-file .env up --build -d
docker compose --env-file .env ps
```

The default host binding is `127.0.0.1:8443`. A same-host campus edge proxy
can forward to that HTTPS listener. For a remote gateway, the deployment
owner must select a private bind address and enforce the corresponding
network/firewall allowlist outside this repository. Do not expose the gateway
or origin to the Internet while setting up or validating access. The only
container-published port is the TLS gateway; the Streamlit process binds to
`127.0.0.1:8501` in the same container and is never published.
Use a currently supported, patched Docker Engine (28.0.0 or newer) and verify
loopback-published-port behavior on the actual host before relying on it for
network isolation.

For live verification, run `python scripts/verify_gateway.py` from an
authorized test client and separately from an unauthorized network. Provide
these variables through the test client's secret manager (not a checked-in
file or command-line argument):

- `OPTIMA_GATEWAY_URL`: the HTTPS gateway origin, with a publicly/privately
  trusted certificate.
- `OPTIMA_TEST_USERNAME` and `OPTIMA_TEST_PASSWORD`: a temporary test account.
- `OPTIMA_ORIGIN_URL`: the host's direct `http://...:8501/` origin target,
  tested from outside the deployment host/network.

The verifier checks unauthenticated and authenticated `GET /`, the application
health route, invalid credentials, WebSocket upgrade responses (401 without
credentials and 101 with valid credentials), TLS 1.2+, cleartext HTTP refusal
or HTTPS redirection, and direct-origin TCP reachability. It does not print
credentials. A successful localhost-only test does not prove remote network
isolation; run the direct-origin and authorized-scanner checks from the actual
evaluation networks. `pytest tests/test_deployment_security.py` validates
configuration structure without starting containers or opening ports.

### Campus identity provider, operations, and cleanup

The project does not identify a campus IdP or deploy-time trust boundary, so
no automatic OIDC/SAML integration is claimed. If campus SSO is required,
place the campus-approved identity-aware gateway in front of this TLS listener,
keep the Basic Auth control enabled unless a separately reviewed authentication
integration replaces it, and verify that the edge also authenticates WebSocket
upgrades. Do not pass client-supplied identity headers as proof of identity.
The upstream gateway must be allowed to reach only the private/loopback Nginx
gateway; the Streamlit origin remains unpublished.

Rotate accounts by replacing the external htpasswd file with newly generated
hashes, then recreate the deployment through the approved change process.
After evaluation, remove temporary accounts and scanner allowlists, revoke
temporary credentials/certificates, stop the test deployment, and remove
temporary data according to the host retention policy. Restrict access to
deployment logs and host configuration.

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
