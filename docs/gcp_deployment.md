# Single-VM production Compose deployment

This is a runbook for a later, explicitly initiated VM deployment. Creating these
files does not provision paths, credentials, firewall rules, or containers. The
initial target is `free-trial-vm`, zone `us-central1-a`, Ubuntu 24.04 / AMD64.
Docker is already installed; do not reinstall it as part of this procedure.

## Inputs and layout

- `deploy/compose.prod.yaml`: versioned runtime rules; no `build` section.
- `deploy/production.env.example`: versioned template containing ordinary values.
- `production.env`: actual VM parameters, supplied with Compose `--env-file`.
- Database and four secret files: provisioned independently on the persistent disk.

Suggested layout under the confirmed VM workspace (not yet created):

```text
/home/yimingshen20000719_gmail_com/agent-lab/
  deploy/
    compose.prod.yaml
    production.env
    production.env.previous      # preserve before replacing a working version
  runtime/
    data/ecommerce.duckdb
    secrets/
      openai_api_key
      langsmith_api_key
      app_access_password
      diagnostic_password
```

Copy the template to `production.env` and fill in the actual absolute paths. The
initial APP_IMAGE is the user-verified release from main commit
`f7f1dd1673935e55346d72c063dc38ea1f6556f3`. Keep the full `@sha256:...` reference;
do not replace it with `latest` or a mutable tag. YAML requires a nonempty image
parameter but does not itself validate digest syntax: review it before commands
and add enforced validation when implementing the automated deployment script.

The env file is for Compose interpolation, not an application `env_file`. Only
the explicitly declared service environment enters the container. No real keys,
passwords, database, or Docker login credentials belong in Git or image layers.
Actual production env files and backups are ignored by Git and Docker; the env
example remains tracked by Git.

## Provisioning and permissions checkpoint

Provision the prepared DuckDB file independently, without executing data
preparation in the running application container. Stop any database writer and
close/checkpoint its connection before copying; validate the destination file
with a read-only connection and compare a transfer checksum. The application
mount is read-only and a missing source must not become a directory. Database
replacement, backup, and restore are separate from image rollout/rollback.

Each secret file is plain UTF-8 containing just its value, not `NAME=value`.
The diagnostic password is included in production even though it was not tested
in the local Compose setup. Supply it privately, independently of login password.
The container user is UID/GID `10001:10001`. Mac tests do not prove Linux access.

On a normal rootful Docker installation, one possible secret layout is a
root-owned host directory with mode `0700`, and individual files owned by
`root:10001` with mode `0440`. Docker's daemon mounts the individual files, so
UID 10001 can read through group permissions inside the container without other
ordinary host users being able to traverse the source directory. Verify the
actual daemon mode and access from inside the container before relying on this
layout; rootless/user-namespace mappings may differ. Do not make secrets
world-readable. File-backed Compose secret uid/gid/mode fields do not remap the
source file's ownership. Docker administrators and root remain trusted.

The database also needs read permission for UID/GID 10001. Do not recursively
change ownership of the checkout or expose real values while checking access.

## Registry access and initial verification

The package may be private. Decide public visibility versus restricted read
access before pulling. A workflow's `GITHUB_TOKEN` is not a permanent VM login.
Do not change visibility or create a token without deciding that access policy.
If private access is used, provision minimum-scope registry credentials separately
and enter them through a non-echoing/password-stdin login flow, not CLI arguments.

Run the following only on the VM after inputs, permissions, and registry access
have been verified. The commands assume the suggested layout and working
directory `~/agent-lab`. Always use the production env explicitly; avoid loading
a development `.env`. Exported shell variables can override env-file values, so
remove stale APP_IMAGE/ADA_DB_FILE/ADA_SECRETS_DIR exports before proceeding.

Validate the resolved model without starting anything:

```sh
docker compose --env-file deploy/production.env -f deploy/compose.prod.yaml config --quiet
```

Check the resolved image with `config --images`; confirm its digest against the
successful GHCR workflow summary. Pulling the image does not replace a running
container:

```sh
docker compose --env-file deploy/production.env -f deploy/compose.prod.yaml pull app
```

Before starting Streamlit, use a one-off non-root container to verify all four
secret files are nonempty/readable without printing values, and open the database
read-only. Check that the database mount is read-only. Do not call external APIs
as part of these permission checks. This preflight is still to be implemented
and executed in the VM stage.

## Start and verify

After the preflight checkpoint, start the pulled image:

```sh
docker compose --env-file deploy/production.env -f deploy/compose.prod.yaml up -d --wait --wait-timeout 180 --pull never
```

This uses the pulled image and inherited web health check, with bounded waiting.
`unless-stopped` restarts an exited process and restores eligible containers when
Docker starts. It does not repair an unhealthy-but-running process, start a stopped
VM, or make a single VM highly available. Confirm the Docker service is enabled
at VM boot. Do not assume Compose health alone validates credentials or queries.

The only published address is VM loopback `127.0.0.1:8501`. Use a separately
verified SSH/IAP tunnel for initial browser checks; no public firewall changes
are part of this runbook. Public HTTPS access is a separate future decision.

Verify container status/health, application login, database preview and an actual
read query, a small OpenAI-backed question, LangSmith trace delivery, and protected
diagnostic access. Live API validation incurs API usage. Logs are available via
`docker compose ... logs --tail 100 app`; avoid exposing sensitive diagnostic data.
The json-file log driver rotates at 10 MB with three files per container. No
container memory limit is imposed merely to fit the current 1 GB VM; measure
host/worker peak memory and OOM behavior before deciding whether to resize.

## Updates and rollback

Before changing a working deployment, preserve its resolved image digest and env
file as `production.env.previous`; keep its registry image and local cache. For
changes to runtime rules also record the working Compose file's Git commit and
preserve its version. Do not overwrite the previous known-good record with an
unverified release. Use one deployment operator at a time until a locking script
is added. Expect a brief interruption and loss of in-memory sessions on this
single-container setup.

Update only APP_IMAGE for an application-only release, pull it, run preflight,
then repeat the bounded start and functional verification. Secrets and database
remain mounted from the host. On failure, use the previous env file and, if
changed, the matching previous Compose rules:

```sh
docker compose --env-file deploy/production.env.previous -f deploy/compose.prod.yaml up -d --wait --wait-timeout 180 --pull never
```

Use that command only with the known-good image still cached and compatible
Compose rules restored. If missing locally, pull the exact previous digest first
after authenticating. Recheck health and functionality after rollback. Once
restored, make the active production.env match the rollback version. This restores
application image/configuration, not changed database contents or secret values.
Do not prune rollback images as part of deployment.

These are manual procedures, not automatic rollback or CD. IAP/IAM setup,
short-lived GitHub-to-GCP authentication, automated preflight/rollout/rollback,
and actual VM validation are subsequent steps. None are performed by this change.
