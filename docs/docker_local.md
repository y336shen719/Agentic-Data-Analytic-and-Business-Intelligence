# Local Docker Compose validation

These files validate the existing local AMD64 image, not a production deployment.
There is no build step or registry pull in Compose. The database remains on the
host and is mounted read-only. If it is missing, Compose must not create a
directory in its place. Only the Mac loopback address publishes port 8501.

## Prepare before starting

- Build and load `agentic-data-analyst:local` for `linux/amd64`.
- Ensure `data/ecommerce.duckdb` is a prepared database file.
- Create a private directory outside the repository/build context, for example
  `$HOME/.config/agentic-data-analyst/secrets`.
- Provide separate plain UTF-8 files named `openai_api_key` and
  `app_access_password`. Each contains only its value, not `NAME=value` syntax.
  Provide `langsmith_api_key` only when using the tracing overlay.
- Enter real credentials privately; never put values in shell commands, Git,
  logs, or chat. This repository does not create or copy real credentials.
- Restrict access to the host directory and files. The non-root container UID/GID
  is `10001:10001`; verify readability from that identity before starting the UI.
  Docker Desktop file sharing and native Linux bind mounts have different
  ownership behavior. Do not assume host mode 0600 is readable by UID 10001, or
  make files world-readable to work around it. File-backed Compose secrets do
  not provide automatic ownership remapping through secret uid/gid/mode fields.

Set `ADA_SECRETS_DIR` to the absolute external directory when invoking Compose.
Use `--env-file /dev/null` so Compose does not load the project's development
`.env`. The application gets secret paths, not the secret values, in its
environment. The inherited image health check checks the Streamlit server only.

## Start after configuration and permissions have been reviewed

Without tracing:

```sh
ADA_SECRETS_DIR="$HOME/.config/agentic-data-analyst/secrets" docker compose --env-file /dev/null -f compose.local.yaml up -d --wait
```

With tracing (the second file adds its secret to the base service's secrets):

```sh
ADA_SECRETS_DIR="$HOME/.config/agentic-data-analyst/secrets" docker compose --env-file /dev/null -f compose.local.yaml -f compose.local.langsmith.yaml up -d --wait
```

Open `http://127.0.0.1:8501` after startup. Verify password login, database reads,
one OpenAI-backed question, and, if enabled, trace delivery in LangSmith. A healthy
container alone does not validate credentials or database queries. OpenAI calls
incur API usage. Container replacement preserves the host database and secrets;
in-memory sessions are transient. Switching back from tracing requires recreating
the container with only the base file, not merely restarting the old container.

## Validation record

The following results were reported by the user during local Docker Desktop
validation on an Apple Silicon Mac using the Linux AMD64 image:

- The image built successfully, including pinned wheel installation and
  `pip check`. Image inspection confirmed `linux/amd64` and user `appuser`.
- A one-off container confirmed UID 10001, readable fake OpenAI/password secret
  files, no corresponding value environment variables, and a read-only DuckDB
  connection executing `SELECT 1`.
- The fake-credential service became healthy. Password login and Database
  Explorer table/row previews worked.
- A one-off container read all three real credential files without printing
  values and confirmed the corresponding value environment variables were absent.
- The service was recreated with real credentials and the LangSmith overlay;
  the user reported successful end-to-end validation. Cloud trace delivery was
  not independently inspected by the assistant.

The full Pytest suite has not run inside the Linux container. Mac mount results
do not validate native Linux UID/permission behavior on the GCP VM. Production
deployment, registry publication, CI/CD, and rollback remain pending.

The diagnostic secret is intentionally absent from local tests. The production
Compose configuration must include it through `DIAGNOSTIC_PASSWORD_FILE`.
