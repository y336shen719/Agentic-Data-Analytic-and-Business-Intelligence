# Runtime secrets

The application accepts runtime credentials without including them in source or
container images. Never commit real credentials or add them to a build context.

## Docker Compose

Configure separate files and grant them to the application service using Compose
secrets. Set these environment variables to the paths inside the container:

- `OPENAI_API_KEY_FILE`
- `LANGSMITH_API_KEY_FILE` (legacy alias: `LANGCHAIN_API_KEY_FILE`)
- `APP_ACCESS_PASSWORD_FILE`
- `DIAGNOSTIC_PASSWORD_FILE` (optional; legacy alias: `ADMIN_PASSWORD_FILE`)

File contents are passed to the application as Python values, without being
copied to environment variables. File permissions must allow the container user
to read them. Actual Compose configuration and mount validation are still pending.

## Streamlit Secrets

Streamlit automatically exports top-level string secrets as environment variables.
Removing application synchronization cannot prevent this platform behavior.
For LangSmith and diagnostic credentials, use a nested section to avoid this
automatic export:

```toml
LANGSMITH_TRACING = "true"
LANGSMITH_PROJECT = "agentic-data-analyst"

[credentials]
LANGSMITH_API_KEY = "replace-with-your-key"
DIAGNOSTIC_PASSWORD = "replace-with-your-diagnostic-password"
```

Remove the corresponding top-level entries and restart the application when
migrating: existing process environment values are deliberately not deleted.
Keep ordinary settings above `[credentials]` or in their appropriate TOML section.

For LangSmith and diagnostic credentials, precedence is configured secret file,
environment value, nested Streamlit credentials, then legacy top-level Streamlit
credentials. Canonical names take priority over aliases within each source.
Legacy top-level credentials remain supported but retain Streamlit's environment
export behavior. This nested section currently applies only to LangSmith and
diagnostic credentials; OpenAI and application-password Streamlit configuration
continue to use their existing top-level entries.

LangSmith uses explicit Clients and graph callback tracers. Existing environment
keys and local `.env` loading remain supported; choosing those sources means the
credentials are present in the process environment. Real cloud trace delivery
must still be verified after deployment.
