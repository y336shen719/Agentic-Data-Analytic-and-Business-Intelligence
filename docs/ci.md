# Continuous integration and GHCR publication

`.github/workflows/ci.yaml` runs for PRs targeting `main`, pushes to `main`, and
manual dispatch once the workflow is available on the default branch. An update
to an open PR triggers PR checks; feature-branch pushes without a PR do not.

The test job uses Ubuntu 24.04, Python 3.11.18, and `requirements-docker.lock`.
The checkout enables Git LFS to download the actual DuckDB file, not its pointer.
The job opens it read-only and checks that tables exist, then runs the whole suite,
including unmarked tests, with `pytest --offline`. Tests do not require the
untracked production raw CSVs. Source row-count validation remains covered by
synthetic CSV tests; database-only primary-key, foreign-key and view checks
remain enabled. Direct production CSV-to-database reconciliation is not part
of this suite.
The image-only build job does not download LFS data because `.dockerignore`
excludes the database from the image. LFS downloads count toward the repository's
applicable Git LFS transfer quota; the workflow adds no external data service.

Offline mode is opt-in and applied before collection. It disables automatic
`.env` loading, clears runtime credential inputs, ignores real Streamlit secrets
files, and blocks IPv4/IPv6 socket connections in the test process. Unix sockets
remain available for local process communication. Tests may explicitly provide
synthetic configuration and clients. Dependency installation and image building
still require network access. This guard is a test aid, not an OS sandbox, and
does not prevent a deliberately spawned executable from accessing the network.

The image stages use the repository Dockerfile and run only after tests pass:

| Event | Image stage | Registry write access |
| --- | --- | --- |
| PR targeting main | Build AMD64, do not publish | None |
| Push to main | Build AMD64 and publish to GHCR | Publish job only |
| Manual dispatch on any branch | Build AMD64, do not publish | None |

Main pushes do not run the separate verification build job, avoiding a duplicate
build before publication. Nothing deploys to GCP. Only the publish job receives
`contents: read` and `packages: write`; it authenticates with the job-scoped
`GITHUB_TOKEN`. No manually created PAT, application API keys, or GCP credentials
are used. Pip and BuildKit caches and job timeouts limit redundant work. Updated
PR checks cancel obsolete runs; an active main publication is not automatically
canceled by a newer push. GitHub concurrency can still replace pending runs, so
this is not a durable release queue guaranteeing publication of every commit.

All Actions are pinned to full commit SHAs verified from their official releases:
checkout v7.0.1, setup-python v7.1.0, setup-buildx v4.4.1, build-push v7.4.0, and
login v4.8.0. Review release changes and update pins intentionally.
On PR runs, `github.sha` normally identifies GitHub's temporary merge commit;
on `main` pushes it identifies the tested and published main commit.

## Published image identity

The image name is `ghcr.io/<lowercase-repository-owner>/agentic-data-analyst`.
For this repository it is `ghcr.io/y336shen719/agentic-data-analyst`. Each image is
tagged `sha-<full-main-commit-SHA>`, with OCI source/revision labels linking it to
the repository. No `latest` tag is published.

Before building, the publish job checks the commit tag. If it exists, the workflow
records its digest and does not overwrite it. Unexpected registry errors fail the
job; a missing manifest permits the first publication. Tags are still mutable to
other registry writers; use the recorded content digest as the authoritative
deployment reference. The workflow summary and publish job outputs contain both
the image name and digest. A rerun reuses the existing published image even if
its Dockerfile/dependency inputs could otherwise yield a different rebuild.

First publication creates a package with GitHub's default private visibility; this
workflow does not change visibility. Confirm the package is connected to this
repository and that Actions has access. If that package name already exists under
the account, its access settings may need adjustment. Before VM pulls, choose
public access or configure appropriately scoped private-package read access.
Do not add runtime OpenAI/LangSmith credentials to GitHub for image publication.

Keep releases needed for rollback. This workflow does not delete package versions
or set a retention policy; review storage/transfer policies and retained versions
as usage grows. Actual GHCR login, first publish, package permissions and remote
digest verification must be checked after merge. No image is published by the PR.

To validate the test stage locally in the project environment:

```sh
python -B -m pytest --offline -q -ra -p no:cacheprovider
```

GitHub-hosted execution and Linux-specific behavior must be verified on the first
PR run. CI does not itself block merging: repository branch protection must require
its checks if merging should be prevented when tests or image builds fail.
