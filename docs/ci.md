# Continuous integration

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

The AMD64 image job runs only after tests pass. It uses the repository Dockerfile,
does not publish images, and does not deploy. CI receives no application API keys,
no registry write permissions, and no GCP credentials. Pip and BuildKit caches,
job timeouts, and cancellation of superseded runs limit redundant work.

Actions currently reference maintained major-version tags; pin reviewed Action
commit SHAs before extending this workflow with publication/deployment privileges.
On PR runs, `github.sha` normally identifies GitHub's temporary merge commit;
on `main` pushes it identifies the pushed commit. The local build tag is for
verification only and is not published to a registry.

To validate the test stage locally in the project environment:

```sh
python -B -m pytest --offline -q -ra -p no:cacheprovider
```

GitHub-hosted execution and Linux-specific behavior must be verified on the first
PR run. CI does not itself block merging: repository branch protection must require
its checks if merging should be prevented when tests or image builds fail.
