# Capsule environment

Use a CPU-only Code Ocean starter environment with **Python 3.13**. Add
`environment/postInstall` as the Environment Editor's post-install script and
build the environment once before requesting a Reproducible Run. No GPU,
secrets, database, or interactive service is required.

`environment/requirements.txt` records the direct dependencies and
`environment/requirements-lock.txt` records the complete resolved environment. CI
fails if either differs from its repository counterpart. The post-install script
uses the environment lock because Code Ocean does not expose `/code` or `/data`
during an environment build.

At run time after Copy from Git, designate `/code/run` as the Capsule's **File to
Run**. If Clone from Git places the complete repository under `/code`, designate
the repository's top-level `run` instead. The driver uses the standard mounts
`/data`, `/results`, and `/scratch`; it never installs packages or downloads code
during a Reproducible Run. The Git repository's `code/` directory contains the
driver, Python source, orchestration script and frozen estimator inputs, so Copy from
Git maps the complete executable analysis into Capsule `/code`.
