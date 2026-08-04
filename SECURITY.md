# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Telco-AIX, please report it privately —
do **not** open a public issue.

- Use GitHub's private vulnerability reporting:
  [Report a vulnerability](https://github.com/open-experiments/Telco-AIX/security/advisories/new)
- Or contact the maintainers listed in the [README](README.md#collaborators).

Please include a description of the issue, steps to reproduce, affected file(s)/component(s),
and any suggested remediation. We aim to acknowledge reports within 7 days.

## Scope

Telco-AIX is a collection of **experimental** AI/ML workloads for telco use cases.
Projects here are reference implementations, not hardened production services.
Reports are welcome for anything in this repository, with priority on:

- Leaked credentials, tokens, or secrets in code, manifests, or history
- Code-injection, path-traversal, XSS, or deserialization issues in the demo apps/UIs
- Vulnerable dependency pins in `requirements*.txt`, `Pipfile.lock`, or container images

## Supported Versions

Only the `main` branch is maintained. There are no versioned releases;
fixes land on `main`.

## Security Practices in This Repository

- **CodeQL** static analysis runs on pushes and pull requests (`.github/workflows/codeql.yml`)
- **Dependency & secret scanning** via CI (`.github/workflows/security-scan.yml`) and Dependabot (`.github/dependabot.yml`)
- **No secrets in git**: Kubernetes/OpenShift Secrets are created out-of-band
  (`oc create secret ... --from-literal=...`); manifests contain placeholders only
- **TLS verification on by default**: lab-only opt-outs are explicit env vars
  (`SME_TLS_VERIFY`, `PERF_TLS_VERIFY`) or `REQUESTS_CA_BUNDLE` for private CAs
