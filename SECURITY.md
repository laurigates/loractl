# Security Policy

## Supported Versions

Only the latest release (and `main`) receives security fixes.

## Reporting a Vulnerability

Report vulnerabilities privately via GitHub's
[private vulnerability reporting](https://github.com/laurigates/loractl/security/advisories/new)
— do not open a public issue for security problems.

Dependency advisories are monitored automatically: `cargo audit` runs in CI on
every dependency change and weekly (`.github/workflows/security-audit.yml`).
