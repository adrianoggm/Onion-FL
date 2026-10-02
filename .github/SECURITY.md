# Security Policy

Onion-FL is a research prototype. Security fixes go into the latest version on `develop`, and from there into the next tagged release on `main`. Older versions are not maintained.

## Reporting a vulnerability

Please do not open a public issue. Report it privately through [GitHub Security Advisories](https://github.com/adrianoggm/Onion-FL/security/advisories/new) instead, and include:

- a description of the vulnerability and its impact;
- the steps to reproduce it;
- any suggested fix or mitigation.

## Known limitations

The current prototype is not hardened: model weights travel as plaintext JSON, the bundled Mosquitto config allows anonymous clients without TLS, and the broker sees every individual client update. Do not deploy it on untrusted networks or with real personal data.
