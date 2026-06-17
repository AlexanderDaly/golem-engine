# AGENTS.md

Scope: entire repository.

Project Golem-Harness is authorized-use-only research infrastructure for operator-owned devices, controlled emulators, and explicitly consenting users. Preserve this scope in code, tests, documentation, and examples.

Durable constraints for future Codex sessions:

- Always run relevant tests before finishing, and report exact commands and results.
- Never log, persist, snapshot, or emit raw telemetry, raw UI XML, raw screenshots, raw text values, credentials, signatures, private keys, auth headers, or PII.
- Never use external cloud APIs for sanitization, NER, OCR, telemetry processing, screenshots, or model inference.
- Keep unsanitized telemetry inside bounded in-memory request scope only.
- Prefer fail-closed behavior at auth, sanitizer, and storage boundaries.
- Add tests for failure paths, especially sanitizer failure and auth rejection paths.
- Document measurable gaps honestly rather than implying production readiness.
- Keep Phase 1 scoped to safe native/system surfaces and synthetic fixtures.
- Do not implement stealth behavior, persistence, anti-detection, Android security bypasses, credential capture, or automation against third-party accounts/devices without authorization.
- Do not automate banking apps, password managers, private messaging, email, medical apps, or other sensitive apps in Phase 1.
- Keep security-sensitive code readable and dependency-light.
