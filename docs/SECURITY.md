# Security Notes

## Authorized Use Only

Golem-Harness is limited to operator-owned devices, controlled emulators, and explicitly consenting test users. It must not be used for third-party accounts, third-party devices, stealth operation, persistence, anti-detection, credential capture, or Android security bypasses.

## Non-Goals

Phase 1 does not automate banking apps, password managers, private messaging, email, medical apps, or other sensitive apps. It also does not implement the Android driver.

## Threat Model

Primary risks are accidental collection of personal data, replayed telemetry, unauthorized device submissions, sensitive app capture, unsafe logs, and unsafe persistence of raw request data.

## Sensitive Package Handling

The sanitizer has a kill-switch list for known sensitive package families and defaults to dropping non-allowlisted packages. Sensitive packages are quarantined or dropped before storage.

## Log Safety

Logs contain only safe metadata such as device ID, trajectory ID, frame ID, sequence number, package name, and reason codes. Logs must never contain raw XML, screenshots, text values, credentials, signatures, private keys, auth headers, or PII.

## Key Handling

Device public keys are configured server-side and mapped by device ID and key ID. Private keys are used only by test clients and should not be committed. mTLS certificate files are loaded from local configuration when enabled.

## Sanitizer Failure Behavior

Sanitizer errors fail closed. Failed frames are not stored. Storage validates that raw fields and detached signatures are absent.

## Known Gaps

- No generated protobuf bindings yet; generation command is documented in `docs/MVP.md`.
- No production local NER or vision redaction model.
- No Parquet writer yet.
- Replay cache is in-memory and should be replaced with a durable bounded cache before multi-process deployment.
