# Golem-Harness Architecture

Project Golem-Harness is an internal, consent-based Android automation research harness for operator-owned devices, controlled emulators, and explicitly consenting test users only. Phase 1 implements only the Go-side server foundation; it does not include an Android AccessibilityService driver.

## Data Flow

1. A trusted test client builds a synthetic `TelemetryFrame` matching `proto/golem/v1/telemetry.proto`.
2. The client signs the canonical JSON payload with an Ed25519 device key.
3. The Go proxy receives the frame over the gRPC scaffold. mTLS can be enabled in configuration for transport-level device/client authentication.
4. The auth boundary rejects missing, malformed, invalid, expired, replayed, oversized, or unauthorized frames before sanitization.
5. Unsanitized frame content exists only in bounded request memory.
6. The sanitizer applies package policy, structural attrition, regex redaction, and placeholder local-only NER/vision interfaces.
7. Only accepted sanitized frames cross into the storage boundary.
8. Phase 1 storage is a JSONL sanitized sink. Parquet is intentionally deferred.

## Trust Boundaries

- **Transport boundary:** gRPC server with optional mTLS.
- **Payload authenticity boundary:** detached Ed25519 verification over canonical JSON with registered device keys.
- **Sanitizer boundary:** fail-closed transformation from request-scoped raw input to sanitized frame.
- **Storage boundary:** storage implementations validate that signatures, pre-storage-only fields, and raw node text values are absent.

## Telemetry Frame Lifecycle

The protobuf contract models protocol version, trajectory/frame identity, timestamp, device metadata, foreground app metadata, allowlist decisions, UI tree structure, intent/action metadata, UI-settle placeholders, screenshot reference metadata without raw bytes, sanitizer metadata, and signature envelope metadata.

Raw text is represented only in the Go adapter as a request-scope `Raw` field to exercise sanitizer tests. The protobuf contract marks future raw extensions as pre-storage only and excludes raw screenshot bytes.

## Pre-Sanitization vs Post-Sanitization

Pre-sanitization data may include synthetic raw node text in memory. It must never be logged or written. Post-sanitization data contains hashes, redaction status, rule IDs, dropped-field metadata, and safe structural state.

## mTLS and Ed25519 Rationale

mTLS authenticates the transport peer and protects traffic in transit. Ed25519 authenticates each payload independently and supports replay checks, key rotation metadata, and safe offline fixture testing.

## Sanitizer Stages

- Package allowlist and kill-switch policy.
- Structural attrition of signatures and pre-storage-only fields.
- Regex redaction for synthetic emails, phone numbers, SSNs, payment-card-like values, addresses, token/API-key-like strings, and long numeric identifiers.
- Local NER interface with a conservative placeholder. No cloud calls are made.
- Vision redaction interface and bounding-box model. OCR/vision is not implemented in Phase 1.

## Storage Boundary

The `storage.Sink` interface accepts only sanitized frames. `MemorySink` supports tests, and `JSONLSink` provides a local sanitized test sink. A Parquet writer remains the next storage milestone.

## Phase 1 Limitations

- No Kotlin driver or Android AccessibilityService implementation.
- No OCR, screenshot bytes, cloud model calls, or production NER model.
- The gRPC service uses a JSON codec scaffold until generated protobuf bindings are added.
- JSONL is used instead of Parquet for the first reviewable milestone.
