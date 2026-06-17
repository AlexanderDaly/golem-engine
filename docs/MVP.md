# MVP Status

## Implemented

- Versioned protobuf telemetry schema at `proto/golem/v1/telemetry.proto`.
- Go gRPC ingestion scaffold with optional mTLS configuration.
- Ed25519 detached signature verification over canonical JSON.
- Device/key registry, timestamp expiry checks, replay checks, and max-frame-size enforcement.
- Fail-closed sanitizer with allowlist, kill-switch, regex redaction, local NER interface, and vision redaction interface.
- Storage interface plus memory and sanitized JSONL sinks.
- Synthetic fixtures and a mock signed client.
- Unit tests for auth, sanitizer, ingest, storage, and config.

## Run Tests

```bash
cd server
go test ./...
```

## Generate Protobuf Bindings Later

When `protoc`, `protoc-gen-go`, and `protoc-gen-go-grpc` are available:

```bash
protoc -I proto \
  --go_out=server/internal/pb --go_opt=paths=source_relative \
  --go-grpc_out=server/internal/pb --go-grpc_opt=paths=source_relative \
  proto/golem/v1/telemetry.proto
```

The current server uses a JSON gRPC codec scaffold so auth, sanitizer, and storage boundaries remain testable without generated code.

## Run Proxy Locally

1. Inspect the deterministic synthetic test public key used by the mock client:

   ```bash
   cd mock-client
   go run . -print-test-key
   ```

2. The example config already contains this deterministic synthetic public key; for real development, copy the config and replace it with a non-committed local test key.
3. Start the proxy:

   ```bash
   cd server
   go run ./cmd/golem-proxy -config testdata/dev-config.example.json
   ```

## Run Mock Client

With the proxy running and matching public key configured:

```bash
cd mock-client
go run . -addr 127.0.0.1:50051
```

The client sends one allowed-package frame and one sensitive-package frame using synthetic data only.

## Before Kotlin Driver Work

Stabilize generated protobuf bindings, add durable replay storage, replace JSONL with sanitized Parquet, add local-only NER/vision implementations or conservative drop policies, and expand integration tests with mTLS dev certificates.
