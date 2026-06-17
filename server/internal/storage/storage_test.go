package storage_test

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/internal/golem-harness/server/internal/sanitize"
	"github.com/internal/golem-harness/server/internal/storage"
	"github.com/internal/golem-harness/server/internal/testutil"
)

func TestStorageOnlyReceivesSanitizedFrames(t *testing.T) {
	sink := &storage.MemorySink{}
	raw := testutil.Frame(time.Now(), testutil.AllowedPackage)
	if err := sink.Store(raw); err == nil {
		t.Fatal("raw frame stored")
	}
	p := sanitize.NewPipeline([]string{testutil.AllowedPackage})
	safe, r, err := p.Sanitize(raw)
	if err != nil || r.Decision != sanitize.Accept {
		t.Fatalf("sanitize: %v %s", err, r.Decision)
	}
	if err := sink.Store(safe); err != nil {
		t.Fatalf("safe frame rejected: %v", err)
	}
}
func TestJSONLOmitsRawPII(t *testing.T) {
	path := t.TempDir() + "/out.jsonl"
	sink, err := storage.NewJSONLSink(path)
	if err != nil {
		t.Fatal(err)
	}
	p := sanitize.NewPipeline([]string{testutil.AllowedPackage})
	f := testutil.Frame(time.Now(), testutil.AllowedPackage)
	f.UITree.Nodes[0].Text.Raw = "alice@example.test"
	safe, _, err := p.Sanitize(f)
	if err != nil {
		t.Fatal(err)
	}
	if err := sink.Store(safe); err != nil {
		t.Fatal(err)
	}
	if err := sink.Close(); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "alice@example.test") {
		t.Fatalf("raw PII in output: %s", string(b))
	}
}
