package ingest_test

import (
	"context"
	"crypto/ed25519"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/internal/golem-harness/server/internal/auth"
	"github.com/internal/golem-harness/server/internal/ingest"
	"github.com/internal/golem-harness/server/internal/sanitize"
	"github.com/internal/golem-harness/server/internal/storage"
	"github.com/internal/golem-harness/server/internal/testutil"
)

func svc(pub ed25519.PublicKey, now time.Time, sink storage.Sink, p *sanitize.Pipeline) *ingest.Service {
	return &ingest.Service{Verifier: &auth.Verifier{Registry: auth.NewStaticRegistry(map[string]ed25519.PublicKey{auth.RegistryKey(testutil.DeviceID, testutil.KeyID): pub}), Replay: auth.NewReplayCache(), Now: func() time.Time { return now }, MaxSkew: time.Minute}, Sanitizer: p, Store: sink, MaxFrameBytes: 4096}
}
func TestOversizedPayloadRejected(t *testing.T) {
	now := time.Unix(1000, 0)
	pub, priv := testutil.Keypair()
	f := testutil.SignedFrame(now, testutil.AllowedPackage, priv)
	s := svc(pub, now, &storage.MemorySink{}, sanitize.NewPipeline([]string{testutil.AllowedPackage}))
	s.MaxFrameBytes = 10
	_, err := s.Ingest(context.Background(), &ingest.Request{Frame: f})
	if err == nil || !strings.Contains(err.Error(), ingest.ErrOversized.Error()) {
		t.Fatalf("got %v", err)
	}
}
func TestSanitizerFailurePreventsStorage(t *testing.T) {
	now := time.Unix(1000, 0)
	pub, priv := testutil.Keypair()
	sink := &storage.MemorySink{}
	p := sanitize.NewPipeline([]string{testutil.AllowedPackage})
	p.ForceFailure = true
	s := svc(pub, now, sink, p)
	f := testutil.SignedFrame(now, testutil.AllowedPackage, priv)
	resp, err := s.Ingest(context.Background(), &ingest.Request{Frame: f})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Accepted {
		t.Fatal("accepted sanitizer failure")
	}
	if len(sink.Frames) != 0 {
		t.Fatal("stored frame after sanitizer failure")
	}
}
func TestSensitivePackageNotStored(t *testing.T) {
	now := time.Unix(1000, 0)
	pub, priv := testutil.Keypair()
	sink := &storage.MemorySink{}
	s := svc(pub, now, sink, sanitize.NewPipeline([]string{testutil.AllowedPackage}))
	f := testutil.SignedFrame(now, "com.google.android.gm", priv)
	resp, err := s.Ingest(context.Background(), &ingest.Request{Frame: f})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Accepted || resp.Decision != string(sanitize.Quarantine) {
		t.Fatalf("resp=%+v", resp)
	}
	if len(sink.Frames) != 0 {
		t.Fatal("stored sensitive package")
	}
}
func TestInvalidSignatureRejectedByIngest(t *testing.T) {
	now := time.Unix(1000, 0)
	pub, priv := testutil.Keypair()
	s := svc(pub, now, &storage.MemorySink{}, sanitize.NewPipeline([]string{testutil.AllowedPackage}))
	f := testutil.SignedFrame(now, testutil.AllowedPackage, priv)
	f.FrameID = "tampered"
	_, err := s.Ingest(context.Background(), &ingest.Request{Frame: f})
	if err == nil || !errors.Is(statusErr(err), auth.ErrInvalidSignature) {
		t.Fatalf("got %v", err)
	}
}
func statusErr(err error) error {
	if err == nil {
		return nil
	}
	if strings.Contains(err.Error(), auth.ErrInvalidSignature.Error()) {
		return auth.ErrInvalidSignature
	}
	return err
}
