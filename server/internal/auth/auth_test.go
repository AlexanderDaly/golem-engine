package auth_test

import (
	"crypto/ed25519"
	"errors"
	"testing"
	"time"

	"github.com/internal/golem-harness/server/internal/auth"
	"github.com/internal/golem-harness/server/internal/testutil"
)

func verifier(pub ed25519.PublicKey, now time.Time) *auth.Verifier {
	return &auth.Verifier{Registry: auth.NewStaticRegistry(map[string]ed25519.PublicKey{auth.RegistryKey(testutil.DeviceID, testutil.KeyID): pub}), Replay: auth.NewReplayCache(), Now: func() time.Time { return now }, MaxSkew: time.Minute}
}
func TestValidSignatureAccepted(t *testing.T) {
	now := time.Unix(1000, 0)
	pub, priv := testutil.Keypair()
	f := testutil.SignedFrame(now, testutil.AllowedPackage, priv)
	if err := verifier(pub, now).Verify(f); err != nil {
		t.Fatalf("valid signature rejected: %v", err)
	}
}
func TestInvalidSignatureRejected(t *testing.T) {
	now := time.Unix(1000, 0)
	pub, priv := testutil.Keypair()
	f := testutil.SignedFrame(now, testutil.AllowedPackage, priv)
	f.TrajectoryID = "tampered"
	err := verifier(pub, now).Verify(f)
	if !errors.Is(err, auth.ErrInvalidSignature) {
		t.Fatalf("got %v", err)
	}
}
func TestMissingSignatureRejected(t *testing.T) {
	now := time.Unix(1000, 0)
	pub, _ := testutil.Keypair()
	f := testutil.Frame(now, testutil.AllowedPackage)
	err := verifier(pub, now).Verify(f)
	if !errors.Is(err, auth.ErrMissingSignature) {
		t.Fatalf("got %v", err)
	}
}
func TestExpiredTimestampRejected(t *testing.T) {
	now := time.Unix(1000, 0)
	pub, priv := testutil.Keypair()
	f := testutil.SignedFrame(now.Add(-2*time.Hour), testutil.AllowedPackage, priv)
	err := verifier(pub, now).Verify(f)
	if !errors.Is(err, auth.ErrExpired) {
		t.Fatalf("got %v", err)
	}
}
func TestReplayRejected(t *testing.T) {
	now := time.Unix(1000, 0)
	pub, priv := testutil.Keypair()
	v := verifier(pub, now)
	f := testutil.SignedFrame(now, testutil.AllowedPackage, priv)
	if err := v.Verify(f); err != nil {
		t.Fatal(err)
	}
	err := v.Verify(f)
	if !errors.Is(err, auth.ErrReplay) {
		t.Fatalf("got %v", err)
	}
}
func TestUnauthorizedRejected(t *testing.T) {
	now := time.Unix(1000, 0)
	pub, _ := testutil.Keypair()
	_, priv2 := testutil.Keypair()
	f := testutil.SignedFrame(now, testutil.AllowedPackage, priv2)
	err := verifier(pub, now).Verify(f)
	if !errors.Is(err, auth.ErrInvalidSignature) {
		t.Fatalf("got %v", err)
	}
	f.Device.DeviceID = "unknown"
	f, _ = auth.SignFrame(f, priv2)
	err = verifier(pub, now).Verify(f)
	if !errors.Is(err, auth.ErrUnauthorized) {
		t.Fatalf("got %v", err)
	}
}
