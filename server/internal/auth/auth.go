package auth

import (
	"crypto/ed25519"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/internal/golem-harness/server/internal/trajectory"
)

var (
	ErrMissingSignature = errors.New("missing signature")
	ErrInvalidSignature = errors.New("invalid signature")
	ErrUnauthorized     = errors.New("unauthorized device or key")
	ErrExpired          = errors.New("expired frame")
	ErrReplay           = errors.New("replayed frame")
	ErrMalformed        = errors.New("malformed auth envelope")
)

type KeyRegistry interface {
	PublicKey(deviceID, keyID string) (ed25519.PublicKey, bool)
}

type StaticRegistry struct{ keys map[string]ed25519.PublicKey }

func NewStaticRegistry(keys map[string]ed25519.PublicKey) *StaticRegistry {
	cp := map[string]ed25519.PublicKey{}
	for k, v := range keys {
		vv := make([]byte, len(v))
		copy(vv, v)
		cp[k] = vv
	}
	return &StaticRegistry{keys: cp}
}
func RegistryKey(deviceID, keyID string) string { return deviceID + ":" + keyID }
func (s *StaticRegistry) PublicKey(deviceID, keyID string) (ed25519.PublicKey, bool) {
	k, ok := s.keys[RegistryKey(deviceID, keyID)]
	return k, ok
}

type ReplayCache struct {
	mu   sync.Mutex
	seen map[string]uint64
}

func NewReplayCache() *ReplayCache { return &ReplayCache{seen: map[string]uint64{}} }
func (r *ReplayCache) CheckAndMark(deviceID, frameID string, seq uint64) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	key := deviceID + ":" + frameID
	if _, ok := r.seen[key]; ok {
		return ErrReplay
	}
	seqKey := deviceID + ":seq"
	if last, ok := r.seen[seqKey]; ok && seq <= last {
		return ErrReplay
	}
	r.seen[key] = seq
	r.seen[seqKey] = seq
	return nil
}

type Verifier struct {
	Registry KeyRegistry
	Replay   *ReplayCache
	Now      func() time.Time
	MaxSkew  time.Duration
}

func (v *Verifier) Verify(f trajectory.Frame) error {
	if len(f.Signature.Signature) == 0 {
		return ErrMissingSignature
	}
	if f.Signature.Algorithm != "Ed25519" || f.Signature.Canonicalization != "golem-json-v1" {
		return ErrMalformed
	}
	if f.Device.DeviceID == "" || f.Signature.KeyID == "" {
		return ErrUnauthorized
	}
	pub, ok := v.Registry.PublicKey(f.Device.DeviceID, f.Signature.KeyID)
	if !ok {
		return ErrUnauthorized
	}
	now := time.Now()
	if v.Now != nil {
		now = v.Now()
	}
	max := v.MaxSkew
	if max == 0 {
		max = 5 * time.Minute
	}
	event := time.UnixMilli(f.EventUnixMillis)
	signed := time.UnixMilli(f.Signature.SignedAtUnixMillis)
	if now.Sub(event) > max || event.Sub(now) > max || now.Sub(signed) > max || signed.Sub(now) > max {
		return ErrExpired
	}
	payload, err := CanonicalPayload(f)
	if err != nil {
		return err
	}
	if !ed25519.Verify(pub, payload, f.Signature.Signature) {
		return ErrInvalidSignature
	}
	if v.Replay != nil {
		return v.Replay.CheckAndMark(f.Device.DeviceID, f.FrameID, f.SequenceNumber)
	}
	return nil
}

func CanonicalPayload(f trajectory.Frame) ([]byte, error) {
	f.Signature.Signature = nil
	return json.Marshal(f)
}
func SignFrame(f trajectory.Frame, priv ed25519.PrivateKey) (trajectory.Frame, error) {
	f.Signature.Algorithm = "Ed25519"
	f.Signature.Canonicalization = "golem-json-v1"
	if f.Signature.SignedAtUnixMillis == 0 {
		f.Signature.SignedAtUnixMillis = f.EventUnixMillis
	}
	p, err := CanonicalPayload(f)
	if err != nil {
		return f, err
	}
	f.Signature.Signature = ed25519.Sign(priv, p)
	return f, nil
}

func DecodePublicKeyB64(s string) (ed25519.PublicKey, error) {
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return nil, err
	}
	if len(b) != ed25519.PublicKeySize {
		return nil, fmt.Errorf("ed25519 public key must be %d bytes", ed25519.PublicKeySize)
	}
	return ed25519.PublicKey(b), nil
}

func LoadServerTLS(certFile, keyFile, caFile string, requireClientCert bool) (*tls.Config, error) {
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, err
	}
	cfg := &tls.Config{MinVersion: tls.VersionTLS12, Certificates: []tls.Certificate{cert}}
	if requireClientCert {
		caPEM, err := os.ReadFile(caFile)
		if err != nil {
			return nil, err
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(caPEM) {
			return nil, errors.New("invalid client CA PEM")
		}
		cfg.ClientCAs = pool
		cfg.ClientAuth = tls.RequireAndVerifyClientCert
	}
	return cfg, nil
}
