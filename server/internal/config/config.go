package config

import (
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/internal/golem-harness/server/internal/auth"
)

type Config struct {
	GRPCAddr            string      `json:"grpc_addr"`
	HTTPAddr            string      `json:"http_addr"`
	MaxFrameBytes       int         `json:"max_frame_bytes"`
	MaxClockSkewSeconds int         `json:"max_clock_skew_seconds"`
	AllowedPackages     []string    `json:"allowed_packages"`
	DeviceKeys          []DeviceKey `json:"device_keys"`
	MTLS                MTLSConfig  `json:"mtls"`
	StorageJSONLPath    string      `json:"storage_jsonl_path"`
}
type DeviceKey struct {
	DeviceID        string `json:"device_id"`
	KeyID           string `json:"key_id"`
	PublicKeyBase64 string `json:"public_key_base64"`
}
type MTLSConfig struct {
	Enabled           bool   `json:"enabled"`
	CertFile          string `json:"cert_file"`
	KeyFile           string `json:"key_file"`
	ClientCAFile      string `json:"client_ca_file"`
	RequireClientCert bool   `json:"require_client_cert"`
}

func Load(path string) (Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Config{}, err
	}
	var c Config
	if err := json.Unmarshal(b, &c); err != nil {
		return Config{}, err
	}
	return c, c.Validate()
}
func (c Config) Validate() error {
	if c.GRPCAddr == "" {
		return errors.New("grpc_addr is required")
	}
	if c.HTTPAddr == "" {
		return errors.New("http_addr is required")
	}
	if c.MaxFrameBytes <= 0 {
		return errors.New("max_frame_bytes must be positive")
	}
	if len(c.AllowedPackages) == 0 {
		return errors.New("at least one allowed package is required")
	}
	if len(c.DeviceKeys) == 0 {
		return errors.New("at least one device key is required")
	}
	if c.MTLS.Enabled && (c.MTLS.CertFile == "" || c.MTLS.KeyFile == "") {
		return errors.New("mtls cert_file and key_file are required when enabled")
	}
	if c.MTLS.RequireClientCert && c.MTLS.ClientCAFile == "" {
		return errors.New("client_ca_file is required when client certs are required")
	}
	_, err := c.Registry()
	return err
}
func (c Config) Registry() (*auth.StaticRegistry, error) {
	keys := map[string]ed25519.PublicKey{}
	for _, dk := range c.DeviceKeys {
		if dk.DeviceID == "" || dk.KeyID == "" {
			return nil, errors.New("device_id and key_id are required")
		}
		raw, err := base64.StdEncoding.DecodeString(dk.PublicKeyBase64)
		if err != nil {
			return nil, fmt.Errorf("invalid public key for %s/%s: %w", dk.DeviceID, dk.KeyID, err)
		}
		if len(raw) != ed25519.PublicKeySize {
			return nil, fmt.Errorf("invalid public key length for %s/%s", dk.DeviceID, dk.KeyID)
		}
		keys[auth.RegistryKey(dk.DeviceID, dk.KeyID)] = ed25519.PublicKey(raw)
	}
	return auth.NewStaticRegistry(keys), nil
}
func (c Config) MaxSkew() time.Duration {
	if c.MaxClockSkewSeconds <= 0 {
		return 5 * time.Minute
	}
	return time.Duration(c.MaxClockSkewSeconds) * time.Second
}
