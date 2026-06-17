package config_test

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"strings"
	"testing"

	"github.com/internal/golem-harness/server/internal/config"
)

func validConfig() config.Config {
	pub, _, _ := ed25519.GenerateKey(rand.Reader)
	return config.Config{GRPCAddr: "127.0.0.1:0", HTTPAddr: "127.0.0.1:0", MaxFrameBytes: 1024, AllowedPackages: []string{"com.example.calculator"}, DeviceKeys: []config.DeviceKey{{DeviceID: "d", KeyID: "k", PublicKeyBase64: base64.StdEncoding.EncodeToString(pub)}}}
}
func TestMissingRequiredConfigFailsClearly(t *testing.T) {
	c := validConfig()
	c.GRPCAddr = ""
	err := c.Validate()
	if err == nil || !strings.Contains(err.Error(), "grpc_addr") {
		t.Fatalf("got %v", err)
	}
}
func TestInvalidKeyMaterialFailsClearly(t *testing.T) {
	c := validConfig()
	c.DeviceKeys[0].PublicKeyBase64 = base64.StdEncoding.EncodeToString([]byte("short"))
	err := c.Validate()
	if err == nil || !strings.Contains(err.Error(), "invalid public key length") {
		t.Fatalf("got %v", err)
	}
}
func TestAllowedPackageConfigParsed(t *testing.T) {
	c := validConfig()
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	if c.AllowedPackages[0] != "com.example.calculator" {
		t.Fatalf("allowed=%v", c.AllowedPackages)
	}
}
