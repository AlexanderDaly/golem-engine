package sanitize_test

import (
	"strings"
	"testing"
	"time"

	"github.com/internal/golem-harness/server/internal/sanitize"
	"github.com/internal/golem-harness/server/internal/testutil"
)

func TestSensitivePackageQuarantined(t *testing.T) {
	p := sanitize.NewPipeline([]string{testutil.AllowedPackage})
	_, r, err := p.Sanitize(testutil.Frame(time.Now(), "com.google.android.gm"))
	if err != nil || r.Decision != sanitize.Quarantine {
		t.Fatalf("decision=%s err=%v", r.Decision, err)
	}
}
func TestNonAllowlistedPackageDropped(t *testing.T) {
	p := sanitize.NewPipeline([]string{testutil.AllowedPackage})
	_, r, err := p.Sanitize(testutil.Frame(time.Now(), "com.example.notallowed"))
	if err != nil || r.Decision != sanitize.Drop {
		t.Fatalf("decision=%s err=%v", r.Decision, err)
	}
}
func TestSyntheticPIIRedacted(t *testing.T) {
	cases := []string{"alice@example.test", "212-555-0100", "123 Main Street", "123-45-6789", "4111 1111 1111 1111", "Bearer abcdefghijklmnop", "123456789012"}
	p := sanitize.NewPipeline([]string{testutil.AllowedPackage})
	for _, raw := range cases {
		f := testutil.Frame(time.Now(), testutil.AllowedPackage)
		f.UITree.Nodes[0].Text.Raw = raw
		out, r, err := p.Sanitize(f)
		if err != nil || r.Decision != sanitize.Accept {
			t.Fatalf("%q decision=%s err=%v", raw, r.Decision, err)
		}
		b := out.UITree.Nodes[0].Text.Raw + out.UITree.Nodes[0].Text.Hash
		if strings.Contains(b, raw) {
			t.Fatalf("raw PII remained for %q", raw)
		}
		if out.UITree.Nodes[0].Text.Status != "redacted" {
			t.Fatalf("%q status=%s", raw, out.UITree.Nodes[0].Text.Status)
		}
	}
}
