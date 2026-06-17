package sanitize

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"regexp"
	"sort"

	"github.com/internal/golem-harness/server/internal/trajectory"
)

type Decision string

const (
	Accept     Decision = "accept"
	Drop       Decision = "drop"
	Quarantine Decision = "quarantine"
)

type Report struct {
	Decision              Decision
	ReasonCodes           []string
	RedactionRulesApplied []string
	FieldsDropped         []string
	SanitizerVersion      string
}
type LocalNER interface{ Findings(text string) []Finding }
type VisionRedactor interface {
	RedactionBoxes(ref trajectory.ScreenshotRef) []trajectory.RedactionBox
}
type Finding struct {
	Start, End int
	Label      string
}
type ConservativeNER struct{}

func (ConservativeNER) Findings(text string) []Finding { return nil }

type Pipeline struct {
	AllowedPackages   map[string]bool
	SensitivePackages map[string]string
	Version           string
	NER               LocalNER
	ForceFailure      bool
}

func NewPipeline(allowed []string) *Pipeline {
	m := map[string]bool{}
	for _, p := range allowed {
		m[p] = true
	}
	return &Pipeline{AllowedPackages: m, SensitivePackages: defaultSensitive(), Version: "sanitize-v1", NER: ConservativeNER{}}
}
func defaultSensitive() map[string]string {
	return map[string]string{"com.android.chrome": "sensitive_browser", "com.google.android.gm": "sensitive_email", "com.whatsapp": "sensitive_messaging", "com.android.providers.contacts": "sensitive_contacts", "com.onepassword.android": "sensitive_password_manager", "com.bank.app": "sensitive_financial"}
}

func (p *Pipeline) Sanitize(in trajectory.Frame) (trajectory.Frame, Report, error) {
	if p.ForceFailure {
		return trajectory.Frame{}, Report{Decision: Drop, ReasonCodes: []string{"sanitizer_failure"}, SanitizerVersion: p.Version}, errors.New("forced sanitizer failure")
	}
	out := in
	out.Signature.Signature = nil
	out.PreStorageOnly = nil
	report := Report{Decision: Accept, SanitizerVersion: p.Version}
	pkg := in.App.ForegroundPackage
	if reason, ok := p.SensitivePackages[pkg]; ok {
		report.Decision = Quarantine
		report.ReasonCodes = append(report.ReasonCodes, "kill_switch", reason)
		out.App.AllowlistDecision.Decision = string(Quarantine)
		out.App.AllowlistDecision.KillSwitchReason = reason
		applyReport(&out, report)
		return out, report, nil
	}
	if !p.AllowedPackages[pkg] {
		report.Decision = Drop
		report.ReasonCodes = append(report.ReasonCodes, "package_not_allowlisted")
		out.App.AllowlistDecision.Decision = string(Drop)
		applyReport(&out, report)
		return out, report, nil
	}
	out.App.AllowlistDecision.Decision = string(Accept)
	out.App.AllowlistDecision.PolicyVersion = "phase1-allowlist"
	for i := range out.UITree.Nodes {
		redactValue(&out.UITree.Nodes[i].Text, &report, "ui_tree.nodes.text")
		redactValue(&out.UITree.Nodes[i].ContentDescription, &report, "ui_tree.nodes.content_description")
	}
	applyReport(&out, report)
	return out, report, nil
}

var rules = []struct {
	name string
	re   *regexp.Regexp
}{
	{"email", regexp.MustCompile(`(?i)[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}`)},
	{"phone", regexp.MustCompile(`\b(?:\+?1[-. ]?)?(?:\(?\d{3}\)?[-. ]?)\d{3}[-. ]?\d{4}\b`)},
	{"ssn", regexp.MustCompile(`\b\d{3}-\d{2}-\d{4}\b`)},
	{"payment_card", regexp.MustCompile(`\b(?:\d[ -]*?){13,19}\b`)},
	{"address", regexp.MustCompile(`(?i)\b\d{1,6}\s+[A-Za-z0-9 .'-]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b`)},
	{"token", regexp.MustCompile(`(?i)\b(?:bearer\s+|api[_-]?key\s*[:=]\s*|token\s*[:=]\s*)[A-Za-z0-9._\-]{12,}\b`)},
	{"long_numeric_identifier", regexp.MustCompile(`\b\d{9,}\b`)},
}

func redactValue(v *trajectory.RedactedValue, r *Report, field string) {
	raw := v.Raw
	if raw == "" {
		v.Status = "absent"
		return
	}
	matched := []string{}
	for _, rule := range rules {
		if rule.re.MatchString(raw) {
			matched = append(matched, rule.name)
		}
	}
	if len(matched) > 0 {
		sort.Strings(matched)
		v.Status = "redacted"
		v.Hash = ""
		v.Raw = ""
		v.RedactionRuleIDs = matched
		r.RedactionRulesApplied = appendUnique(r.RedactionRulesApplied, matched...)
		return
	}
	sum := sha256.Sum256([]byte(raw))
	v.Status = "hashed"
	v.Hash = hex.EncodeToString(sum[:])
	v.Raw = ""
	r.FieldsDropped = appendUnique(r.FieldsDropped, field+".raw")
}
func applyReport(f *trajectory.Frame, r Report) {
	f.Sanitizer.SanitizerVersion = r.SanitizerVersion
	f.Sanitizer.RedactionRulesApplied = append([]string{}, r.RedactionRulesApplied...)
	f.Sanitizer.FieldsDropped = append([]string{}, r.FieldsDropped...)
	f.Sanitizer.ReasonCodes = append([]string{}, r.ReasonCodes...)
	f.Signature.Signature = nil
	f.PreStorageOnly = nil
}
func appendUnique(in []string, vals ...string) []string {
	seen := map[string]bool{}
	for _, v := range in {
		seen[v] = true
	}
	for _, v := range vals {
		if !seen[v] {
			in = append(in, v)
			seen[v] = true
		}
	}
	return in
}
