package trajectory

import "time"

type Frame struct {
	ProtocolVersion uint32            `json:"protocol_version"`
	TrajectoryID    string            `json:"trajectory_id"`
	FrameID         string            `json:"frame_id"`
	SequenceNumber  uint64            `json:"sequence_number"`
	EventUnixMillis int64             `json:"event_unix_millis"`
	Device          DeviceMetadata    `json:"device"`
	App             AppContext        `json:"app"`
	UITree          UITreeSnapshot    `json:"ui_tree"`
	Intent          IntentMetadata    `json:"intent"`
	Action          ActionMetadata    `json:"action"`
	UISettle        UISettleMetadata  `json:"ui_settle"`
	Screenshot      ScreenshotRef     `json:"screenshot"`
	Sanitizer       SanitizerMetadata `json:"sanitizer"`
	Signature       SignatureEnvelope `json:"signature"`
	PreStorageOnly  map[string]string `json:"pre_storage_only,omitempty"`
}

type DeviceMetadata struct {
	DeviceID              string `json:"device_id"`
	AndroidSDKVersion     uint32 `json:"android_sdk_version"`
	BuildFingerprintHash  string `json:"build_fingerprint_hash"`
	RedactedBuildMetadata string `json:"redacted_build_metadata"`
}
type AppContext struct {
	ForegroundPackage  string            `json:"foreground_package"`
	ForegroundActivity string            `json:"foreground_activity"`
	AllowlistDecision  AllowlistDecision `json:"allowlist_decision"`
}
type AllowlistDecision struct {
	Decision         string   `json:"decision"`
	PolicyVersion    string   `json:"policy_version"`
	ReasonCodes      []string `json:"reason_codes"`
	KillSwitchReason string   `json:"kill_switch_reason"`
}
type UITreeSnapshot struct {
	SnapshotID string   `json:"snapshot_id"`
	Nodes      []UINode `json:"nodes"`
	TreeHash   string   `json:"tree_hash"`
}
type UINode struct {
	StableNodeID       string        `json:"stable_node_id"`
	Bounds             Bounds        `json:"bounds"`
	ClassName          string        `json:"class_name"`
	PackageName        string        `json:"package_name"`
	ResourceIDHash     string        `json:"resource_id_hash"`
	Text               RedactedValue `json:"text"`
	ContentDescription RedactedValue `json:"content_description"`
	Clickable          bool          `json:"clickable"`
	Enabled            bool          `json:"enabled"`
	Focused            bool          `json:"focused"`
	Selected           bool          `json:"selected"`
	Checkable          bool          `json:"checkable"`
}
type Bounds struct{ Left, Top, Right, Bottom int32 }
type RedactedValue struct {
	Status           string   `json:"status"`
	Hash             string   `json:"hash,omitempty"`
	RedactionRuleIDs []string `json:"redaction_rule_ids,omitempty"`
	Raw              string   `json:"raw,omitempty"`
}
type IntentMetadata struct {
	IntentID              string   `json:"intent_id"`
	OperatorIntentHash    string   `json:"operator_intent_hash"`
	AllowedActionFamilies []string `json:"allowed_action_families"`
}
type ActionMetadata struct {
	ActionID           string `json:"action_id"`
	ActionType         string `json:"action_type"`
	TargetStableNodeID string `json:"target_stable_node_id"`
	ParametersHash     string `json:"parameters_hash"`
}
type UISettleMetadata struct {
	Observed            bool   `json:"observed"`
	SettleTimeoutMillis uint32 `json:"settle_timeout_millis"`
	StableWindowMillis  uint32 `json:"stable_window_millis"`
	HeuristicVersion    string `json:"heuristic_version"`
}
type ScreenshotRef struct {
	Present             bool           `json:"present"`
	SanitizedArtifactID string         `json:"sanitized_artifact_id"`
	PerceptualHash      string         `json:"perceptual_hash"`
	RedactionBoxes      []RedactionBox `json:"redaction_boxes"`
}
type RedactionBox struct {
	Left, Top, Right, Bottom int32
	ReasonCode               string `json:"reason_code"`
}
type SanitizerMetadata struct {
	SanitizerVersion      string   `json:"sanitizer_version"`
	RedactionRulesApplied []string `json:"redaction_rules_applied"`
	FieldsDropped         []string `json:"fields_dropped"`
	ReasonCodes           []string `json:"reason_codes"`
}
type SignatureEnvelope struct {
	KeyID              string `json:"key_id"`
	Algorithm          string `json:"algorithm"`
	Signature          []byte `json:"signature"`
	SignedAtUnixMillis int64  `json:"signed_at_unix_millis"`
	Canonicalization   string `json:"canonicalization"`
}

func NowMillis() int64 { return time.Now().UnixMilli() }
