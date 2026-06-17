package testutil

import (
	"crypto/ed25519"
	"crypto/rand"
	"time"

	"github.com/internal/golem-harness/server/internal/auth"
	"github.com/internal/golem-harness/server/internal/trajectory"
)

const DeviceID = "test-device-001"
const KeyID = "test-key-001"
const AllowedPackage = "com.example.calculator"

func Keypair() (ed25519.PublicKey, ed25519.PrivateKey) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		panic(err)
	}
	return pub, priv
}
func Frame(now time.Time, pkg string) trajectory.Frame {
	return trajectory.Frame{ProtocolVersion: 1, TrajectoryID: "traj-synth-001", FrameID: "frame-synth-001", SequenceNumber: 1, EventUnixMillis: now.UnixMilli(), Device: trajectory.DeviceMetadata{DeviceID: DeviceID, AndroidSDKVersion: 35, BuildFingerprintHash: "sha256:test-build"}, App: trajectory.AppContext{ForegroundPackage: pkg, ForegroundActivity: ".SyntheticActivity"}, UITree: trajectory.UITreeSnapshot{SnapshotID: "snap-001", TreeHash: "sha256:tree", Nodes: []trajectory.UINode{{StableNodeID: "node-1", ClassName: "android.widget.TextView", PackageName: pkg, ResourceIDHash: "sha256:res", Text: trajectory.RedactedValue{Raw: "hello synthetic"}, ContentDescription: trajectory.RedactedValue{Raw: "button synthetic"}, Clickable: true, Enabled: true}}}, Intent: trajectory.IntentMetadata{IntentID: "intent-1", OperatorIntentHash: "sha256:intent", AllowedActionFamilies: []string{"tap"}}, Action: trajectory.ActionMetadata{ActionID: "action-1", ActionType: "tap", TargetStableNodeID: "node-1"}, UISettle: trajectory.UISettleMetadata{Observed: true, SettleTimeoutMillis: 1000, StableWindowMillis: 250, HeuristicVersion: "placeholder-v1"}, Signature: trajectory.SignatureEnvelope{KeyID: KeyID, SignedAtUnixMillis: now.UnixMilli()}}
}
func SignedFrame(now time.Time, pkg string, priv ed25519.PrivateKey) trajectory.Frame {
	f := Frame(now, pkg)
	sf, err := auth.SignFrame(f, priv)
	if err != nil {
		panic(err)
	}
	return sf
}
