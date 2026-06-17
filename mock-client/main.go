package main

import (
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/internal/golem-harness/server/internal/auth"
	"github.com/internal/golem-harness/server/internal/ingest"
	"github.com/internal/golem-harness/server/internal/testutil"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:50051", "proxy gRPC address")
	printKeys := flag.Bool("print-test-key", false, "print a generated config entry and exit")
	flag.Parse()

	seed := sha256.Sum256([]byte("golem-harness-synthetic-dev-key-v1"))
	priv := ed25519.NewKeyFromSeed(seed[:])
	pub := priv.Public().(ed25519.PublicKey)
	if *printKeys {
		fmt.Printf("device_id=%s key_id=%s public_key_base64=%s\n", testutil.DeviceID, testutil.KeyID, base64.StdEncoding.EncodeToString(pub))
		return
	}

	conn, err := grpc.NewClient(*addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatal(err)
	}
	defer conn.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	accepted := testutil.SignedFrame(time.Now(), testutil.AllowedPackage, priv)
	resp, err := ingest.InvokeIngest(ctx, conn, accepted)
	fmt.Printf("allowed-package response: resp=%+v err=%v\n", resp, err)

	rejected := testutil.Frame(time.Now(), "com.google.android.gm")
	rejected.FrameID = "frame-sensitive-001"
	rejected.SequenceNumber = 2
	rejected, _ = auth.SignFrame(rejected, priv)
	resp, err = ingest.InvokeIngest(ctx, conn, rejected)
	fmt.Printf("sensitive-package response: resp=%+v err=%v\n", resp, err)
}
