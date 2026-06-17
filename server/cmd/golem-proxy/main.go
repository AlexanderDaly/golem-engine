package main

import (
	"context"
	"flag"
	"log/slog"
	"os"
	"os/signal"
	"sync"

	"github.com/internal/golem-harness/server/internal/auth"
	"github.com/internal/golem-harness/server/internal/config"
	"github.com/internal/golem-harness/server/internal/ingest"
	"github.com/internal/golem-harness/server/internal/sanitize"
	"github.com/internal/golem-harness/server/internal/storage"
	"google.golang.org/grpc/credentials"
)

func main() {
	cfgPath := flag.String("config", "", "path to JSON config")
	flag.Parse()
	if *cfgPath == "" {
		slog.Error("missing -config")
		os.Exit(2)
	}
	cfg, err := config.Load(*cfgPath)
	if err != nil {
		slog.Error("invalid config", "error", err)
		os.Exit(2)
	}
	reg, err := cfg.Registry()
	if err != nil {
		slog.Error("invalid registry", "error", err)
		os.Exit(2)
	}
	sink, err := storage.NewJSONLSink(cfg.StorageJSONLPath)
	if err != nil {
		slog.Error("storage init failed", "error", err)
		os.Exit(1)
	}
	defer sink.Close()
	var creds credentials.TransportCredentials
	if cfg.MTLS.Enabled {
		tlsCfg, err := auth.LoadServerTLS(cfg.MTLS.CertFile, cfg.MTLS.KeyFile, cfg.MTLS.ClientCAFile, cfg.MTLS.RequireClientCert)
		if err != nil {
			slog.Error("tls init failed", "error", err)
			os.Exit(1)
		}
		creds = credentials.NewTLS(tlsCfg)
	}
	svc := &ingest.Service{Verifier: &auth.Verifier{Registry: reg, Replay: auth.NewReplayCache(), MaxSkew: cfg.MaxSkew()}, Sanitizer: sanitize.NewPipeline(cfg.AllowedPackages), Store: sink, MaxFrameBytes: cfg.MaxFrameBytes, Log: slog.Default()}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		if err := ingest.ServeHTTP(ctx, cfg.HTTPAddr, func() bool { return true }); err != nil {
			slog.Error("http server failed", "error", err)
			stop()
		}
	}()
	go func() {
		defer wg.Done()
		if err := ingest.ServeGRPC(ctx, cfg.GRPCAddr, svc, creds); err != nil {
			slog.Error("grpc server failed", "error", err)
			stop()
		}
	}()
	wg.Wait()
}
