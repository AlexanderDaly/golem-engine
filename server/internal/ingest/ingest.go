package ingest

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"

	"github.com/internal/golem-harness/server/internal/auth"
	"github.com/internal/golem-harness/server/internal/sanitize"
	"github.com/internal/golem-harness/server/internal/storage"
	"github.com/internal/golem-harness/server/internal/trajectory"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/encoding"
	"google.golang.org/grpc/status"
)

const MaxDefaultFrameBytes = 1 << 20

var ErrOversized = errors.New("oversized payload")

type Request struct {
	Frame trajectory.Frame `json:"frame"`
}
type Response struct {
	Accepted    bool     `json:"accepted"`
	Decision    string   `json:"decision"`
	ReasonCodes []string `json:"reason_codes,omitempty"`
}

type Service struct {
	Verifier  *auth.Verifier
	Sanitizer interface {
		Sanitize(trajectory.Frame) (trajectory.Frame, sanitize.Report, error)
	}
	Store         storage.Sink
	MaxFrameBytes int
	Log           *slog.Logger
}

func (s *Service) Ingest(ctx context.Context, req *Request) (*Response, error) {
	_ = ctx
	max := s.MaxFrameBytes
	if max <= 0 {
		max = MaxDefaultFrameBytes
	}
	b, err := json.Marshal(req.Frame)
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, "malformed frame")
	}
	if len(b) > max {
		return nil, status.Error(codes.ResourceExhausted, ErrOversized.Error())
	}
	if err := s.Verifier.Verify(req.Frame); err != nil {
		return nil, status.Error(codes.Unauthenticated, err.Error())
	}
	safe, report, err := s.Sanitizer.Sanitize(req.Frame)
	if err != nil {
		s.safeLog("sanitizer failure", req.Frame, []string{"sanitizer_failure"})
		return &Response{Accepted: false, Decision: string(sanitize.Drop), ReasonCodes: []string{"sanitizer_failure"}}, nil
	}
	if report.Decision != sanitize.Accept {
		s.safeLog("frame not accepted", req.Frame, report.ReasonCodes)
		return &Response{Accepted: false, Decision: string(report.Decision), ReasonCodes: report.ReasonCodes}, nil
	}
	if err := s.Store.Store(safe); err != nil {
		return nil, status.Error(codes.Internal, "storage rejected frame")
	}
	s.safeLog("frame accepted", safe, report.ReasonCodes)
	return &Response{Accepted: true, Decision: string(report.Decision), ReasonCodes: report.ReasonCodes}, nil
}
func (s *Service) safeLog(msg string, f trajectory.Frame, reasons []string) {
	l := s.Log
	if l == nil {
		l = slog.Default()
	}
	l.Info(msg, "device_id", f.Device.DeviceID, "trajectory_id", f.TrajectoryID, "frame_id", f.FrameID, "sequence", f.SequenceNumber, "package", f.App.ForegroundPackage, "reasons", reasons)
}

type grpcService interface {
	Ingest(context.Context, *Request) (*Response, error)
}

func RegisterGRPC(g *grpc.Server, svc *Service) {
	g.RegisterService(&grpc.ServiceDesc{ServiceName: "golem.v1.TelemetryIngest", HandlerType: (*grpcService)(nil), Methods: []grpc.MethodDesc{{MethodName: "IngestFrame", Handler: func(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
		in := new(Request)
		if err := dec(in); err != nil {
			return nil, err
		}
		if interceptor == nil {
			return srv.(*Service).Ingest(ctx, in)
		}
		info := &grpc.UnaryServerInfo{Server: srv, FullMethod: "/golem.v1.TelemetryIngest/IngestFrame"}
		handler := func(ctx context.Context, req interface{}) (interface{}, error) {
			return srv.(*Service).Ingest(ctx, req.(*Request))
		}
		return interceptor(ctx, in, info, handler)
	}}}}, svc)
}

func ServeGRPC(ctx context.Context, addr string, svc *Service, tlsCfg credentials.TransportCredentials) error {
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	opts := []grpc.ServerOption{grpc.ForceServerCodec(jsonCodec{})}
	if tlsCfg != nil {
		opts = append(opts, grpc.Creds(tlsCfg))
	}
	srv := grpc.NewServer(opts...)
	RegisterGRPC(srv, svc)
	go func() { <-ctx.Done(); srv.GracefulStop() }()
	return srv.Serve(lis)
}
func ServeHTTP(ctx context.Context, addr string, ready func() bool) error {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) { _, _ = io.WriteString(w, "ok\n") })
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		if ready != nil && !ready() {
			http.Error(w, "not ready", 503)
			return
		}
		_, _ = io.WriteString(w, "ready\n")
	})
	srv := &http.Server{Addr: addr, Handler: mux}
	go func() { <-ctx.Done(); _ = srv.Shutdown(context.Background()) }()
	err := srv.ListenAndServe()
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}

type jsonCodec struct{}

func (jsonCodec) Marshal(v interface{}) ([]byte, error)      { return json.Marshal(v) }
func (jsonCodec) Unmarshal(data []byte, v interface{}) error { return json.Unmarshal(data, v) }
func (jsonCodec) Name() string                               { return "json" }
func init()                                                  { encoding.RegisterCodec(jsonCodec{}) }

func InvokeIngest(ctx context.Context, cc *grpc.ClientConn, f trajectory.Frame) (*Response, error) {
	out := new(Response)
	err := cc.Invoke(ctx, "/golem.v1.TelemetryIngest/IngestFrame", &Request{Frame: f}, out, grpc.ForceCodec(jsonCodec{}))
	if err != nil {
		return nil, fmt.Errorf("ingest rpc: %w", err)
	}
	return out, nil
}
