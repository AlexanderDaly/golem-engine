package storage

import (
	"bufio"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"sync"

	"github.com/internal/golem-harness/server/internal/trajectory"
)

var ErrUnsafeFrame = errors.New("storage accepts sanitized frames only")

type Sink interface {
	Store(frame trajectory.Frame) error
}

type MemorySink struct {
	mu     sync.Mutex
	Frames []trajectory.Frame
}

func (m *MemorySink) Store(f trajectory.Frame) error {
	if err := ValidateSanitized(f); err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.Frames = append(m.Frames, f)
	return nil
}

type JSONLSink struct {
	mu   sync.Mutex
	file *os.File
	w    *bufio.Writer
}

func NewJSONLSink(path string) (*JSONLSink, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0600)
	if err != nil {
		return nil, err
	}
	return &JSONLSink{file: f, w: bufio.NewWriter(f)}, nil
}
func (s *JSONLSink) Store(f trajectory.Frame) error {
	if err := ValidateSanitized(f); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	b, err := json.Marshal(f)
	if err != nil {
		return err
	}
	if _, err = s.w.Write(append(b, '\n')); err != nil {
		return err
	}
	return s.w.Flush()
}
func (s *JSONLSink) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.w.Flush(); err != nil {
		_ = s.file.Close()
		return err
	}
	return s.file.Close()
}

func ValidateSanitized(f trajectory.Frame) error {
	if len(f.Signature.Signature) > 0 || len(f.PreStorageOnly) > 0 {
		return ErrUnsafeFrame
	}
	if f.Sanitizer.SanitizerVersion == "" {
		return ErrUnsafeFrame
	}
	for _, n := range f.UITree.Nodes {
		if strings.TrimSpace(n.Text.Raw) != "" || strings.TrimSpace(n.ContentDescription.Raw) != "" {
			return ErrUnsafeFrame
		}
	}
	return nil
}
