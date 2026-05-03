package tracing

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"

	"github.com/ingero-io/ingero/pkg/contract"
)

const tracerName = "ingero"

// Config wires the dormant otlp: block in configs/ingero.yaml to a real
// OTLP/HTTP traces exporter. Enabled=false yields a non-recording tracer
// with zero allocations on the hot path.
type Config struct {
	Enabled        bool
	Endpoint       string
	Insecure       bool
	NodeID         string
	ClusterID      string
	ServiceVersion string
}

// Init returns a tracer plus a shutdown function. When cfg.Enabled is
// false, the returned tracer is non-recording and shutdown is a no-op.
func Init(ctx context.Context, cfg Config) (trace.Tracer, func(context.Context) error, error) {
	tracer, _, shutdown, err := initWithProvider(ctx, cfg)
	return tracer, shutdown, err
}

// initWithProvider exposes the underlying *sdktrace.TracerProvider so tests
// can drive ForceFlush synchronously. Returns a nil provider when Enabled
// is false.
func initWithProvider(ctx context.Context, cfg Config) (trace.Tracer, *sdktrace.TracerProvider, func(context.Context) error, error) {
	if !cfg.Enabled {
		return noop.NewTracerProvider().Tracer(tracerName), nil, func(context.Context) error { return nil }, nil
	}

	endpoint, insecure, err := parseEndpoint(cfg.Endpoint, cfg.Insecure)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("tracing: endpoint: %w", err)
	}

	opts := []otlptracehttp.Option{otlptracehttp.WithEndpoint(endpoint)}
	if insecure {
		opts = append(opts, otlptracehttp.WithInsecure())
	}
	exp, err := otlptracehttp.New(ctx, opts...)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("tracing: exporter: %w", err)
	}

	res := resource.NewWithAttributes(
		semconv.SchemaURL,
		semconv.ServiceName(tracerName),
		semconv.ServiceVersion(cfg.ServiceVersion),
		semconv.K8SNodeName(cfg.NodeID),
		attribute.String(contract.AttrNodeID, cfg.NodeID),
		attribute.String(contract.AttrClusterID, cfg.ClusterID),
	)

	bsp := sdktrace.NewBatchSpanProcessor(exp)
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithSpanProcessor(bsp),
		sdktrace.WithResource(res),
	)
	return tp.Tracer(tracerName), tp, tp.Shutdown, nil
}

// parseEndpoint returns the host:port form expected by otlptracehttp.WithEndpoint
// and a flag indicating whether to use HTTP (insecure). Accepts either a
// bare host:port or a full URL; for full URLs the scheme overrides the
// caller's Insecure preference.
func parseEndpoint(raw string, insecure bool) (string, bool, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", false, errors.New("empty endpoint")
	}
	if strings.Contains(raw, "://") {
		u, err := url.Parse(raw)
		if err != nil {
			return "", false, err
		}
		if u.Host == "" {
			return "", false, fmt.Errorf("missing host in %q", raw)
		}
		switch u.Scheme {
		case "http":
			return u.Host, true, nil
		case "https":
			return u.Host, false, nil
		default:
			return "", false, fmt.Errorf("unsupported scheme %q", u.Scheme)
		}
	}
	if !strings.Contains(raw, ":") {
		return "", false, fmt.Errorf("endpoint must be host:port or URL: %q", raw)
	}
	return raw, insecure, nil
}
