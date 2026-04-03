package dashboard

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"embed"
	"encoding/pem"
	"fmt"
	"io/fs"
	"math/big"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/ingero-io/ingero/internal/discover"
	"github.com/ingero-io/ingero/internal/store"
)

//go:embed static
var staticFiles embed.FS

// Server is the dashboard HTTPS server.
type Server struct {
	store    *store.Store
	addr     string
	certFile string
	keyFile  string
	noTLS    bool     // serve plain HTTP instead of HTTPS
	gpuInfo  *gpuInfo // cached at startup, nil if no GPU
}

// New creates a dashboard server backed by the given SQLite store.
func New(s *store.Store, addr, certFile, keyFile string) *Server {
	srv := &Server{
		store:    s,
		addr:     addr,
		certFile: certFile,
		keyFile:  keyFile,
	}
	// Cache GPU info at startup (doesn't change at runtime).
	srv.gpuInfo = probeGPUInfo()
	return srv
}

// SetNoTLS configures the server to serve plain HTTP instead of HTTPS.
// Used for fleet queries on trusted networks (private subnets, VPNs, WireGuard).
func (s *Server) SetNoTLS(noTLS bool) {
	s.noTLS = noTLS
}

// probeGPUInfo queries nvidia-smi for GPU model, driver version, and CUDA version.
func probeGPUInfo() *gpuInfo {
	g := &gpuInfo{}
	if result := discover.CheckGPUModel(); result.OK {
		g.Model = result.Value
	}
	if result := discover.CheckNVIDIA(); result.OK {
		g.DriverVersion = result.Value
	}
	g.CUDAVersion = discover.CUDAVersion()
	if g.Model == "" && g.DriverVersion == "" && g.CUDAVersion == "" {
		return nil
	}
	return g
}

// Start begins serving the dashboard over HTTPS. Blocks until ctx is done.
func (s *Server) Start(ctx context.Context) error {
	mux := http.NewServeMux()

	// API endpoints.
	mux.HandleFunc("/api/v1/overview", s.handleOverview)
	mux.HandleFunc("/api/v1/ops", s.handleOps)
	mux.HandleFunc("/api/v1/chains", s.handleChains)
	mux.HandleFunc("/api/v1/snapshots", s.handleSnapshots)
	mux.HandleFunc("/api/v1/capabilities", s.handleCapabilities)
	mux.HandleFunc("/api/v1/graph-metrics", s.handleGraphMetrics)
	mux.HandleFunc("/api/v1/graph-events", s.handleGraphEvents)
	mux.HandleFunc("/api/v1/query", s.handleQuery)
	mux.HandleFunc("/api/v1/time", s.handleTime)

	// Static files (embedded HTML/JS).
	staticSub, err := fs.Sub(staticFiles, "static")
	if err != nil {
		return fmt.Errorf("embedding static files: %w", err)
	}
	mux.Handle("/", http.FileServer(http.FS(staticSub)))

	// For fleet queries on trusted networks, allow any Host header.
	// For HTTPS mode, guard against DNS rebinding.
	var handler http.Handler
	if s.noTLS {
		handler = mux
	} else {
		handler = hostGuard(mux)
	}

	// Plain HTTP mode (--no-tls) for fleet queries on trusted networks.
	if s.noTLS {
		httpSrv := &http.Server{
			Addr:    s.addr,
			Handler: handler,
		}
		go func() {
			<-ctx.Done()
			httpSrv.Close()
		}()

		fmt.Fprintf(os.Stderr, "Dashboard HTTP server listening on %s (plain HTTP, no TLS)\n", s.addr)
		fmt.Fprintf(os.Stderr, "  WARNING: no encryption — use only on trusted networks\n")

		ln, err := net.Listen("tcp", s.addr)
		if err != nil {
			return err
		}
		defer ln.Close()

		err = httpSrv.Serve(ln)
		if err == http.ErrServerClosed {
			return nil
		}
		return err
	}

	// TLS 1.3 minimum.
	tlsCfg := &tls.Config{
		MinVersion: tls.VersionTLS13,
	}

	if s.certFile != "" && s.keyFile != "" {
		cert, err := tls.LoadX509KeyPair(s.certFile, s.keyFile)
		if err != nil {
			return fmt.Errorf("loading TLS certificate: %w", err)
		}
		tlsCfg.Certificates = []tls.Certificate{cert}
		fmt.Fprintf(os.Stderr, "  TLS certificate: %s\n", s.certFile)
	} else {
		cert, fingerprint, err := generateSelfSignedCert()
		if err != nil {
			return fmt.Errorf("generating self-signed certificate: %w", err)
		}
		tlsCfg.Certificates = []tls.Certificate{cert}
		fmt.Fprintf(os.Stderr, "  Generated ephemeral self-signed certificate (valid 24h)\n")
		fmt.Fprintf(os.Stderr, "  SHA-256 fingerprint: %s\n", fingerprint)
	}

	httpSrv := &http.Server{
		Addr:      s.addr,
		Handler:   handler,
		TLSConfig: tlsCfg,
	}

	// Graceful shutdown.
	go func() {
		<-ctx.Done()
		httpSrv.Close()
	}()

	fmt.Fprintf(os.Stderr, "Dashboard HTTPS server listening on %s (TLS 1.3)\n", s.addr)
	fmt.Fprintf(os.Stderr, "  Open https://localhost%s in your browser\n", s.addr)
	fmt.Fprintf(os.Stderr, "  Remote: ssh -L %s:localhost:%s user@gpu-vm\n",
		strings.TrimPrefix(s.addr, ":"), strings.TrimPrefix(s.addr, ":"))

	ln, err := tls.Listen("tcp", s.addr, tlsCfg)
	if err != nil {
		return err
	}
	defer ln.Close()

	err = httpSrv.Serve(ln)
	if err == http.ErrServerClosed {
		return nil
	}
	return err
}

// generateSelfSignedCert creates an ephemeral ECDSA P-256 certificate valid
// for 24 hours, bound to localhost/127.0.0.1/::1.
func generateSelfSignedCert() (tls.Certificate, string, error) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return tls.Certificate{}, "", err
	}

	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return tls.Certificate{}, "", err
	}

	tmpl := &x509.Certificate{
		SerialNumber:          serial,
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(24 * time.Hour),
		KeyUsage:              x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IPAddresses:           []net.IP{net.IPv4(127, 0, 0, 1), net.IPv6loopback},
		DNSNames:              []string{"localhost"},
	}

	certDER, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		return tls.Certificate{}, "", err
	}

	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})
	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		return tls.Certificate{}, "", err
	}
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER})

	tlsCert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		return tls.Certificate{}, "", err
	}

	hash := sha256.Sum256(certDER)
	parts := make([]string, sha256.Size)
	for i, b := range hash {
		parts[i] = fmt.Sprintf("%02X", b)
	}
	fingerprint := strings.Join(parts, ":")

	return tlsCert, fingerprint, nil
}

// hostGuard rejects requests whose Host header does not match localhost,
// 127.0.0.1, or [::1]. Mitigates DNS rebinding attacks.
func hostGuard(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		host := r.Host
		if h, _, err := net.SplitHostPort(host); err == nil {
			host = h
		}
		switch host {
		case "localhost", "127.0.0.1", "::1":
			next.ServeHTTP(w, r)
		default:
			http.Error(w, "forbidden: invalid Host header", http.StatusForbidden)
		}
	})
}
