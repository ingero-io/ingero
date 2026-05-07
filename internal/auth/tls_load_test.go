package auth

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeTestCertKeyPair(t *testing.T, dir string, certName, keyName string, keyMode os.FileMode) (string, string) {
	t.Helper()
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: "test"},
		NotBefore:    time.Now(),
		NotAfter:     time.Now().Add(time.Hour),
	}
	derBytes, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &priv.PublicKey, priv)
	if err != nil {
		t.Fatalf("CreateCertificate: %v", err)
	}
	certPath := filepath.Join(dir, certName)
	keyPath := filepath.Join(dir, keyName)
	if err := os.WriteFile(certPath, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: derBytes}), 0o644); err != nil {
		t.Fatalf("write cert: %v", err)
	}
	keyBytes, err := x509.MarshalECPrivateKey(priv)
	if err != nil {
		t.Fatalf("MarshalECPrivateKey: %v", err)
	}
	if err := os.WriteFile(keyPath, pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyBytes}), keyMode); err != nil {
		t.Fatalf("write key: %v", err)
	}
	return certPath, keyPath
}

func TestLoadTLSKeyPair_HappyPath(t *testing.T) {
	dir := t.TempDir()
	cert, key := writeTestCertKeyPair(t, dir, "cert.pem", "key.pem", 0o600)
	if _, err := LoadTLSKeyPair(cert, key); err != nil {
		t.Errorf("happy path failed: %v", err)
	}
}

func TestLoadTLSKeyPair_RejectsLooseKeyPerms(t *testing.T) {
	dir := t.TempDir()
	cert, key := writeTestCertKeyPair(t, dir, "cert.pem", "key.pem", 0o644)
	_, err := LoadTLSKeyPair(cert, key)
	if err == nil {
		t.Fatal("expected error for 0o644 key perms")
	}
	if !strings.Contains(err.Error(), "loose permissions") {
		t.Errorf("error message should mention loose permissions: %v", err)
	}
}

func TestLoadTLSKeyPair_OptOutEnvAllowsLoosePerms(t *testing.T) {
	dir := t.TempDir()
	cert, key := writeTestCertKeyPair(t, dir, "cert.pem", "key.pem", 0o644)
	t.Setenv("INGERO_TLS_ALLOW_LOOSE_KEY_PERMS", "1")
	if _, err := LoadTLSKeyPair(cert, key); err != nil {
		t.Errorf("opt-out env should bypass perm check: %v", err)
	}
}

func TestLoadTLSKeyPair_RejectsCertIsDirectory(t *testing.T) {
	dir := t.TempDir()
	cert, key := writeTestCertKeyPair(t, dir, "cert.pem", "key.pem", 0o600)
	_ = cert
	_, err := LoadTLSKeyPair(dir, key)
	if err == nil {
		t.Fatal("expected error when cert path is a directory")
	}
	if !strings.Contains(err.Error(), "directory") {
		t.Errorf("error should mention directory: %v", err)
	}
}

func TestLoadTLSKeyPair_RejectsKeyIsDirectory(t *testing.T) {
	dir := t.TempDir()
	cert, _ := writeTestCertKeyPair(t, dir, "cert.pem", "key.pem", 0o600)
	_, err := LoadTLSKeyPair(cert, dir)
	if err == nil {
		t.Fatal("expected error when key path is a directory")
	}
}

func TestLoadTLSKeyPair_RejectsMissingFile(t *testing.T) {
	_, err := LoadTLSKeyPair("/nonexistent/cert", "/nonexistent/key")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestLoadTLSKeyPair_RejectsEmptyPath(t *testing.T) {
	if _, err := LoadTLSKeyPair("", "/some/key"); err == nil {
		t.Errorf("empty cert path should error")
	}
	if _, err := LoadTLSKeyPair("/some/cert", ""); err == nil {
		t.Errorf("empty key path should error")
	}
}
