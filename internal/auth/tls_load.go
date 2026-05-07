package auth

import (
	"crypto/tls"
	"errors"
	"fmt"
	"io/fs"
	"os"
)

// LoadTLSKeyPair loads a TLS certificate + private-key pair from
// the given file paths, applying defense-in-depth checks the bare
// tls.LoadX509KeyPair does not perform:
//
//  1. The key file MUST NOT be world-readable (mode & 0o077 != 0).
//     Private keys leak through casual `cat`, accidental tarball
//     creation, or careless backup tooling. Refuse to load instead
//     of silently allowing a misconfigured deployment. Set the
//     INGERO_TLS_ALLOW_LOOSE_KEY_PERMS=1 env var to opt out (e.g.,
//     for ephemeral CI environments where chmod isn't available).
//  2. Neither path may be a directory; tls.LoadX509KeyPair would
//     return an opaque "is a directory" error. Surface a clear
//     error here so the operator immediately understands the
//     configuration mistake.
//
// v0.15 item D (folds v0.14 R3 ★3).
func LoadTLSKeyPair(certPath, keyPath string) (tls.Certificate, error) {
	if err := preflightTLSPath(certPath, "cert"); err != nil {
		return tls.Certificate{}, err
	}
	if err := preflightTLSPath(keyPath, "key"); err != nil {
		return tls.Certificate{}, err
	}
	if err := checkKeyFilePermissions(keyPath); err != nil {
		return tls.Certificate{}, err
	}
	return tls.LoadX509KeyPair(certPath, keyPath)
}

// preflightTLSPath verifies the path exists and is a regular file
// (not a directory). label is used in error messages to distinguish
// the cert vs key surface.
func preflightTLSPath(path, label string) error {
	if path == "" {
		return fmt.Errorf("TLS %s path is empty", label)
	}
	st, err := os.Stat(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return fmt.Errorf("TLS %s file does not exist: %s", label, path)
		}
		return fmt.Errorf("TLS %s file stat: %w", label, err)
	}
	if st.IsDir() {
		return fmt.Errorf("TLS %s path is a directory, expected a file: %s", label, path)
	}
	return nil
}

// checkKeyFilePermissions enforces the 0o077-clear rule unless the
// operator has set INGERO_TLS_ALLOW_LOOSE_KEY_PERMS=1.
func checkKeyFilePermissions(keyPath string) error {
	if os.Getenv("INGERO_TLS_ALLOW_LOOSE_KEY_PERMS") == "1" {
		return nil
	}
	st, err := os.Stat(keyPath)
	if err != nil {
		return fmt.Errorf("TLS key file stat: %w", err)
	}
	mode := st.Mode().Perm()
	if mode&0o077 != 0 {
		return fmt.Errorf("TLS key file %s has loose permissions %#o (group/world readable); chmod 0600 or set INGERO_TLS_ALLOW_LOOSE_KEY_PERMS=1 to opt out",
			keyPath, mode)
	}
	return nil
}
