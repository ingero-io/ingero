package annotate

import (
	"fmt"
	"net"

	"golang.org/x/sys/unix"

	"github.com/ingero-io/ingero/pkg/annotate"
)

// peerCred extracts the SO_PEERCRED credentials (uid, gid, pid) of the
// process on the other end of a Unix-domain connection. The annotation
// socket is an inbound trust boundary, so every accepted connection's
// writer identity is captured and recorded as provenance on every row -
// a poisoned or surprising annotation is then traceable to the writing
// process even when group access is enabled.
//
// Returns a zero Provenance and an error when the connection is not a
// *net.UnixConn or the getsockopt fails. The caller continues with zero
// provenance rather than dropping the connection.
func peerCred(conn net.Conn) (annotate.Provenance, error) {
	uc, ok := conn.(*net.UnixConn)
	if !ok {
		return annotate.Provenance{}, fmt.Errorf("annotate: connection is not a unix socket")
	}
	raw, err := uc.SyscallConn()
	if err != nil {
		return annotate.Provenance{}, fmt.Errorf("annotate: SyscallConn: %w", err)
	}

	var cred *unix.Ucred
	var credErr error
	controlErr := raw.Control(func(fd uintptr) {
		cred, credErr = unix.GetsockoptUcred(int(fd),
			unix.SOL_SOCKET, unix.SO_PEERCRED)
	})
	if controlErr != nil {
		return annotate.Provenance{}, fmt.Errorf("annotate: raw control: %w", controlErr)
	}
	if credErr != nil {
		return annotate.Provenance{}, fmt.Errorf("annotate: SO_PEERCRED: %w", credErr)
	}
	return annotate.Provenance{
		PeerUID: cred.Uid,
		PeerGID: cred.Gid,
		PeerPID: uint32(cred.Pid),
	}, nil
}
