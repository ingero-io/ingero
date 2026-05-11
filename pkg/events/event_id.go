package events

import (
	"encoding/binary"
	"hash/fnv"
)

// DeterministicID computes a 16-byte FNV-128a digest of the workload-key
// tuple plus a domain salt. The salt lets callers derive distinct digests
// from the same (pid, streamHandle, tsNanos) context without an extra
// round of hashing - e.g., the OTel TraceID and SpanID for one outlier
// share workload identity but need different bytes on the wire.
//
// Used as a deterministic fallback when crypto/rand is unavailable
// (sandboxed runtimes that deny /dev/urandom). Without a deterministic
// fallback the historical behavior was an all-zeros TraceID / EventID,
// which Tempo and OTel collectors collapse into a single bucket so every
// failure-window emit looks like one downstream row.
//
// Collision-resistant within one process across distinct outliers because
// tsNanos moves forward and the (pid, streamHandle) tuple is unique per
// workload. Cross-process collisions are possible but the fallback path
// fires only on the rare CSPRNG failure and the workload-key still
// disambiguates the most useful operator queries.
func DeterministicID(pid uint32, streamHandle uint64, tsNanos int64, salt byte) [16]byte {
	h := fnv.New128a()
	var buf [21]byte
	binary.BigEndian.PutUint32(buf[0:4], pid)
	binary.BigEndian.PutUint64(buf[4:12], streamHandle)
	binary.BigEndian.PutUint64(buf[12:20], uint64(tsNanos))
	buf[20] = salt
	h.Write(buf[:])
	var out [16]byte
	h.Sum(out[:0])
	return out
}
