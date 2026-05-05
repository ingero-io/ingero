// Package docs exposes documentation files that ship inside the ingero
// binary so subcommands can use them as offline fallbacks.
package docs

import _ "embed"

// GPURatesMD is the contents of gpu_rates.md, used by `ingero rates update`
// when the canonical fleet URL is unreachable. The CLI extracts the
// embedded ```yaml code block from this file and writes it to disk.
//
//go:embed gpu_rates.md
var GPURatesMD []byte
