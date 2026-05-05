package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	gomcp "github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/ingero-io/ingero/internal/alerter"
)

// customDetailsLimit caps the serialized custom_details size. PagerDuty's
// 512 KiB total payload limit is the hard ceiling; 256 KiB leaves headroom
// for the surrounding envelope without a second round-trip on rejection.
const customDetailsLimit = 256 * 1024

type PagerDutyTriggerInput struct {
	Summary       string         `json:"summary" jsonschema:"One-line incident summary,required"`
	Severity      string         `json:"severity" jsonschema:"PD severity,required,enum=info,enum=warning,enum=error,enum=critical"`
	Source        string         `json:"source,omitempty" jsonschema:"Identifies origin (default: ingero)"`
	DedupKey      string         `json:"dedup_key,omitempty" jsonschema:"PD dedup key; auto-generated UUID when omitted"`
	CustomDetails map[string]any `json:"custom_details,omitempty" jsonschema:"Free-form context (max 256 KiB serialized)"`
}

type PagerDutyTriggerOutput struct {
	DedupKey string `json:"dedup_key"`
	Status   string `json:"status"`
}

// RegisterPagerDutyTool wires the pagerduty_trigger tool into the MCP server.
// getPD is called at handler invocation (not registration) so the server can
// register the tool once and have its PagerDuty backend wired in later via
// a setter. The closure may return nil; in that case the handler returns an
// actionable "not configured" error. Tool is always registered so the
// inventory is stable across nodes regardless of config.
//
// v0.14 R3 ★4: this tool emits an outbound webhook at operator cost.
// When the agent's MCP listener has no caller-identity check (the v0.14
// default; bearer auth on the agent MCP is Sec ★2 #2 in the deferred
// pile), any local user or compromised on-host process could trigger
// PagerDuty pages at operator cost. The handler gates on `enabledFn()`;
// when it returns false the handler returns an actionable error
// instructing the operator to enable via `--enable-mcp-pagerduty` and
// pair with `--mcp-bearer-token` on the agent MCP listener. Registration
// stays so AI callers introspecting the inventory see the tool exists
// (and the disabled-state error explains the path forward).
func RegisterPagerDutyTool(server *gomcp.Server, getPD func() *alerter.PagerDuty, enabledFn func() bool) {
	gomcp.AddTool(server, &gomcp.Tool{
		Name:        "pagerduty_trigger",
		Description: "Create a PagerDuty incident with rich context. For AI-driven escalation during investigations.",
	}, func(ctx context.Context, req *gomcp.CallToolRequest, input PagerDutyTriggerInput) (*gomcp.CallToolResult, any, error) {
		if enabledFn != nil && !enabledFn() {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{&gomcp.TextContent{Text: "pagerduty_trigger is disabled on this agent MCP listener (v0.14 default). Enable with --enable-mcp-pagerduty AND --mcp-bearer-token to gate access. See agent docs/commands.md."}},
				IsError: true,
			}, nil, nil
		}
		var pd *alerter.PagerDuty
		if getPD != nil {
			pd = getPD()
		}
		out, err := handlePagerDutyTrigger(ctx, pd, input)
		if err != nil {
			return &gomcp.CallToolResult{
				Content: []gomcp.Content{&gomcp.TextContent{Text: err.Error()}},
				IsError: true,
			}, nil, nil
		}
		body, _ := json.Marshal(out)
		return &gomcp.CallToolResult{
			Content: []gomcp.Content{&gomcp.TextContent{Text: string(body)}},
		}, out, nil
	})
}

func handlePagerDutyTrigger(ctx context.Context, pd *alerter.PagerDuty, input PagerDutyTriggerInput) (*PagerDutyTriggerOutput, error) {
	if pd == nil {
		return nil, errors.New("pagerduty not configured: set alerter.pagerduty.routing_key in your configs/ingero.yaml or pass --pagerduty-routing-key to the mcp command")
	}
	switch input.Severity {
	case "info", "warning", "error", "critical":
	default:
		return nil, fmt.Errorf("invalid severity %q: must be one of info|warning|error|critical", input.Severity)
	}
	if input.CustomDetails != nil {
		raw, err := json.Marshal(input.CustomDetails)
		if err != nil {
			return nil, fmt.Errorf("custom_details: marshal: %w", err)
		}
		if len(raw) > customDetailsLimit {
			return nil, fmt.Errorf("custom_details exceeds 256 KiB limit (%d bytes)", len(raw))
		}
	}
	dedup, err := pd.Trigger(ctx, alerter.TriggerParams{
		Summary:       input.Summary,
		Severity:      input.Severity,
		Source:        input.Source,
		DedupKey:      input.DedupKey,
		CustomDetails: input.CustomDetails,
	})
	if err != nil {
		return nil, err
	}
	return &PagerDutyTriggerOutput{DedupKey: dedup, Status: "accepted"}, nil
}
