# gpu_rates.yaml

Reference for the cost-of-problem catalog consumed by the v0.11+
Prometheus recording rules and the v0.13+ `ingero rates update`
CLI subcommand.

## Where the canonical file lives

The canonical `gpu_rates.yaml` ships in the `ingero-fleet` repo at
`examples/gpu_rates.yaml`. `ingero rates update` fetches it from:

```
https://raw.githubusercontent.com/ingero-io/ingero-fleet/main/examples/gpu_rates.yaml
```

When that URL is unreachable (network failure, HTTP error, malformed
response), the CLI falls back to the YAML block embedded in this
file. The fallback ships with every `ingero` binary, so a fresh box
with no network can still author a usable rates file.

A separate public catalog repo (`ingero-io/gpu-rates`) is planned
for a future release. When it lands, the default URL will move and
the fallback embedded here will be retired.

## Schema

```
currency_name: USD               # ISO 4217 code; single per file.
currency_symbol: "$"             # Symbol surfaced in dashboards.

providers:
  <provider_id>:                 # Free-form key (ec2 / gcp / lambda / ...).
    "<gpu_model_string>": <hourly_rate>
    ...
  ...

fallback_rate: 0.0               # Used when (provider, gpu_model) misses.
                                 # Set low so missing rows never flatter
                                 # the dashboard.
```

The fallback block at the bottom of this file is the only ```yaml
fence. The CLI's offline path keys on that fence — please do not
add other yaml fences to this document.

`<gpu_model_string>` MUST match exactly the value the agent emits
as `ingero.gpu_model` (sourced from `nvidia-smi --query-gpu=name`).
When in doubt, run on a node:

```
nvidia-smi --query-gpu=name --format=csv,noheader
```

and copy the string verbatim into the YAML.

## How to edit

1. Open `ingero-fleet/examples/gpu_rates.yaml` (or the file
   `ingero rates update` wrote locally).
2. Add or amend rows under `providers.<provider_id>` using public
   list prices unless your contracted rate is intended to flow into
   the dashboard.
3. Validate by running `ingero rates update --url file://<path>`
   (validation runs before write; malformed files are rejected).

Public list prices are an **upper bound** on real exposure: spot,
RIs, committed-use discounts, and negotiated rates are not modeled.
Operators with non-list pricing maintain a private fork.

## Embedded fallback

The block below is the fallback used when the canonical URL is
unreachable. Keep it byte-for-byte in sync with the canonical file
in `ingero-fleet/examples/gpu_rates.yaml` at release time.

```yaml
currency_name: USD
currency_symbol: "$"

providers:
  ec2:
    "NVIDIA H100 80GB HBM3":     12.29
    "NVIDIA A100-SXM4-80GB":      4.10
    "NVIDIA A100-SXM4-40GB":      3.27
    "NVIDIA L4 24GB":             0.85
  gcp:
    "NVIDIA H100 80GB HBM3":     11.06
    "NVIDIA A100-SXM4-80GB":      4.12
    "NVIDIA L4 24GB":             0.85
  azure:
    "NVIDIA H100 80GB HBM3":     12.29
    "NVIDIA A100-SXM4-80GB":      3.40
  lambda:
    "NVIDIA H100 80GB HBM3":      2.49
    "NVIDIA A100-SXM4-80GB":      1.29
    "NVIDIA GH200 480GB":         2.29
    "NVIDIA L40S":                1.29
  coreweave:
    "NVIDIA H100 80GB HBM3":      4.25
    "NVIDIA A100-SXM4-80GB":      2.39
    "NVIDIA L40S":                1.50

fallback_rate: 0.0
```
