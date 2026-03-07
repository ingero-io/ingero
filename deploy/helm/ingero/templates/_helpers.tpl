{{/*
Chart name.
*/}}
{{- define "ingero.name" -}}
ingero
{{- end -}}

{{/*
Fully qualified name: <release>-ingero, truncated to 63 chars.
*/}}
{{- define "ingero.fullname" -}}
{{- printf "%s-ingero" .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels applied to every resource.
*/}}
{{- define "ingero.labels" -}}
app.kubernetes.io/name: {{ include "ingero.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{/*
Selector labels (subset of common labels, immutable on Deployments/DaemonSets).
*/}}
{{- define "ingero.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ingero.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
