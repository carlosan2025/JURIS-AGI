{{/*
Expand the name of the chart.
*/}}
{{- define "juris-agi.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "juris-agi.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "juris-agi.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "juris-agi.labels" -}}
helm.sh/chart: {{ include "juris-agi.chart" . }}
{{ include "juris-agi.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "juris-agi.selectorLabels" -}}
app.kubernetes.io/name: {{ include "juris-agi.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "juris-agi.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "juris-agi.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Redis host
*/}}
{{- define "juris-agi.redisHost" -}}
{{- if .Values.redis.enabled }}
{{- printf "%s-redis-master" (include "juris-agi.fullname" .) }}
{{- else }}
{{- .Values.externalRedis.host }}
{{- end }}
{{- end }}

{{/*
Redis URL
*/}}
{{- define "juris-agi.redisUrl" -}}
{{- $host := include "juris-agi.redisHost" . }}
{{- $port := .Values.redis.enabled | ternary 6379 .Values.externalRedis.port }}
{{- printf "redis://%s:%d/0" $host (int $port) }}
{{- end }}
