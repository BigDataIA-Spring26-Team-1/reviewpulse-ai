param(
    [string]$ProjectId = "reviewpulse-492923",
    [string]$Region = "us-central1",
    [string]$Service = "reviewpulse-api"
)

$ErrorActionPreference = "Stop"

$gcloud = Get-Command gcloud -ErrorAction SilentlyContinue
if (-not $gcloud) {
    $candidate = Join-Path $env:LOCALAPPDATA "Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
    if (Test-Path $candidate) {
        $gcloudPath = $candidate
    } else {
        throw "gcloud was not found on PATH or under LOCALAPPDATA."
    }
} else {
    $gcloudPath = $gcloud.Source
}

$tag = Get-Date -Format "yyyyMMdd-HHmmss"
$image = "gcr.io/$ProjectId/$Service`:$tag"

function Assert-LastCommandSucceeded {
    param([string]$Step)

    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

Write-Host "Building $image with Cloud Build..."
& $gcloudPath builds submit `
    --project $ProjectId `
    --config deploy/gcp/cloudbuild.api.yaml `
    --substitutions "_IMAGE=$image" `
    .
Assert-LastCommandSucceeded "Cloud Build"

Write-Host "Deploying $Service to Cloud Run..."
& $gcloudPath run deploy $Service `
    --project $ProjectId `
    --region $Region `
    --image $image `
    --platform managed
Assert-LastCommandSucceeded "Cloud Run deploy"

$url = & $gcloudPath run services describe $Service `
    --project $ProjectId `
    --region $Region `
    --format "value(status.url)"
Assert-LastCommandSucceeded "Cloud Run describe"

Write-Host "Deployed URL: $url"
Write-Host "Health:"
Invoke-RestMethod -Uri "$url/health" -TimeoutSec 30
