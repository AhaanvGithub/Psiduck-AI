# Psiduck-AI overnight eval: one command, unattended, resumable.
#
#   powershell -ExecutionPolicy Bypass -File .\run_overnight.ps1
#
# Smoke test -> lock N -> GSM8K latent -> GSM8K baseline -> AIME latent -> AIME baseline -> SUMMARY.md.
# Re-running the same command after a crash or reboot resumes where it stopped.
param(
    [switch]$AllowBattery,     # skip waiting for AC power before the smoke test
    [int]$MaxRestarts = 10
)

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Host "No .venv found. See README 'Setup'."; exit 1 }
New-Item -ItemType Directory -Force -Path results | Out-Null

# Keep Windows awake for as long as this script runs (the Python driver also does this).
Add-Type -Namespace Psiduck -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
[void][Psiduck.Power]::SetThreadExecutionState([uint32]"0x80000041")  # CONTINUOUS | SYSTEM_REQUIRED | AWAYMODE

# ntfy topic: random, generated once, reused on resume.
$topicFile = "results\ntfy_topic.txt"
if (-not (Test-Path $topicFile) -or -not (Get-Content $topicFile -Raw).Trim()) {
    $hex = -join ((1..16) | ForEach-Object { '{0:x}' -f (Get-Random -Maximum 16) })
    Set-Content -Path $topicFile -Value "psiduck-$hex" -Encoding ascii
}
$topic = (Get-Content $topicFile -Raw).Trim()

function Send-Ntfy([string]$msg, [string]$priority = "default") {
    try {
        Invoke-RestMethod -Method Post -Uri "https://ntfy.sh/$topic" -Body $msg `
            -Headers @{ Title = "Psiduck eval"; Priority = $priority } -TimeoutSec 15 | Out-Null
    } catch { Write-Host "(ntfy failed, continuing)" }
}

Write-Host "=================================================================="
Write-Host " ntfy topic: $topic"
Write-Host " Subscribe in the ntfy app, or open https://ntfy.sh/$topic"
Write-Host " Live status: results\STATUS.txt    Log: results\run.log"
Write-Host "=================================================================="

try {
    $battery = Get-CimInstance Win32_Battery -ErrorAction Stop
    if ($battery -and $battery.BatteryStatus -eq 1) {
        Write-Warning "Laptop is ON BATTERY. Plug it in, and keep the lid open (closing it can still sleep the machine)."
        Send-Ntfy "WARNING: laptop is on battery. Plug it in." "high"
    }
} catch {}
Write-Host "Keep the lid open: lid-close sleep is a power setting this script does not change."

Send-Ntfy "Psiduck eval starting (topic $topic)."

# Make sure the model + data are on disk before the long run (has its own retries).
& $py scripts\prefetch.py
if ($LASTEXITCODE -ne 0) { Send-Ntfy "Prefetch failed; the driver will retry downloads itself." "high" }

$runArgs = @("scripts\run_eval.py")
if ($AllowBattery) { $runArgs += "--allow-battery" }

for ($i = 0; $i -le $MaxRestarts; $i++) {
    & $py @runArgs
    $code = $LASTEXITCODE
    if ($code -eq 0) { break }
    $msg = "Driver exited with code $code (restart $($i + 1)/$MaxRestarts); resuming in 30 s."
    Write-Warning $msg
    Add-Content -Path results\run.log -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Send-Ntfy $msg "high"
    Start-Sleep -Seconds 30
}
if ($code -ne 0) { Send-Ntfy "Gave up after $MaxRestarts restarts. Check results\run.log." "urgent" }

[void][Psiduck.Power]::SetThreadExecutionState([uint32]"0x80000000")
Write-Host "Finished. See SUMMARY.md and results\STATUS.txt."
