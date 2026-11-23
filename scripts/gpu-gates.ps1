<#
.SYNOPSIS
  Runs the GPU enablement gates in order and tells you exactly where you stand.

.DESCRIPTION
  Implements the gates from notes/0002-gpu-enablement-checklist.md as one command.
  Stops at the first gate that fails and prints the specific fix, so you are never
  debugging driver + WSL + Docker at the same time.

  Read-only: it never installs or changes anything.

.PARAMETER From
  Start at a given gate (1-5). Use after fixing something to resume.

.PARAMETER IncludeMjx
  Also run gate 5 (JAX/MJX on device). Slow: uses the mjx image.

.EXAMPLE
  .\scripts\gpu-gates.ps1
  .\scripts\gpu-gates.ps1 -From 3
  .\scripts\gpu-gates.ps1 -IncludeMjx
#>
[CmdletBinding()]
param(
  [ValidateRange(1, 5)]
  [int]$From = 1,

  [switch]$IncludeMjx
)

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $PSScriptRoot
$Compose = @('-f', 'docker-compose.yml', '-f', 'docker-compose.gpu.yml')
$results = @()

function Run-Cmd {
  param([string]$File, [string[]]$ArgList, [string]$Dir = $Root)

  $outFile = [System.IO.Path]::GetTempFileName()
  $errFile = [System.IO.Path]::GetTempFileName()
  try {
    $proc = Start-Process -FilePath $File -ArgumentList $ArgList -NoNewWindow `
                          -Wait -PassThru -WorkingDirectory $Dir `
                          -RedirectStandardOutput $outFile -RedirectStandardError $errFile
    $text = ''
    if (Test-Path $outFile) { $text += Get-Content $outFile -Raw -ErrorAction SilentlyContinue }
    if (Test-Path $errFile) { $text += "`n" + (Get-Content $errFile -Raw -ErrorAction SilentlyContinue) }
    [pscustomobject]@{ Code = $proc.ExitCode; Out = ($text -replace "`r", '') }
  }
  finally {
    Remove-Item $outFile, $errFile -Force -ErrorAction SilentlyContinue
  }
}

function Gate {
  param([int]$Number, [string]$Name, [bool]$Passed, [string]$Detail, [string]$Fix)

  $script:results += [pscustomobject]@{ Gate = $Number; Name = $Name; Passed = $Passed }
  if ($Passed) {
    Write-Host "[PASS] gate $Number - $Name" -ForegroundColor Green
    if ($Detail) { Write-Host "       $Detail" -ForegroundColor DarkGray }
    return $true
  }
  Write-Host "[FAIL] gate $Number - $Name" -ForegroundColor Red
  if ($Detail) { Write-Host "       $Detail" -ForegroundColor Yellow }
  if ($Fix)    { Write-Host "       FIX: $Fix" -ForegroundColor Cyan }
  return $false
}

Write-Host "GPU enablement gates -- $($env:COMPUTERNAME)" -ForegroundColor White
Write-Host "project: $Root`n"

# --- Gate 1: Windows driver, and specifically the R580 branch ----------------
if ($From -le 1) {
  $r = Run-Cmd 'nvidia-smi' @('--query-gpu=driver_version,name', '--format=csv,noheader')
  $line = ($r.Out -split "`n" | Where-Object { $_ -match '\S' } | Select-Object -First 1)
  if ($line -match '^([0-9]+)\.([0-9.]+)\s*,\s*(.+)$') {
    $drv  = [int]$matches[1]
    $name = $matches[3].Trim()
    if ($drv -ge 580 -and $drv -lt 595) {
      $ok = Gate 1 "Windows driver on the R580 branch" $true "$name, driver $line" ''
    }
    elseif ($drv -ge 595) {
      Gate 1 "Windows driver on the R580 branch" $false `
        "$name reports driver $line -- R595/R620 DROPPED Pascal, this GPU is unsupported there" `
        'Roll back to the highest 580-branch build (580-582.x). See the branch history page.' | Out-Null
      exit 1
    }
    else {
      Gate 1 "Windows driver on the R580 branch" $false `
        "$name reports driver $line. CUDA-on-WSL needs R470+; Pascal dies after R580." `
        'Install 582.99 (R580 U12), Quadro/RTX Enterprise, Windows 10/11 64-bit. Custom -> Clean install, then reboot.' | Out-Null
      exit 1
    }
  }
  else {
    Gate 1 "Windows driver on the R580 branch" $false `
      "nvidia-smi did not report a GPU" 'Check Device Manager for a disabled or missing display adapter.' | Out-Null
    exit 1
  }
}

# --- Gate 2: WSL sees the GPU -----------------------------------------------
if ($From -le 2) {
  $lib = Run-Cmd 'wsl' @('-d', 'Ubuntu', '--', 'ls', '/usr/lib/wsl/lib')
  $hasCuda = $lib.Out -match 'libcuda'
  $smi = Run-Cmd 'wsl' @('-d', 'Ubuntu', '--', 'which', 'nvidia-smi')
  if ($hasCuda -and $smi.Code -eq 0) {
    Gate 2 'WSL sees the GPU' $true 'nvidia-smi present, libcuda.so.1 in /usr/lib/wsl/lib' '' | Out-Null
  }
  else {
    Gate 2 'WSL sees the GPU' $false `
      ('libcuda in /usr/lib/wsl/lib: {0}; nvidia-smi in WSL: {1}' -f $hasCuda, ($smi.Code -eq 0)) `
      'Run: wsl --shutdown  (WSL keeps the OLD driver mounted until every instance is stopped), then reopen Ubuntu.' | Out-Null
    exit 1
  }
}

# --- Gate 3: Docker sees the GPU --------------------------------------------
if ($From -le 3) {
  $r = Run-Cmd 'docker' @('run', '--rm', '--gpus', 'all', 'ubuntu:24.04', 'nvidia-smi',
                          '--query-gpu=name', '--format=csv,noheader')
  if ($r.Code -eq 0 -and $r.Out -match 'Quadro|NVIDIA') {
    Gate 3 'Docker can run --gpus all' $true ($r.Out -split "`n" | Where-Object { $_ -match 'Quadro|NVIDIA' } | Select-Object -First 1) '' | Out-Null
  }
  else {
    $why = if ($r.Out -match 'no adapters were found') {
             'nvidia-container-cli still sees no WSL adapters'
           } elseif ($r.Out -match 'docker:.*command not found|Cannot connect') {
             'cannot reach the Docker daemon'
           } else { ($r.Out -split "`n" | Where-Object { $_ -match 'error|Error' } | Select-Object -First 1) }
    Gate 3 'Docker can run --gpus all' $false $why `
      'Fully quit Docker Desktop (tray -> Quit), run wsl --shutdown, relaunch Docker. Then check Settings -> Resources -> WSL Integration has Ubuntu enabled.' | Out-Null
    exit 1
  }
}

# --- Gate 4: the container actually has libcuda -----------------------------
if ($From -le 4) {
  $r = Run-Cmd 'docker' ($Compose + @('run', '--rm', 'sim', 'python', 'docker/doctor.py'))
  if ($r.Out -match 'libcuda visible') {
    Gate 4 'container has libcuda' $true (($r.Out -split "`n" | Where-Object { $_ -match 'libcuda visible' }) -join '') '' | Out-Null
  }
  else {
    $detail = ($r.Out -split "`n" | Where-Object { $_ -match '\[FAIL\] gpu' } | Select-Object -First 1)
    Gate 4 'container has libcuda' $false `
      $(if ($detail) { $detail.Trim() } else { 'doctor.py produced no gpu line' }) `
      'runtime: nvidia starts containers WITHOUT a GPU on this host and does not warn, so a clean run means nothing. Re-check gates 1-3.' | Out-Null
    exit 1
  }
}

# --- Gate 5: JAX / MJX on the device ---------------------------------------
if ($From -le 5) {
  if (-not $IncludeMjx) {
    Write-Host "`n[skip] gate 5 - JAX/MJX on device (re-run with -IncludeMjx; uses the mjx image)" -ForegroundColor DarkGray
  }
  else {
    $r = Run-Cmd 'docker' ($Compose + @('run', '--rm', 'mjx-shell', 'python', 'docker/mjx_check.py'))
    if ($r.Out -match 'GPU PATH READY') {
      Gate 5 'JAX + MJX run on the GPU' $true (($r.Out -split "`n" | Where-Object { $_ -match 'jax.devices' }) -join '') '' | Out-Null
    }
    else {
      $detail = ($r.Out -split "`n" | Where-Object { $_ -match '^\[FAIL\]|Falling back to cpu|cuda12' } | Select-Object -First 2)
      Gate 5 'JAX + MJX run on the GPU' $false ($detail -join ' ') `
        'The mjx image installs plain CPU jax. Switch the mjx stage to "mujoco-mjx==4.0.1" + "jax[cuda12]" (never cuda13: needs CC 7.5, this GPU is 6.1), rebuild, retry.' | Out-Null
      exit 1
    }
  }
}

Write-Host "`nAll gates passed -- GPU is live." -ForegroundColor Green
Write-Host 'Start viewing with:' -ForegroundColor White
Write-Host "  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up studio   -> http://localhost:9080/"
exit 0
