<#
.SYNOPSIS
    Makes a working ffmpeg reachable as `ffmpeg` on PATH for this conda env.

.DESCRIPTION
    Two ways ffmpeg goes wrong on Windows, both seen here:

    1. conda-forge `ffmpeg` installs an ffmpeg.exe that dies immediately with
       exit code -1073741511 (0xC0000139, STATUS_ENTRYPOINT_NOT_FOUND) because
       its DLL set conflicts with other packages in the env. It is on PATH, so
       tools find it and get an empty error message.
    2. imageio-ffmpeg ships a self-contained build that works, but its filename
       carries a version suffix (ffmpeg-win-x86_64-v9.0.2.exe), so
       shutil.which("ffmpeg") does not find it and mediapy reports
       "Program 'ffmpeg' is not found".

    This creates a shim directory holding ffmpeg.exe, hard-linked to the
    working imageio-ffmpeg binary (same volume, so no disk cost), and installs
    conda activation hooks that put the directory first on PATH. The
    conda-managed tree is otherwise untouched: nothing is deleted or overwritten.

    Run once per environment, with that environment active.

.EXAMPLE
    conda activate robotics_muj
    .\scripts\setup_ffmpeg_shim.ps1
#>
param(
    # Conda environment prefix. Defaults to the active environment.
    [string]$Prefix = $env:CONDA_PREFIX
)

$ErrorActionPreference = "Stop"

if (-not $Prefix) {
    Write-Error "No conda environment active. Run 'conda activate <env>' first, or pass -Prefix."
}
if (-not (Test-Path $Prefix)) {
    Write-Error "Prefix not found: $Prefix"
}

$python = Join-Path $Prefix "python.exe"
if (-not (Test-Path $python)) {
    Write-Error "No python.exe in $Prefix"
}

# Ask the env's own python where its ffmpeg lives; do not assume the path.
$source = (& $python -c "import imageio_ffmpeg,sys; sys.stdout.write(imageio_ffmpeg.get_ffmpeg_exe())").Trim()
if (-not (Test-Path $source)) {
    Write-Error "imageio-ffmpeg binary not found at: $source`nInstall it:  python -m pip install imageio-ffmpeg"
}

# Confirm it is actually runnable before wiring it into PATH.
& $source -version *>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "The imageio-ffmpeg binary itself fails (exit $LASTEXITCODE). Not installing the shim."
}
$version = ((& $source -version 2>&1 | Select-Object -First 1))

$shimDir = Join-Path $Prefix "shims"
New-Item -ItemType Directory -Force -Path $shimDir | Out-Null

$shim = Join-Path $shimDir "ffmpeg.exe"
if (Test-Path $shim) { Remove-Item $shim -Force }
# Hard link: same bytes on disk, but appears as an independent file named ffmpeg.exe.
New-Item -ItemType HardLink -Path $shim -Target $source | Out-Null

& $shim -version *>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Shim does not run (exit $LASTEXITCODE)."
}

# Activation hooks. 'zz' sorts last, so this prepend wins over earlier ones.
$activateDir = Join-Path $Prefix "etc\conda\activate.d"
$deactivateDir = Join-Path $Prefix "etc\conda\deactivate.d"
New-Item -ItemType Directory -Force -Path $activateDir, $deactivateDir | Out-Null

@'
@echo off
set "_ML_OLD_PATH=%PATH%"
set "PATH={SHIM};%PATH%"
'@ -replace '\{SHIM\}', $shimDir | Set-Content (Join-Path $activateDir "zz-mujoco-learning-ffmpeg.bat") -Encoding ASCII

@'
$env:_ML_OLD_PATH = $env:PATH
$env:PATH = "{SHIM};" + $env:PATH
'@ -replace '\{SHIM\}', $shimDir | Set-Content (Join-Path $activateDir "zz-mujoco-learning-ffmpeg.ps1") -Encoding ASCII

@'
@echo off
if defined _ML_OLD_PATH set "PATH=%_ML_OLD_PATH%"
set "_ML_OLD_PATH="
'@ | Set-Content (Join-Path $deactivateDir "zz-mujoco-learning-ffmpeg.bat") -Encoding ASCII

@'
if (Test-Path env:_ML_OLD_PATH) { $env:PATH = $env:_ML_OLD_PATH; Remove-Item env:_ML_OLD_PATH }
'@ | Set-Content (Join-Path $deactivateDir "zz-mujoco-learning-ffmpeg.ps1") -Encoding ASCII

# Activation hooks only cover shells that were activated. A Jupyter kernel, an
# IDE run configuration, or a direct python.exe path never sees them, and then
# mediapy reports "Program 'ffmpeg' is not found" for reasons that look like a
# broken tutorial. usercustomize.py is imported by site.py for every process
# using this interpreter, so it closes that gap. It is chosen over
# sitecustomize.py because packages ship the latter and would be clobbered.
# purelib is the canonical site-packages directory; site.getsitepackages()[0]
# is not it (on Windows that is the environment root).
$sitePackages = (& $python -c "import sysconfig,sys; sys.stdout.write(sysconfig.get_paths()['purelib'])").Trim()
$uc = Join-Path $sitePackages "usercustomize.py"

# Placeholder must not occur anywhere else: PowerShell's -replace is
# case-insensitive, so a bare 'SHIM' also rewrote the word "shim" in the
# surrounding comment.
$block = @"

# --- mujoco-learning: make the env's ffmpeg visible to every process ---------
# Added by scripts/setup_ffmpeg_shim.ps1. Delete this block to undo.
_ML_FFMPEG_DIR = r"@@SHIMDIR@@"
if os.path.isdir(_ML_FFMPEG_DIR) and _ML_FFMPEG_DIR not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _ML_FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")
# --- end mujoco-learning block ----------------------------------------------
"@ -replace '@@SHIMDIR@@', $shimDir

if (Test-Path $uc) {
    $existing = Get-Content $uc -Raw
    if ($existing -match "mujoco-learning: make the env's ffmpeg visible") {
        # Replace the previous block rather than stacking duplicates.
        $pattern = "(?s)\r?\n# --- mujoco-learning: make the env's ffmpeg visible.*?# --- end mujoco-learning block -+`r?`n"
        $new = [regex]::Replace($existing, $pattern, "`r`n") + $block
        Set-Content $uc $new -Encoding UTF8 -NoNewline
        Write-Host "  usercustomize.py block refreshed"
    }
    else {
        Set-Content $uc ($existing + $block) -Encoding UTF8 -NoNewline
        Write-Host "  usercustomize.py block appended"
    }
}
else {
    Set-Content $uc ("import os" + $block) -Encoding UTF8 -NoNewline
    Write-Host "  usercustomize.py created"
}

# Prove it: a child interpreter with no activation and a scrubbed PATH must find
# ffmpeg through site.py alone. Scrub before launching - overwriting PATH inside
# the probe process would erase what usercustomize just added and always report
# NOT FOUND.
$prevPath = $env:PATH
$env:PATH = "$Prefix;$env:SystemRoot\System32"
$probe = (& $python -c "import shutil,sys; sys.stdout.write(shutil.which('ffmpeg') or 'NOT FOUND')").Trim()
$env:PATH = $prevPath

Write-Host ""
Write-Host "ffmpeg shim installed"
Write-Host "  target     : $source"
Write-Host "  shim       : $shim"
Write-Host "  version    : $version"
Write-Host "  cold probe : $probe"
Write-Host ""
Write-Host "Re-activate the environment to pick it up in this shell:"
Write-Host "    conda deactivate; conda activate $(Split-Path $Prefix -Leaf)"
