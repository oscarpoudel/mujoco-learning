<#
.SYNOPSIS
  Launch MuJoCo Studio in a graphics mode that actually works on this machine.

.DESCRIPTION
  MuJoCo 4.0.1's native Studio window crashes on the first frame with
  "ValueError: Target not found:" because native_viewer.py never registers an
  'out' render target in hardware graphics modes. See
  notes/0001-mujoco4-studio-native-viewer.md for the analysis.

  This script defaults to --gfx=opengl_software, which takes the code path that
  does register its target and so avoids the bug. Use -Gfx web for the browser
  viewer instead.

.PARAMETER Model
  Path to an MJCF/URDF/MJB/MJZ model file. Omit to open Studio with no model.

.PARAMETER Gfx
  Graphics mode. opengl_software (default) gives a native desktop window.
  web streams to your browser at http://localhost:8081/ with no encryption,
  so do not leave it running on a shared network.

.PARAMETER HotReload
  Reload model/plugins on save: '', 'model', 'plugins' or 'all'.

.PARAMETER Python
  Interpreter to run. Defaults to $env:CONDA_PREFIX, falling back to the
  robotics_muj env.

.EXAMPLE
  .\scripts\studio.ps1
  .\scripts\studio.ps1 repos\Tutorial_Mujoco_RL\1tutorial_ur5e\model\universal_robots_ur5e\scene.xml
  .\scripts\studio.ps1 my_robot.xml -Gfx web -HotReload all

.NOTES
  MuJoCo 4.x has no `mujoco` command-line binary; `mujoco simulate x.xml` is not a thing.
  For the classic 3.x-style viewer use:  python -m mujoco.viewer
#>
[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [string]$Model = '',

  [ValidateSet('opengl_software', 'web', 'webgl', 'vulkan', 'vulkan_software', 'opengl', 'opengl_headless')]
  [string]$Gfx = 'opengl_software',

  [ValidateSet('', 'model', 'plugins', 'all')]
  [string]$HotReload = '',

  [string]$Python = ''
)

$ErrorActionPreference = 'Stop'

# --- resolve the interpreter -------------------------------------------------
if (-not $Python) {
  if ($env:CONDA_PREFIX -and (Test-Path "$env:CONDA_PREFIX\python.exe")) {
    $Python = "$env:CONDA_PREFIX\python.exe"
  }
  else {
    $candidate = Join-Path $HOME 'miniconda3\envs\robotics_muj\python.exe'
    if (Test-Path $candidate) { $Python = $candidate }
  }
}

if (-not (Test-Path $Python)) {
  Write-Error "Python not found at '$Python'. Activate your env first: conda activate robotics_muj"
  exit 1
}

# Fail early with a clear message rather than a traceback from deep in studio.
$version = & $Python -c "import mujoco; print(mujoco.__version__)" 2>$null
if (-not $version) {
  Write-Error "mujoco is not importable by $Python. Try: `"$Python`" -m pip install mujoco"
  exit 1
}

# --- resolve the model -------------------------------------------------------
if ($Model -and -not (Test-Path $Model)) {
  Write-Error "Model file not found: $Model"
  exit 1
}
if ($Model) { $Model = (Resolve-Path $Model).Path }

Write-Host "python : $Python (mujoco $version)" -ForegroundColor DarkGray
Write-Host "gfx    : $Gfx" -ForegroundColor DarkGray
Write-Host "model  : $(if ($Model) { $Model } else { '<none>' })" -ForegroundColor DarkGray

if ($Gfx -in @('web', 'webgl')) {
  Write-Warning "Web viewer has NO auth or encryption. Anyone reaching this port can see your logs and file paths."
}

# --- launch ------------------------------------------------------------------
$argv = @('-m', 'mujoco.studio.viewer', '--gfx', $Gfx)
if ($Model)        { $argv += @('--model', $Model) }
if ($HotReload)    { $argv += @('--hot_reload', $HotReload) }

& $Python @argv
