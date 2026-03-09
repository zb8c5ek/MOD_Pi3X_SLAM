# BURNSCRIPT_ENTRY_SLAM.ps1 - PowerShell entry for Pi3X SLAM pipeline
# Activates the sfm3rV2 environment and runs the SLAM pipeline.
#
# Usage:
#   .\BURNSCRIPT_ENTRY_SLAM.ps1                              # uses default config
#   .\BURNSCRIPT_ENTRY_SLAM.ps1 _configs_slam/config_slam_test.yaml

param(
    [string]$ConfigPath = "_configs_slam/config_slam_test.yaml"
)

$env:HF_HOME     = "D:\_HUBs\HuggingFace"
$env:TORCH_HOME  = "D:\_HUBs\Torch"
$env:HF_HUB_CACHE = "D:\_HUBs\HuggingFace\hub"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Running Pi3X SLAM with config: $ConfigPath"
Write-Host "Script directory: $ScriptDir"

Set-Location $ScriptDir
micromamba run -n sfm3rV2 python BURNPIPE_Pi3X_SLAM.py $ConfigPath
