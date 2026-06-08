# Convenience wrapper: run a NIAH context-length sweep over a checkpoint and chart it.
# run_niah.py already handles multiple context lengths in one process; this just
# supplies sensible defaults.
#
#   .\scripts\run_sweep.ps1 -Ckpt out-niah-demo\ckpt.pt
#   .\scripts\run_sweep.ps1 -Ckpt out-niah-demo\ckpt.pt -ContextLengths 64,128,256 -Device cpu

param(
    [Parameter(Mandatory = $true)][string]$Ckpt,
    [int[]]$ContextLengths = @(64, 128, 256),
    [int]$Samples = 5,
    [int]$Choices = 4,
    [int]$Seed = 42,
    [string]$Device = "",
    [string]$Tokenizer = "gpt2",
    [string]$OutDir = "results"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $Ckpt)) { throw "Checkpoint not found: $Ckpt" }

$argsList = @(
    "scripts/run_niah.py",
    "--ckpt", $Ckpt,
    "--context-length"
) + ($ContextLengths | ForEach-Object { "$_" }) + @(
    "--samples", "$Samples",
    "--num-needles", "$Choices",
    "--seed", "$Seed",
    "--tokenizer", $Tokenizer,
    "--csv-out", (Join-Path $OutDir "niah_sweep.csv"),
    "--chart", $OutDir
)
if ($Device) { $argsList += @("--device", $Device) }

Write-Host "==> NIAH sweep over $($ContextLengths -join ', ')"
python @argsList
