param(
    [ValidateSet('Prepare', 'Relay', 'Status', 'Stop')]
    [string]$Action = 'Status',
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$Session = 'claim-check-extension',
    [string]$Distribution = 'Ubuntu',
    [int]$Port = 8000
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$linuxHome = (& wsl -d $Distribution -- printenv HOME).Trim()
if ($LASTEXITCODE -ne 0 -or -not $linuxHome.StartsWith('/')) {
    throw 'Ubuntu WSL must be installed and able to run Python 3.'
}
$cli = "$linuxHome/.local/share/claim-check-colab-cli/bin/colab"
$python = "$linuxHome/.local/share/claim-check-colab-cli/bin/python"
$linuxRepo = (& wsl -d $Distribution -- wslpath -a -u $repoRoot).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Could not resolve this checkout in WSL.' }

function Invoke-Colab([string[]]$CliArgs) {
    & wsl -d $Distribution -- $cli @CliArgs
    if ($LASTEXITCODE -ne 0) { throw "Colab command failed (exit $LASTEXITCODE). Read its output above." }
}

switch ($Action) {
    'Prepare' {
        & wsl -d $Distribution -- python3 -m venv "$linuxHome/.local/share/claim-check-colab-cli"
        if ($LASTEXITCODE -ne 0) { throw 'Install Python 3 and python3-venv in Ubuntu WSL.' }
        & wsl -d $Distribution -- $python -m pip install --quiet google-colab-cli==0.7.5
        if ($LASTEXITCODE -ne 0) { throw 'Could not install the Colab CLI.' }
        & wsl -d $Distribution -- $cli status -s $Session
        if ($LASTEXITCODE -ne 0) { Invoke-Colab -CliArgs @('new', '-s', $Session) }
        Invoke-Colab -CliArgs @('drivemount', '-s', $Session)
        $localFolder = Join-Path $repoRoot 'local'
        New-Item -ItemType Directory -Path $localFolder -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $repoRoot 'notebooks/extension_backend_cpu_colab.ipynb') -Destination (Join-Path $localFolder "colab-$Session.ipynb")
        Invoke-Colab -CliArgs @('exec', '-s', $Session, '--timeout', '600', '-f', "$linuxRepo/local/colab-$Session.ipynb")
        Write-Host 'Backend ready. Run this script with -Action Relay in a terminal you can leave open.'
    }
    'Relay' {
        & wsl -d $Distribution -- $python "$linuxRepo/apps/backend/colab_relay.py" --colab $cli --session $Session --port $Port --state-file "$linuxRepo/local/relay-$Session.json"
        if ($LASTEXITCODE -ne 0) { throw 'Relay failed. Check that Prepare completed on this session.' }
    }
    'Status' { Invoke-Colab -CliArgs @('status', '-s', $Session) }
    'Stop' {
        & wsl -d $Distribution -- $python "$linuxRepo/apps/backend/colab_relay.py" --stop --state-file "$linuxRepo/local/relay-$Session.json"
        if ($LASTEXITCODE -ne 0) { throw 'Could not stop the managed local relay.' }
        Invoke-Colab -CliArgs @('stop', '-s', $Session)
    }
}
