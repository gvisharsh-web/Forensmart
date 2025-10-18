# install_drivers_template.ps1
# Run as Administrator. This is a template that will attempt to install any INF files found
# in the folder provided. It does NOT download drivers. You must place official OEM driver INF files in the folder.
Param(
    [string]$DriverFolder = ".\drivers"
)

if (-not (Test-Path $DriverFolder)) {
    Write-Error "Driver folder not found: $DriverFolder"
    exit 1
}

Write-Host "Installing INF drivers from: $DriverFolder"
Get-ChildItem -Path $DriverFolder -Filter *.inf -Recurse | ForEach-Object {
    $inf = $_.FullName
    Write-Host "Installing driver: $inf"
    try {
        pnputil /add-driver $inf /install | Out-Null
        Write-Host "Installed: $inf"
    } catch {
        Write-Warning "Failed to install $inf"
    }
}
Write-Host "Done. Replug devices or reboot if needed."
