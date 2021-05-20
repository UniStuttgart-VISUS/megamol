#
# MegaMol Plugin Instantiation Wizard
# Copyright 2018-2019 by MegaMol Team
# Alle Rechte vorbehalten.
#

# greeting
Write-Host "`n"
Write-Host "MegaMol(TM) Plugin Instantiation Wizard"
Write-Host "Copyright 2010-2015 by MegaMol Team"
Write-Host "Alle Rechte vorbehalten.`n`n"

# get plugin name
do {
    $pluginname = Read-Host -Prompt "Enter plugin name"
    if ($pluginname -notmatch '^[a-zA-Z][0-9a-zA-Z_]*') {
        Write-Host "ERROR: filename is invalid`n";
        Write-Host "       A plugin filename must start with a character and must only contain characters, numbers and underscores.`n";
    }
} while ($pluginname -notmatch '^[a-zA-Z][0-9a-zA-Z_]*')

# create GUID
# $guid = New-Guid
# $usegenguid = $false
# $dummyguid = $guid

# do {
#     $inputguid = Read-Host -Prompt "Enter well-formed GUID [$guid]"
#     if ([string]::IsNullOrEmpty($inputguid)) {
#         $usegenguid = $true
#         break
#     }
#     if (-not [System.Guid]::TryParse($inputguid, [ref]$dummyguid)) {
#         Write-Host "ERROR: GUID is ill-formed`n"
#     }
# } while (-not [System.Guid]::TryParse($inputguid, [ref]$dummyguid))

# if (-not $usegenguid) {
#     $guid = [System.Guid]::Parse($inputguid)
# }

# # paranoia confirmation
# $accept = Read-Host -Prompt "Generating plugin '$pluginname' with GUID '$guid'. OK? [ /n]"
# if (-not [string]::IsNullOrEmpty($accept)) {
#     Write-Host "Aborting"
#     exit
# }

# paranoia confirmation
$accept = Read-Host -Prompt "Generating plugin '$pluginname'. OK? [ /n]"
if (-not [string]::IsNullOrEmpty($accept)) {
    Write-Host "Aborting"
    exit
}

# perform instantiation
Move-Item -Path ".\include\MegaMolPlugin\" -Destination ".\include\$pluginname\"
$uppername = $pluginname.ToUpper()

$fn = ".\include\$pluginname\$pluginname.h"
Move-Item -Path ".\include\$pluginname\MegaMolPlugin.h" -Destination $fn
$temp = [IO.File]::ReadAllText($fn)
$temp = $temp -creplace "MegaMolPlugin", "$pluginname"
$temp = $temp -creplace "MEGAMOLPLUGIN", "$uppername"
[IO.File]::WriteAllText($fn, $temp)

$fn = ".\src\$pluginname.cpp"
Move-Item -Path ".\src\MegaMolPlugin.cpp" -Destination $fn
$temp = [IO.File]::ReadAllText($fn)
$temp = $temp -creplace "MegaMolPlugin", "$pluginname"
$temp = $temp -creplace "MEGAMOLPLUGIN", "$uppername"
[IO.File]::WriteAllText($fn, $temp)

$fn = ".\src\stdafx.h";
$temp = [IO.File]::ReadAllText($fn)
$temp = $temp -creplace "MEGAMOLPLUGIN", "$uppername"
[IO.File]::WriteAllText($fn, $temp)


#  - Cmake files

$fn = ".\CMakeLists.txt";
$temp = [IO.File]::ReadAllText($fn)
$temp = $temp -creplace "MegaMolPlugin", "$pluginname"
[IO.File]::WriteAllText($fn, $temp)

# Completed
Write-Host "`n== Instantiation complete ==`n`n"
Write-Host "You should do now:`n"
Write-Host "  * Delete instawiz.pl`n"
Write-Host "  * Delete instawiz.ps1`n"
Write-Host "  * Commit changes to git`n"
Write-Host "  * Start implementing`n`n"
