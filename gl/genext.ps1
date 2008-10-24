#
# Generator file for 'glh_genext.h'
# Copyright (C) 2008 by Universitaet Stuttgart (VISUS)
# Alle Rechte vorbehalten.
#

# Original Commands 32 bit
# powershell -nologo -noprofile -noninteractive -command "%1""\include\glh\genextfile.ps1" "%1""\include\GL\glext.h" "%1""\include\glh\extfile.txt"
# set INCLUDE="$(InputDir)\include\";%INCLUDE%
# "$(VCInstallDir)\vcpackages\vcbuild.exe" "$(InputDir)\include\glh\glh.vcproj" $(ConfigurationName) /useenv /nologo
# @echo Generating $(InputDir)\include\glh\glh_genext.h" from "$(InputDir)\include\glh\extfile.txt" ...
# "$(InputDir)\include\glh\$(ConfigurationName)\glh.exe" "$(InputDir)\include\glh\glh_genext.h" "$(InputDir)\include\glh\extfile.txt"

# Original Commands 32 bit
# powershell -nologo -noprofile -noninteractive -command "$(InputDir)\include\glh\genextfile.ps1" "$(InputDir)\include\GL\glext.h" "$(InputDir)\include\glh\extfile.txt"
# powershell -nologo -noprofile -noninteractive -command "[xml]$proj = gc '$(InputDir)\include\glh\glh.vcproj'; $proj.GetElementsByTagName('Tool') | where { $_.Name -eq 'VCCLCompilerTool' } | foreach { $_.SetAttribute('AdditionalIncludeDirectories', '$(InputDir)\include\;' + $_.AdditionalIncludeDirectories) }; $proj.Save('$(InputDir)\include\glh\glh64crowbar.vcproj')"
# "$(VCInstallDir)\vcpackages\vcbuild.exe" "$(InputDir)\include\glh\glh64crowbar.vcproj" $(ConfigurationName) /nologo /platform:Win32
# @echo Generating $(InputDir)\include\glh\glh_genext.h" from "$(InputDir)\include\glh\extfile.txt" ...
# "$(InputDir)\include\glh\$(ConfigurationName)\glh.exe" "$(InputDir)\include\glh\glh_genext.h" "$(InputDir)\include\glh\extfile.txt"
# del "$(InputDir)\include\glh\glh64crowbar.vcproj"


# Check arguments
if ($Args.Length -ne 4) {
    echo "Error: Argument list invalid."
    echo "Must be: InputDir Bits ConfigurationName VCInstallDir"
    exit -1
}

$InputDir = $Args[0]
$Bits = $Args[1]
$ConfigurationName = $Args[2]
$VCInstallDir = $Args[3]

if (($Bits -ne "32") -and ($Bits -ne "64"))  {
    echo "Error: Bits (Arg 2) is invalid"
    exit -2
}


# Generate 'extfile.txt'
& "$InputDir\include\glh\genextfile.ps1" "$InputDir\include\GL\glext.h" "$InputDir\include\glh\extfile.txt"


# Include the path to wglext.h in the environment INCLUDE variable.
$incPath = "`"$InputDir\include\glh`";" 
$incPath += gc Env:INCLUDE
si -path Env:INCLUDE -value "$incPath"
echo $Env:INCLUDE


# compile 'glh.exe'
$projName = "$InputDir\" + "include\glh\glh$Bits" + "crowbar.vcproj"

[xml]$proj = gc "$InputDir\include\glh\glh.vcproj";
if ($Bits -eq "64") {
    $proj.GetElementsByTagName('Tool') | where { $_.Name -eq 'VCCLCompilerTool' } | foreach {
        $_.SetAttribute('AdditionalIncludeDirectories', '$InputDir\include\;$InputDir\include\glh\;' + $_.AdditionalIncludeDirectories);
    };
}
$proj.Save($projName);

& "$VCInstallDir\vcpackages\vcbuild.exe" $projName /upgrade
& "$VCInstallDir\vcpackages\vcbuild.exe" $projName "$ConfigurationName" /useenv /nologo /platform:Win32

del $projName


# Call 'glh.exe' to generate 'glh_genext.h'
& "$InputDir\include\glh\$ConfigurationName\glh.exe" "$InputDir\include\glh\glh_genext.h" "$InputDir\include\glh\extfile.txt"

exit 0
