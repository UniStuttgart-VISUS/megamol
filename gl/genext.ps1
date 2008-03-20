#
# Generator file for 'glh_genext.h'
# Copyright (C) 2008 by Universitaet Stuttgart (VISUS)
# Alle Rechte vorbehalten.
#

#check arguments
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

# generate 'extfile.txt'
& "$InputDir\include\glh\genextfile.ps1" "$InputDir\include\GL\glext.h" "$InputDir\include\glh\extfile.txt"

# compile 'glh.exe'
if ($Bits -eq "32") {

    # commands 32
    # powershell -nologo -noprofile -noninteractive -command "%1""\include\glh\genextfile.ps1" "%1""\include\GL\glext.h" "%1""\include\glh\extfile.txt"
    # set INCLUDE="$(InputDir)\include\";%INCLUDE%
    # "$(VCInstallDir)\vcpackages\vcbuild.exe" "$(InputDir)\include\glh\glh.vcproj" $(ConfigurationName) /useenv /nologo
    # @echo Generating $(InputDir)\include\glh\glh_genext.h" from "$(InputDir)\include\glh\extfile.txt" ...
    # "$(InputDir)\include\glh\$(ConfigurationName)\glh.exe" "$(InputDir)\include\glh\glh_genext.h" "$(InputDir)\include\glh\extfile.txt"

    & "$VCInstallDir\vcpackages\vcbuild.exe" "$InputDir\include\glh\glh.vcproj" "$ConfigurationName" /useenv /nologo
}

if ($Bits -eq "64") {

    # commands 64
    # powershell -nologo -noprofile -noninteractive -command "$(InputDir)\include\glh\genextfile.ps1" "$(InputDir)\include\GL\glext.h" "$(InputDir)\include\glh\extfile.txt"
    # powershell -nologo -noprofile -noninteractive -command "[xml]$proj = gc '$(InputDir)\include\glh\glh.vcproj'; $proj.GetElementsByTagName('Tool') | where { $_.Name -eq 'VCCLCompilerTool' } | foreach { $_.SetAttribute('AdditionalIncludeDirectories', '$(InputDir)\include\;' + $_.AdditionalIncludeDirectories) }; $proj.Save('$(InputDir)\include\glh\glh64crowbar.vcproj')"
    # "$(VCInstallDir)\vcpackages\vcbuild.exe" "$(InputDir)\include\glh\glh64crowbar.vcproj" $(ConfigurationName) /nologo /platform:Win32
    # @echo Generating $(InputDir)\include\glh\glh_genext.h" from "$(InputDir)\include\glh\extfile.txt" ...
    # "$(InputDir)\include\glh\$(ConfigurationName)\glh.exe" "$(InputDir)\include\glh\glh_genext.h" "$(InputDir)\include\glh\extfile.txt"
    # del "$(InputDir)\include\glh\glh64crowbar.vcproj"

    [xml]$proj = gc "$InputDir\include\glh\glh.vcproj";
    $proj.GetElementsByTagName('Tool') | where { $_.Name -eq 'VCCLCompilerTool' } | foreach {
        $_.SetAttribute('AdditionalIncludeDirectories', '$InputDir\include\;' + $_.AdditionalIncludeDirectories);
    };
    $proj.Save("$InputDir\include\glh\glh64crowbar.vcproj");

    & "$VCInstallDir\vcpackages\vcbuild.exe" "$InputDir\include\glh\glh64crowbar.vcproj" "$ConfigurationName" /nologo /platform:Win32

    del "$InputDir\include\glh\glh64crowbar.vcproj"
}

# call 'glh.exe' to generate 'glh_genext.h'
& "$InputDir\include\glh\$ConfigurationName\glh.exe" "$InputDir\include\glh\glh_genext.h" "$InputDir\include\glh\extfile.txt"

# finished
exit 0
