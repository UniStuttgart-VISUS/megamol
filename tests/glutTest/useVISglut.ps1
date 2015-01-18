#
# useVISglut.ps1
#
# Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
#

# Greeting
echo ""
echo "    VISlib Use-VISglut/freeGlut powershell script"
echo "    Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten."
echo ""

# visglut
$visglutWinPath = read-host "Enter windows path to the VISglut/freeGlut directory"
$visglutLinPath = read-host "Enter unix path to the VISglut/freeGlut directory"

# configure the file
$files = "glutInclude.visglut.h", "glutInclude.visglut.mk"

write-host ""
write-host "Creating files:";

foreach ($file in $files) {
    $filenameparts = $file.Split('.');
    if ($filenameparts.Length -lt 3) {
        write-host -foregroundcolor red "Error!"
        exit
    }
    $filename = "";
    for ($i = 0; $i -lt $filenameparts.Length; $i++) {
        if ($i + 2 -ne $filenameparts.Length) {
            if ($filename.Length -gt 0) {
                $filename += ".";
            }
            $filename += $filenameparts[$i];
        }
    }

    write-host "    $file => $filename, ";

    # read input template
    $data = get-content $file;

    # find and replace variables
    for ($i = 0; $i -lt $data.length; $i++) {
        $data[$i] = $data[$i] -ireplace "%visglutWinPath%", $visglutWinPath
        $data[$i] = $data[$i] -ireplace "%visglutLinPath%", $visglutLinPath
    }

    new-item . -name $filename -force -type "file" -value ($data | out-string) >> $null

}

write-host "    Done.";
