$exitcode = 0

# Check for files containing CRLF line ending
$bad_line_ending_files = Get-ChildItem -Recurse | Select-String "`r`n" -List | Select-Object Path
if ($bad_line_ending_files.Count -gt 0) {

    Write-Output "############################################################"
    Write-Output " ERROR: Files with CRLF line ending found!"
    Write-Output "############################################################"
    $bad_line_ending_files
    
    $exitcode = 1
} else {
    Write-Output "Good: No CRLF line endings found."
}

# Check for files not encoded in UTF-8
$bad_encoding_files = @()
Get-ChildItem -Recurse -File | ForEach-Object { 
    $Encoding = New-Object -TypeName System.IO.StreamReader -ArgumentList $_.FullName -OutVariable Stream | Select-Object -ExpandProperty CurrentEncoding | Select-Object -ExpandProperty BodyName;
    if ($Encoding -ne "utf-8") {
        $bad_encoding_files += $_;
    }
    $Stream.Dispose();
}
if ($bad_encoding_files.Count -gt 0) {

    Write-Output "############################################################"
    Write-Output " ERROR: The following text files are not UTF-8 encoded!"
    Write-Output "############################################################"
    $bad_encoding_files
    
    $exitcode = 1
} else {
    Write-Output "Good: All text files are UTF-8 encoded."
}

exit $exitcode
