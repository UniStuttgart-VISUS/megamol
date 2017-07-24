# filename: all_uncertainty_input_data.ps1
# call script:
# .\all_uncertainty_input_data.ps1 | Out-File ".\all_uncertainty_input_data.log"

Clear

Write-Host "===================================================================="
Write-Host "Executing `"UncertaintyInputData.py`" for all PDB-files in `"../cache`""
#Write-Host "Writing output to `"all_uncertainty_input_data.log`" "
Write-Host "===================================================================="

# Delete old files ...
cd ..\cache
Remove-Item *.stride*
Remove-Item *.dssp*
Remove-Item *.prosign*
Remove-Item *.uid*
Remove-Item *.csv*
Remove-Item *.pdb~

cd ..\Resource
Remove-Item .\all_uncertainty_input_data.log

$fileList = Get-Item "..\cache\*" | Select-Object "Name" | Sort "Name"
ForEach ($file in $fileList) {
    
    $id = ($file.Name).SubString(0,4)

    Write-Output "====================================================================" | Tee-Object -FilePath .\all_uncertainty_input_data.log -Append 
    Write-Output "Processing PDB-ID: $id                                              " | Tee-Object -FilePath .\all_uncertainty_input_data.log -Append 
    Write-Output "====================================================================" | Tee-Object -FilePath .\all_uncertainty_input_data.log -Append 
    python.exe .\UncertaintyInputData.py  $id -o -c                                *>&1 | Tee-Object -FilePath .\all_uncertainty_input_data.log -Append 
}

Write-Host
Write-Host
Write-Host "===================================================================="
Write-Host "DONE"
Write-Host "===================================================================="