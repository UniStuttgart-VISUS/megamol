#
# checkdumpbin.ps1  12.09.2006 (mueller)
#
# Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
#

# Use Set-ExecutionPolicy unrestricted for easy execution on Powershell


# Path to the batch script that sets the Visual Studio paths and variables
$VSEnvScript = "D:\Program Files (x86)\Microsoft Visual Studio 8\Common7\Tools\vsvars32.bat"


if ($Args.Length -eq 1) {
	$LibPath = $Args[0]
} else {
	echo "usage: checkdumpbin.ps1 <libpath>"
	exit
}

# Define the platforms and the include and exclude list that is evaluated for 
# the output of dumpbin.
$Platforms = @{ 
	"32" = @{ "Include" = "machine\s+\(x86\)"; "Exclude" = "machine\s+\(x64\)" };
	"64" = @{ "Include" = "machine\s+\(x64\)"; "Exclude" = "machine\s+\(x86\)" };
}

# Define the debug and release version and the two lists for them.
$Debugs = @{ 
	"" = @{ "Include" = @(); "Exclude" = "\.debug" };
	"d" = @{ "Include" = "\.debug"; "Exclude" = @() };
}

# Add the Visual Studio Variables from $VSEnvScript to the path environment of
# the Powershell
$VSPath = (gc $VSEnvScript | select-string "set\s+path") -replace "(.+\s+path\s*=\s*)", "" -replace "%path%", ""
if ("$env:path" -inotmatch ($VSPath -replace "\\", "\\" -replace "\(", "\(" -replace "\)", "\)")) {
	# $VSPath must be at begin for dumpbin to work
	$env:path = $VSPath + $env:path
}

foreach ($p in $Platforms.Keys) {
	foreach ($d in $Debugs.Keys) {
	
		# List the the libraries that match
		$Libs = gci $LibPath\* -i *$p$c$d.lib
		foreach ($l in $Libs)  {
			echo "Checking $l ..."
			$failed = $FALSE
			$out = dumpbin /all $l.FullName

			# Check platform
			foreach ($i in $Platforms[$p]["Include"]) {
				if ("$out" -inotmatch "$i") { 
					echo "Include '$i' for $p FAILED"
					$failed = $TRUE
				}
			}
			
			foreach ($e in $Platforms[$p]["Exclude"]) {
				if ("$out" -imatch "$e") { 
					echo "Exclude '$e' for $p FAILED"
					$failed = $TRUE
				}
			}
			
			# Check debug symbols
			foreach ($i in $Debugs[$d]["Include"]) {
				if ("$out" -inotmatch "$i") { 
					echo "Include '$i' for $d FAILED"
					$failed = $TRUE
				}
			}
			
			foreach ($e in $Debugs[$d]["Exclude"]) {
				if ("$out" -imatch "$e") { 
					echo "Exclude '$e' for $d FAILED"
					$failed = $TRUE
				}
			}	
			
			if ($failed) {
				$filename = $l.Name -replace "(\.lib)$", ".txt"
				echo "Writing output of dumpbin to $filename  ..."
				ni . -name $filename -force -type "file" -value ($out | out-string) >> $null
			}
		} # end foreach ($l in $Libs)
	} # end foreach ($d in $Debugs.Keys)
} # end foreach ($p in $Platforms.Keys)
