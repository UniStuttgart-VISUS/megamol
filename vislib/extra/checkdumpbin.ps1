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

# SIG # Begin signature block
# MIINsAYJKoZIhvcNAQcCoIINoTCCDZ0CAQExCzAJBgUrDgMCGgUAMGkGCisGAQQB
# gjcCAQSgWzBZMDQGCisGAQQBgjcCAR4wJgIDAQAABBAfzDtgWUsITrck0sYpfvNR
# AgEAAgEAAgEAAgEAAgEAMCEwCQYFKw4DAhoFAAQU0iAnXVu8dvufq+syXCrHOCou
# 6eegggn7MIIJ9zCCB9+gAwIBAgIKYXFZ9AAAAAAACTANBgkqhkiG9w0BAQsFADBw
# MRIwEAYKCZImiZPyLGQBGRYCZGUxHTAbBgoJkiaJk/IsZAEZFg11bmktc3R1dHRn
# YXJ0MRUwEwYKCZImiZPyLGQBGRYFdmlzdXMxJDAiBgNVBAMTG1ZJU1VTIENlcnRp
# ZmljYXRlIEF1dGhvcml0eTAeFw0wOTA4MjgxMTI1NTVaFw0xMTA4MjgxMTM1NTVa
# MIGtMRIwEAYKCZImiZPyLGQBGRYCZGUxHTAbBgoJkiaJk/IsZAEZFg11bmktc3R1
# dHRnYXJ0MRUwEwYKCZImiZPyLGQBGRYFdmlzdXMxDjAMBgNVBAMTBVVzZXJzMRow
# GAYDVQQDDBFDaHJpc3RvcGggTcO8bGxlcjE1MDMGCSqGSIb3DQEJARYmY2hyaXN0
# b3BoLm11ZWxsZXJAdmlzLnVuaS1zdHV0dGdhcnQuZGUwggIiMA0GCSqGSIb3DQEB
# AQUAA4ICDwAwggIKAoICAQDjlqAj3WZstwWvxgjEOSjeyHUSOVVsqgGf+H4T06eL
# RfyZRjNNbLc/UIOV+FHMziQbwWAEu2o4BivJm3DUI4s/A5DIqMgwsi3LhLB1AaQ7
# jTSfPhXVZaqyG2cComqjHnF+pHOXGTif3SXm0nk4YoAWcStRVhitv7dw6+i1DBfR
# JSUJ2J7axHfp2OVUnO5TNdDt0OulaRw6AmU7Myo7zOY5JQZ5m5ToM0Otj/WXtZm5
# aPyQpQsJfVayQ1UwCgRkiJe5t38oGkrAM7k9fv1akYDrQ8DeOqpJ77Zr9GezAeBi
# j+ZTl3dQqfwD4weugf3ltacBlxGLHjsnOyrn/LTHFptu45KRxksJGpvPuVUPwGp9
# N5OEvYFD0ReQc0yc7JhHBYGZVD7zcgCEQQ8vHMCgsv4ff80hLZg1V61VT1L/Qh6K
# ufQyJ0lmZnaRq5bKGr577H94gcojwht3IC3vkLMlAqvruB+PDdz1fTjBbEuqJ3JZ
# wlspV7iAsR62dTdyWjWdQNJDSI6RQTKRVtMAVtOjqQ4ePs4Scns6opjAtbsNSXiT
# EPvDPR3HyEDYkup7+Dm+ofXZ0dYBmGepohyOVO+CQpd9LeSPWKZqHX1v7m/CXRo2
# YznbPDbp0yFHXMhWF374V686yH0/1Rm7EtCmEyxAeR5m/rTzin1pXXLctQLCeIvv
# 9QIDAQABo4IEUzCCBE8wDgYDVR0PAQH/BAQDAgWgMDwGCSsGAQQBgjcVBwQvMC0G
# JSsGAQQBgjcVCIaDkj+G7aobgpGfJYOYmSys9W1EhInOf4GIswACAWQCAQIwRAYJ
# KoZIhvcNAQkPBDcwNTAOBggqhkiG9w0DAgICAIAwDgYIKoZIhvcNAwQCAgCAMAcG
# BSsOAwIHMAoGCCqGSIb3DQMHMB0GA1UdDgQWBBQrIWEaw7/rKVXMtXiurlQ5a7JE
# sjAfBgNVHSMEGDAWgBRzN/rHq5JtzSKAenACE5Q23cBlPTCCAT4GA1UdHwSCATUw
# ggExMIIBLaCCASmgggElhoHQbGRhcDovLy9DTj1WSVNVUyUyMENlcnRpZmljYXRl
# JTIwQXV0aG9yaXR5LENOPW5vaCxDTj1DRFAsQ049UHVibGljJTIwS2V5JTIwU2Vy
# dmljZXMsQ049U2VydmljZXMsQ049Q29uZmlndXJhdGlvbixEQz12aXN1cyxEQz11
# bmktc3R1dHRnYXJ0LERDPWRlP2NlcnRpZmljYXRlUmV2b2NhdGlvbkxpc3Q/YmFz
# ZT9vYmplY3RDbGFzcz1jUkxEaXN0cmlidXRpb25Qb2ludIZQaHR0cDovL25vaC52
# aXN1cy51bmktc3R1dHRnYXJ0LmRlL0NlcnRFbnJvbGwvVklTVVMlMjBDZXJ0aWZp
# Y2F0ZSUyMEF1dGhvcml0eS5jcmwwggFaBggrBgEFBQcBAQSCAUwwggFIMIHMBggr
# BgEFBQcwAoaBv2xkYXA6Ly8vQ049VklTVVMlMjBDZXJ0aWZpY2F0ZSUyMEF1dGhv
# cml0eSxDTj1BSUEsQ049UHVibGljJTIwS2V5JTIwU2VydmljZXMsQ049U2Vydmlj
# ZXMsQ049Q29uZmlndXJhdGlvbixEQz12aXN1cyxEQz11bmktc3R1dHRnYXJ0LERD
# PWRlP2NBQ2VydGlmaWNhdGU/YmFzZT9vYmplY3RDbGFzcz1jZXJ0aWZpY2F0aW9u
# QXV0aG9yaXR5MHcGCCsGAQUFBzAChmtodHRwOi8vbm9oLnZpc3VzLnVuaS1zdHV0
# dGdhcnQuZGUvQ2VydEVucm9sbC9ub2gudmlzdXMudW5pLXN0dXR0Z2FydC5kZV9W
# SVNVUyUyMENlcnRpZmljYXRlJTIwQXV0aG9yaXR5LmNydDAzBgNVHSUELDAqBggr
# BgEFBQcDBAYKKwYBBAGCNwoDBAYIKwYBBQUHAwMGCCsGAQUFBwMCMEEGCSsGAQQB
# gjcVCgQ0MDIwCgYIKwYBBQUHAwQwDAYKKwYBBAGCNwoDBDAKBggrBgEFBQcDAzAK
# BggrBgEFBQcDAjBhBgNVHREEWjBYoC4GCisGAQQBgjcUAgOgIAwebXVlbGxlckB2
# aXN1cy51bmktc3R1dHRnYXJ0LmRlgSZjaHJpc3RvcGgubXVlbGxlckB2aXMudW5p
# LXN0dXR0Z2FydC5kZTANBgkqhkiG9w0BAQsFAAOCAgEAfbCIiE0qntw/iXHWWpjV
# CU5r12VXwAfPx9C+c9JU4vX7pnUBujnrG5ehC9Yzitxp0HPhlQOFffLTmbx6ZL28
# FkrI8j7ii7hKewXMGDOnbsl3Zekdak/tDdqyi8R2+LTaJB9NLIeTuXKNaU3Vkj5m
# oYHjftVALUcYcv3owSbqIqUlPFTrxhTRa5HV502GI/8YGYxx0IpJCACHs5QfWQAu
# hTMpl3mMN8lIqWYBUfI9fRx0qxbp3BbgM3zC5+wJNH4o+m5wHG39zCNEvqOfY23W
# WbS+orOXJ0o0HKgyM1PQqRlFlYJESRFA62BhorZgAmOqScVhxzjzFDy0ypV/BJhL
# u5SgkEQxHOMvVxzFi+wOSVbBSgs6OVCFBczvMWTVcL7gE1C3xDDrGxz3uhoJE++e
# 80tTUjz1JaOgncek11AwuGdIiK2Zaoq1nxgOvWaYBaFPDASkiPSri5DEG8PtcGXh
# n1+fzkst+x46k/Zt9fijZaRn4NXjQbjYV/ig4Oh5TAFQTfzBq0vRbRFrkG6sau69
# VcW8JK1TO7UMW1i6ZCHFksKN/bLZC9VPnaYcEaIeYWha+Wta/Nhm7UHPH1PDS9Nf
# oza8udGfXSPosgmLkh4CXKBwJqjv5F0a7uiVTFWiIIjUlv4MegCSwAfX+8SBQNl5
# xba5aEsVonIIwLGmNoG64WYxggMfMIIDGwIBATB+MHAxEjAQBgoJkiaJk/IsZAEZ
# FgJkZTEdMBsGCgmSJomT8ixkARkWDXVuaS1zdHV0dGdhcnQxFTATBgoJkiaJk/Is
# ZAEZFgV2aXN1czEkMCIGA1UEAxMbVklTVVMgQ2VydGlmaWNhdGUgQXV0aG9yaXR5
# AgphcVn0AAAAAAAJMAkGBSsOAwIaBQCgeDAYBgorBgEEAYI3AgEMMQowCKACgACh
# AoAAMBkGCSqGSIb3DQEJAzEMBgorBgEEAYI3AgEEMBwGCisGAQQBgjcCAQsxDjAM
# BgorBgEEAYI3AgEVMCMGCSqGSIb3DQEJBDEWBBT9VNHCoA1sj/DiAdKg5TjcS/bV
# SzANBgkqhkiG9w0BAQEFAASCAgBCfeKQHm4MAn91ecQ4UoAPw+kt35TLKo6lx9Hx
# QDiqmS3yqy6R82YjpVQhLARoAycrfJwGiTfIkJ5ZHRk+5Ai6Vd88fk0q65r4yz/V
# 94HkDvlhuygHizQEq+8bUQMEMlgPmsps8P5vRtdeW6XHoFKjfQWEYowb7XCrTjA9
# 8ystagEFTmoU42SXTSKAeaD9sdyEsy15AdZM3u5Gi1gvXomrE/xM6rrVKq9D6R+f
# HsAc9Nrc+mbRejCAjg6f2+4nE7EFpxEWMTPQyYG1fuJWmrzILQkmyUM7i4um1bw0
# UyiH1mhOeVJAJPWgOfHzwv7TtG5oNl95WNJI9NdCArWR65jneN4DH0sdXmvEuRHx
# ulv3QKWMFWtfPC+l6SfZzA5RR0VA9KyAeFY1zvEur3OuUQcE5aLwt17FyK4pl46u
# HBMHyzrEtCNRGs1IL6qJ22FfTEUC8973i5lP0vidOlarFNN0zrZAlCxO7Hfv8wAB
# NF/8YqNK1o3icaWPsVxrzNaxTLB3Y1t0N2V/+jYIZ6IALB1KERA/qaNfdDHOf+JI
# FqXiJTAr1L9VO2rTumVwW5eeUuWaWWY9j+HwouRMWrceT7HTteYEwhOOLBEiuZE8
# TxnwHNkG93IERpSkAF454SiyZM3PHT18TvPvcVNMkNv6+5R14kZbaUK0g4ZkgSIH
# rDr/AA==
# SIG # End signature block
