#
# createclass.ps1
#
# Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
#

# Greeting
echo ""
echo "    VISlib CreateClass powershell script"
echo "    Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten."
echo ""

# Check arguments
if ($Args.Length -eq 3) {
	$Project = $Args[0]
	$Mode = $Args[1]
	$ClassName = $Args[2]
	$Namespaces = $Args[2].Split(":", [StringSplitOptions]::RemoveEmptyEntries)
	
	if ($Namespaces.Length -eq 1) {
		$Namespaces = "vislib", $Project
	} else {
		$ClassName = $Namespaces[$Namespaces.Length - 1]
		$Namespaces = $Namespaces[0..($Namespaces.Length - 2)]
	}
	
	$FullyQualifiedNamespace = $Namespaces[0]
	for ($i = 1; $i -lt $Namespaces.Length; $i++) {
		$FullyQualifiedNamespace += "::" + $Namespaces[$i]
	}
	
} else {
	echo "Usage: createclass.ps1 <project> [public|private] <classname>"
    echo ""
	exit
}

$curYear = get-date -uformat "%Y"

# Check creation mode and set path variables
$CPPPath = ".\$Project\src\"
if ($Mode -eq "public") {
    $HPath = ".\$Project\include\vislib\"
} elseif ($Mode -eq "private") {
    $HPath = ".\$Project\src\"
} else {
    echo "Error: You must specify either `"public`" or `"private`" as creation mode."
    echo ""
    exit
}

# Check Project
if (! (test-path $CPPPath)) {
    echo "Error: No project named `"$Project`" found."
    echo ""
    exit
}

# Make file names with path
$CPPFile = $CPPPath + $ClassName + ".cpp"
$HFile = $HPath + $ClassName + ".h"

# Check if classes already exists
if ((test-path $CPPFile) -or (test-path $HFile)) {
    echo "Error: Project `"$Project`" has already a class named `"$ClassName`"."
    echo ""
    exit
}

# Summary and confirmation
echo "Creating $Mode class `"$ClassName`" in namespace `"$FullyQualifiedNamespace`" in project `"$Project`"";
$confirmation = read-host "Press 'm' to make it so"
if ($confirmation -ne "m") {
    echo ""
    exit
}

# Writing CPP file
$CPPContent = "/*
 * " + $ClassName + ".cpp
 *
 * Copyright (C) 2006 - "
 $CPPContent += $curYear
 $CPPContent += " by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include `""

if ($Mode -eq "public") {
    $CPPContent += "vislib/"
}

$CPPContent += $ClassName + ".h`"


/*
 * " + $FullyQualifiedNamespace + "::" + $ClassName + "::" + $ClassName + "
 */
" + $FullyQualifiedNamespace + "::" + $ClassName + "::" + $ClassName + "(void) {
    // TODO: Implement
}


/*
 * " + $FullyQualifiedNamespace + "::" + $ClassName + "::~" + $ClassName + "
 */
" + $FullyQualifiedNamespace + "::" + $ClassName + "::~" + $ClassName + "(void) {
    // TODO: Implement
}"

# Writing H file
$HContent = "/*
 * " + "$ClassName" + ".h
 *
 * Copyright (C) 2006 - "
 $HContent += $curYear
 $HContent += " by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_" 
$HContent += $ClassName.ToUpper()
$HContent += "_H_INCLUDED
#define VISLIB_"
$HContent += $ClassName.ToUpper()
$HContent += "_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
"

if ($Mode -eq "public") {
    $HContent += "#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */
"
}

$HContent += "

"
foreach ($Namespace in $Namespaces) {
	$HContent += "namespace " + $Namespace + " {
"
}
$HContent += "

    /**
     * TODO: comment class
     */
    class " + $ClassName + " {

    public:

        /** Ctor. */
        " + $ClassName + "(void);

        /** Dtor. */
        ~" + $ClassName + "(void);

    protected:

    private:

    };
    
"
for ($i = $Namespaces.length - 1; $i -ge 0; $i--) {
	$HContent += "} /* end namespace " + $Namespaces[$i] + " */
"
}
$HContent += "
"

if ($Mode -eq "public") {
    $HContent += "#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
"
}

$HContent += "#endif /* VISLIB_" + $ClassName.ToUpper() + "_H_INCLUDED */
"

#echo $HContent > $HFile
ni . -name $HFile -force -type "file" -value ($HContent | out-string) >> $null
ni . -name $CPPFile -force -type "file" -value ($CPPContent | out-string) >> $null

echo "Done."
echo ""

# SIG # Begin signature block
# MIINsAYJKoZIhvcNAQcCoIINoTCCDZ0CAQExCzAJBgUrDgMCGgUAMGkGCisGAQQB
# gjcCAQSgWzBZMDQGCisGAQQBgjcCAR4wJgIDAQAABBAfzDtgWUsITrck0sYpfvNR
# AgEAAgEAAgEAAgEAAgEAMCEwCQYFKw4DAhoFAAQUYV719qPhMYg7YoSQ5L2vubPb
# UX2gggn7MIIJ9zCCB9+gAwIBAgIKYXFZ9AAAAAAACTANBgkqhkiG9w0BAQsFADBw
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
# BgorBgEEAYI3AgEVMCMGCSqGSIb3DQEJBDEWBBQfER12ZuUY85Mf7GyGlOoQU6ln
# wzANBgkqhkiG9w0BAQEFAASCAgBCgkPUeGOlcTSCwDDSd36/QolmPLEqkHDTNYcb
# AEC4vq4sRQXb6MKb25equwRxUoEIl8WvDdBO9yYMDlSl6kTXhDSdDaryRhhB44R2
# wyC96FNhL4udeTBLl3JA/ZC/O+IWBj/h38TpAqASR785WiMF5mUS8wmdiaTSTGjE
# gSPIrT5wMY0n8bY5+1WoJ0+p7u+VocUgc674+WyfIfhS/M3MphMjO/wJDuIUTpHS
# LkIu1VI1qFB1riTgkuEN2nENuxoMTTpVYvSLCVfz1OUfOGMmlMd7WUj5Jv5q3OAA
# JCOlnSh6W8cMukQumBteWUR/HFpJOahuogp/w4wQMhyvaWkHiWjXe08TQLwUQed0
# ik4c8Ye/6eE7/O85PQuRphS59FzeY+j9Cfy9zBZKd31y1MotYA+zuFsfRlLeWBBP
# q8yVkEaQoosveNK8x2ciY80embUTzUcDVumsylZwMM2NZ4twpc68eHpJ+oYiDYxj
# m2ubyxjbBCqvIvlSpPtUCfLVHqSZzrKEgo35nyRQbDIHErdwBSBD0uGsk3Ctk+og
# tB7gyukQw2f1G11xSW6gkdBmc1UQDdTOAnoYitVYq30vCmhxKPSVLhvfZh9tnE0o
# xauD+d5Oed5X3I8vtty0YgeVqQtNunVCWSEFo0D1Xx0P5Fad1kfdDIokhFqRwtGP
# QIJy/Q==
# SIG # End signature block
