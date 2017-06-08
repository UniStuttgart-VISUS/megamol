#
# sign-script.ps1
#
# Copyright (C) 2008 by Universität Stuttgart (VISUS). Alle Rechte vorbehalten.
#

if ($Args.Length -eq 2) {
    $file = $Args[0]
    $user = $Args[1]

    if (Test-Path $file) {
        $search = "CN\s*=\s*{0}" -f $user
        $matches = @(gci -r cert:\CurrentUser\My -codesigning | where { $_.Subject -imatch $search })

        if ($matches.Length -eq 1) {
            Set-AuthenticodeSignature $file $matches[0]
        } else {
            Write-Output ("Your query `"{0}`" should match exactly one certificate, but actually matches {1}:" -f $user, $matches.Length)
            $matches | foreach { Write-Output ("`"{0}`"" -f $_.Subject) }
        }

    } else {
        Write-Output "The specified file `"$file`" does not exist." 
    }

} else {
    Write-Output ("Usage: {0} <Script> <Certificate CN>" -f ($MyInvocation.MyCommand).Name)
}

# SIG # Begin signature block
# MIINsAYJKoZIhvcNAQcCoIINoTCCDZ0CAQExCzAJBgUrDgMCGgUAMGkGCisGAQQB
# gjcCAQSgWzBZMDQGCisGAQQBgjcCAR4wJgIDAQAABBAfzDtgWUsITrck0sYpfvNR
# AgEAAgEAAgEAAgEAAgEAMCEwCQYFKw4DAhoFAAQUmuHa3sXAQAzfmcTP5UuZ9L7k
# fk2gggn7MIIJ9zCCB9+gAwIBAgIKYXFZ9AAAAAAACTANBgkqhkiG9w0BAQsFADBw
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
# BgorBgEEAYI3AgEVMCMGCSqGSIb3DQEJBDEWBBQmpsS/lEMMwzEIcrFFutOAuqi8
# TDANBgkqhkiG9w0BAQEFAASCAgA0ZkkKwwjg7ZcjdPfwCrX4feQwzF5xViuzkoRm
# asMt/U8LsklTid9n1epRKQqIvrcyMohzgDcFA6EAs7FghlNO5HYCg9VrQnOlwg35
# t+fEpp9ZnIeGbBRulayx6RmesnSgfkDgYOdnbhRYcNvCMjWK36E7dDCwEBIgH8oH
# cjR2HfTXGedAMcCIW8i/eWAnK+zURohWXHHFoy+L8y2pZ+WhuA6oKoq8JJatPvpv
# QaP5heHuzvsrs8fuB5RdD0sndrQp+p0D/wJ2ol3Zjcmq5bI7YA7Vfv9M0hDhCtia
# k78Dpi6TEW0AWYdzyfSyGyZ1ZDqSwSomlx0a8Zy36jPaKguzvXb/muiSx2nmGp4N
# Vyn3RxFPoPgTeyiv5NB0vm3Dlv5I69HMuaTkRKrXNcxwVsEBs3iJ720bd7Fyx4GW
# qxiuxx8hchekXOq8qfRbJ5ENGrDlAKdG5gS/uNPhsmOARIU25kzuvdy6YD9GanwD
# 80Rh8gsnNcFq15PNr8490IB/ANwsX0ctrtlc6WVvLLd5DAIMhxq4//UUU/fhZ8Ur
# u8aYt1CaX9WVAF4LOITYZewwrgVWE2N4KPlajrayt3AYnJIIlJpCg4L927JSF/jU
# rBh0J31yuP9xeRJCSKGMEI534pgUznRJVde2bi4Gf1wuRobP6YpbiOTFdwpAW1z7
# SVvMuQ==
# SIG # End signature block
