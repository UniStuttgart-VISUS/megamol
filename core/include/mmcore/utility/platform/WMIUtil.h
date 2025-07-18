/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#ifdef _WIN32

#include <iostream>
#include <wil/com.h>

#define _WIN32_DCOM
#include <Wbemidl.h>
#include <comdef.h>

#pragma comment(lib, "wbemuuid.lib")

namespace megamol::core::utility::platform {

// from https://docs.microsoft.com/en-us/windows/win32/wmisdk/example--getting-wmi-data-from-the-local-computer
class WMIUtil {
public:
    WMIUtil();
    ~WMIUtil();

    std::string get_value(const std::string& wmi_class, const std::string& attribute);

private:
    //IWbemLocator* locator = nullptr;
    //IWbemServices* service = nullptr;
    wil::com_ptr<IWbemServices> service;
    //wil::unique_couninitialize_call cleanup;
};
} // namespace megamol::core::utility::platform

#endif
