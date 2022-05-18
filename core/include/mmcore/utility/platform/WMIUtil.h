#pragma once

#ifdef _WIN32

#define _WIN32_DCOM
#include <Wbemidl.h>
#include <comdef.h>
#include <iostream>

#pragma comment(lib, "wbemuuid.lib")

namespace megamol::core::utility::platform {

// from https://docs.microsoft.com/en-us/windows/win32/wmisdk/example--getting-wmi-data-from-the-local-computer
class WMIUtil {
public:
    WMIUtil();
    ~WMIUtil();

    std::string get_value(const std::string& wmi_class, const std::string& attribute);

private:
    IWbemLocator* locator = nullptr;
    IWbemServices* service = nullptr;
};
} // namespace megamol::core::utility::platform

#endif
