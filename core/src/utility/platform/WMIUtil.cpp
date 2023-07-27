/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/platform/WMIUtil.h"

#include "mmcore/utility/log/Log.h"

#ifdef _WIN32

megamol::core::utility::platform::WMIUtil::WMIUtil() {
    HRESULT hres;

    // Step 1: --------------------------------------------------
    // Initialize COM. ------------------------------------------

    hres = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hres)) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "WMIUtil: Failed to initialize COM library. Is COM already initialized? Error code = %#010X", hres);
    }

    // Step 2: --------------------------------------------------
    // Set general COM security levels --------------------------

    hres = CoInitializeSecurity(nullptr,
        -1,                          // COM authentication
        nullptr,                     // Authentication services
        nullptr,                     // Reserved
        RPC_C_AUTHN_LEVEL_DEFAULT,   // Default authentication
        RPC_C_IMP_LEVEL_IMPERSONATE, // Default Impersonation
        nullptr,                     // Authentication info
        EOAC_NONE,                   // Additional capabilities
        nullptr                      // Reserved
    );


    if (FAILED(hres)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "WMIUtil: Failed to initialize security. Error code = %#010X", hres);
        CoUninitialize();
        return;
    }

    // Step 3: ---------------------------------------------------
    // Obtain the initial locator to WMI -------------------------

    hres = CoCreateInstance(
        CLSID_WbemLocator, nullptr, CLSCTX_INPROC_SERVER, IID_IWbemLocator, reinterpret_cast<LPVOID*>(&locator));

    if (FAILED(hres)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "WMIUtil: Failed to create IWbemLocator object. Error code = %#010X", hres);
        CoUninitialize();
        return;
    }

    // Step 4: -----------------------------------------------------
    // Connect to WMI through the IWbemLocator::ConnectServer method

    // Connect to the root\cimv2 namespace with
    // the current user and obtain pointer pSvc
    // to make IWbemServices calls.
    hres = locator->ConnectServer(_bstr_t(L"ROOT\\CIMV2"), // Object path of WMI namespace
        nullptr,                                           // User name. NULL = current user
        nullptr,                                           // User password. NULL = current
        nullptr,                                           // Locale. NULL indicates current
        NULL,                                              // Security flags.
        nullptr,                                           // Authority (for example, Kerberos)
        nullptr,                                           // Context object
        &service                                           // pointer to IWbemServices proxy
    );

    if (FAILED(hres)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "WMIUtil: Could not connect. Error code = %#010X", hres);
        locator->Release();
        locator = nullptr;
        CoUninitialize();
        return;
    }

    // cout << "Connected to ROOT\\CIMV2 WMI namespace" << endl;

    // Step 5: --------------------------------------------------
    // Set security levels on the proxy -------------------------

    hres = CoSetProxyBlanket(service, // Indicates the proxy to set
        RPC_C_AUTHN_WINNT,            // RPC_C_AUTHN_xxx
        RPC_C_AUTHZ_NONE,             // RPC_C_AUTHZ_xxx
        nullptr,                      // Server principal name
        RPC_C_AUTHN_LEVEL_CALL,       // RPC_C_AUTHN_LEVEL_xxx
        RPC_C_IMP_LEVEL_IMPERSONATE,  // RPC_C_IMP_LEVEL_xxx
        nullptr,                      // client identity
        EOAC_NONE                     // proxy capabilities
    );

    if (FAILED(hres)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "WMIUtil: Could not set proxy blanket. Error code = %#010X", hres);
        service->Release();
        service = nullptr;
        locator->Release();
        locator = nullptr;
        CoUninitialize();
        return;
    }
}

megamol::core::utility::platform::WMIUtil::~WMIUtil() {
    // Cleanup
    // ========

    if (service)
        service->Release();
    if (locator)
        locator->Release();
    CoUninitialize();
}

std::string megamol::core::utility::platform::WMIUtil::get_value(
    const std::string& wmi_class, const std::string& attribute) {

    IEnumWbemClassObject* enumerator = nullptr;

    // Step 6: --------------------------------------------------
    // Use the IWbemServices pointer to make requests of WMI ----

    if (!service)
        return "";
    std::string query = "SELECT * FROM " + wmi_class;
    HRESULT hres = service->ExecQuery(bstr_t("WQL"), bstr_t(query.c_str()),
        WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY, nullptr, &enumerator);

    if (FAILED(hres)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "WMIUtil: Query for %s has failed. Error code = %#010X", wmi_class.c_str(), hres);
    }


    // Step 7: -------------------------------------------------
    // Get the data from the query in step 6 -------------------

    IWbemClassObject* pclsObj = nullptr;
    ULONG uReturn = 0;
    std::string ret;

    // conversion from https://stackoverflow.com/questions/6284524/bstr-to-stdstring-stdwstring-and-vice-versa

    auto wslen = ::MultiByteToWideChar(CP_ACP, 0 /* no flags */, attribute.data(), attribute.length(), nullptr, 0);

    BSTR ws_attribute = ::SysAllocStringLen(nullptr, wslen);
    ::MultiByteToWideChar(CP_ACP, 0 /* no flags */, attribute.data(), attribute.length(), ws_attribute, wslen);

    while (enumerator) {
        HRESULT hr = enumerator->Next(WBEM_INFINITE, 1, &pclsObj, &uReturn);

        if (0 == uReturn) {
            break;
        }

        VARIANT vtProp;

        // Get the value of the Name property
        hr = pclsObj->Get(ws_attribute, 0, &vtProp, nullptr, nullptr);
        auto wslen = ::SysStringLen(vtProp.bstrVal);
        auto len = ::WideCharToMultiByte(CP_ACP, 0, vtProp.bstrVal, wslen, nullptr, 0, nullptr, nullptr);
        std::string dblstr(len, '\0');
        len = ::WideCharToMultiByte(CP_ACP, 0 /* no flags */, vtProp.bstrVal, wslen /* not necessary NULL-terminated */,
            &dblstr[0], len, nullptr, nullptr /* no default char */);
        ret = dblstr;
        VariantClear(&vtProp);

        pclsObj->Release();
    }
    enumerator->Release();
    return ret;
}

#endif
