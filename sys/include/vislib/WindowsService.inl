/*
 * WindowsService.inl
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#define HKSERVICESA "System\\CurrentControlSet\\Services"
#define HKSERVICESW L"System\\CurrentControlSet\\Services"
#define VALNAMEDESCA "Description"
#define VALNAMEDESCW L"Description"

#define IMPLEMENT_WINDOWS_SERVICE_INSTALL1(strType)                            \
    SC_HANDLE hSCMgr = NULL;                                                   \
    SC_HANDLE hSvc = NULL;                                                     \
                                                                               \
    if ((hSCMgr = ::OpenSCManager##strType(NULL, NULL, SC_MANAGER_ALL_ACCESS)) \
            == NULL) {                                                         \
        throw the::system::system_exception(__FILE__, __LINE__);                             \
    }                                                                          \
                                                                               \
    if ((hSvc = ::CreateService##strType(hSCMgr, svcName.c_str(), displayName.c_str(),         \
            SERVICE_ALL_ACCESS, svcType, startType, SERVICE_ERROR_NORMAL,      \
            binaryPath.c_str(), NULL, NULL, NULL, NULL, NULL)) == NULL) {              \
        ::CloseServiceHandle(hSCMgr);                                          \
        throw the::system::system_exception(__FILE__, __LINE__);                             \
    }                                                                          \
                                                                               \
    ::CloseServiceHandle(hSvc);                                                \
    ::CloseServiceHandle(hSCMgr); 


#define IMPLEMENT_WINDOWS_SERVICE_SETDESCRIPTION(strType)                      \
    HKEY hKeyServices = NULL;                                                  \
    HKEY hKeySvc = NULL;                                                       \
    bool retval = false;                                                       \
                                                                               \
    if (::RegOpenKey##strType(HKEY_LOCAL_MACHINE, HKSERVICES##strType,         \
            &hKeyServices) != ERROR_SUCCESS) {                                 \
        return false;                                                          \
    }                                                                          \
                                                                               \
    if (::RegOpenKey##strType(hKeyServices, svcName.c_str(), &hKeySvc)                 \
            != ERROR_SUCCESS) {                                                \
        ::RegCloseKey(hKeyServices);                                           \
        return false;                                                          \
    }                                                                          \
                                                                               \
    retval = (::RegSetValueEx##strType(hKeySvc, VALNAMEDESC##strType, 0,       \
        REG_SZ, reinterpret_cast<const uint8_t *>(desc.c_str()),             \
        static_cast<DWORD>(desc.size() * sizeof(Char))) == ERROR_SUCCESS);    \
                                                                               \
    ::RegCloseKey(hKeyServices);                                               \
    ::RegCloseKey(hKeySvc);                                                    \
    return retval;


#define IMPLEMENT_WINDOWS_SERVICE_INSTALL2(strType)                            \
    String binaryPath(MAX_PATH, ' ');                                                \
                                                                               \
    if (::GetModuleFileName##strType(NULL, const_cast<Char*>(binaryPath.c_str()),\
            MAX_PATH)) {                                                       \
        WindowsService::Install(binaryPath.c_str(), this->name.c_str(),                        \
            this->displayName.c_str(), this->status.dwServiceType, startType);         \
    } else {                                                                   \
        throw the::system::system_exception(__FILE__, __LINE__);                             \
    }


#define IMPLEMENT_WINDOWS_SERVICE_UNINSTALL(strType)                           \
    SC_HANDLE hSCMgr = NULL;                                                   \
    SC_HANDLE hSvc = NULL;                                                     \
                                                                               \
    if ((hSCMgr = ::OpenSCManager##strType(NULL, NULL, SC_MANAGER_ALL_ACCESS)) \
            == NULL) {                                                         \
        throw the::system::system_exception(__FILE__, __LINE__);                             \
    }                                                                          \
                                                                               \
    if ((hSvc = ::OpenService##strType(hSCMgr, this->name.c_str(),                     \
            SERVICE_ALL_ACCESS)) == NULL) {                                    \
        ::CloseServiceHandle(hSCMgr);                                          \
        throw the::system::system_exception(__FILE__, __LINE__);                             \
    }                                                                          \
                                                                               \
    if (!::DeleteService(hSvc)) {                                              \
        ::CloseServiceHandle(hSvc);                                            \
        ::CloseServiceHandle(hSCMgr);                                          \
        throw the::system::system_exception(__FILE__, __LINE__);                             \
    }                                                                          \
                                                                               \
    ::CloseServiceHandle(hSvc);                                                \
    ::CloseServiceHandle(hSCMgr);


#define IMPLEMENT_WINDOWS_SERVICE_RUN(strType)                                 \
    DWORD errorCode = 0;                                                       \
    SERVICE_TABLE_ENTRY##strType svcTable[] = {                                \
        { const_cast<Char *>(this->name.c_str()), reinterpret_cast<       \
LPSERVICE_MAIN_FUNCTION##strType>(&WindowsService::serviceMain) },             \
        { NULL, NULL}                                                          \
    };                                                                         \
                                                                               \
    if (WindowsService::instance == NULL) {                                    \
        WindowsService::instance = this;                                       \
    } else {                                                                   \
        throw the::invalid_operation_exception("There is already a Windows Service"       \
            "instance.", __FILE__, __LINE__);                                  \
    }                                                                          \
                                                                               \
    if (::StartServiceCtrlDispatcher##strType(svcTable)) {                     \
        /* Service is running. */                                              \
        return true;                                                           \
                                                                               \
    } else {                                                                   \
        WindowsService::instance = NULL;                                       \
                                                                               \
        if ((errorCode = ::GetLastError())                                     \
                == ERROR_FAILED_SERVICE_CONTROLLER_CONNECT) {                  \
            /* Running as normal application. */                               \
            return false;                                                      \
        } else {                                                               \
            /* Could not start service for other reason. */                    \
            throw the::system::system_exception(errorCode, __FILE__, __LINE__);              \
        }                                                                      \
    }                                                                           


#define IMPLEMENT_WINDOWS_SERVICE_HANDLER_EX                                   \
    THE_ASSERT(context != NULL);                                                   \
    DWORD retval = NO_ERROR;                                                   \
    WindowsService *svc = static_cast<WindowsService *>(context);              \
                                                                               \
    switch (control) {                                                         \
        case SERVICE_CONTROL_CONTINUE:                                         \
            try {                                                              \
                svc->setStatus(SERVICE_CONTINUE_PENDING);                      \
                if ((retval = svc->OnContinue()) == NO_ERROR) {                \
                    svc->setStatus(SERVICE_RUNNING);                           \
                }                                                              \
            } catch (the::system::system_exception e) {                                      \
                retval = e.get_error().native_error();                                     \
            }                                                                  \
            break;                                                             \
                                                                               \
        case SERVICE_CONTROL_INTERROGATE:                                      \
            if (!::SetServiceStatus(svc->hStatus, &svc->status)) {             \
                retval = ::GetLastError();                                     \
            }                                                                  \
            break;                                                             \
                                                                               \
        case SERVICE_CONTROL_NETBINDADD:                                       \
            retval = svc->OnNetBindAdd();                                      \
            break;                                                             \
                                                                               \
        case SERVICE_CONTROL_NETBINDDISABLE:                                   \
            retval = svc->OnNetBindDisable();                                  \
            break;                                                             \
                                                                               \
        case SERVICE_CONTROL_NETBINDENABLE:                                    \
            retval = svc->OnNetBindEnable();                                   \
            break;                                                             \
                                                                               \
        case SERVICE_CONTROL_NETBINDREMOVE:                                    \
            retval = svc->OnNetBindRemove();                                   \
            break;                                                             \
                                                                               \
        case SERVICE_CONTROL_PARAMCHANGE:                                      \
            retval = svc->OnParamChange();                                     \
            break;                                                             \
                                                                               \
        case SERVICE_CONTROL_PAUSE:                                            \
            try {                                                              \
                svc->setStatus(SERVICE_PAUSE_PENDING);                         \
                if ((retval = svc->OnPause()) == NO_ERROR) {                   \
                    svc->setStatus(SERVICE_PAUSED);                            \
                }                                                              \
            } catch (the::system::system_exception e) {                                      \
                retval = e.get_error().native_error();                                     \
            }                                                                  \
            break;                                                             \
                                                                               \
        case SERVICE_CONTROL_SHUTDOWN:                                         \
            try {                                                              \
                svc->setStatus(SERVICE_STOP_PENDING);                          \
                if ((retval = svc->OnShutdown()) == NO_ERROR) {                \
                    svc->setStatus(SERVICE_STOPPED);                           \
                }                                                              \
            } catch (the::system::system_exception e) {                                      \
                retval = e.get_error().native_error();                                     \
            }                                                                  \
            break;                                                             \
                                                                               \
        case SERVICE_CONTROL_STOP:                                             \
            try {                                                              \
                svc->setStatus(SERVICE_STOP_PENDING);                          \
                if ((retval = svc->OnStop()) == NO_ERROR) {                    \
                    svc->setStatus(SERVICE_STOPPED);                           \
                }                                                              \
            } catch (the::system::system_exception e) {                                      \
                retval = e.get_error().native_error();                                     \
            }                                                                  \
            break;                                                             \
                                                                               \
        case SERVICE_CONTROL_DEVICEEVENT:                                      \
            retval = svc->OnDeviceEvent(eventType, eventData);                 \
            break;                                                             \
                                                                               \
        case SERVICE_CONTROL_HARDWAREPROFILECHANGE:                            \
            retval = svc->OnHardwareProfileChanged(eventType);                 \
            break;                                                             \
                                                                               \
        case SERVICE_CONTROL_POWEREVENT:                                       \
            retval = svc->OnPowerEvent(eventType, eventData);                  \
            break;                                                             \
                                                                               \
        case SERVICE_CONTROL_SESSIONCHANGE:                                    \
            retval = svc->OnSessionChange(eventType, eventData);               \
            break;                                                             \
                                                                               \
        default:                                                               \
            if ((control >= 128) && (control <= 255)) {                        \
                retval = svc->OnUserControl(control);                          \
            } else {                                                           \
                retval = ERROR_CALL_NOT_IMPLEMENTED;                           \
            }                                                                  \
            break;                                                             \
    }                                                                          \
                                                                               \
    return retval;


#define IMPLEMENT_WINDOWS_SERVICE_SERVICE_MAIN(strType)                        \
    THE_ASSERT(WindowsService::instance != NULL);                                  \
                                                                               \
    SERVICE_STATUS_HANDLE& hStatus = WindowsService::instance->hStatus;        \
    SERVICE_STATUS& status = WindowsService::instance->status;                 \
                                                                               \
    if ((hStatus = ::RegisterServiceCtrlHandlerEx##strType(NULL,               \
            WindowsService::handlerEx, WindowsService::instance)) != NULL) {   \
        status.dwCurrentState = SERVICE_RUNNING;                               \
        ::SetServiceStatus(hStatus, &status);                                  \
                                                                               \
        status.dwServiceSpecificExitCode                                       \
            = WindowsService::instance->OnRun(argc,                            \
            const_cast<const Char **>(argv));                                  \
        if (status.dwServiceSpecificExitCode == 0) {                           \
            status.dwWin32ExitCode = 0;                                        \
        } else {                                                               \
            status.dwWin32ExitCode = ERROR_SERVICE_SPECIFIC_ERROR;             \
        }                                                                      \
                                                                               \
        status.dwCurrentState = SERVICE_STOPPED;                               \
        ::SetServiceStatus(hStatus, &status);                                  \
    }                                                                          \
                                                                               \
    ::CloseServiceHandle(reinterpret_cast<SC_HANDLE>(hStatus));                \
    hStatus = NULL;


////////////////////////////////////////////////////////////////////////////////
// BEGIN OF PARTIAL TEMPLATE SPECIALISATION FOR CharTraitsA

/*
 * vislib::sys::WindowsService<the::astring>::Install
 */
void vislib::sys::WindowsService<the::astring>::Install(
        const String& binaryPath, const String& svcName, 
        const String& displayName, const DWORD svcType, const DWORD startType) {
    IMPLEMENT_WINDOWS_SERVICE_INSTALL1(A);
}


/*
 * vislib::sys::WindowsService<the::astring>::SetDescription
 */
bool vislib::sys::WindowsService<the::astring>::SetDescription(
        const String& svcName, const String& desc) {
    IMPLEMENT_WINDOWS_SERVICE_SETDESCRIPTION(A);
}


/*
 * vislib::sys::WindowsService<the::astring>::~WindowsService
 */
vislib::sys::WindowsService<the::astring>::~WindowsService(void) {
    if (this->hStatus != NULL) {
        ::CloseServiceHandle(reinterpret_cast<SC_HANDLE>(this->hStatus));
    }
}


/*
 * vislib::sys::WindowsService<the::astring>::Install
 */
void vislib::sys::WindowsService<the::astring>::Install(
        const DWORD startType) {
    IMPLEMENT_WINDOWS_SERVICE_INSTALL2(A);
}


/*
 * vislib::sys::WindowsService<the::astring>::Run
 */
bool vislib::sys::WindowsService<the::astring>::Run(void) {
    IMPLEMENT_WINDOWS_SERVICE_RUN(A);
}


/*
 * vislib::sys::WindowsService<the::astring>::OnContinue
 */
DWORD vislib::sys::WindowsService<the::astring>::OnContinue(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::astring>::OnDeviceEvent
 */
DWORD vislib::sys::WindowsService<the::astring>::OnDeviceEvent(
        const DWORD eventType, void *eventData) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::astring>::OnHardwareProfileChanged
 */
DWORD 
vislib::sys::WindowsService<the::astring>::OnHardwareProfileChanged(
        const DWORD eventType) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::astring>::OnNetBindAdd
 */
DWORD vislib::sys::WindowsService<the::astring>::OnNetBindAdd(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::astring>::OnNetBindDisable
 */
DWORD vislib::sys::WindowsService<the::astring>::OnNetBindDisable(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::astring>::OnNetBindEnable
 */
DWORD vislib::sys::WindowsService<the::astring>::OnNetBindEnable(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::astring>::OnNetBindRemove
 */
DWORD vislib::sys::WindowsService<the::astring>::OnNetBindRemove(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/* 
 * vislib::sys::WindowsService<the::astring>::OnParamChange
 */
DWORD vislib::sys::WindowsService<the::astring>::OnParamChange(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::astring>::OnPause
 */
DWORD vislib::sys::WindowsService<the::astring>::OnPause(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::astring>::OnPowerEvent
 */
DWORD vislib::sys::WindowsService<the::astring>::OnPowerEvent(
        const DWORD eventType, void *eventData) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::astring>::OnSessionChange
 */
DWORD vislib::sys::WindowsService<the::astring>::OnSessionChange(
        const DWORD eventType, void *eventData) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::astring>::OnShutdown
 */
DWORD vislib::sys::WindowsService<the::astring>::OnShutdown(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::astring>::OnStop
 */
DWORD vislib::sys::WindowsService<the::astring>::OnStop(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::astring>::OnUserControl
 */
DWORD vislib::sys::WindowsService<the::astring>::OnUserControl(
        const DWORD control) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::astring>::Uninstall
 */
void vislib::sys::WindowsService<the::astring>::Uninstall(void) {
    IMPLEMENT_WINDOWS_SERVICE_UNINSTALL(A);
}


/*
 * vislib::sys::WindowsService<the::astring>::handlerEx
 */
DWORD WINAPI vislib::sys::WindowsService<the::astring>::handlerEx(
        DWORD control, DWORD eventType, void *eventData, void *context) {
    IMPLEMENT_WINDOWS_SERVICE_HANDLER_EX;
}


/*
 * vislib::sys::WindowsService<the::astring>::serviceMain
 */
void WINAPI vislib::sys::WindowsService<the::astring>::serviceMain(
        DWORD argc, Char **argv) {
    IMPLEMENT_WINDOWS_SERVICE_SERVICE_MAIN(A);
}


/*
 * vislib::sys::WindowsService<the::astring>::instance
 */
vislib::sys::WindowsService<the::astring> 
*vislib::sys::WindowsService<the::astring>::instance = NULL;


/*
 * vislib::sys::WindowsService<the::astring>::operator =
 */
vislib::sys::WindowsService<the::astring>& 
vislib::sys::WindowsService<the::astring>::operator =(
        const WindowsService& rhs) {
    if (this != &rhs) {
        throw the::argument_exception("rhs", __FILE__, __LINE__);
    }

    return *this;
}

// END OF PARTIAL TEMPLATE SPECIALISATION FOR CharTraitsA
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// BEGIN OF PARTIAL TEMPLATE SPECIALISATION FOR CharTraitsW

/*
 * vislib::sys::WindowsService<the::wstring>::Install
 */
void vislib::sys::WindowsService<the::wstring>::Install(
        const String& binaryPath, const String& svcName, 
        const String& displayName, const DWORD svcType, const DWORD startType) {
    IMPLEMENT_WINDOWS_SERVICE_INSTALL1(W);
}


/*
 * vislib::sys::WindowsService<the::wstring>::SetDescription
 */
bool vislib::sys::WindowsService<the::wstring>::SetDescription(
        const String& svcName, const String& desc) {
    IMPLEMENT_WINDOWS_SERVICE_SETDESCRIPTION(W);
}



/*
 * vislib::sys::WindowsService<the::wstring>::~WindowsService
 */
vislib::sys::WindowsService<the::wstring>::~WindowsService(void) {
    if (this->hStatus != NULL) {
        ::CloseServiceHandle(reinterpret_cast<SC_HANDLE>(this->hStatus));
    }
}


/*
 * vislib::sys::WindowsService<the::wstring>::Install
 */
void vislib::sys::WindowsService<the::wstring>::Install(
        const DWORD startType) {
    IMPLEMENT_WINDOWS_SERVICE_INSTALL2(W);
}


/*
 * vislib::sys::WindowsService<the::wstring>::Run
 */
bool vislib::sys::WindowsService<the::wstring>::Run(void) {
    IMPLEMENT_WINDOWS_SERVICE_RUN(W);
}


/*
 * vislib::sys::WindowsService<the::wstring>::OnContinue
 */
DWORD vislib::sys::WindowsService<the::wstring>::OnContinue(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::wstring>::OnDeviceEvent
 */
DWORD vislib::sys::WindowsService<the::wstring>::OnDeviceEvent(
        const DWORD eventType, void *eventData) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::wstring>::OnHardwareProfileChanged
 */
DWORD 
vislib::sys::WindowsService<the::wstring>::OnHardwareProfileChanged(
        const DWORD eventType) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::wstring>::OnNetBindAdd
 */
DWORD vislib::sys::WindowsService<the::wstring>::OnNetBindAdd(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::wstring>::OnNetBindDisable
 */
DWORD vislib::sys::WindowsService<the::wstring>::OnNetBindDisable(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::wstring>::OnNetBindEnable
 */
DWORD vislib::sys::WindowsService<the::wstring>::OnNetBindEnable(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::wstring>::OnNetBindRemove
 */
DWORD vislib::sys::WindowsService<the::wstring>::OnNetBindRemove(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/* 
 * vislib::sys::WindowsService<the::wstring>::OnParamChange
 */
DWORD vislib::sys::WindowsService<the::wstring>::OnParamChange(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::wstring>::OnPause
 */
DWORD vislib::sys::WindowsService<the::wstring>::OnPause(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::wstring>::OnPowerEvent
 */
DWORD vislib::sys::WindowsService<the::wstring>::OnPowerEvent(
        const DWORD eventType, void *eventData) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::wstring>::OnSessionChange
 */
DWORD vislib::sys::WindowsService<the::wstring>::OnSessionChange(
        const DWORD eventType, void *eventData) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::wstring>::OnShutdown
 */
DWORD vislib::sys::WindowsService<the::wstring>::OnShutdown(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::wstring>::OnStop
 */
DWORD vislib::sys::WindowsService<the::wstring>::OnStop(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::wstring>::OnUserControl
 */
DWORD vislib::sys::WindowsService<the::wstring>::OnUserControl(
        const DWORD control) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<the::wstring>::Uninstall
 */
void vislib::sys::WindowsService<the::wstring>::Uninstall(void) {
    IMPLEMENT_WINDOWS_SERVICE_UNINSTALL(W);
}


/*
 * vislib::sys::WindowsService<the::wstring>::handlerEx
 */
DWORD WINAPI vislib::sys::WindowsService<the::wstring>::handlerEx(
        DWORD control, DWORD eventType, void *eventData, void *context) {
    IMPLEMENT_WINDOWS_SERVICE_HANDLER_EX;
}


/*
 * vislib::sys::WindowsService<the::wstring>::serviceMain
 */
void WINAPI vislib::sys::WindowsService<the::wstring>::serviceMain(
        DWORD argc, Char **argv) {
    IMPLEMENT_WINDOWS_SERVICE_SERVICE_MAIN(W);
}


/*
 * vislib::sys::WindowsService<the::wstring>::instance
 */
vislib::sys::WindowsService<the::wstring> 
*vislib::sys::WindowsService<the::wstring>::instance = NULL;


/*
 * vislib::sys::WindowsService<the::wstring>::operator =
 */
vislib::sys::WindowsService<the::wstring>& 
vislib::sys::WindowsService<the::wstring>::operator =(
        const WindowsService& rhs) {
    if (this != &rhs) {
        throw the::argument_exception("rhs", __FILE__, __LINE__);
    }

    return *this;
}

// END OF PARTIAL TEMPLATE SPECIALISATION FOR CharTraitsW
////////////////////////////////////////////////////////////////////////////////

#undef IMPLEMENT_WINDOWS_SERVICE_INSTALL1
#undef IMPLEMENT_WINDOWS_SERVICE_INSTALL2
#undef IMPLEMENT_WINDOWS_SERVICE_UNINSTALL
#undef IMPLEMENT_WINDOWS_SERVICE_RUN
#undef IMPLEMENT_WINDOWS_SERVICE_HANDLER_EX
#undef IMPLEMENT_WINDOWS_SERVICE_SERVICE_MAIN
