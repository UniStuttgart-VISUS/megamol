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
        throw SystemException(__FILE__, __LINE__);                             \
    }                                                                          \
                                                                               \
    if ((hSvc = ::CreateService##strType(hSCMgr, svcName, displayName,         \
            SERVICE_ALL_ACCESS, svcType, startType, SERVICE_ERROR_NORMAL,      \
            binaryPath, NULL, NULL, NULL, NULL, NULL)) == NULL) {              \
        ::CloseServiceHandle(hSCMgr);                                          \
        throw SystemException(__FILE__, __LINE__);                             \
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
    if (::RegOpenKey##strType(hKeyServices, svcName, &hKeySvc)                 \
            != ERROR_SUCCESS) {                                                \
        ::RegCloseKey(hKeyServices);                                           \
        return false;                                                          \
    }                                                                          \
                                                                               \
    retval = (::RegSetValueEx##strType(hKeySvc, VALNAMEDESC##strType, 0,       \
        REG_SZ, reinterpret_cast<const BYTE *>(desc.PeekBuffer()),             \
        desc.Length() * CharTraits##strType::CharSize()) == ERROR_SUCCESS);    \
                                                                               \
    ::RegCloseKey(hKeyServices);                                               \
    ::RegCloseKey(hKeySvc);                                                    \
    return retval;


#define IMPLEMENT_WINDOWS_SERVICE_INSTALL2(strType)                            \
    String##strType binaryPath;                                                \
                                                                               \
    if (::GetModuleFileName##strType(NULL, binaryPath.AllocateBuffer(MAX_PATH),\
            MAX_PATH)) {                                                       \
        WindowsService::Install(binaryPath, this->name,                        \
            this->displayName, this->status.dwServiceType, startType);         \
    } else {                                                                   \
        throw SystemException(__FILE__, __LINE__);                             \
    }


#define IMPLEMENT_WINDOWS_SERVICE_UNINSTALL(strType)                           \
    SC_HANDLE hSCMgr = NULL;                                                   \
    SC_HANDLE hSvc = NULL;                                                     \
                                                                               \
    if ((hSCMgr = ::OpenSCManager##strType(NULL, NULL, SC_MANAGER_ALL_ACCESS)) \
            == NULL) {                                                         \
        throw SystemException(__FILE__, __LINE__);                             \
    }                                                                          \
                                                                               \
    if ((hSvc = ::OpenService##strType(hSCMgr, this->name,                     \
            SERVICE_ALL_ACCESS)) == NULL) {                                    \
        ::CloseServiceHandle(hSCMgr);                                          \
        throw SystemException(__FILE__, __LINE__);                             \
    }                                                                          \
                                                                               \
    if (!::DeleteService(hSvc)) {                                              \
        ::CloseServiceHandle(hSvc);                                            \
        ::CloseServiceHandle(hSCMgr);                                          \
        throw SystemException(__FILE__, __LINE__);                             \
    }                                                                          \
                                                                               \
    ::CloseServiceHandle(hSvc);                                                \
    ::CloseServiceHandle(hSCMgr);


#define IMPLEMENT_WINDOWS_SERVICE_RUN(strType)                                 \
    DWORD errorCode = 0;                                                       \
    SERVICE_TABLE_ENTRY##strType svcTable[] = {                                \
        { const_cast<Char *>(this->name.PeekBuffer()), reinterpret_cast<       \
LPSERVICE_MAIN_FUNCTION##strType>(&WindowsService::serviceMain) },             \
        { NULL, NULL}                                                          \
    };                                                                         \
                                                                               \
    if (WindowsService::instance == NULL) {                                    \
        WindowsService::instance = this;                                       \
    } else {                                                                   \
        throw IllegalStateException("There is already a Windows Service"       \
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
            throw SystemException(errorCode, __FILE__, __LINE__);              \
        }                                                                      \
    }                                                                           


#define IMPLEMENT_WINDOWS_SERVICE_HANDLER_EX                                   \
    ASSERT(context != NULL);                                                   \
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
            } catch (SystemException e) {                                      \
                retval = e.GetErrorCode();                                     \
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
            } catch (SystemException e) {                                      \
                retval = e.GetErrorCode();                                     \
            }                                                                  \
            break;                                                             \
                                                                               \
        case SERVICE_CONTROL_SHUTDOWN:                                         \
            try {                                                              \
                svc->setStatus(SERVICE_STOP_PENDING);                          \
                if ((retval = svc->OnShutdown()) == NO_ERROR) {                \
                    svc->setStatus(SERVICE_STOPPED);                           \
                }                                                              \
            } catch (SystemException e) {                                      \
                retval = e.GetErrorCode();                                     \
            }                                                                  \
            break;                                                             \
                                                                               \
        case SERVICE_CONTROL_STOP:                                             \
            try {                                                              \
                svc->setStatus(SERVICE_STOP_PENDING);                          \
                if ((retval = svc->OnStop()) == NO_ERROR) {                    \
                    svc->setStatus(SERVICE_STOPPED);                           \
                }                                                              \
            } catch (SystemException e) {                                      \
                retval = e.GetErrorCode();                                     \
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
    ASSERT(WindowsService::instance != NULL);                                  \
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
 * vislib::sys::WindowsService<vislib::CharTraitsA>::Install
 */
void vislib::sys::WindowsService<vislib::CharTraitsA>::Install(
        const String& binaryPath, const String& svcName, 
        const String& displayName, const DWORD svcType, const DWORD startType) {
    IMPLEMENT_WINDOWS_SERVICE_INSTALL1(A);
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::SetDescription
 */
bool vislib::sys::WindowsService<vislib::CharTraitsA>::SetDescription(
        const String& svcName, const String& desc) {
    IMPLEMENT_WINDOWS_SERVICE_SETDESCRIPTION(A);
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::~WindowsService
 */
vislib::sys::WindowsService<vislib::CharTraitsA>::~WindowsService(void) {
    if (this->hStatus != NULL) {
        ::CloseServiceHandle(reinterpret_cast<SC_HANDLE>(this->hStatus));
    }
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::Install
 */
void vislib::sys::WindowsService<vislib::CharTraitsA>::Install(
        const DWORD startType) {
    IMPLEMENT_WINDOWS_SERVICE_INSTALL2(A);
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::Run
 */
bool vislib::sys::WindowsService<vislib::CharTraitsA>::Run(void) {
    IMPLEMENT_WINDOWS_SERVICE_RUN(A);
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::OnContinue
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsA>::OnContinue(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::OnDeviceEvent
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsA>::OnDeviceEvent(
        const DWORD eventType, void *eventData) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::OnHardwareProfileChanged
 */
DWORD 
vislib::sys::WindowsService<vislib::CharTraitsA>::OnHardwareProfileChanged(
        const DWORD eventType) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::OnNetBindAdd
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsA>::OnNetBindAdd(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::OnNetBindDisable
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsA>::OnNetBindDisable(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::OnNetBindEnable
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsA>::OnNetBindEnable(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::OnNetBindRemove
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsA>::OnNetBindRemove(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/* 
 * vislib::sys::WindowsService<vislib::CharTraitsA>::OnParamChange
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsA>::OnParamChange(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::OnPause
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsA>::OnPause(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::OnPowerEvent
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsA>::OnPowerEvent(
        const DWORD eventType, void *eventData) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::OnSessionChange
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsA>::OnSessionChange(
        const DWORD eventType, void *eventData) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::OnShutdown
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsA>::OnShutdown(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::OnStop
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsA>::OnStop(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::OnUserControl
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsA>::OnUserControl(
        const DWORD control) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::Uninstall
 */
void vislib::sys::WindowsService<vislib::CharTraitsA>::Uninstall(void) {
    IMPLEMENT_WINDOWS_SERVICE_UNINSTALL(A);
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::handlerEx
 */
DWORD WINAPI vislib::sys::WindowsService<vislib::CharTraitsA>::handlerEx(
        DWORD control, DWORD eventType, void *eventData, void *context) {
    IMPLEMENT_WINDOWS_SERVICE_HANDLER_EX;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::serviceMain
 */
void WINAPI vislib::sys::WindowsService<vislib::CharTraitsA>::serviceMain(
        DWORD argc, Char **argv) {
    IMPLEMENT_WINDOWS_SERVICE_SERVICE_MAIN(A);
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::instance
 */
vislib::sys::WindowsService<vislib::CharTraitsA> 
*vislib::sys::WindowsService<vislib::CharTraitsA>::instance = NULL;


/*
 * vislib::sys::WindowsService<vislib::CharTraitsA>::operator =
 */
vislib::sys::WindowsService<vislib::CharTraitsA>& 
vislib::sys::WindowsService<vislib::CharTraitsA>::operator =(
        const WindowsService& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}

// END OF PARTIAL TEMPLATE SPECIALISATION FOR CharTraitsA
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// BEGIN OF PARTIAL TEMPLATE SPECIALISATION FOR CharTraitsW

/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::Install
 */
void vislib::sys::WindowsService<vislib::CharTraitsW>::Install(
        const String& binaryPath, const String& svcName, 
        const String& displayName, const DWORD svcType, const DWORD startType) {
    IMPLEMENT_WINDOWS_SERVICE_INSTALL1(W);
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::SetDescription
 */
bool vislib::sys::WindowsService<vislib::CharTraitsW>::SetDescription(
        const String& svcName, const String& desc) {
    IMPLEMENT_WINDOWS_SERVICE_SETDESCRIPTION(W);
}



/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::~WindowsService
 */
vislib::sys::WindowsService<vislib::CharTraitsW>::~WindowsService(void) {
    if (this->hStatus != NULL) {
        ::CloseServiceHandle(reinterpret_cast<SC_HANDLE>(this->hStatus));
    }
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::Install
 */
void vislib::sys::WindowsService<vislib::CharTraitsW>::Install(
        const DWORD startType) {
    IMPLEMENT_WINDOWS_SERVICE_INSTALL2(W);
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::Run
 */
bool vislib::sys::WindowsService<vislib::CharTraitsW>::Run(void) {
    IMPLEMENT_WINDOWS_SERVICE_RUN(W);
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::OnContinue
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsW>::OnContinue(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::OnDeviceEvent
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsW>::OnDeviceEvent(
        const DWORD eventType, void *eventData) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::OnHardwareProfileChanged
 */
DWORD 
vislib::sys::WindowsService<vislib::CharTraitsW>::OnHardwareProfileChanged(
        const DWORD eventType) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::OnNetBindAdd
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsW>::OnNetBindAdd(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::OnNetBindDisable
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsW>::OnNetBindDisable(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::OnNetBindEnable
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsW>::OnNetBindEnable(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::OnNetBindRemove
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsW>::OnNetBindRemove(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/* 
 * vislib::sys::WindowsService<vislib::CharTraitsW>::OnParamChange
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsW>::OnParamChange(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::OnPause
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsW>::OnPause(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::OnPowerEvent
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsW>::OnPowerEvent(
        const DWORD eventType, void *eventData) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::OnSessionChange
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsW>::OnSessionChange(
        const DWORD eventType, void *eventData) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::OnShutdown
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsW>::OnShutdown(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::OnStop
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsW>::OnStop(void) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::OnUserControl
 */
DWORD vislib::sys::WindowsService<vislib::CharTraitsW>::OnUserControl(
        const DWORD control) {
    return ERROR_CALL_NOT_IMPLEMENTED;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::Uninstall
 */
void vislib::sys::WindowsService<vislib::CharTraitsW>::Uninstall(void) {
    IMPLEMENT_WINDOWS_SERVICE_UNINSTALL(W);
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::handlerEx
 */
DWORD WINAPI vislib::sys::WindowsService<vislib::CharTraitsW>::handlerEx(
        DWORD control, DWORD eventType, void *eventData, void *context) {
    IMPLEMENT_WINDOWS_SERVICE_HANDLER_EX;
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::serviceMain
 */
void WINAPI vislib::sys::WindowsService<vislib::CharTraitsW>::serviceMain(
        DWORD argc, Char **argv) {
    IMPLEMENT_WINDOWS_SERVICE_SERVICE_MAIN(W);
}


/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::instance
 */
vislib::sys::WindowsService<vislib::CharTraitsW> 
*vislib::sys::WindowsService<vislib::CharTraitsW>::instance = NULL;


/*
 * vislib::sys::WindowsService<vislib::CharTraitsW>::operator =
 */
vislib::sys::WindowsService<vislib::CharTraitsW>& 
vislib::sys::WindowsService<vislib::CharTraitsW>::operator =(
        const WindowsService& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
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
