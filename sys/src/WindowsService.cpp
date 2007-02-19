/*
 * WindowsService.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/WindowsService.h"


/*
 * This file is intentionally empty.
 */


#if defined(TEMPLATE_CROWBAR_ENABLED) && (TEMPLATE_CROWBAR_ENABLED != 0)

namespace vislib {
namespace sys {

    /*
     * vislib::sys::WindowsService<CharTraitsA>::Install
     */
    void WindowsService<CharTraitsA>::Install(const String& binaryPath,
            const String& svcName, const String& displayName, 
            const DWORD svcType, const DWORD startType) {
        IMPLEMENT_WINDOWS_SERVICE_INSTALL1(A);
    }


    /*
     * vislib::sys::WindowsService<CharTraitsA>::~WindowsService
     */
    WindowsService<CharTraitsA>::~WindowsService(void) {
        if (this->hStatus != NULL) {
            ::CloseServiceHandle(reinterpret_cast<SC_HANDLE>(this->hStatus));
        }
    }


    /*
     * vislib::sys::WindowsService<CharTraitsA>::Install
     */
    void WindowsService<CharTraitsA>::Install(const DWORD svcType, 
            const DWORD startType) {
        IMPLEMENT_WINDOWS_SERVICE_INSTALL2(A);
    }


    /*
     * vislib::sys::WindowsService<CharTraitsA>::Run
     */
    bool WindowsService<CharTraitsA>::Run(void) {
        IMPLEMENT_WINDOWS_SERVICE_RUN(A);
    }


    /*
     * vislib::sys::WindowsService<CharTraitsA>::OnContinue
     */
    DWORD WindowsService<CharTraitsA>::OnContinue(void) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsA>::OnDeviceEvent
     */
    DWORD WindowsService<CharTraitsA>::OnDeviceEvent(const DWORD eventType, 
                                                     void *eventData) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }

    
    /*
     * vislib::sys::WindowsService<CharTraitsA>::OnHardwareProfileChanged
     */
    DWORD WindowsService<CharTraitsA>::OnHardwareProfileChanged(
            const DWORD eventType) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsA>::OnNetBindAdd
     */
    DWORD WindowsService<CharTraitsA>::OnNetBindAdd(void) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsA>::OnNetBindDisable
     */
    DWORD WindowsService<CharTraitsA>::OnNetBindDisable(void) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsA>::OnNetBindEnable
     */
    DWORD WindowsService<CharTraitsA>::OnNetBindEnable(void) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsA>::OnNetBindRemove
     */
    DWORD WindowsService<CharTraitsA>::OnNetBindRemove(void) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /* 
     * vislib::sys::WindowsService<CharTraitsA>::OnParamChange
     */
    DWORD WindowsService<CharTraitsA>::OnParamChange(void) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsA>::OnPause
     */
    DWORD WindowsService<CharTraitsA>::OnPause(void) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsA>::OnPowerEvent
     */
    DWORD WindowsService<CharTraitsA>::OnPowerEvent(const DWORD eventType,
                                                    void *eventData) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsA>::OnSessionChange
     */
    DWORD WindowsService<CharTraitsA>::OnSessionChange(const DWORD eventType, 
                                                       void *eventData) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsA>::OnShutdown
     */
    DWORD WindowsService<CharTraitsA>::OnShutdown(void) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsA>::OnStop
     */
    DWORD WindowsService<CharTraitsA>::OnStop(void) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsA>::OnUserControl
     */
    DWORD WindowsService<CharTraitsA>::OnUserControl(const DWORD control) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsA>::Uninstall
     */
    void WindowsService<CharTraitsA>::Uninstall(void) {
        IMPLEMENT_WINDOWS_SERVICE_UNINSTALL(A);
    }


    /*
     * vislib::sys::WindowsService<CharTraitsA>::handlerEx
     */
    DWORD WINAPI WindowsService<CharTraitsA>::handlerEx(DWORD control,
            DWORD eventType, void *eventData, void *context) {
        IMPLEMENT_WINDOWS_SERVICE_HANDLER_EX;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsA>::serviceMain
     */
    void WINAPI WindowsService<CharTraitsA>::serviceMain(DWORD argc, 
            Char *argv) {
        IMPLEMENT_WINDOWS_SERVICE_SERVICE_MAIN(A);
    }


    /*
     * vislib::sys::WindowsService<CharTraitsA>::instance
     */
    WindowsService<CharTraitsA> *WindowsService<CharTraitsA>::instance = NULL;


    /*
     * vislib::sys::WindowsService<CharTraitsA>::operator =
     */
    WindowsService<CharTraitsA>& WindowsService<CharTraitsA>::operator =(
            const WindowsService& rhs) {
        if (this != &rhs) {
            throw IllegalParamException("rhs", __FILE__, __LINE__);
        }

        return *this;
    }

#endif /* defined(TEMPLATE_CROWBAR_ENABLED) && ... */
    
#if defined(TEMPLATE_CROWBAR_ENABLED) && (TEMPLATE_CROWBAR_ENABLED != 0)

    /*
     * vislib::sys::WindowsService<CharTraitsW>::Install
     */
    void WindowsService<CharTraitsW>::Install(const String& binaryPath,
            const String& svcName, const String& displayName, 
            const DWORD svcType, const DWORD startType) {
        IMPLEMENT_WINDOWS_SERVICE_INSTALL1(W);
    }


    /*
     * vislib::sys::WindowsService<CharTraitsW>::~WindowsService
     */
    WindowsService<CharTraitsW>::~WindowsService(void) {
        if (this->hStatus != NULL) {
            ::CloseServiceHandle(reinterpret_cast<SC_HANDLE>(this->hStatus));
        }
    }


    /*
     * vislib::sys::WindowsService<CharTraitsW>::Install
     */
    void WindowsService<CharTraitsW>::Install(const DWORD svcType, 
            const DWORD startType) {
        IMPLEMENT_WINDOWS_SERVICE_INSTALL2(W);
    }


    /*
     * vislib::sys::WindowsService<CharTraitsW>::Run
     */
    bool WindowsService<CharTraitsW>::Run(void) {
        IMPLEMENT_WINDOWS_SERVICE_RUN(W);
    }


    /*
     * vislib::sys::WindowsService<CharTraitsW>::OnContinue
     */
    DWORD WindowsService<CharTraitsW>::OnContinue(void) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsW>::OnDeviceEvent
     */
    DWORD WindowsService<CharTraitsW>::OnDeviceEvent(const DWORD eventType, 
                                                     void *eventData) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }

    
    /*
     * vislib::sys::WindowsService<CharTraitsW>::OnHardwareProfileChanged
     */
    DWORD WindowsService<CharTraitsW>::OnHardwareProfileChanged(
            const DWORD eventType) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsW>::OnNetBindAdd
     */
    DWORD WindowsService<CharTraitsW>::OnNetBindAdd(void) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsW>::OnNetBindDisable
     */
    DWORD WindowsService<CharTraitsW>::OnNetBindDisable(void) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsW>::OnNetBindEnable
     */
    DWORD WindowsService<CharTraitsW>::OnNetBindEnable(void) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsW>::OnNetBindRemove
     */
    DWORD WindowsService<CharTraitsW>::OnNetBindRemove(void) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /* 
     * vislib::sys::WindowsService<CharTraitsW>::OnParamChange
     */
    DWORD WindowsService<CharTraitsW>::OnParamChange(void) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsW>::OnPause
     */
    DWORD WindowsService<CharTraitsW>::OnPause(void) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsW>::OnPowerEvent
     */
    DWORD WindowsService<CharTraitsW>::OnPowerEvent(const DWORD eventType,
                                                    void *eventData) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsW>::OnSessionChange
     */
    DWORD WindowsService<CharTraitsW>::OnSessionChange(const DWORD eventType, 
                                                       void *eventData) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsW>::OnShutdown
     */
    DWORD WindowsService<CharTraitsW>::OnShutdown(void) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsW>::OnStop
     */
    DWORD WindowsService<CharTraitsW>::OnStop(void) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsW>::OnUserControl
     */
    DWORD WindowsService<CharTraitsW>::OnUserControl(const DWORD control) {
        return ERROR_CALL_NOT_IMPLEMENTED;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsW>::Uninstall
     */
    void WindowsService<CharTraitsW>::Uninstall(void) {
        IMPLEMENT_WINDOWS_SERVICE_UNINSTALL(W);
    }


    /*
     * vislib::sys::WindowsService<CharTraitsW>::handlerEx
     */
    DWORD WINAPI WindowsService<CharTraitsW>::handlerEx(DWORD control,
            DWORD eventType, void *eventData, void *context) {
        IMPLEMENT_WINDOWS_SERVICE_HANDLER_EX;
    }


    /*
     * vislib::sys::WindowsService<CharTraitsW>::serviceMain
     */
    void WINAPI WindowsService<CharTraitsW>::serviceMain(DWORD argc, 
            Char *argv) {
        IMPLEMENT_WINDOWS_SERVICE_SERVICE_MAIN(W);
    }


    /*
     * vislib::sys::WindowsService<CharTraitsW>::instance
     */
    WindowsService<CharTraitsW> *WindowsService<CharTraitsW>::instance = NULL;


    /*
     * vislib::sys::WindowsService<CharTraitsW>::operator =
     */
    WindowsService<CharTraitsW>& WindowsService<CharTraitsW>::operator =(
            const WindowsService& rhs) {
        if (this != &rhs) {
            throw IllegalParamException("rhs", __FILE__, __LINE__);
        }

        return *this;
    }

} /* end namespace sys */
} /* end namespace vislib */

#endif /* defined(TEMPLATE_CROWBAR_ENABLED) && ... */