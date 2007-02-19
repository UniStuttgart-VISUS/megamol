/*
 * WindowsService.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_WINDOWSSERVICE_H_INCLUDED
#define VISLIB_WINDOWSSERVICE_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifdef _WIN32

#include <windows.h>

#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"
#include "vislib/IllegalStateException.h"
#include "vislib/String.h"
#include "vislib/SystemException.h"
#include "vislib/UnsupportedOperationException.h"


#define IMPLEMENT_WINDOWS_SERVICE_INSTALL1(strType)                            \
	SC_HANDLE hSCMgr = NULL;                                                   \
	SC_HANDLE hSvc = NULL;		                                               \
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


#define IMPLEMENT_WINDOWS_SERVICE_INSTALL2(strType)                            \
    String##strType binaryPath;                                                \
                                                                               \
    if (::GetModuleFileName##strType(NULL, binaryPath.AllocateBuffer(MAX_PATH),\
            MAX_PATH)) {                                                       \
        WindowsService::Install(binaryPath, this->name,                        \
            this->displayName, svcType, startType);                            \
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
        ::ZeroMemory(&status, sizeof(SERVICE_STATUS));                         \
        status.dwServiceType = SERVICE_WIN32;                                  \
        status.dwCurrentState = SERVICE_RUNNING;                               \
        ::SetServiceStatus(hStatus, &status);                                  \
                                                                               \
        WindowsService::instance->OnRun(argc, argv);                           \
                                                                               \
        status.dwCurrentState = SERVICE_STOPPED;                               \
        ::SetServiceStatus(hStatus, &status);                                  \
    }                                                                          \


namespace vislib {
namespace sys {

    /**
     * The WindowsService class serves as superclass for objects that implement
     * a Windows service. 
     *
     * This template is a really big crowbar dissecting the ANSI and Unicode 
     * Windows API functions.
     */
    template<class T> class WindowsService {
    };

    ////////////////////////////////////////////////////////////////////////////
    // BEGIN OF PARTIAL TEMPLATE SPECIALISATION FOR CharTraitsA

    /**
     * Partial template specialisation of WindowsService for ANSI characters.
     */
    template<> class WindowsService<CharTraitsA> {

    public: 

        /** Characters to use in this class. */
        typedef CharTraitsA::Char Char;

        /** String to use in this class. */
        typedef String<CharTraitsA> String;

        /**
         * Install 'binaryPath' as Windows service using 'svcName' as 
         * identifying name.
         *
         * @param binaryPath  Path to the executable to install as service.
         * @param svcName     The service name.
         * @param displayName The display name shown to the user in the Control
         *                    panel.
         * @param svcType     The service type. See Win32 API documentation of
         *                    CreateService for valid values.
         * @param startType   The service start type. See Win32 API 
         *                    documentation of CreateService for valid values.
         *
         * @throws SystemException If the operation fails.
         */
        static void Install(const String& binaryPath, const String& svcName, 
            const String& displayName, 
            const DWORD svcType = SERVICE_WIN32_OWN_PROCESS,
            const DWORD startType = SERVICE_AUTO_START);


        /** Dtor. */
        virtual ~WindowsService(void);

        /**
         * Answer the display name of the service.
         *
         * @return The display name of the service.
         */
        inline const String& GetDisplayName(void) const {
            return this->displayName;
        }

        /**
         * Answer the name of the service.
         *
         * @return The name of the service.
         */
        inline const String& GetName(void) const {
            return this->name;
        }

        /**
         * Install this executable as service on the current machine.
         *
         * @param svcType   The service type.
         * @param startType The startup type of the service.
         *
         * @throws SystemException If the service could not be installed.
         */
        void Install(const DWORD svcType = SERVICE_WIN32_OWN_PROCESS,
            const DWORD startType = SERVICE_AUTO_START);

        /**
         * Start the service main function. If the method succeeds, it does not
         * return. If the method returns, it is running as normal application 
         * and returns false.
         *
         * @return false, if the executable is running as application and not as
         *         service; true, never.
         *
         * @throws SystemException       If the executable is running as 
         *                               service, but the service could not be 
         *                               started.
         * @throws IllegalStateException If another service has already been
         *                               started.
         */
        bool Run(void);

        /**
         * Handle resume of paused service.
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnContinue(void);

        /**
         * Handle device detection etc.
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @param eventType The type of the device event.
         * @param eventData Lparam of WM_DEVICECHANGE.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnDeviceEvent(const DWORD eventType, void *eventData);

        /**
         * Handle hardware profile changes.
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @param eventType The type of profile change.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnHardwareProfileChanged(const DWORD eventType);

        /**
         * Notifies a network service that there is a new component for binding.
         * The service should bind to the new component. 
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnNetBindAdd(void);

        /**
         * Notifies a network service that one of its bindings has been 
         * disabled. The service should reread its binding information and 
         * remove the binding.
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnNetBindDisable(void); 

        /**
         * Notifies a network service that a disabled binding has been enabled.
         * The service should reread its binding information and add the new 
         * binding. 
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnNetBindEnable(void); 

        /**
         * Notifies a network service that a component for binding has been
         * removed. The service should reread its binding information and 
         * unbind from the removed component. 
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnNetBindRemove(void); 

        /**
         * Notifies a service that service-specific startup parameters have
         * changed. The service should reread its startup parameters.
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnParamChange(void);

        /**
         * Notifies a service that it should pause.
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnPause(void);

        /**
         * This method performs the service's task. It is called once the 
         * service has been initialised. If it is called, the service is in 
         * running state.
         *
         * @param argc The number of command line arguments passed to the 
         *             service.
         * @param argv The command line arguments passed to the service.
         */
        virtual void OnRun(const DWORD argc, const Char *argv) = 0;

        /**
         * Notifies a service of system power events.
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @param eventType The type of the device event.
         * @param eventData Lparam of WM_POWERBROADCAST.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnPowerEvent(const DWORD eventType, void *eventData);

        /**
         * Notifies a service of session change events.
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @param eventType The type of the device event.
         * @param eventData A WTSSESSION_NOTIFICATION structure.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnSessionChange(const DWORD eventType, void *eventData);

        /**
         * Notifies a service that the system is shutting down so the service 
         * can perform cleanup tasks. 
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnShutdown(void);
        
        /**
         * Notifies a service that it should stop.
         * If a service accepts this control code, it must stop upon receipt. 
         * After the SCM sends this control code, it will not send other 
         * control codes.
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnStop(void);

        /**
         * Handles user control messages.s.
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @param control The user defined control code.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnUserControl(const DWORD control);

        /**
         * Uninstall the service.
         *
         * @throws SystemException If the operation fails.
         */
        void Uninstall(void);

    protected:

        /**
         * Initialise the service.
         *
         * @param name
         * @param displayName
         * @param controlsAccepted
         */
        inline WindowsService(const String& name, const String& displayName, 
                const DWORD controlsAccepted)
                : displayName(displayName), name(name), hStatus(NULL) {
            this->status.dwControlsAccepted = controlsAccepted;
        }

        /**
         * Inform the service control manager about a state change of the 
         * service.
         *
         * @param currentState The new state of the service.
         *
         * @throws SystemException If the status could not be updated.
         */
        inline void setStatus(const DWORD currentState) {
            this->status.dwCurrentState = SERVICE_STOPPED;
            if (!::SetServiceStatus(this->hStatus, &this->status)) {
                throw SystemException(__FILE__, __LINE__);
            }
        }

    private:

        /**
         * Handles events for our services.
         *
         * @param control   Control code.
         * @param eventType The type of event that occurred. This parameter is 
         *                  used if 'control' is SERVICE_CONTROL_DEVICEEVENT, 
         *                  SERVICE_CONTROL_HARDWAREPROFILECHANGE, 
         *                  SERVICE_CONTROL_POWEREVENT, or 
         *                  SERVICE_CONTROL_SESSIONCHANGE. Otherwise, it is 
         *                  zero.
         * @param eventData Additional device information, if required. The 
         *                  format of this data depends on the value of the 
         *                  'control' and 'eventData' parameters. 
         * @param context   Pointer to the service object receiving the event-
         *
         * @return The return code of the service object's event callback.
         */
        static DWORD WINAPI handlerEx(DWORD control, DWORD eventType, 
            void *eventData, void *context);

        /**
         * The entry point of the service.
         *
         * @param argc The size of the 'argv' array.
         * @param argv The array of command line arguments.
         */
        static void WINAPI serviceMain(DWORD argc, Char *argv);

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        inline WindowsService(const WindowsService& rhs) {
            throw UnsupportedOperationException("WindowsService", __FILE__, 
                __LINE__);
        }

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @throws IllegalParamException If (&rhs != this).
         */
        WindowsService& operator =(const WindowsService& rhs);

        /** Pointer to the service instance that is running. */
        static WindowsService *instance;

        /** The display name of the service. */
        String displayName;

        /** The name of the service. */
        String name;

        /** The status handle to manipulate the service status. */
        SERVICE_STATUS_HANDLE hStatus;

        /** The status of the Windows service. */
        SERVICE_STATUS status;

    };


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
    
    // END OF PARTIAL TEMPLATE SPECIALISATION FOR CharTraitsA
    ////////////////////////////////////////////////////////////////////////////


    ////////////////////////////////////////////////////////////////////////////
    // BEGIN OF PARTIAL TEMPLATE SPECIALISATION FOR CharTraitsW


    /**
     * Partial template specialisation of WindowsService for wide characters.
     */
    template<> class WindowsService<CharTraitsW> {
        typedef CharTraitsW::Char Char;
        typedef String<CharTraitsW> String;
        /**
         * Install 'binaryPath' as Windows service using 'svcName' as 
         * identifying name.
         *
         * @param binaryPath  Path to the executable to install as service.
         * @param svcName     The service name.
         * @param displayName The display name shown to the user in the Control
         *                    panel.
         * @param svcType     The service type. See Win32 API documentation of
         *                    CreateService for valid values.
         * @param startType   The service start type. See Win32 API 
         *                    documentation of CreateService for valid values.
         *
         * @throws SystemException If the operation fails.
         */
        static void Install(const String& binaryPath, const String& svcName, 
            const String& displayName, 
            const DWORD svcType = SERVICE_WIN32_OWN_PROCESS,
            const DWORD startType = SERVICE_AUTO_START);


        /** Dtor. */
        virtual ~WindowsService(void);

        /**
         * Answer the display name of the service.
         *
         * @return The display name of the service.
         */
        inline const String& GetDisplayName(void) const {
            return this->displayName;
        }

        /**
         * Answer the name of the service.
         *
         * @return The name of the service.
         */
        inline const String& GetName(void) const {
            return this->name;
        }

        /**
         * Install this executable as service on the current machine.
         *
         * @param svcType   The service type.
         * @param startType The startup type of the service.
         *
         * @throws SystemException If the service could not be installed.
         */
        void Install(const DWORD svcType = SERVICE_WIN32_OWN_PROCESS,
            const DWORD startType = SERVICE_AUTO_START);

        /**
         * Start the service main function. If the method succeeds, it does not
         * return. If the method returns, it is running as normal application 
         * and returns false.
         *
         * @return false, if the executable is running as application and not as
         *         service; true, never.
         *
         * @throws SystemException       If the executable is running as 
         *                               service, but the service could not be 
         *                               started.
         * @throws IllegalStateException If another service has already been
         *                               started.
         */
        bool Run(void);

        /**
         * Handle resume of paused service.
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnContinue(void);

        /**
         * Handle device detection etc.
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @param eventType The type of the device event.
         * @param eventData Lparam of WM_DEVICECHANGE.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnDeviceEvent(const DWORD eventType, void *eventData);

        /**
         * Handle hardware profile changes.
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @param eventType The type of profile change.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnHardwareProfileChanged(const DWORD eventType);

        /**
         * Notifies a network service that there is a new component for binding.
         * The service should bind to the new component. 
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnNetBindAdd(void);

        /**
         * Notifies a network service that one of its bindings has been 
         * disabled. The service should reread its binding information and 
         * remove the binding.
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnNetBindDisable(void); 

        /**
         * Notifies a network service that a disabled binding has been enabled.
         * The service should reread its binding information and add the new 
         * binding. 
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnNetBindEnable(void); 

        /**
         * Notifies a network service that a component for binding has been
         * removed. The service should reread its binding information and 
         * unbind from the removed component. 
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnNetBindRemove(void); 

        /**
         * Notifies a service that service-specific startup parameters have
         * changed. The service should reread its startup parameters.
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnParamChange(void);

        /**
         * Notifies a service that it should pause.
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnPause(void);

        /**
         * This method performs the service's task. It is called once the 
         * service has been initialised. If it is called, the service is in 
         * running state.
         *
         * @param argc The number of command line arguments passed to the 
         *             service.
         * @param argv The command line arguments passed to the service.
         */
        virtual void OnRun(const DWORD argc, const Char *argv) = 0;

        /**
         * Notifies a service of system power events.
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @param eventType The type of the device event.
         * @param eventData Lparam of WM_POWERBROADCAST.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnPowerEvent(const DWORD eventType, void *eventData);

        /**
         * Notifies a service of session change events.
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @param eventType The type of the device event.
         * @param eventData A WTSSESSION_NOTIFICATION structure.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnSessionChange(const DWORD eventType, void *eventData);

        /**
         * Notifies a service that the system is shutting down so the service 
         * can perform cleanup tasks. 
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnShutdown(void);
        
        /**
         * Notifies a service that it should stop.
         * If a service accepts this control code, it must stop upon receipt. 
         * After the SCM sends this control code, it will not send other 
         * control codes.
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnStop(void);

        /**
         * Handles user control messages.s.
         *
         * The default implementation always returns ERROR_CALL_NOT_IMPLEMENTED.
         *
         * @param control The user defined control code.
         *
         * @return NO_ERROR in case of success, an error code otherwise.
         */
        virtual DWORD OnUserControl(const DWORD control);

        /**
         * Uninstall the service.
         *
         * @throws SystemException If the operation fails.
         */
        void Uninstall(void);

    protected:

        /**
         * Initialise the service.
         *
         * @param name
         * @param displayName
         * @param controlsAccepted
         */
        inline WindowsService(const String& name, const String& displayName, 
                const DWORD controlsAccepted)
                : displayName(displayName), name(name), hStatus(NULL) {
            this->status.dwControlsAccepted = controlsAccepted;
        }

        /**
         * Inform the service control manager about a state change of the 
         * service.
         *
         * @param currentState The new state of the service.
         *
         * @throws SystemException If the status could not be updated.
         */
        inline void setStatus(const DWORD currentState) {
            this->status.dwCurrentState = SERVICE_STOPPED;
            if (!::SetServiceStatus(this->hStatus, &this->status)) {
                throw SystemException(__FILE__, __LINE__);
            }
        }

    private:

        /**
         * Handles events for our services.
         *
         * @param control   Control code.
         * @param eventType The type of event that occurred. This parameter is 
         *                  used if 'control' is SERVICE_CONTROL_DEVICEEVENT, 
         *                  SERVICE_CONTROL_HARDWAREPROFILECHANGE, 
         *                  SERVICE_CONTROL_POWEREVENT, or 
         *                  SERVICE_CONTROL_SESSIONCHANGE. Otherwise, it is 
         *                  zero.
         * @param eventData Additional device information, if required. The 
         *                  format of this data depends on the value of the 
         *                  'control' and 'eventData' parameters. 
         * @param context   Pointer to the service object receiving the event-
         *
         * @return The return code of the service object's event callback.
         */
        static DWORD WINAPI handlerEx(DWORD control, DWORD eventType, 
            void *eventData, void *context);

        /**
         * The entry point of the service.
         *
         * @param argc The size of the 'argv' array.
         * @param argv The array of command line arguments.
         */
        static void WINAPI serviceMain(DWORD argc, Char *argv);

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        inline WindowsService(const WindowsService& rhs) {
            throw UnsupportedOperationException("WindowsService", __FILE__, 
                __LINE__);
        }

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @throws IllegalParamException If (&rhs != this).
         */
        WindowsService& operator =(const WindowsService& rhs);

        /** Pointer to the service instance that is running. */
        static WindowsService *instance;

        /** The display name of the service. */
        String displayName;

        /** The name of the service. */
        String name;

        /** The status handle to manipulate the service status. */
        SERVICE_STATUS_HANDLE hStatus;

        /** The status of the Windows service. */
        SERVICE_STATUS status;

    };


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

    // END OF PARTIAL TEMPLATE SPECIALISATION FOR CharTraitsW
    ////////////////////////////////////////////////////////////////////////////


    /** Instantiation of WindowsService for ANSI strings. */
    typedef WindowsService<CharTraitsA> WindowsServiceA;

    /** Instantiation of WindowsService for Unicode strings. */
    typedef WindowsService<CharTraitsW> WindowsServiceW;

} /* end namespace sys */
} /* end namespace vislib */



#endif /* _WIN32 */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_WINDOWSSERVICE_H_INCLUDED */

