/*
 * WindowsService.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_WINDOWSSERVICE_H_INCLUDED
#define VISLIB_WINDOWSSERVICE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
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

        /**
         * Set a description for the service with the specified name. The 
         * service must have been installed before.
         *
         * @param svcName Name of the service to set the description for.
         * @param desc    The description text.
         *
         * @return true, if the description was successfully set, false 
         *         otherwise.
         */
        static bool SetDescription(const String& svcName, const String& desc);


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
         * @param startType The startup type of the service.
         *
         * @throws SystemException If the service could not be installed.
         */
        void Install(const DWORD startType = SERVICE_AUTO_START);

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
        virtual DWORD OnRun(const DWORD argc, const Char **argv) = 0;

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
         * @param svcType   The service type.
         * @param controlsAccepted
         */
        inline WindowsService(const String& name, const String& displayName, 
                const DWORD svcType, const DWORD controlsAccepted)
                : displayName(displayName), name(name), hStatus(NULL) {
            ::ZeroMemory(&this->status, sizeof(SERVICE_STATUS));
            this->status.dwServiceType = svcType;
            this->status.dwCurrentState = SERVICE_STOPPED;
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
        static void WINAPI serviceMain(DWORD argc, Char **argv);

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
    
    // END OF PARTIAL TEMPLATE SPECIALISATION FOR CharTraitsA
    ////////////////////////////////////////////////////////////////////////////


    ////////////////////////////////////////////////////////////////////////////
    // BEGIN OF PARTIAL TEMPLATE SPECIALISATION FOR CharTraitsW


    /**
     * Partial template specialisation of WindowsService for wide characters.
     */
    template<> class WindowsService<CharTraitsW> {

    public: 

        /** Characters to use in this class. */
        typedef CharTraitsW::Char Char;

        /** String to use in this class. */
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

        /**
         * Set a description for the service with the specified name. The 
         * service must have been installed before.
         *
         * @param svcName Name of the service to set the description for.
         * @param desc    The description text.
         *
         * @return true, if the description was successfully set, false 
         *         otherwise.
         */
        static bool SetDescription(const String& svcName, const String& desc);


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
         * @param startType The startup type of the service.
         *
         * @throws SystemException If the service could not be installed.
         */
        void Install(const DWORD startType = SERVICE_AUTO_START);

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
         *
         * @return The exit code of the service.
         */
        virtual DWORD OnRun(const DWORD argc, const Char **argv) = 0;

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
         * @param svcType   The service type.
         * @param controlsAccepted
         */
        inline WindowsService(const String& name, const String& displayName, 
                const DWORD svcType, const DWORD controlsAccepted)
                : displayName(displayName), name(name), hStatus(NULL) {
            ::ZeroMemory(&this->status, sizeof(SERVICE_STATUS));
            this->status.dwServiceType = svcType;
            this->status.dwCurrentState = SERVICE_STOPPED;
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
        static void WINAPI serviceMain(DWORD argc, Char **argv);

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

    // END OF PARTIAL TEMPLATE SPECIALISATION FOR CharTraitsW
    ////////////////////////////////////////////////////////////////////////////


    /** Instantiation of WindowsService for ANSI strings. */
    typedef WindowsService<CharTraitsA> WindowsServiceA;

    /** Instantiation of WindowsService for Unicode strings. */
    typedef WindowsService<CharTraitsW> WindowsServiceW;

} /* end namespace sys */
} /* end namespace vislib */


//#include "vislib/WindowsService.inl"

#endif /* _WIN32 */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_WINDOWSSERVICE_H_INCLUDED */

