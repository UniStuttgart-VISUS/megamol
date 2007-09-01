/*
 * Process.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_PROCESS_H_INCLUDED
#define VISLIB_PROCESS_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#ifdef _WIN32 
#include <windows.h>
#else /* _WIN32 */
#include <unistd.h>
#endif /* _WIN32 */

#include "vislib/Environment.h"
#include "vislib/memutils.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {


    /**
     * TODO: comment class
     */
    class Process {

    public:

        /** System-dependent process ID type. */
#ifdef _WIN32
        typedef DWORD PID;
#else /* _WIN32 */
        typedef pid_t PID;
#endif /* _WIN32 */

        /**
         * Answer the ID of the calling process.
         *
         * @eturn The ID of the calling process.
         */
        static inline DWORD CurrentID(void) {
#ifdef _WIN32
            return ::GetCurrentProcessId();
#else /* _WIN32 */
            return ::getpid();
#endif /* _WIN32 */
        }

        /**
         * Check whether a process with the specified ID exists.
         *
         * @param processID The ID of the process to check.
         *
         * @return true if a process with the specified ID exists, false 
         *         otherwise.
         *
         * @throws SystemInformation If the requested information could not be
         *                           retrieved.
         */
        static bool Exists(const PID processID);

        /**
         * Exit the calling process.
         *
         * @param exitCode The exit code to return.
         */
        static void Exit(const DWORD exitCode);

        /** 
         * This constant is an empty environment, which can be used to make a 
         * process inherit the current environment.
         */
        static const Environment::Snapshot EMPTY_ENVIRONMENT;

        /** Ctor. */
        Process(void);

        /** Dtor. */
        ~Process(void);

        /**
         * Create a new process.
         *
         * @param command          The command to be executed.
         * @param arguments        The command line arguments. This array must 
         *                         be terminated with a NULL pointer as guard.
         *                         The name of the executable should not be
         *                         included as first element, it will be added
         *                         by the method.
         * @param environment      The environment of the new process. If 
         *                         EMPTY_ENVIRONMENT is specified, the new 
         *                         process will inherit the environment of the
         *                         calling process.
         * @param currentDirectory The working directory of the new process. If
         *                         NULL is specified, the new process will 
         *                         inherit the working directory of the calling
         *                         process.
         *
         * @throws IllegalStateException If another process has already been
         *                               created using this object.
         * @throws SystemException If the creation of the process failed.
         */
        inline void Create(const char *command, const char *arguments[] = NULL, 
                const Environment::Snapshot& environment = EMPTY_ENVIRONMENT, 
                const char *currentDirectory = NULL) {
            this->create(command, arguments, NULL, NULL, NULL, environment, 
                currentDirectory);

        }

        //inline void Create(const char *command, const char *arguments[],
        //        const char *user, const char *domain, const char *password,
        //        const Environment::Snapshot& environment = EMPTY_ENVIRONMENT,
        //        const char *currentDirectory = NULL) {
        //    this->create(command, arguments, user, domain, password, 
        //        environment, currentDirectory);
        //}

        /**
         * Answer the ID of this process.
         *
         * @return The ID of the process.
         *
         * @throws SystemException If no process has been created or the process
         *                         already terminated.
         */
        PID GetID(void) const;

        /**
         * Forcefully terminate a process previousely started using one of the
         * Create methods. 
         *
         * It is not safe to call this method if no process has been created.
         * The method will also fail if the process already exited.
         *
         * On Windows, 'exitCode' is set as exit code of the terminated process.
         * Note that the state of global data maintained by DLLs may be 
         * compromised by this operation according to MSDN.
         *
         * On Linux, 'exitCode' has no meaning. The process is terminated using
         * SIGKILL. The method might terminate another process than intended if
         * the process already has been terminated and its PID has been reused.
         *
         * @param exitCode The exit code to set for the process.
         *
         * @throws SystemException If the process could not be terminated.
         */
        void Terminate(const DWORD exitCode = 0);

    private:

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        Process(const Process& rhs);

        // TODO: Must find a way to make Linux version fail, if exec in forked
        // process fails.
        void create(const char *command, const char *arguments[],
            const char *user, const char *domain, const char *password,
            const Environment::Snapshot& environment, 
            const char *currentDirectory);

        /**
         * Forbidden assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this
         *
         * @throws IllegalParamException If (this != &rhs).
         */
        Process& operator =(const Process& rhs);

#ifdef _WIN32
        HANDLE hProcess;
#else /* _WIN32 */
        pid_t pid;
#endif /* _WIN32 */

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_PROCESS_H_INCLUDED */
