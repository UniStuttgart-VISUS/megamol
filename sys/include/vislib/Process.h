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
