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


#ifdef _WIN32 
#include <windows.h>
#else /* _WIN32 */
#include <unistd.h>
#endif /* _WIN32 */

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
         * This class provides a platform independent way of specifying the 
         * environment for a new process. It accepts an arbitrary number of
         * "variable=value" pairs in its constructor and builds a system 
         * dependent input for the process creation. The class can be casted
         * to the system dependent environment format.
         *
         * The default constructor creates an empty environment which implies
         * that a newly created process inherits the environment of the 
         * calling process.
         */
        class Environment {

        public:

            /** Create an empty environment. */
            Environment(void);
            
            /**
             * Create an environment with user-specified variables. The 
             * variables must be passed as separate strings in the form
             * "name=value". The ellipsis must be terminated with a NULL
             * pointer!
             *
             * @param variable The first variable to set. NOTE THAT THE LAST
             *                 PARAMETER MUST BE A NULL POINTER!
             *
             * @throws std::bad_alloc If the environment block cannot be
             *                        allocated.
             */
            Environment(const char *variable, ...);

            /** Dtor. */
            ~Environment(void);

            /** 
             * Answer whether the environment is empty.
             *
             * @return true, if no variables are set, false otherwise. 
             */
            inline bool IsEmpty(void) const {
                return (this->data == NULL);
            }

#ifdef _WIN32
            /**
             * Answer the internal data which can be used as environment
             * input for Win32 API CreateProcess.
             *
             * @return The environment data.
             */
            inline operator void *(void) const {
                return this->IsEmpty() ? NULL : this->data;
            }
#else /* _WIN32 */
            /**
             * Answer the internal data in a form that can be used as
             * environment input for execve.
             *
             * @return The environment data.
             */
            inline operator char *const *(void) const {
                return this->IsEmpty() ? NULL 
                    : const_cast<char *const *>(this->data);
            }
#endif /* _WIN32 */

        private:

            /**
             * Forbidden copy ctor.
             *
             * @param rhs The object to be cloned.
             *
             * @throws UnsupportedOperationException Unconditionally.
             */
            Environment(const Environment& rhs);

            /**
             * Forbidden assignment.
             *
             * @param rhs The right hand side operand.
             *
             * @return *this
             *
             * @throws IllegalParamException If this != &rhs.
             */
            Environment& operator =(const Environment& rhs);
            
            /** This raw storage block contains the environment data. */
#ifdef _WIN32
            void *data;
#else /* _WIN32 */
            char **data;
#endif /* _WIN32 */
        };

        /** 
         * This constant is an empty environment, which can be used to make a 
         * process inherit the current environment.
         */
        static const Environment EMPTY_ENVIRONMENT;

        /** Ctor. */
        Process(void);

        /** Dtor. */
        ~Process(void);

        inline void Create(const char *command, const char *arguments[] = NULL, 
                const Environment& environment = EMPTY_ENVIRONMENT, 
                const char *currentDirectory = NULL) {
            this->create(command, arguments, NULL, NULL, NULL, environment, 
                currentDirectory);

        }

        inline void Create(const char *command, const char *arguments[],
                const char *user, const char *domain, const char *password,
                const Environment& environment = EMPTY_ENVIRONMENT,
                const char *currentDirectory = NULL) {
            this->create(command, arguments, user, domain, password, 
                environment, currentDirectory);
        }

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
            const Environment& environment, const char *currentDirectory);


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

#endif /* VISLIB_PROCESS_H_INCLUDED */

