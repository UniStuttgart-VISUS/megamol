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


#include "vislib/RawStorage.h"
#include "vislib/memutils.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {


    /**
     * TODO: comment class
     */
    class Process {

    public:

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
            inline operator const void *(void) const {
                return this->IsEmpty() ? NULL : this->data;
            }
#else /* _WIN32 */
            /**
             * Answer the internal data in a form that can be used as
             * environment input for execve.
             *
             * @return The environment data.
             */
            inline operator const char **(void) const {
                return this->IsEmpty() ? NULL 
                    : const_cast<const char **>(this->data);
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

        void Create(const char *command, const char *arguments, 
            const Environment& environment = EMPTY_ENVIRONMENT, 
            const char *workingDirectory = NULL); 

    private:

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        Process(const Process& rhs);

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

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#endif /* VISLIB_PROCESS_H_INCLUDED */

