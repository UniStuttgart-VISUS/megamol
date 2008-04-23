/*
 * Environment.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ENVIRONMENT_H_INCLUDED
#define VISLIB_ENVIRONMENT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/MultiSz.h"
#include "vislib/String.h"


namespace vislib {
namespace sys {


    /**
     * This class provides access to the environment of the calling process.
     */
    class Environment {

    public:

        /**
         * Objects of this class represent a snapshot of the current 
         * environment.
         *
         * The environment snapshot is stored as wide characters on Windows,
         * because Unicode characters are the native storage format of the
         * OS. On Linux, ANSI characters are used because the Unicode
         * support on Linux is crap.
         */
        class Snapshot {
        
        public:

            /** Create an empty environment snapshot. */
            Snapshot(void);

            /**
             * Create an environment with user-specified variables. The 
             * variables must be passed as separate strings in the form
             * "name=value". The ellipsis must be terminated with a NULL
             * pointer! Wide and ANSI parameters must not be mixed!
             *
             * @param variable The first variable to set. NOTE THAT THE LAST
             *                 PARAMETER MUST BE A NULL POINTER!
             *
             * @throws std::bad_alloc If the environment block could not be
             *                        allocated.
             */
            Snapshot(const char *variable, ...);

            /**
             * Create an environment with user-specified variables. The 
             * variables must be passed as separate strings in the form
             * "name=value". The ellipsis must be terminated with a NULL
             * pointer! Wide and ANSI parameters must not be mixed!
             *
             * @param variable The first variable to set. NOTE THAT THE LAST
             *                 PARAMETER MUST BE A NULL POINTER!
             *
             * @throws std::bad_alloc If the environment block could not be
             *                        allocated.
             */
            Snapshot(const wchar_t *variable, ...);

            /**
             * Clone 'rhs'.
             *
             * @param rhs The object to be cloned.
             *
             * @throws std::bad_alloc If the required memory could not be 
             *                        allocated.
             */
            Snapshot(const Snapshot& rhs);

            /** Dtor. */
            ~Snapshot(void);

            /**
             * Clear all variables in the the environment snapshot.
             */
            void Clear(void);

            /**
             * Answer the number of environment variables set in the snapshot.
             *
             * @return The number of environment variables set.
             */
            inline SIZE_T Count(void) const {
#ifdef _WIN32
                return this->data.Count();
#else /* _WIN32 */
                return Snapshot::count(const_cast<const char **>(this->data));
#endif /* _WIN32 */
            }

            /**
             * Get the 'idx'th variable and its value.
             *
             * @param idx      The index of the variable to retrieve. This must 
             *                 be a value within [0, Count()[.
             * @param outName  Receives the name of the specified variable.
             * @param outValue Receives the value of the specified variable.
             *
             * @throws OutOfRangeException If 'idx' is not valid.
             */
            void GetAt(const SIZE_T idx, StringA& outName, StringA& outValue);

            /**
             * Get the 'idx'th variable and its value.
             *
             * @param idx      The index of the variable to retrieve. This must 
             *                 be a value within [0, Count()[.
             * @param outName  Receives the name of the specified variable.
             * @param outValue Receives the value of the specified variable.
             *
             * @throws OutOfRangeException If 'idx' is not valid.
             */
            void GetAt(const SIZE_T idx, StringW& outName, StringW& outValue);

            /**
             * Answer the variable with the name 'name'. If such a variable does
             * not exist, an empty string is returned.
             *
             * @param name The name of the variable to retrieve. It is safe to 
             *             pass a NULL pointer. The caller remains owner of the
             *             memory designated by the pointer.
             *
             * @return The value of the specified variable.
             */
            StringA GetVariable(const char *name) const;

            /**
             * Answer the variable with the name 'name'. If such a variable does
             * not exist, an empty string is returned.
             *
             * @param name The name of the variable to retrieve.
             *
             * @return The value of the specified variable.
             */
            inline StringA GetVariable(const StringA& name) const {
                return this->GetVariable(name.PeekBuffer());
            }

            /**
             * Answer the variable with the name 'name'. If such a variable does
             * not exist, an empty string is returned.
             *
             * @param name The name of the variable to retrieve. It is safe to 
             *             pass a NULL pointer. The caller remains owner of the
             *             memory designated by the pointer.
             *
             * @return The value of the specified variable.
             */
            StringW GetVariable(const wchar_t *name) const;

            /**
             * Answer the variable with the name 'name'. If such a variable does
             * not exist, an empty string is returned.
             *
             * @param name The name of the variable to retrieve.
             *
             * @return The value of the specified variable.
             */
            inline StringW GetVariable(const StringW& name) const {
                return this->GetVariable(name.PeekBuffer());
            }

            /**
             * Answer whether no variable is set in the snapshot.
             *
             * @return true if no variable is set, false otherwise.
             */
            inline bool IsEmpty(void) const {
#ifdef _WIN32
                return this->data.IsEmpty();
#else /* _WIN32 */
                return (this->data == NULL);
#endif /* _WIN32 */
            }

            /**
             * Answer whether the specified environment variable is set in
             * the snapshot.
             *
             * @param name The name of the variable to test.
             *
             * @return true if the variable is set, false otherwise.
             */
            bool IsSet(const char *name) const;

            /**
             * Answer whether the specified environment variable is set in
             * the snapshot.
             *
             * @param name The name of the variable to test.
             *
             * @return true if the variable is set, false otherwise.
             */
            bool IsSet(const wchar_t *name) const;

            /**
             * Answer whether the specified environment variable is set in
             * the snapshot.
             *
             * @param name The name of the variable to test.
             *
             * @return true if the variable is set, false otherwise.
             */
            inline bool IsSet(const vislib::StringA& name) const {
                return this->IsSet(name.PeekBuffer());
            }

            /**
             * Answer whether the specified environment variable is set in
             * the snapshot.
             *
             * @param name The name of the variable to test.
             *
             * @return true if the variable is set, false otherwise.
             */
            inline bool IsSet(const vislib::StringW& name) const {
                return this->IsSet(name.PeekBuffer());
            }

            /**
             * Assignment.
             *
             * @param rhs The right hand side operand.
             *
             * @return *this.
             *
             * @throws std::bad_alloc If the required memory could not be 
             *                        allocated.
             */
            Snapshot& operator =(const Snapshot& rhs);

            /**
             * Answer a pointer to the naked environment block in the OS's 
             * native format.
             * The native format is as follows: On Windows, it is a 
             * double-zero-terminated, CreateProcess-compatible string as 
             * returned by GetEnvironmentStringsW holding variables in the
             * format "variable=name". On Linux, it is an array of strings
             * as in the '::environ' global variable holding strings in the
             * format "variable=name".
             *
             * @return A pointer to the environment block.
             */
            inline operator const void *(void) const {
#ifdef _WIN32
                return static_cast<const void *>(this->data.PeekBuffer());
#else /* _WIN32 */
                return this->data;
#endif /* _WIN32 */
            }

        private:

#ifndef _WIN32
            /**
             * Count the variables in a Linux environment block. This 
             * environment block is an array of strings as in the 
             * '::environ' global variable holding strings in the format 
             * "variable=name".
             *
             * @param data The environment block to count the variables of.
             *             The caller remains owner of the memory designated 
             *             by this pointer.
             */
            static SIZE_T count(const char **const data);
#endif /* !_WIN32 */

            /**
             * Find the variable with the name 'name' in the environment block
             * 'data', which must be in the OS's native format.
             *
             * The native format is as follows: On Windows, it is a 
             * double-zero-terminated, CreateProcess-compatible string as 
             * returned by GetEnvironmentStringsW holding variables in the
             * format "variable=name". On Linux, it is an array of strings
             * as in the '::environ' global variable holding strings in the
             * format "variable=name".
             *
             * @param name The name of the variable to find. The caller remains
             *             owner of the memory designated by this pointer.
             * @param data The environment block to search for the variable.
             *             The caller remains owner of the memory designated 
             *             by this pointer.
             *
             * @return A pointer to the entry, i. e. to the begin of the 
             *         variable name in the environment block 'data', or NULL if
             *         the variable is not in the environment block.
             */
#ifdef _WIN32
            static const wchar_t *find(const wchar_t *name, 
                const wchar_t *data);
#else /* _WIN32 */
            static const char *find(const char *name, 
                const char **const data);
#endif /* _WIN32 */

#ifndef _WIN32
            /**
             * Assign new snapshot data from a Linux environment block. This 
             * environment block is an array of strings as in the 
             * '::environ' global variable holding strings in the format 
             * "variable=name".
             *
             * It is safe to assign to an already allocated environment block
             * as any existing data will be cleared before copying.
             *
             * @param data The environment block to assign. The caller remains 
             *             owner of the memory designated by this pointer.
             */
            void assign(const char **const data);
#endif /* !_WIN32 */

#ifdef _WIN32
            /** 
             * The environment data in a CreateProcess-compatible format. The
             * snapshot uses wide characters to store the data.
             * We assume a NULL pointer representing an empty environment block.
             */
            MultiSzW data;
#else /* _WIN32 */
            /** 
             * The environment data in a execve-compatible format. 
             * We assume a NULL pointer representing an empty environment block.
             */
            char **data;
#endif /* _WIN32 */

            /* Allow Environment to call the initialisation ctor. */
            friend class Environment;

        }; /* end class Snapshot */

        /**
         * Create a snapshot of the current process environment.
         *
         * @return The environment snapshot.
         *
         * @throws std::bad_alloc In case of low memory
         * @throws SystemException If the current environment could not be 
         *                         retrieved.
         */
        static Snapshot CreateSnapshot(void);

        /**
         * Answer the value of the environment variable with the specified name.
         * Note: If no such variable exists, an exception is thrown, except 
         * 'isLenient' is specified. If 'isLenient' is true, an empty string 
         * will be returned if a variable does not exist. However, in case of
         * another error an exception will be thrown.
         *
         * @param name      The name of the environment variable to retrieve.
         * @param isLenient If set true, no exception will be thrown if the 
         *                  specified variable does not exist.
         *
         * @return The value of the specified environment variable.
         *
         * @throws SystemException If the variable could not be retrieved, 
         *                         e. g. because it is not set.
         */
        static vislib::StringA GetVariable(const char *name, 
            const bool isLenient = false);

        /**
         * Answer the value of the environment variable with the specified name.
         * Note: If no such variable exists, an exception is thrown, except 
         * 'isLenient' is specified. If 'isLenient' is true, an empty string 
         * will be returned if a variable does not exist. However, in case of
         * another error an exception will be thrown.
         *
         * @param name      The name of the environment variable to retrieve.
         * @param isLenient If set true, no exception will be thrown if the 
         *                  specified variable does not exist.
         *
         * @return The value of the specified environment variable.
         *
         * @throws SystemException If the variable could not be retrieved, 
         *                         e. g. because it is not set.
         */
        static vislib::StringW GetVariable(const wchar_t *name, 
            const bool isLenient = false);

        /**
         * Answer the value of the environment variable with the specified name.
         * Note: If no such variable exists, an exception is thrown, except 
         * 'isLenient' is specified. If 'isLenient' is true, an empty string 
         * will be returned if a variable does not exist. However, in case of
         * another error an exception will be thrown.
         *
         * @param name      The name of the environment variable to retrieve.
         * @param isLenient If set true, no exception will be thrown if the 
         *                  specified variable does not exist.
         *
         * @return The value of the specified environment variable.
         *
         * @throws SystemException If the variable could not be retrieved, 
         *                         e. g. because it is not set.
         */
        inline static vislib::StringA GetVariable(const vislib::StringA& name,
                const bool isLenient = false) {
            return Environment::GetVariable(name.PeekBuffer(), isLenient);
        }

        /**
         * Answer the value of the environment variable with the specified name.
         * Note: If no such variable exists, an exception is thrown, except 
         * 'isLenient' is specified. If 'isLenient' is true, an empty string 
         * will be returned if a variable does not exist. However, in case of
         * another error an exception will be thrown.
         *
         * @param name      The name of the environment variable to retrieve.
         * @param isLenient If set true, no exception will be thrown if the 
         *                  specified variable does not exist.
         *
         * @return The value of the specified environment variable.
         *
         * @throws SystemException If the variable could not be retrieved, 
         *                         e. g. because it is not set.
         */
        inline static vislib::StringW GetVariable(const vislib::StringW& name,
                const bool isLenient = false) {
            return Environment::GetVariable(name.PeekBuffer(), isLenient);
        }

        /**
         * Answer whether the specified environment variable is set.
         *
         * @param name The name of the variable to test.
         *
         * @return true if the variable is set, false otherwise.
         *
         * @throws SystemException If an error occurred during the system call.
         */
        static bool IsSet(const char *name);

        /**
         * Answer whether the specified environment variable is set.
         *
         * @param name The name of the variable to test.
         *
         * @return true if the variable is set, false otherwise.
         *
         * @throws SystemException If an error occurred during the system call.
         */
        static bool IsSet(const wchar_t *name);

        /**
         * Answer whether the specified environment variable is set.
         *
         * @param name The name of the variable to test.
         *
         * @return true if the variable is set, false otherwise.
         *
         * @throws SystemException If an error occurred during the system call.
         */
        inline static bool IsSet(const vislib::StringA& name) {
            return Environment::IsSet(name.PeekBuffer());
        }

        /**
         * Answer whether the specified environment variable is set.
         *
         * @param name The name of the variable to test.
         *
         * @return true if the variable is set, false otherwise.
         *
         * @throws SystemException If an error occurred during the system call.
         */
        inline static bool IsSet(const vislib::StringW& name) {
            return Environment::IsSet(name.PeekBuffer());
        }

        /**
         * Set the environment variable 'name' to value 'value'. If the 
         * variable already exists, it is overwritten. If 'value' is a NULL
         * pointer, the variable is deleted from the environment.
         *
         * @param name  The name of the environment variable.
         * @param value The new value of the variable.
         *
         * @throws SystemException If settings the environment variable failed.
         */
        static void SetVariable(const char *name, const char *value);

        /**
         * Set the environment variable 'name' to value 'value'. If the 
         * variable already exists, it is overwritten. If 'value' is a NULL
         * pointer, the variable is deleted from the environment.
         *
         * @param name  The name of the environment variable.
         * @param value The new value of the variable.
         *
         * @throws SystemException If settings the environment variable failed.
         */
        static void SetVariable(const wchar_t *name, const wchar_t *value);

        /**
         * Set the environment variable 'name' to value 'value'. If the 
         * variable already exists, it is overwritten.
         *
         * @param name  The name of the environment variable.
         * @param value The new value of the variable.
         *
         * @throws SystemException If settings the environment variable failed.
         */
        inline static void SetVariable(const vislib::StringA& name, 
                                       const vislib::StringA& value) {
            Environment::SetVariable(name.PeekBuffer(), value.PeekBuffer());
        }

        /**
         * Set the environment variable 'name' to value 'value'. If the 
         * variable already exists, it is overwritten.
         *
         * @param name  The name of the environment variable.
         * @param value The new value of the variable.
         *
         * @throws SystemException If settings the environment variable failed.
         */
        inline static void SetVariable(const vislib::StringW& name, 
                                       const vislib::StringW& value) {
            Environment::SetVariable(name.PeekBuffer(), value.PeekBuffer());
        }

        /** Dtor. */
        ~Environment(void);

    private:

        /** Disallow instances. */
        Environment(void);

    };
    
} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ENVIRONMENT_H_INCLUDED */

