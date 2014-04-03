/*
 * VolatileIPCStringTable.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_VOLATILEIPCSTRINGTABLE_H_INCLUDED
#define VISLIB_VOLATILEIPCSTRINGTABLE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "the/string.h"
#include "vislib/SmartPtr.h"


namespace vislib {
namespace sys {


    /**
     * This class represents a system-user-wide, volatile string table meant 
     * for configuration, nagotiation and publication purpose for IPC. Do not 
     * use this class to transfer larger data.
     *
     * System-user-wide means that each user logged into the local system has
     * his own string table.
     *
     * This class cannot be instanciated. To create an entry in this string 
     * table an application calls 'Create' with a unique name characterising 
     * the application. If this succeeds, the value of this string table entry 
     * can be changed using the returned 'Entry' object, and can be received 
     * by any application running on the local computer as the same user using 
     * 'GetValue'. 
     *
     * The entry into the string table is only guaranteed to exist as long as 
     * the corresponding 'Entry' object exists. As soon as the 'Entry' object 
     * is deleted, the operating system should remove the entry from the 
     * string table. This is, however, not guaranteed. So your application must
     * be aware that information from this table might be outdated.
     *
     * Remark:
     * The class supports ANSI and unicode strings for names and values. 
     * However since some systems might have problems with unicode it is
     * strongly recommended to only use 7 bit ascii characters.
     */
    class VolatileIPCStringTable {
    private:

        /**
         * private nested class implementing the functionality of 'Entry'
         */
        class EntryImplementation {
        public:

            /**
             * Ctor.
             *
             * @param name The name of the entry
             */
            EntryImplementation(const char* name);

            /**
             * Ctor.
             *
             * @param name The name of the entry
             */
            EntryImplementation(const wchar_t* name);

            /** Dtor. */
            ~EntryImplementation(void);

            /** 
             * Creates the entry.
             *
             * @throw the::already_exists_exception if there already is an entry
             *        in the string tabe with the specified name.
             * @throw the::exception in case of an generic error.
             */
            void Create(void);

            /**
             * Sets the value of the entry.
             *
             * @param value The new value of the entry.
             */
            void SetValue(const char* value);

            /**
             * Sets the value of the entry.
             *
             * @param value The new value of the entry.
             */
            void SetValue(const wchar_t* value);

            /**
             * Answer the name of the entry.
             *
             * @return The name of the entry.
             */
            void GetName(the::astring& outName) const;

            /**
             * Answer the name of the entry.
             *
             * @return The name of the entry.
             */
            void GetName(the::wstring& outName) const;

        private:

#ifdef _WIN32

            /** The name of the entry */
            the::wstring name;

            /** The registry key handle of the entry */
            HKEY key;

#else /* _WIN32 */

            /** The name of the entry */
            the::astring name;

#endif /* _WIN32 */

        };

    public:

        /**
         * Nested class representing a key-value pair entry into the volatile, 
         * system-wide string table.
         */
        class Entry {
        public:
            friend class VolatileIPCStringTable;

            /**
             * Copy ctor.
             *
             * @param rhs The right hand side operand
             */
            Entry(const Entry& rhs);

            /** Dtor */
            ~Entry();

            /**
             * Sets the value of the entry.
             *
             * @param value The new value of the entry.
             */
            void SetValue(const char *value);

            /**
             * Sets the value of the entry.
             *
             * @param value The new value of the entry.
             */
            inline void SetValue(const the::astring& value) {
                this->SetValue(value.c_str());
            }

            /**
             * Sets the value of the entry.
             *
             * @param value The new value of the entry.
             */
            void SetValue(const wchar_t *value);

            /**
             * Sets the value of the entry.
             *
             * @param value The new value of the entry.
             */
            inline void SetValue(const the::wstring& value) {
                this->SetValue(value.c_str());
            }

            /**
             * Answer the name of the entry as ANSI string.
             *
             * @return The name of the entry as ANSI string.
             */
            the::astring NameA() const;

            /**
             * Answer the name of the entry as unicode string.
             *
             * @return The name of the entry as unicode string.
             */
            the::wstring NameW() const;

            /**
             * Assignment operator.
             *
             * @param rhs The right hand side operand.
             *
             * @return 'this'
             */
            Entry& operator=(const Entry& rhs);

            /**
             * Test for equality
             *
             * @param rhs The right hand side operand.
             *
             * @return 'true' if 'this' and 'rhs' represent the same entry,
             *         'false' otherwise.
             */
            bool operator==(const Entry& rhs);

        private:

            /** Ctor */
            Entry(void);

            /** The implementing object */
            vislib::SmartPtr<EntryImplementation> impl;

        };

        /**
         * Answer the value of the entry with the given name. If there is no
         * entry with this name, an empty string is returned. These Nams are
         * not case sensitive.
         *
         * @param name The name of the entry to be returned.
         *
         * @return The value of the entry with the given name.
         */
        static the::astring GetValue(const char *name);

        /**
         * Answer the value of the entry with the given name. If there is no
         * entry with this name, an empty string is returned. These Nams are
         * not case sensitive.
         *
         * @param name The name of the entry to be returned.
         *
         * @return The value of the entry with the given name.
         */
        static inline the::astring GetValue(const the::astring& name) {
            return VolatileIPCStringTable::GetValue(name.c_str());
        }

        /**
         * Answer the value of the entry with the given name. If there is no
         * entry with this name, an empty string is returned. These Nams are
         * not case sensitive.
         *
         * @param name The name of the entry to be returned.
         *
         * @return The value of the entry with the given name.
         */
        static the::wstring GetValue(const wchar_t *name);

        /**
         * Answer the value of the entry with the given name. If there is no
         * entry with this name, an empty string is returned. These Nams are
         * not case sensitive.
         *
         * @param name The name of the entry to be returned.
         *
         * @return The value of the entry with the given name.
         */
        static the::wstring GetValue(const the::wstring& name) {
            return VolatileIPCStringTable::GetValue(name.c_str());
        }

        /**
         * Creates a new entry into the string table under the given name and
         * initialised with the given value. The entry will remain valid as
         * long as the returned 'Entry' object and all other 'Entry' objects
         * created using the copy ctor or assigned from the returned object 
         * exist.
         *
         * @param name The name of the entry to be created. These Nams are not
         *             case sensitive. Must not be an empty string or NULL.
         * @param value The initial value for the entry. Can be NULL.
         *
         * @return An 'Entry' object representing the newly created entry.
         *
         * @throw the::already_exists_exception if there already is an entry
         *        in the string tabe with the specified name.
         * @throw the::exception in case of an generic error.
         */
        static Entry Create(const char *name, const char *value = NULL);

        /**
         * Creates a new entry into the string table under the given name and
         * initialised with the given value. The entry will remain valid as
         * long as the returned 'Entry' object and all other 'Entry' objects
         * created using the copy ctor or assigned from the returned object 
         * exist.
         *
         * @param name The name of the entry to be created. These Nams are not
         *             case sensitive. Must not be an empty string or NULL.
         * @param value The initial value for the entry. Can be NULL.
         *
         * @return An 'Entry' object representing the newly created entry.
         *
         * @throw the::already_exists_exception if there already is an entry
         *        in the string tabe with the specified name.
         * @throw the::exception in case of an generic error.
         */
        static inline Entry Create(const the::astring& name, 
                const char *value = NULL) {
            return VolatileIPCStringTable::Create(name.c_str(), value);
        }

        /**
         * Creates a new entry into the string table under the given name and
         * initialised with the given value. The entry will remain valid as
         * long as the returned 'Entry' object and all other 'Entry' objects
         * created using the copy ctor or assigned from the returned object 
         * exist.
         *
         * @param name The name of the entry to be created. These Nams are not
         *             case sensitive. Must not be an empty string or NULL.
         * @param value The initial value for the entry. Can be NULL.
         *
         * @return An 'Entry' object representing the newly created entry.
         *
         * @throw the::already_exists_exception if there already is an entry
         *        in the string tabe with the specified name.
         * @throw the::exception in case of an generic error.
         */
        static inline Entry Create(const char *name, const the::astring& value) {
            return VolatileIPCStringTable::Create(name, value.c_str());
        }

        /**
         * Creates a new entry into the string table under the given name and
         * initialised with the given value. The entry will remain valid as
         * long as the returned 'Entry' object and all other 'Entry' objects
         * created using the copy ctor or assigned from the returned object 
         * exist.
         *
         * @param name The name of the entry to be created. These Nams are not
         *             case sensitive. Must not be an empty string or NULL.
         * @param value The initial value for the entry. Can be NULL.
         *
         * @return An 'Entry' object representing the newly created entry.
         *
         * @throw the::already_exists_exception if there already is an entry
         *        in the string tabe with the specified name.
         * @throw the::exception in case of an generic error.
         */
        static inline Entry Create(const the::astring& name, const the::astring& value) {
            return VolatileIPCStringTable::Create(name.c_str(), 
                value.c_str());
        }

        /**
         * Creates a new entry into the string table under the given name and
         * initialised with the given value. The entry will remain valid as
         * long as the returned 'Entry' object and all other 'Entry' objects
         * created using the copy ctor or assigned from the returned object 
         * exist.
         *
         * @param name The name of the entry to be created. These Nams are not
         *             case sensitive. Must not be an empty string or NULL.
         * @param value The initial value for the entry. Can be NULL.
         *
         * @return An 'Entry' object representing the newly created entry.
         *
         * @throw the::already_exists_exception if there already is an entry
         *        in the string tabe with the specified name.
         * @throw the::exception in case of an generic error.
         */
        static Entry Create(const wchar_t *name, const wchar_t *value = NULL);

        /**
         * Creates a new entry into the string table under the given name and
         * initialised with the given value. The entry will remain valid as
         * long as the returned 'Entry' object and all other 'Entry' objects
         * created using the copy ctor or assigned from the returned object 
         * exist.
         *
         * @param name The name of the entry to be created. These Nams are not
         *             case sensitive. Must not be an empty string or NULL.
         * @param value The initial value for the entry. Can be NULL.
         *
         * @return An 'Entry' object representing the newly created entry.
         *
         * @throw the::already_exists_exception if there already is an entry
         *        in the string tabe with the specified name.
         * @throw the::exception in case of an generic error.
         */
        static inline Entry Create(const the::wstring& name, 
                const wchar_t *value = NULL) {
            return VolatileIPCStringTable::Create(name.c_str(), value);
        }

        /**
         * Creates a new entry into the string table under the given name and
         * initialised with the given value. The entry will remain valid as
         * long as the returned 'Entry' object and all other 'Entry' objects
         * created using the copy ctor or assigned from the returned object 
         * exist.
         *
         * @param name The name of the entry to be created. These Nams are not
         *             case sensitive. Must not be an empty string or NULL.
         * @param value The initial value for the entry. Can be NULL.
         *
         * @return An 'Entry' object representing the newly created entry.
         *
         * @throw the::already_exists_exception if there already is an entry
         *        in the string tabe with the specified name.
         * @throw the::exception in case of an generic error.
         */
        static inline Entry Create(const wchar_t *name, const the::wstring& value) {
            return VolatileIPCStringTable::Create(name, value.c_str());
        }

        /**
         * Creates a new entry into the string table under the given name and
         * initialised with the given value. The entry will remain valid as
         * long as the returned 'Entry' object and all other 'Entry' objects
         * created using the copy ctor or assigned from the returned object 
         * exist.
         *
         * @param name The name of the entry to be created. These Nams are not
         *             case sensitive. Must not be an empty string or NULL.
         * @param value The initial value for the entry. Can be NULL.
         *
         * @return An 'Entry' object representing the newly created entry.
         *
         * @throw the::already_exists_exception if there already is an entry
         *        in the string tabe with the specified name.
         * @throw the::exception in case of an generic error.
         */
        static inline Entry Create(const the::wstring& name, const the::wstring& value) {
            return VolatileIPCStringTable::Create(name.c_str(), 
                value.c_str());
        }

    private:

        /**
         * Forbidden Ctor, to prohibit instantiating of the class.
         */
        VolatileIPCStringTable(void);

    };

} /* end namespace sys */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_VOLATILEIPCSTRINGTABLE_H_INCLUDED */
