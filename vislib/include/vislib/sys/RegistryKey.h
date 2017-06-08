/*
 * RegistryKey.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_REGISTRYKEY_H_INCLUDED
#define VISLIB_REGISTRYKEY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#ifdef _WIN32

#include <windows.h>
#include "vislib/Array.h"
#include "vislib/MultiSz.h"
#include "vislib/RawStorage.h"
#include "vislib/String.h"
#include "vislib/types.h"

namespace vislib {
namespace sys {


    /**
     * Class managing a windows registry key
     */
    class RegistryKey {
    public:

        /** Possible registry value kinds */
        enum RegValueType {
            REGVAL_BINARY,
            REGVAL_DWORD,
            REGVAL_EXPAND_SZ,
            REGVAL_MULTI_SZ,
            REGVAL_NONE,
            REGVAL_QWORD,
            REGVAL_STRING
        };

        /**
         * Answer the predefined root key for 'HKEY_CLASSES_ROOT' for reading
         *
         * @return The predefined root key for 'HKEY_CLASSES_ROOT'
         */
        static const RegistryKey& HKeyClassesRoot(void);

        /**
         * Answer the predefined root key for 'HKEY_CURRENT_USER' for reading,
         * opened with 'RegOpenCurrentUser'.
         *
         * @return The predefined root key for 'HKEY_CURRENT_USER'
         */
        static const RegistryKey& HKeyCurrentUser(void);

        /**
         * Answer the predefined root key for 'HKEY_LOCAL_MACHINE' for reading
         *
         * @return The predefined root key for 'HKEY_LOCAL_MACHINE'
         */
        static const RegistryKey& HKeyLocalMachine(void);

        /**
         * Answer the predefined root key for 'HKEY_USERS' for reading
         *
         * @return The predefined root key for 'HKEY_USERS'
         */
        static const RegistryKey& HKeyUsers(void);

        /**
         * Default Ctor.
         */
        RegistryKey(void);

        /**
         * Ctor.
         *
         * @param key The win api key handle to use
         * @param duplicate If true, the key handle will be duplicated, if
         *                  false the class will take ownership of 'key' and
         *                  will free it on destruction
         * @param write Flag whether or not the duplicated key should have
         *              write rights.
         */
        RegistryKey(HKEY key, bool duplicate = true, bool write = false);

        /**
         * Copy Ctor.
         *
         * @param src The object to clone from
         * @param sam The new security settings (if zero, will use the
         *            settings from this object).
         */
        RegistryKey(const RegistryKey& src, REGSAM sam = 0);

        /** Dtor. */
        ~RegistryKey(void);

        /** Closes this registry key */
        void Close(void);

        /**
         * Creates a new sub key
         *
         * @param outKey The key object to receive the newly created subkey
         * @param name The name for the new subkey
         * @param sam The new security settings (if zero, will use the
         *            settings from this object).
         *
         * @return The error code
         */
        DWORD CreateSubKey(RegistryKey& outKey,
            const vislib::StringA& name, REGSAM sam = 0);

        /**
         * Creates a new sub key
         *
         * @param outKey The key object to receive the newly created subkey
         * @param name The name for the new subkey
         * @param sam The new security settings (if zero, will use the
         *            settings from this object).
         *
         * @return The error code
         */
        DWORD CreateSubKey(RegistryKey& outKey,
            const vislib::StringW& name, REGSAM sam = 0);

        /**
         * Deletes a sub key
         *
         * @param name The name of the subkey to be deleted
         *
         * @return The error code
         */
        DWORD DeleteSubKey(const vislib::StringA& name);

        /**
         * Deletes a sub key
         *
         * @param name The name of the subkey to be deleted
         *
         * @return The error code
         */
        DWORD DeleteSubKey(const vislib::StringW& name);

        /**
         * Deletes a value
         *
         * @param name The name of the value to be deleted
         *
         * @return The error code
         */
        DWORD DeleteValue(const vislib::StringA& name);

        /**
         * Deletes a value
         *
         * @param name The name of the value to be deleted
         *
         * @return The error code
         */
        DWORD DeleteValue(const vislib::StringW& name);

        /**
         * Gets the names of the subkeys of this key
         *
         * @return The names of the subkeys of this key
         */
        vislib::Array<vislib::StringA> GetSubKeysA(void) const;

        /**
         * Gets the names of the subkeys of this key
         *
         * @return The names of the subkeys of this key
         */
        vislib::Array<vislib::StringW> GetSubKeysW(void) const;

        /**
         * Gets the names of the values of this key. The empty name is omitted.
         *
         * @return The names of the values of this key
         */
        vislib::Array<vislib::StringA> GetValueNamesA(void) const;

        /**
         * Gets the names of the values of this key. The empty name is omitted.
         *
         * @return The names of the values of this key
         */
        vislib::Array<vislib::StringW> GetValueNamesW(void) const;

        /**
         * Gets the value type of a value of this key.
         *
         * @param name The name of the value to receive its type
         *
         * @return The value type of a value of this key.
         */
        RegValueType GetValueType(const vislib::StringA& name) const;

        /**
         * Gets the value type of a value of this key.
         *
         * @param name The name of the value to receive its type
         *
         * @return The value type of a value of this key.
         */
        RegValueType GetValueType(const vislib::StringW& name) const;

        /**
         * Gets a value of type REGVAL_BINARY (or any other type)
         *
         * @param name The name of the value to receive
         * @param outData The RawStorage object to receive the data
         *
         * @return The error code
         */
        DWORD GetValue(const vislib::StringA& name,
            vislib::RawStorage& outData) const;

        /**
         * Gets a value of type REGVAL_BINARY (or any other type)
         *
         * @param name The name of the value to receive
         * @param outData The RawStorage object to receive the data
         *
         * @return The error code
         */
        DWORD GetValue(const vislib::StringW& name,
            vislib::RawStorage& outData) const;

        /**
         * Gets a value of type REGVAL_BINARY (or any other type)
         *
         * @param name The name of the value to receive
         * @param outData The pointer to the memory to recive the data
         * @param dataSize The number of bytes 'outData' points to
         *
         * @return The error code
         */
        DWORD GetValue(const vislib::StringA& name, void* outData,
            SIZE_T dataSize) const;

        /**
         * Gets a value of type REGVAL_BINARY (or any other type)
         *
         * @param name The name of the value to receive
         * @param outData The pointer to the memory to recive the data
         * @param dataSize The number of bytes 'outData' points to
         *
         * @return The error code
         */
        DWORD GetValue(const vislib::StringW& name, void* outData,
            SIZE_T dataSize) const;

        /**
         * Gets a value of type REGVAL_STRING or REGVAL_EXPAND_SZ
         *
         * @param name The name of the value to receive
         * @param outStr The object to receive the data
         *
         * @return The error code
         */
        DWORD GetValue(const vislib::StringA& name,
            vislib::StringA& outStr) const;

        /**
         * Gets a value of type REGVAL_STRING or REGVAL_EXPAND_SZ
         *
         * @param name The name of the value to receive
         * @param outStr The object to receive the data
         *
         * @return The error code
         */
        DWORD GetValue(const vislib::StringW& name,
            vislib::StringW& outStr) const;

        /**
         * Gets a value of type REGVAL_STRING or REGVAL_EXPAND_SZ
         *
         * @param name The name of the value to receive
         * @param outStr The object to receive the data
         *
         * @return The error code
         */
        DWORD GetValue(const vislib::StringA& name,
            vislib::StringW& outStr) const;

        /**
         * Gets a value of type REGVAL_STRING or REGVAL_EXPAND_SZ
         *
         * @param name The name of the value to receive
         * @param outStr The object to receive the data
         *
         * @return The error code
         */
        DWORD GetValue(const vislib::StringW& name,
            vislib::StringA& outStr) const;

        /**
         * Gets a value of type REGVAL_STRING, REGVAL_EXPAND_SZ, or
         * REGVAL_MULTI_SZ
         *
         * @param name The name of the value to receive
         * @param outStrs The object to receive the data
         *
         * @return The error code
         */
        DWORD GetValue(const vislib::StringA& name,
            vislib::MultiSzA& outStrs) const;

        /**
         * Gets a value of type REGVAL_STRING, REGVAL_EXPAND_SZ, or
         * REGVAL_MULTI_SZ
         *
         * @param name The name of the value to receive
         * @param outStrs The object to receive the data
         *
         * @return The error code
         */
        DWORD GetValue(const vislib::StringW& name,
            vislib::MultiSzW& outStrs) const;

        /**
         * Gets a value of type REGVAL_STRING, REGVAL_EXPAND_SZ, or
         * REGVAL_MULTI_SZ
         *
         * @param name The name of the value to receive
         * @param outStrs The object to receive the data
         *
         * @return The error code
         */
        DWORD GetValue(const vislib::StringA& name,
            vislib::MultiSzW& outStrs) const;

        /**
         * Gets a value of type REGVAL_STRING, REGVAL_EXPAND_SZ, or
         * REGVAL_MULTI_SZ
         *
         * @param name The name of the value to receive
         * @param outStrs The object to receive the data
         *
         * @return The error code
         */
        DWORD GetValue(const vislib::StringW& name,
            vislib::MultiSzA& outStrs) const;

        /**
         * Gets a value of type REGVAL_STRING, REGVAL_EXPAND_SZ, or
         * REGVAL_MULTI_SZ
         *
         * @param name The name of the value to receive
         * @param outStrs The object to receive the data
         *
         * @return The error code
         */
        DWORD GetValue(const vislib::StringA& name,
            vislib::Array<vislib::StringA>& outStrs) const;

        /**
         * Gets a value of type REGVAL_STRING, REGVAL_EXPAND_SZ, or
         * REGVAL_MULTI_SZ
         *
         * @param name The name of the value to receive
         * @param outStrs The object to receive the data
         *
         * @return The error code
         */
        DWORD GetValue(const vislib::StringW& name,
            vislib::Array<vislib::StringW>& outStrs) const;

        /**
         * Gets a value of type REGVAL_STRING, REGVAL_EXPAND_SZ, or
         * REGVAL_MULTI_SZ
         *
         * @param name The name of the value to receive
         * @param outStrs The object to receive the data
         *
         * @return The error code
         */
        DWORD GetValue(const vislib::StringA& name,
            vislib::Array<vislib::StringW>& outStrs) const;

        /**
         * Gets a value of type REGVAL_STRING, REGVAL_EXPAND_SZ, or
         * REGVAL_MULTI_SZ
         *
         * @param name The name of the value to receive
         * @param outStrs The object to receive the data
         *
         * @return The error code
         */
        DWORD GetValue(const vislib::StringW& name,
            vislib::Array<vislib::StringA>& outStrs) const;

        /**
         * Gets a value of type REGVAL_DWORD or REGVAL_QWORD
         *
         * @param name The name of the value to receive
         * @param outVal The object to receive the data
         *
         * @return The error code
         */
        DWORD GetValue(const vislib::StringA& name, UINT32& outVal) const;

        /**
         * Gets a value of type REGVAL_DWORD or REGVAL_QWORD
         *
         * @param name The name of the value to receive
         * @param outVal The object to receive the data
         *
         * @return The error code
         */
        DWORD GetValue(const vislib::StringW& name, UINT32& outVal) const;

        /**
         * Gets a value of type REGVAL_DWORD or REGVAL_QWORD
         *
         * @param name The name of the value to receive
         * @param outVal The object to receive the data
         *
         * @return The error code
         */
        DWORD GetValue(const vislib::StringA& name, UINT64& outVal) const;

        /**
         * Gets a value of type REGVAL_DWORD or REGVAL_QWORD
         *
         * @param name The name of the value to receive
         * @param outVal The object to receive the data
         *
         * @return The error code
         */
        DWORD GetValue(const vislib::StringW& name, UINT64& outVal) const;

        /**
         * Gets the size of the data of a value in bytes
         *
         * @param name The name of the value
         *
         * @return The size of the data in bytes
         */
        SIZE_T GetValueSize(const vislib::StringA& name) const;

        /**
         * Gets the size of the data of a value in bytes
         *
         * @param name The name of the value
         *
         * @return The size of the data in bytes
         */
        SIZE_T GetValueSize(const vislib::StringW& name) const;

        /**
         * Answer whether or not this object represents a valid key handle
         *
         * @return True if this object represents a valid key handle
         */
        inline bool IsValid(void) const {
            return (this->key != INVALID_HANDLE_VALUE);
        }

        /**
         * Opens a subkey of this key
         *
         * @param outKey Object to receive the newly created subkey
         * @param name Name of the subkey to open
         * @param sam The new security settings (if zero, will use the
         *            settings from this object).
         *
         * @return The error code
         */
        DWORD OpenSubKey(RegistryKey& outKey, const vislib::StringA& name,
            REGSAM sam = 0) const;

        /**
         * Opens a subkey of this key
         *
         * @param outKey Object to receive the newly created subkey
         * @param name Name of the subkey to open
         * @param sam The new security settings (if zero, will use the
         *            settings from this object).
         *
         * @return The error code
         */
        DWORD OpenSubKey(RegistryKey& outKey, const vislib::StringW& name,
            REGSAM sam = 0) const;

        /**
         * Reopens an open key into this object (e.g. to get writing rights on
         * a key opened for reading).
         *
         * @param key The key to be reopend
         * @param sam The new security settings
         *
         * @return The error code
         */
        DWORD ReopenKey(const RegistryKey& key, REGSAM sam);

        /**
         * Sets a value of this key of type 'REGVAL_BINARY'
         *
         * @param name The name of the value to be set
         * @param data The data to be set
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringA& name,
            const vislib::RawStorage& data);

        /**
         * Sets a value of this key of type 'REGVAL_BINARY'
         *
         * @param name The name of the value to be set
         * @param data The data to be set
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringW& name,
            const vislib::RawStorage& data);

        /**
         * Sets a value of this key of type 'REGVAL_BINARY'
         *
         * @param name The name of the value to be set
         * @param data The data to be set
         * @param size The size of 'data' in bytes
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringA& name, const void* data,
            SIZE_T size);

        /**
         * Sets a value of this key of type 'REGVAL_BINARY'
         *
         * @param name The name of the value to be set
         * @param data The data to be set
         * @param size The size of 'data' in bytes
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringW& name, const void* data,
            SIZE_T size);

        /**
         * Sets a value of this key of type 'REGVAL_STRING' or
         * 'REGVAL_EXPAND_SZ'
         *
         * @param name The name of the value to be set
         * @param str The data to be set
         * @param expandable If true will use 'REGVAL_EXPAND_SZ', otherwise
         *                   will use 'REGVAL_STRING'
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringA& name,
            const vislib::StringA& str, bool expandable = false);

        /**
         * Sets a value of this key of type 'REGVAL_STRING' or
         * 'REGVAL_EXPAND_SZ'
         *
         * @param name The name of the value to be set
         * @param str The data to be set
         * @param expandable If true will use 'REGVAL_EXPAND_SZ', otherwise
         *                   will use 'REGVAL_STRING'
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringW& name,
            const vislib::StringW& str, bool expandable = false);

        /**
         * Sets a value of this key of type 'REGVAL_STRING' or
         * 'REGVAL_EXPAND_SZ'
         *
         * @param name The name of the value to be set
         * @param str The data to be set
         * @param expandable If true will use 'REGVAL_EXPAND_SZ', otherwise
         *                   will use 'REGVAL_STRING'
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringA& name,
            const vislib::StringW& str, bool expandable = false);

        /**
         * Sets a value of this key of type 'REGVAL_STRING' or
         * 'REGVAL_EXPAND_SZ'
         *
         * @param name The name of the value to be set
         * @param str The data to be set
         * @param expandable If true will use 'REGVAL_EXPAND_SZ', otherwise
         *                   will use 'REGVAL_STRING'
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringW& name,
            const vislib::StringA& str, bool expandable = false);

        /**
         * Sets a value of this key of type 'REGVAL_MULTI_SZ'
         *
         * @param name The name of the value to be set
         * @param strs The data to be set
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringA& name,
            const vislib::MultiSzA& strs);

        /**
         * Sets a value of this key of type 'REGVAL_MULTI_SZ'
         *
         * @param name The name of the value to be set
         * @param strs The data to be set
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringW& name,
            const vislib::MultiSzW& strs);

        /**
         * Sets a value of this key of type 'REGVAL_MULTI_SZ'
         *
         * @param name The name of the value to be set
         * @param strs The data to be set
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringA& name,
            const vislib::MultiSzW& strs);

        /**
         * Sets a value of this key of type 'REGVAL_MULTI_SZ'
         *
         * @param name The name of the value to be set
         * @param strs The data to be set
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringW& name,
            const vislib::MultiSzA& strs);

        /**
         * Sets a value of this key of type 'REGVAL_MULTI_SZ'. Empty strings
         * in the array will be omitted.
         *
         * @param name The name of the value to be set
         * @param strs The data to be set
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringA& name,
            const vislib::Array<vislib::StringA>& strs);

        /**
         * Sets a value of this key of type 'REGVAL_MULTI_SZ'. Empty strings
         * in the array will be omitted.
         *
         * @param name The name of the value to be set
         * @param strs The data to be set
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringW& name,
            const vislib::Array<vislib::StringW>& strs);

        /**
         * Sets a value of this key of type 'REGVAL_MULTI_SZ'. Empty strings
         * in the array will be omitted.
         *
         * @param name The name of the value to be set
         * @param strs The data to be set
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringA& name,
            const vislib::Array<vislib::StringW>& strs);

        /**
         * Sets a value of this key of type 'REGVAL_MULTI_SZ'. Empty strings
         * in the array will be omitted.
         *
         * @param name The name of the value to be set
         * @param strs The data to be set
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringW& name,
            const vislib::Array<vislib::StringA>& strs);

        /**
         * Sets a value of this key of type 'REGVAL_DWORD'
         *
         * @param name The name of the value to be set
         * @param strs The data to be set
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringA& name, INT32 val);

        /**
         * Sets a value of this key of type 'REGVAL_DWORD'
         *
         * @param name The name of the value to be set
         * @param strs The data to be set
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringW& name, INT32 val);

        /**
         * Sets a value of this key of type 'REGVAL_DWORD'
         *
         * @param name The name of the value to be set
         * @param strs The data to be set
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringA& name, UINT32 val);

        /**
         * Sets a value of this key of type 'REGVAL_DWORD'
         *
         * @param name The name of the value to be set
         * @param strs The data to be set
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringW& name, UINT32 val);

        /**
         * Sets a value of this key of type 'REGVAL_QWORD'
         *
         * @param name The name of the value to be set
         * @param strs The data to be set
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringA& name, INT64 val);

        /**
         * Sets a value of this key of type 'REGVAL_QWORD'
         *
         * @param name The name of the value to be set
         * @param strs The data to be set
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringW& name, INT64 val);

        /**
         * Sets a value of this key of type 'REGVAL_QWORD'
         *
         * @param name The name of the value to be set
         * @param strs The data to be set
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringA& name, UINT64 val);

        /**
         * Sets a value of this key of type 'REGVAL_QWORD'
         *
         * @param name The name of the value to be set
         * @param strs The data to be set
         *
         * @return The error code
         */
        DWORD SetValue(const vislib::StringW& name, UINT64 val);

        /**
         * Assignment operator. This will duplicate the key 'rhs' represents
         *
         * @param rhs The right hand side operand.
         *
         * @return A reference to this
         */
        RegistryKey& operator=(const RegistryKey& rhs);

    private:

        /** const for invalid keys */
        static const HKEY INVALID_HKEY;

        /** The win api key handle */
        HKEY key;

        /** The security settings used when creating this key */
        REGSAM sam;

    };

} /* end namespace sys */
} /* end namespace vislib */

#endif /* _WIN32 */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_REGISTRYKEY_H_INCLUDED */

