/*
 * RegistrySerialiser.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_REGISTRYSERIALISER_H_INCLUDED
#define VISLIB_REGISTRYSERIALISER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifdef _WIN32

#include <windows.h>

#include "vislib/assert.h"
#include "vislib/Serialiser.h"
#include "vislib/String.h"
#include "vislib/types.h"


namespace vislib {
namespace sys {


    /**
     * TODO: comment class
     *
     * TODO: Think about a more general serialisation concept.
     */
    class RegistrySerialiser : public Serialiser {

    public:

        /**
         * TODO: documentation
         */
        RegistrySerialiser(const char *subKey, HKEY hKey = HKEY_CURRENT_USER);

        /**
         * TODO: documentation
         */
        RegistrySerialiser(const wchar_t *subKey, 
            HKEY hKey = HKEY_CURRENT_USER);

        /** Dtor. */
        ~RegistrySerialiser(void);

        virtual void Deserialise(bool& outValue, 
            const char *name = NULL);

        virtual void Deserialise(bool& outValue, 
            const wchar_t *name = NULL);

        virtual void Deserialise(char& outValue, 
            const char *name = NULL);

        virtual void Deserialise(char& outValue, 
            const wchar_t *name = NULL);

        virtual void Deserialise(wchar_t& outValue, 
            const char *name = NULL);

        virtual void Deserialise(wchar_t& outValue, 
            const wchar_t *name = NULL);

        virtual void Deserialise(INT8& outValue, 
            const char *name = NULL);

        virtual void Deserialise(INT8& outValue, 
            const wchar_t *name = NULL);

        virtual void Deserialise(UINT8& outValue, 
            const char *name = NULL);

        virtual void Deserialise(UINT8& outValue, 
            const wchar_t *name = NULL);

        virtual void Deserialise(INT16& outValue, 
            const char *name = NULL);

        virtual void Deserialise(INT16& outValue, 
            const wchar_t *name = NULL);

        virtual void Deserialise(UINT16& outValue, 
            const char *name = NULL);

        virtual void Deserialise(UINT16& outValue, 
            const wchar_t *name = NULL);

        virtual void Deserialise(INT32& outValue, 
            const char *name = NULL);

        virtual void Deserialise(INT32& outValue, 
            const wchar_t *name = NULL);

        virtual void Deserialise(UINT32& outValue, 
            const char *name = NULL);

        virtual void Deserialise(UINT32& outValue, 
            const wchar_t *name = NULL);

        virtual void Deserialise(INT64& outValue, 
            const char *name = NULL);

        virtual void Deserialise(INT64& outValue, 
            const wchar_t *name = NULL);

        virtual void Deserialise(UINT64& outValue, 
            const char *name = NULL);

        virtual void Deserialise(UINT64& outValue, 
            const wchar_t *name = NULL);

        virtual void Deserialise(StringA& outValue, 
            const char *name = NULL);

        virtual void Deserialise(StringA& outValue, 
            const wchar_t *name = NULL);

        virtual void Deserialise(StringW& outValue, 
            const char *name = NULL);

        virtual void Deserialise(StringW& outValue, 
            const wchar_t *name = NULL);

        virtual void Serialise(const bool value, 
            const char *name = NULL);

        virtual void Serialise(const bool value, 
            const wchar_t *name = NULL);

        virtual void Serialise(const char value,
            const char *name = NULL);

        virtual void Serialise(const char value,
            const wchar_t *name = NULL);

        virtual bool Serialise(const wchar_t value,
            const char *name = NULL);

        virtual bool Serialise(const wchar_t value,
            const wchar_t *name = NULL);

        virtual void Serialise(const INT8 value,
            const char *name = NULL);

        virtual void Serialise(const INT8 value,
            const wchar_t *name = NULL);

        virtual void Serialise(const UINT8 value,
            const char *name = NULL);

        virtual void Serialise(const UINT8 value,
            const wchar_t *name = NULL);

        virtual void Serialise(const INT16 value,
            const char *name = NULL);

        virtual void Serialise(const INT16 value,
            const wchar_t *name = NULL);

        virtual void Serialise(const UINT16 value,
            const char *name = NULL);

        virtual void Serialise(const UINT16 value,
            const wchar_t *name = NULL);

        virtual void Serialise(const INT32 value,
            const char *name = NULL);

        virtual void Serialise(const INT32 value,
            const wchar_t *name = NULL);

        virtual void Serialise(const UINT32 value,
            const char *name = NULL);

        virtual void Serialise(const UINT32 value,
            const wchar_t *name = NULL);

        virtual void Serialise(const INT64 value,
            const char *name = NULL);

        virtual void Serialise(const INT64 value,
            const wchar_t *name = NULL);

        virtual void Serialise(const UINT64 value,
            const char *name = NULL);

        virtual void Serialise(const UINT64 value,
            const wchar_t *name = NULL);

        virtual void Serialise(const StringA& value,
            const char *name = NULL);

        virtual void Serialise(const StringA& value,
            const wchar_t *name = NULL);

        virtual void Serialise(const StringW& value,
            const char *name = NULL);

        virtual void Serialise(const StringW& value,
            const wchar_t *name = NULL);

    private:

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        RegistrySerialiser(const RegistrySerialiser& rhs);

        /**
         * Generic deserialisation of integral types that are at most as large
         * as a DWORD. The method just delegates the job to the DWORD 
         * deserialisation. The intent of this template is not to duplicate the
         * implementation of serialisation methods for small types.
         *
         * The template parameters have the following meaning:
         * T - An integral type of the target variable, which must be at most as
         *     large as a DWORD.
         * C - char or wchar_t
         *
         * @param outValue Recevies the deserialised value.
         * @param name     The name of the stored value.
         *
         * @throws
         */
        template<class T, class C> 
        inline void deserialiseAsDword(T& outValue, const C *name) {
            ASSERT(sizeof(T) <= sizeof(DWORD));
            UINT32 value;
            this->Deserialise(value, name);
            outValue = static_cast<T>(value);
        }

        /**
         * Generic serialisation of integral types that are at most as large
         * as a DWORD. The method just delegates the job to the DWORD 
         * serialisation. The intent of this template is not to duplicate the
         * implementation of serialisation methods for small types.
         *
         * The template parameters have the following meaning:
         * T - An integral type of the target variable, which must be at most as
         *     large as a DWORD.
         * C - char or wchar_t
         *
         * @param value The value to save.
         * @param name  The name of the stored value.
         *
         * @throws
         */
        template<class T, class C>
        inline void serialiseAsDword(const T& value, const C *name) {
            ASSERT(sizeof(T) <= sizeof(DWORD));
            UINT32 v = static_cast<UINT32>(value);
            this->Serialise(v, name);
        }

        /**
         * Forbidden assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throw IllegalParamException If (this != &rhs).
         */
        RegistrySerialiser& operator =(const RegistrySerialiser& rhs);

        /** Handle of the base key that is parent of the serialised elements. */
        HKEY hBaseKey;
    };

} /* end namespace sys */
} /* end namespace vislib */

#endif /* _WIN32 */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_REGISTRYSERIALISER_H_INCLUDED */
