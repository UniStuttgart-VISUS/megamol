/*
 * RegistrySerialiser.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_REGISTRYSERIALISER_H_INCLUDED
#define VISLIB_REGISTRYSERIALISER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifdef _WIN32

#include <windows.h>

#include "the/assert.h"
#include "vislib/RegistryKey.h"
#include "vislib/Serialiser.h"
#include "vislib/Stack.h"
#include "vislib/String.h"
#include "the/types.h"


namespace vislib {
namespace sys {


    /**
     * Serialises the given object into the Windows registry.
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

        /**
         * TODO: documentation
         */
        RegistrySerialiser(const vislib::StringA& subKey, 
            HKEY hKey = HKEY_CURRENT_USER);

        /**
         * TODO: documentation
         */
        RegistrySerialiser(const vislib::StringW& subKey, 
            HKEY hKey = HKEY_CURRENT_USER);

        /**
         * Copy ctor.
         *
         * The key stack state of the serialiser is not cloned. The serialiser 
         * is is the well-defined inital state after construction.
         *
         * @param rhs The object to be cloned.
         *
         * @throw SystemException If the registry handle could not be 
         *                        duplicated.
         */
        RegistrySerialiser(const RegistrySerialiser& rhs);

        /** Dtor. */
        ~RegistrySerialiser(void);

        /**
         * Removes all subkeys of the key that is currently on top of the
         * key stack.
         * 
         * @param includeValues If true, the values of the key currently on
         *                      top of the key stack will also be deleted.
         */
        void ClearKey(const bool includeValues = false);

        virtual void Deserialise(bool& outValue, 
            const char *name = NULL);

        virtual void Deserialise(bool& outValue, 
            const wchar_t *name);

        virtual void Deserialise(wchar_t& outValue, 
            const char *name = NULL);

        virtual void Deserialise(wchar_t& outValue, 
            const wchar_t *name);

        virtual void Deserialise(int8_t& outValue, 
            const char *name = NULL);

        virtual void Deserialise(int8_t& outValue, 
            const wchar_t *name);

        virtual void Deserialise(uint8_t& outValue, 
            const char *name = NULL);

        virtual void Deserialise(uint8_t& outValue, 
            const wchar_t *name);

        virtual void Deserialise(int16_t& outValue, 
            const char *name = NULL);

        virtual void Deserialise(int16_t& outValue, 
            const wchar_t *name);

        virtual void Deserialise(uint16_t& outValue, 
            const char *name = NULL);

        virtual void Deserialise(uint16_t& outValue, 
            const wchar_t *name);

        virtual void Deserialise(int32_t& outValue, 
            const char *name = NULL);

        virtual void Deserialise(int32_t& outValue, 
            const wchar_t *name);

        virtual void Deserialise(uint32_t& outValue, 
            const char *name = NULL);

        virtual void Deserialise(uint32_t& outValue, 
            const wchar_t *name);

        virtual void Deserialise(int64_t& outValue, 
            const char *name = NULL);

        virtual void Deserialise(int64_t& outValue, 
            const wchar_t *name);

        virtual void Deserialise(uint64_t& outValue, 
            const char *name = NULL);

        virtual void Deserialise(uint64_t& outValue, 
            const wchar_t *name);

        virtual void Deserialise(float& outValue, 
            const char *name = NULL);

        virtual void Deserialise(float& outValue, 
            const wchar_t *name);

        virtual void Deserialise(double& outValue, 
            const char *name = NULL);

        virtual void Deserialise(double& outValue, 
            const wchar_t *name);

        virtual void Deserialise(StringA& outValue, 
            const char *name = NULL);

        virtual void Deserialise(StringA& outValue, 
            const wchar_t *name);

        virtual void Deserialise(StringW& outValue, 
            const char *name = NULL);

        virtual void Deserialise(StringW& outValue, 
            const wchar_t *name);

        /**
         * Remove a registry key from the stack and use the next one. It is only
         * possible to pop all user-added keys, i.e. the base key defined in the
         * ctor must remain.
         *
         * @param isSilent If set true, do not throw an exception if the stack
         *                 underflows, i.e. the call would remove the base key
         *                 from the stack.
         */
        void PopKey(const bool isSilent = false);

        /**
         * Add a new registry with the specified name to the stack.
         *
         * @param name The name of the new key.
         */
        void PushKey(const char *name);

        /**
         * Add a new registry with the specified name to the stack.
         *
         * @param name The name of the new key.
         */
        void PushKey(const wchar_t *name);

        /**
         * Add a new registry with the specified name to the stack.
         *
         * @param name The name of the new key.
         */
        inline void PushKey(const StringA& name) {
            this->PushKey(name.PeekBuffer());
        }

        /**
         * Add a new registry with the specified name to the stack.
         *
         * @param name The name of the new key.
         */
        inline void PushKey(const StringW& name) {
            this->PushKey(name.PeekBuffer());
        }

        virtual void Serialise(const bool value, 
            const char *name = NULL);

        virtual void Serialise(const bool value, 
            const wchar_t *name);

        virtual void Serialise(const wchar_t value,
            const char *name = NULL);

        virtual void Serialise(const wchar_t value,
            const wchar_t *name);

        virtual void Serialise(const int8_t value,
            const char *name = NULL);

        virtual void Serialise(const int8_t value,
            const wchar_t *name);

        virtual void Serialise(const uint8_t value,
            const char *name = NULL);

        virtual void Serialise(const uint8_t value,
            const wchar_t *name);

        virtual void Serialise(const int16_t value,
            const char *name = NULL);

        virtual void Serialise(const int16_t value,
            const wchar_t *name);

        virtual void Serialise(const uint16_t value,
            const char *name = NULL);

        virtual void Serialise(const uint16_t value,
            const wchar_t *name);

        virtual void Serialise(const int32_t value,
            const char *name = NULL);

        virtual void Serialise(const int32_t value,
            const wchar_t *name);

        virtual void Serialise(const uint32_t value,
            const char *name = NULL);

        virtual void Serialise(const uint32_t value,
            const wchar_t *name);

        virtual void Serialise(const int64_t value,
            const char *name = NULL);

        virtual void Serialise(const int64_t value,
            const wchar_t *name);

        virtual void Serialise(const uint64_t value,
            const char *name = NULL);

        virtual void Serialise(const uint64_t value,
            const wchar_t *name);

        virtual void Serialise(const float value,
            const char *name = NULL);

        virtual void Serialise(const float value,
            const wchar_t *name);

        virtual void Serialise(const double value,
            const char *name = NULL);

        virtual void Serialise(const double value,
            const wchar_t *name);

        virtual void Serialise(const StringA& value,
            const char *name = NULL);

        virtual void Serialise(const StringA& value,
            const wchar_t *name);

        virtual void Serialise(const StringW& value,
            const char *name = NULL);

        virtual void Serialise(const StringW& value,
            const wchar_t *name);

        /**
         * Assignment operator.
         *
         * Assignment does not include the state of the key stack, but only the
         * base key. The serialiser is in the well-defined base state after 
         * assignment.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throw SystemException If the registry handle could not be 
         *                        duplicated.
         */
        RegistrySerialiser& operator =(const RegistrySerialiser& rhs);

    private:

        /**
         * Close the whole stack of registry keys.
         */
        void closeAllRegistry(void);

        /**
         * Generic convenience method for deserialisation of integral types 
         * that are at most as large as a DWORD (uint32_t). The method just 
         * delegates the job to the uint32_t deserialisation. The intent of 
         * this template is not to duplicate the implementation of 
         * serialisation methods for small types. It can, however, not be used
         * for all types. E.g. bool deserialisation produces a compiler warning
         * if done via this method.
         *
         * The template parameters have the following meaning:
         * T: An integral type of the target variable, which must be at most as
         *    large as a DWORD.
         * R: Before casting the returned DWORD to T, the address of the value
         *    is reinterpret_casted to a pointer to R. This intent of this cast
         *    is to perform singed/unsigned or integral/floating point 
         *    conversions.
         * C: char or wchar_t
         *
         * @param outValue Recevies the deserialised value.
         * @param name     The name of the stored value.
         *
         * @throws IllegalParamException If 'name' is NULL or the type of the 
         *                               value parameter does not match the 
         *                               type of the stored element.
         * @throws SystemException If the registry access failed.
         */
        template<class T, class R, class C>
        inline void deserialiseAsDword(T& outValue, const C *name) {
            THE_ASSERT(sizeof(T) <= sizeof(DWORD));
            THE_ASSERT(sizeof(DWORD) == sizeof(uint32_t));
            THE_ASSERT(sizeof(DWORD) == sizeof(R));
            uint32_t value;
            this->Deserialise(value, name);
            outValue = static_cast<T>(*reinterpret_cast<R *>(&value));
        }

        /**
         * Further simplification of deserialiseAsDword for use with signed 
         * integer types of at most sizeof(DWORD) bytes.
         *
         * The template parameters have the following meaning:
         * T: An integral type of the target variable, which must be at most as
         *    large as a DWORD.
         * C: char or wchar_t
         *
         * @param outValue Recevies the deserialised value.
         * @param name     The name of the stored value.
         *
         * @throws IllegalParamException If 'name' is NULL or the type of the 
         *                               value parameter does not match the 
         *                               type of the stored element.
         * @throws SystemException If the registry access failed.
         */
        template<class T, class C>
        inline void deserialiseSignedAsDword(T& outValue, const C *name) {
            this->deserialiseAsDword<T, int32_t, C>(outValue, name);
        }

        /**
         * Further simplification of deserialiseAsDword for use with unsigned 
         * integer types of at most sizeof(DWORD) bytes.
         *
         * The template parameters have the following meaning:
         * T: An integral type of the target variable, which must be at most as
         *    large as a DWORD.
         * C: char or wchar_t
         *
         * @param outValue Recevies the deserialised value.
         * @param name     The name of the stored value.
         *
         * @throws IllegalParamException If 'name' is NULL or the type of the 
         *                               value parameter does not match the 
         *                               type of the stored element.
         * @throws SystemException If the registry access failed.
         */
        template<class T, class C>
        inline void deserialiseUnsignedAsDword(T& outValue, const C *name) {
            this->deserialiseAsDword<T, uint32_t, C>(outValue, name);
        }

        /**
         * Initialises the class. This should be called from the ctors.
         *
         * @param subKey Name of the subkey to write to.
         * @param hKey   Key to create the subkey at.
         *
         * @throws SystemException In case of an error.
         */
        void initialise(const char *subKey, HKEY hKey);

        /**
         * Initialises the class. This should be called from the ctors.
         *
         * @param subKey Name of the subkey to write to.
         * @param hKey   Key to create the subkey at.
         *
         * @throws SystemException In case of an error.
         */
        void initialise(const wchar_t *subKey, HKEY hKey);

        /**
         * This is a simplified version of serialiseAsDword0 which assumes that
         * no reinterpretation of the value is required. It should only be used
         * for unsigned types.
         *
         * @param value The value to save.
         * @param name  The name of the stored value.
         *
         * @throws SystemException If the registry access failed.
         */
        template<class T, class C>
        inline void serialiseAsDword(const T& value, const C *name) {
            this->serialiseAsDword0<T, T, C>(value, name);
        }

        /**
         * Generic serialisation of integral types that are at most as large
         * as a DWORD (uint32_t). The method just delegates the job to the uint32_t
         * serialisation. The intent of this template is not to duplicate the
         * implementation of serialisation methods for small types.
         *
         * The template parameters have the following meaning:
         * T: An integral type of the target variable, which must be at most as
         *    large as a uint32_t.
         * R: Before casting the value to uint32_t, its address is 
         *    reinterpret_casted to a pointer to R. This allows serialising
         *    signed integral and floating point variables using this method.
         * C: char or wchar_t
         *
         * @param value The value to save.
         * @param name  The name of the stored value.
         *
         * @throws SystemException If the registry access failed.
         */
        template<class T, class R, class C>
        inline void serialiseAsDword0(const T& value, const C *name) {
            THE_ASSERT(sizeof(T) <= sizeof(DWORD));
            THE_ASSERT(sizeof(DWORD) == sizeof(uint32_t));
            THE_ASSERT(sizeof(T) == sizeof(R));
            uint32_t v = static_cast<uint32_t>(*reinterpret_cast<const R *>(
                &value));
            this->Serialise(v, name);
        }

        /**
         * Handle of the base key that is parent of the serialised elements.
         * This is a duplicate of the bottom element in the 'keyStack'. It is
         * required to duplicate a serialiser instance.
         */
        HKEY hBaseKey;

        /** The stack of recursive registry keys opened by the user. */
        vislib::Stack<HKEY> keyStack;
    };

} /* end namespace sys */
} /* end namespace vislib */

#endif /* _WIN32 */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_REGISTRYSERIALISER_H_INCLUDED */
