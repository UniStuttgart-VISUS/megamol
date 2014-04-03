/*
 * Serialiser.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SERIALISER_H_INCLUDED
#define VISLIB_SERIALISER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "the/string.h"


namespace vislib {


    /**
     * This is the superclass for all serialisers. It defines the interface for 
     * them.
     *
     * The serialiser must implement the serialisation for built-in types and
     * some commonly used VISlib types like string. Classes that implement the
     * Serialisable interface can use these methods to serialise themselves into
     * a given serialiser.
     *
     * Serialises may be stateful or stateless and use the variable names or 
     * ignore them. E.g. an implementing class could serialise and deserialise 
     * the values solely depending on the order the bult-in types occur. The 
     * caller is responsible for maintaining the order from serialisation to
     * deserialisation in this case. Implementing classes must document this
     * behaviour. Generally, classes implementing the serialisable interface
     * should assume that the serialiser can be both, stateful and stateless, 
     * and will use the name.
     */
    class Serialiser {

    public:

        /**
         * Flag for signaling that the serialiser supports the use of names for
         * values to be identified.
         */
        static const uint32_t SERIALISER_SUPPORTS_NAMES;

        /**
         * Flag for signaling that the serialiser requires the use of names. 
         * This implies that it also supports names.
         */
        static const uint32_t SERIALISER_REQUIRES_NAMES;

        /**
         * Flag for signaling that the serialiser requires the serialisation 
         * and deserialisation calls to be made in the same order.
         */
        static const uint32_t SERIALISER_REQUIRES_ORDER;

        /** Dtor. */
        virtual ~Serialiser(void);

        /**
         * Deserialise the bool element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(bool& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the bool element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(bool& outValue, 
            const wchar_t *name) = 0;

        /**
         * Deserialise the wchar_t element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(wchar_t& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the wchar_t element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(wchar_t& outValue, 
            const wchar_t *name) = 0;

        /**
         * Deserialise the int8_t element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(int8_t& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the int8_t element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(int8_t& outValue, 
            const wchar_t *name) = 0;

        /**
         * Deserialise the uint8_t element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(uint8_t& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the uint8_t element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(uint8_t& outValue, 
            const wchar_t *name) = 0;

        /**
         * Deserialise the int16_t element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(int16_t& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the int16_t element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(int16_t& outValue, 
            const wchar_t *name) = 0;

        /**
         * Deserialise the uint16_t element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(uint16_t& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the uint16_t element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(uint16_t& outValue, 
            const wchar_t *name) = 0;

        /**
         * Deserialise the int32_t element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(int32_t& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the int32_t element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(int32_t& outValue, 
            const wchar_t *name) = 0;

        /**
         * Deserialise the uint32_t element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(uint32_t& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the uint32_t element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(uint32_t& outValue, 
            const wchar_t *name) = 0;

        /**
         * Deserialise the int64_t element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(int64_t& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the int64_t element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(int64_t& outValue, 
            const wchar_t *name) = 0;

        /**
         * Deserialise the uint64_t element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(uint64_t& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the uint64_t element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(uint64_t& outValue, 
            const wchar_t *name) = 0;

        /**
         * Deserialise the float element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(float& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the float element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(float& outValue, 
            const wchar_t *name) = 0;

        /**
         * Deserialise the double element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(double& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the double element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(double& outValue, 
            const wchar_t *name) = 0;

        /**
         * Deserialise the the::astring element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(the::astring& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the the::astring element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(the::astring& outValue, 
            const wchar_t *name) = 0;

        /**
         * Deserialise the the::wstring element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(the::wstring& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the the::wstring element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(the::wstring& outValue, 
            const wchar_t *name) = 0;

        /**
         * Answer the characteristics of the serialiser.
         *
         * @return The properties bitmask of the serialiser.
         */
        inline uint32_t GetProperties(void) const {
            return this->properties;
        }

        /**
         * Serialise the bool variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const bool value, 
            const char *name = NULL) = 0;

        /**
         * Serialise the bool variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const bool value, 
            const wchar_t *name) = 0;

        /**
         * Serialise the wchar_t variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const wchar_t value,
            const char *name = NULL) = 0;

        /**
         * Serialise the wchar_t variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const wchar_t value,
            const wchar_t *name) = 0;

        /**
         * Serialise the int8_t variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const int8_t value,
            const char *name = NULL) = 0;

        /**
         * Serialise the int8_t variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const int8_t value,
            const wchar_t *name) = 0;

        /**
         * Serialise the uint8_t variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const uint8_t value,
            const char *name = NULL) = 0;

        /**
         * Serialise the uint8_t variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const uint8_t value,
            const wchar_t *name) = 0;

        /**
         * Serialise the int16_t variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const int16_t value,
            const char *name = NULL) = 0;

        /**
         * Serialise the int16_t variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const int16_t value,
            const wchar_t *name) = 0;

        /**
         * Serialise the uint16_t variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const uint16_t value,
            const char *name = NULL) = 0;

        /**
         * Serialise the uint16_t variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const uint16_t value,
            const wchar_t *name) = 0;

        /**
         * Serialise the int32_t variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const int32_t value,
            const char *name = NULL) = 0;

        /**
         * Serialise the int32_t variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const int32_t value,
            const wchar_t *name) = 0;

        /**
         * Serialise the uint32_t variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const uint32_t value,
            const char *name = NULL) = 0;

        /**
         * Serialise the uint32_t variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const uint32_t value,
            const wchar_t *name) = 0;

        /**
         * Serialise the int64_t variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const int64_t value,
            const char *name = NULL) = 0;

        /**
         * Serialise the int64_t variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const int64_t value,
            const wchar_t *name) = 0;

        /**
         * Serialise the uint64_t variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const uint64_t value,
            const char *name = NULL) = 0;

        /**
         * Serialise the uint64_t variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const uint64_t value,
            const wchar_t *name) = 0;

        /**
         * Serialise the float variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const float value,
            const char *name = NULL) = 0;

        /**
         * Serialise the float variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const float value,
            const wchar_t *name) = 0;

        /**
         * Serialise the double variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const double value,
            const char *name = NULL) = 0;

        /**
         * Serialise the double variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const double value,
            const wchar_t *name) = 0;

        /**
         * Serialise the the::astring variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const the::astring& value,
            const char *name = NULL) = 0;

        /**
         * Serialise the the::astring variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const the::astring& value,
            const wchar_t *name) = 0;

        /**
         * Serialise the the::wstring variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const the::wstring& value,
            const char *name = NULL) = 0;

        /**
         * Serialise the the::wstring variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const the::wstring& value,
            const wchar_t *name) = 0;

        /**
         * TODO: Document
         */
        template<class T> inline Serialiser& operator<<(const T& d) {
            this->Serialise(d);
            return *this;
        }

        /**
         * TODO: Document
         */
        template<class T> inline Serialiser& operator>>(T& d) {
            this->Deserialise(d);
            return *this;
        }

    protected:

        /** 
         * Ctor.
         *
         * @param properties The properties of the serialiser.
         */
        Serialiser(const uint32_t properties);

        /**
         * Copy ctor.
         *
         * @param rhs The object to be cloned.
         */
        Serialiser(const Serialiser& rhs);

        /**
         * Assignment operator.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         */
        Serialiser& operator =(const Serialiser& rhs);

    private:

        /** The behaviour properties of the serialiser. */
        uint32_t properties;
    };

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SERIALISER_H_INCLUDED */
