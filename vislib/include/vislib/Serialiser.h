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


#include "vislib/String.h"


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
        static const UINT32 SERIALISER_SUPPORTS_NAMES;

        /**
         * Flag for signaling that the serialiser requires the use of names. 
         * This implies that it also supports names.
         */
        static const UINT32 SERIALISER_REQUIRES_NAMES;

        /**
         * Flag for signaling that the serialiser requires the serialisation 
         * and deserialisation calls to be made in the same order.
         */
        static const UINT32 SERIALISER_REQUIRES_ORDER;

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
         * Deserialise the INT8 element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(INT8& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the INT8 element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(INT8& outValue, 
            const wchar_t *name) = 0;

        /**
         * Deserialise the UINT8 element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(UINT8& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the UINT8 element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(UINT8& outValue, 
            const wchar_t *name) = 0;

        /**
         * Deserialise the INT16 element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(INT16& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the INT16 element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(INT16& outValue, 
            const wchar_t *name) = 0;

        /**
         * Deserialise the UINT16 element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(UINT16& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the UINT16 element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(UINT16& outValue, 
            const wchar_t *name) = 0;

        /**
         * Deserialise the INT32 element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(INT32& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the INT32 element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(INT32& outValue, 
            const wchar_t *name) = 0;

        /**
         * Deserialise the UINT32 element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(UINT32& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the UINT32 element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(UINT32& outValue, 
            const wchar_t *name) = 0;

        /**
         * Deserialise the INT64 element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(INT64& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the INT64 element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(INT64& outValue, 
            const wchar_t *name) = 0;

        /**
         * Deserialise the UINT64 element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(UINT64& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the UINT64 element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(UINT64& outValue, 
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
         * Deserialise the StringA element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(StringA& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the StringA element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(StringA& outValue, 
            const wchar_t *name) = 0;

        /**
         * Deserialise the StringW element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(StringW& outValue, 
            const char *name = NULL) = 0;

        /**
         * Deserialise the StringW element with name 'name' into 'outValue'.
         *
         * @param outValue Receives the deserialised value.
         * @param name     The name of the value to be deserialises. 
         *                 Implementing classes may choose to ignore this value
         *                 and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Deserialise(StringW& outValue, 
            const wchar_t *name) = 0;

        /**
         * Answer the characteristics of the serialiser.
         *
         * @return The properties bitmask of the serialiser.
         */
        inline UINT32 GetProperties(void) const {
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
         * Serialise the INT8 variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const INT8 value,
            const char *name = NULL) = 0;

        /**
         * Serialise the INT8 variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const INT8 value,
            const wchar_t *name) = 0;

        /**
         * Serialise the UINT8 variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const UINT8 value,
            const char *name = NULL) = 0;

        /**
         * Serialise the UINT8 variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const UINT8 value,
            const wchar_t *name) = 0;

        /**
         * Serialise the INT16 variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const INT16 value,
            const char *name = NULL) = 0;

        /**
         * Serialise the INT16 variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const INT16 value,
            const wchar_t *name) = 0;

        /**
         * Serialise the UINT16 variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const UINT16 value,
            const char *name = NULL) = 0;

        /**
         * Serialise the UINT16 variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const UINT16 value,
            const wchar_t *name) = 0;

        /**
         * Serialise the INT32 variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const INT32 value,
            const char *name = NULL) = 0;

        /**
         * Serialise the INT32 variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const INT32 value,
            const wchar_t *name) = 0;

        /**
         * Serialise the UINT32 variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const UINT32 value,
            const char *name = NULL) = 0;

        /**
         * Serialise the UINT32 variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const UINT32 value,
            const wchar_t *name) = 0;

        /**
         * Serialise the INT64 variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const INT64 value,
            const char *name = NULL) = 0;

        /**
         * Serialise the INT64 variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const INT64 value,
            const wchar_t *name) = 0;

        /**
         * Serialise the UINT64 variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const UINT64 value,
            const char *name = NULL) = 0;

        /**
         * Serialise the UINT64 variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const UINT64 value,
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
         * Serialise the StringA variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const StringA& value,
            const char *name = NULL) = 0;

        /**
         * Serialise the StringA variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const StringA& value,
            const wchar_t *name) = 0;

        /**
         * Serialise the StringW variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const StringW& value,
            const char *name = NULL) = 0;

        /**
         * Serialise the StringW variable using the name 'name'.
         *
         * @param value The value to be serialised.
         * @param name  The name for this value.
         *              Implementing classes may choose to ignore this value
         *              and use the order of calls instead.
         *
         * @throws Exception Implementing classes may throw an exception to 
         *                   indicated failure.
         */
        virtual void Serialise(const StringW& value,
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
        Serialiser(const UINT32 properties);

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
        UINT32 properties;
    };

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SERIALISER_H_INCLUDED */
