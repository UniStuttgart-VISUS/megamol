/*
 * Serialiser.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SERIALISER_H_INCLUDED
#define VISLIB_SERIALISER_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/String.h"


namespace vislib {


    /**
     * This is the superclass for all serialisers. It defines the interface for 
     * them. 
     *
     * Serialises may be stateful or stateless and use the variable names or 
     * ignore them.
     */
    class Serialiser {

    public:

        /** Ctor. */
        Serialiser(void);

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
            const wchar_t *name = NULL) = 0;

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
            const wchar_t *name = NULL) = 0;

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
            const wchar_t *name = NULL) = 0;

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
            const char *name = NULL);

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
            const wchar_t *name = NULL);

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
            const wchar_t *name = NULL) = 0;

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
            const wchar_t *name = NULL) = 0;

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
            const wchar_t *name = NULL) = 0;

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
            const wchar_t *name = NULL) = 0;

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
            const wchar_t *name = NULL) = 0;

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
            const wchar_t *name = NULL) = 0;

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
            const wchar_t *name = NULL) = 0;

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
            const wchar_t *name = NULL) = 0;

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
            const wchar_t *name = NULL) = 0;

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
            const wchar_t *name = NULL) = 0;

        virtual void Serialise(const bool value, 
            const char *name = NULL) = 0;

        virtual void Serialise(const bool value, 
            const wchar_t *name = NULL) = 0;

        virtual void Serialise(const wchar_t value,
            const char *name = NULL) = 0;

        virtual void Serialise(const wchar_t value,
            const wchar_t *name = NULL) = 0;

        virtual void Serialise(const INT8 value,
            const char *name = NULL);

        virtual void Serialise(const INT8 value,
            const wchar_t *name = NULL);

        virtual void Serialise(const UINT8 value,
            const char *name = NULL);

        virtual void Serialise(const UINT8 value,
            const wchar_t *name = NULL);

        virtual void Serialise(const INT16 value,
            const char *name = NULL) = 0;

        virtual void Serialise(const INT16 value,
            const wchar_t *name = NULL) = 0;

        virtual void Serialise(const UINT16 value,
            const char *name = NULL) = 0;

        virtual void Serialise(const UINT16 value,
            const wchar_t *name = NULL) = 0;

        virtual void Serialise(const INT32 value,
            const char *name = NULL) = 0;

        virtual void Serialise(const INT32 value,
            const wchar_t *name = NULL) = 0;

        virtual void Serialise(const UINT32 value,
            const char *name = NULL) = 0;

        virtual void Serialise(const UINT32 value,
            const wchar_t *name = NULL) = 0;

        virtual void Serialise(const INT64 value,
            const char *name = NULL) = 0;

        virtual void Serialise(const INT64 value,
            const wchar_t *name = NULL) = 0;

        virtual void Serialise(const UINT64 value,
            const char *name = NULL) = 0;

        virtual void Serialise(const UINT64 value,
            const wchar_t *name = NULL) = 0;

        virtual void Serialise(const float value,
            const char *name = NULL) = 0;

        virtual void Serialise(const float value,
            const wchar_t *name = NULL) = 0;

        virtual void Serialise(const double value,
            const char *name = NULL) = 0;

        virtual void Serialise(const double value,
            const wchar_t *name = NULL) = 0;

        virtual void Serialise(const StringA& value,
            const char *name = NULL) = 0;

        virtual void Serialise(const StringA& value,
            const wchar_t *name = NULL) = 0;

        virtual void Serialise(const StringW& value,
            const char *name = NULL) = 0;

        virtual void Serialise(const StringW& value,
            const wchar_t *name = NULL) = 0;

    protected:

        Serialiser(const Serialiser& rhs);

        Serialiser& operator =(const Serialiser& rhs);

    };

} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SERIALISER_H_INCLUDED */

