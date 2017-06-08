/*
 * StringConverter.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_STRINGCONVERTER_H_INCLUDED
#define VISLIB_STRINGCONVERTER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/assert.h"
#include "vislib/CharTraits.h"
#include "vislib/IllegalParamException.h"
#include "vislib/memutils.h"
#include "vislib/types.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * String conversion macros for temporary conversion from ANSI to wide 
 * characters and vice versa. The converted string is only valid until the
 * next sequence point. Note, that the conversion is possibly very slow.
 */

// Note: Explicits casts are required for Linux.
#ifndef A2W
#define A2W(str) static_cast<const WCHAR *>(vislib::StringConverter<\
    vislib::CharTraitsA, vislib::CharTraitsW>(str))
#endif /* !A2W */
#ifndef W2A
#define W2A(str) static_cast<const CHAR *>(vislib::StringConverter<\
    vislib::CharTraitsW, vislib::CharTraitsA>(str))
#endif /* !W2A */

#if defined(UNICODE) || defined(_UNICODE)
#ifndef A2T
#define A2T(str) A2W(str)
#endif /* !A2T */
#ifndef T2A
#define T2A(str) W2A(str)
#endif /* !T2A */
#ifndef W2T
#define W2T(str) (str)
#endif /* !W2T */
#ifndef T2W
#define T2W(str) (str)
#endif /* !T2W */
#else /* defined(UNICODE) || defined(_UNICODE) */
#ifndef A2T
#define A2T(str) (str)
#endif /* !A2T */
#ifndef T2A
#define T2A(str) (str)
#endif /* !T2A */
#ifndef W2T
#define W2T(str) W2A(str)
#endif /* !W2T */
#ifndef T2W
#define T2W(str) A2W(str)
#endif /* !T2W */
#endif /* defined(UNICODE) || defined(_UNICODE) */


namespace vislib {

    /**
     * Helper class for string conversion macros. The class can only be used for
     * conversions that are "directly consumed", i. e. the pointer returned by
     * the cast operator cannot be stored, as it is only valid as long as the
     * object exists (until the next sequence point is passed). The memory for 
     * the converted string is allocated when the object is constructed and 
     * freed, if it is destroyed. If the string has less than B characters,
     * a static buffer is used for conversion.
     *
     * Note, that the conversion is slow due to dynamic memory allocation and
     * deallocation and string copying.
     *
     * The template parameters are:
     * S: CharTraits for the source character type.
     * T: CharTraits for the target character type.
     * B: The static buffer size.
     *
     * @author Christoph Mueller
     */
    template<class S, class T, INT32 B = 64> class StringConverter {

    public:

        /** Define a local name for the source character type. */
        typedef typename S::Char SrcChar;

        /** Define a local name for the target character type. */
        typedef typename T::Char DstChar;

        /** 
         * Convert 'str'. 
         *
         * @param str The string to be converted. The caller remains owner
         *            of the memory designated by 'str'.
         */
        StringConverter(const SrcChar *str);

        /** Dtor. */
        ~StringConverter(void);

        /**
         * Answer the converted string. This operator is used as a trick
         * in order to allow access to the buffer member without any call
         * to an operator or method. Implicit conversion is used here.
         *
         * @return The converted string. The object remains owner of the
         *         memory designated by the return value.
         */
        inline operator const DstChar *(void) const {
            return this->buffer;
        }

    private:

        /**
         * Forbidden copy ctor.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        inline StringConverter(const StringConverter& rhs) {
            throw UnsupportedOperationException(
                "vislib::StringConverter::StringConverter", __FILE__, 
                __LINE__);
        }

        /**
         * Assignemnt operator.
         *
         * @param rhs The right hand side operand.
         *
         * @throws IllegalParamException If (this != rhs).
         */
        StringConverter& operator =(const StringConverter& rhs);

        /** The buffer for the temporary, converted string. */
        DstChar *buffer;

        /** Static buffer that is used, if string is short enough. */
        DstChar staticBuffer[B];
    };


    /*
     * StringConverter<S, T, B>::StringConverter
     */
    template<class S, class T, INT32 B>
    StringConverter<S, T, B>::StringConverter(const SrcChar *str) 
            : buffer(NULL) {
        typename S::Size bufLen = S::SafeStringLength(str) + 1;
        this->buffer = (bufLen <= B) ? this->staticBuffer : new DstChar[bufLen];
        S::Convert(this->buffer, bufLen, str);
    }


    /*
     * StringConverter<S, T, B>::~StringConverter
     */
    template<class S, class T, INT32 B>
    StringConverter<S, T, B>::~StringConverter(void) {
        if (this->buffer != this->staticBuffer) {
            ARY_SAFE_DELETE(this->buffer);
        }
    }


    /*
     * StringConverter<S, T, B>::operator =
     */
    template<class S, class T, INT32 B>
    StringConverter<S, T, B>& StringConverter<S, T, B>::operator =(
            const StringConverter& rhs) {
        if (this != &rhs) {
            throw IllegalParamException("vislib::StringConverT::operator =",
                __FILE__, __LINE__);
        }
        return *this;
    }
}

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_STRINGCONVERTER_H_INCLUDED */
