/*
 * CharTraits.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 - 2006 by Christoph Mueller. All rights reserved.
 */

#ifndef VISLIB_CHARTRAITS_H_INCLUDED
#define VISLIB_CHARTRAITS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <cwctype>

#ifndef _WIN32
#include <langinfo.h>
#include <iconv.h>
#endif /* !_WIN32 */

#include "vislib/assert.h"
#include "vislib/memutils.h"
#include "vislib/types.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/FormatException.h"
#include "vislib/IllegalParamException.h"


namespace vislib {

    /** forward declaration of a friend from another namespace */
    namespace sys {
        class Log;
    }


    /**
     * This is an identifier for the currently supported char traits. It is used
     * to identify the char traits at runtime.
     */
    enum CharType {
        UNDEFINED_CHAR = 0,
        ANSI_CHAR,
        UNICODE_CHAR
    };


    /**
     * This class is the basis class for character trait descriptor classes. The
     * character trait classes are used to instantiate the string template.
     *
     * Usage notes: Creating instances of this class is not allowed. There are
     * static methods that provide the required functionality. These methods are
     * either implemented in this superclass or in the specialisations for wide
     * and ANSI characters. All methods are protected and can only be accessed
     * by friend classes like the StringConverter and the String class itself.
     * Other classes should not be allowed to access these method as it may be
     * unsafe.
     *
     * Rationale: The CharTraitsBase class is used to share implementation 
     * between the partial template specialisations of CharTraits. It is the 
     * same "crowbar pattern" as used for Vectors etc. in vislib::math.
     *
     * @author Christoph Mueller
     */
    template<class T> class CharTraitsBase {

    public:

        /** Define an type-independent name for the char traits. */
        typedef T Char;

        /** Define a string size type. */
        typedef INT32 Size;

        /**
         * Answer the size of a single character in bytes.
         *
         * @return The size of a single character in bytes.
         */
        inline static Size CharSize(void) {
            return sizeof(T);
        }

        /**
         * Answer the string length. 'str' can be a NULL pointer.
         *
         * @param str A zero-terminated string.
         *
         * @return The number of characters 'str' consists of excluding the
         *         terminating zero.
         */
        inline static Size SafeStringLength(const Char *str) {
            return (str != NULL) ? CharTraitsBase::StringLength(str) : 0;
        }

        /**
         * Converts the string str to the returned 64 bit integer value.
         *
         * @param str The input string.
         *
         * @return The parsed integer value.
         *
         * @throw IllegalParamException if str is NULL.
         * @throw FormatException if the string could not be parsed to an
         *        integer value.
         */
        inline static INT64 ParseInt64(const Char *str) {
            INT64 sig = 1;
            if (str == NULL) {
                throw IllegalParamException("str", __FILE__, __LINE__);
            }
            if (*str == static_cast<Char>('-')) {
                sig = -1;
                str++;
            }
            INT64 value = static_cast<INT64>(ParseUInt64(str));
            if (value < 0) {
                throw FormatException("Overflow parsing Int64", 
                    __FILE__, __LINE__);
            }
            return sig;
        }

        /**
         * Converts the string str to the returned 64 bit unsigned integer 
         * value.
         *
         * @param str The input string.
         *
         * @return The parsed integer value.
         *
         * @throw IllegalParamException if str is NULL.
         * @throw FormatException if the string could not be parsed to an
         *        integer value.
         */
        inline static UINT64 ParseUInt64(const Char *str) {
            UINT64 value = 0;
            UINT64 v;
            if (str == NULL) {
                throw IllegalParamException("str", __FILE__, __LINE__);
            }
            while ((*str != 0) && (*str >= static_cast<Char>('0')) 
                    && (*str <= static_cast<Char>('9'))) {
                v = value * 10 + (*str - static_cast<Char>('0'));
                str++;
                if (v < value) {
                    throw FormatException("Overflow parsing Int64", 
                        __FILE__, __LINE__);
                }
                value = v;
            }
            return value;
        }

        /** Empty character sequence constant. */
        static const T EMPTY_STRING[];

    protected:

        /** Forbidden Ctor. */
        inline CharTraitsBase(void) {
            throw UnsupportedOperationException("CharTraitsBase", __FILE__, 
                __LINE__);
        }

        /**
         * Copy the zero-terminated string 'src' to 'dst'.
         *
         * @param dst Receives the copy
         * @param src The string to be copied.
         *
         * @return 'dst'.
         */
        inline static Char *StringCopy(Char *dst, const Char *src) {
            ASSERT(dst != NULL);
            ASSERT(src != NULL);

            Char *tmp = dst;
            while ((*tmp++ = *src++));
            return dst;
        }

#ifdef _WIN32
        /** 
         * Load an ANSI string from the embedded resources.
         *
         * @param hInst  The instance handle to use the string table of.
         * @param id     The resource ID.
         * @param outStr Receives the string.
         * @param cntStr The size of the 'outStr' buffer in characters.
         *
         * @return The number of characters, not including the terminating zero,
         *         in case of success, 0 otherwise.
         */
        inline static INT StringFromResource(HINSTANCE hInst, const UINT id, 
                char *outStr, const INT cntStr) {
            return ::LoadStringA(hInst, id, outStr, cntStr);
        }

        /** 
         * Load an Unicode string from the embedded resources.
         *
         * @param hInst  The instance handle to use the string table of.
         * @param id     The resource ID.
         * @param outStr Receives the string.
         * @param cntStr The size of the 'outStr' buffer in characters.
         *
         * @return The number of characters, not including the terminating zero,
         *         in case of success, 0 otherwise.
         */
        inline static INT StringFromResource(HINSTANCE hInst, const UINT id, 
                wchar_t *outStr, const INT cntStr) {
            return ::LoadStringW(hInst, id, outStr, cntStr);
        }
#endif /* _WIN32 */

        /**
         * Answer the string length. 'str' must not be NULL.
         *
         *
         * @param str A zero-terminated string.
         *
         * @return The number of characters 'str' consists of excluding the
         *         terminating zero.
         */
        inline static Size StringLength(const Char *str) {
            ASSERT(str != NULL);

            const Char *end = str;
            while (*end++ != 0);
            return static_cast<Size>(end - str - 1);
        }

        /**
         * Answer the size of the string including the trailing zero in bytes, 
         * i. e. the size of the memory block that is required to store the
         * string designated by 'str'.
         *
         * @param str A pointer to a string, which can be NULL.
         *
         * @return The size in bytes that is required to store 'str' including
         *         its trailing zero.
         */
        inline static Size StringSize(const Char *str) {
            if (str != NULL) {
                return (StringLength(str) + 1) * CharSize();
            } else {
                return 0;
            }
            
        }

#ifndef _WIN32

        /** The code identifier for normal chars. */
        static const char *ICONV_CODE_CHAR;

        /**The code identifier for wide chars. */
        static const char *ICONV_CODE_WCHAR;

        /** Constant identifying an invalid iconv_t. */
        static const iconv_t INVALID_ICONV_T;
#endif /* !_WIN32 */

    }; /* end class CharTraitsBase */


    /*
     * vislib::CharTraitsBase<T>::EMPTY_STRING
     */
    template<class T> 
    //const typename CharTraitsBase<T>::Char CharTraitsBase<T>::EMPTY_STRING[1]
    const T CharTraitsBase<T>::EMPTY_STRING[] = { static_cast<T>(0) };


#ifndef _WIN32
    /*
     * vislib::CharTraitsBase<T>::ICONV_CODE_CHAR
     */
    template<class T>
    const char *CharTraitsBase<T>::ICONV_CODE_CHAR = "MS-ANSI";

    
    /*
     * vislib::CharTraitsBase<T>::ICONV_CODE_WCHAR
     */
    template<class T>
    const char *CharTraitsBase<T>::ICONV_CODE_WCHAR = "WCHAR_T";


    /*
     * vislib::CharTraitsBase<T>::INVALID_ICONV_T
     */
    template<class T> 
    const iconv_t CharTraitsBase<T>::INVALID_ICONV_T 
        = reinterpret_cast<iconv_t>(-1);
#endif /* !_WIN32 */



    /**
     * This is the general CharTraits implementation. Specialisations for 
     * ANSI and Unicode charachter sets are realised via partial template
     * specialisation in order to allow dynamic instantiation in the String
     * class (required for conversion).
     */
    template<class T> class CharTraits : public CharTraitsBase<T> {

    protected:
        /** Forbidden Ctor. */
        inline CharTraits(void) {
            throw UnsupportedOperationException("CharTraits", __FILE__, 
                __LINE__);
        }
    }; /* end class CharTraits */



    /**
     * Specialisation for ANSI character traits.
     *
     * @author Christoph Mueller
     */
    template<> class CharTraits<char> : public CharTraitsBase<char> {

    public:

        /**
         * Answer the type identifier of the characters.
         *
         * @return The type identifier.
         */
        inline static vislib::CharType CharType(void) {
            return ANSI_CHAR;
        }

        /**
         * Convert character 'src' to an ANSI character saved to 'dst'.
         *
         * @param dst The variable receiving the converted character.
         * @param src The character to be converted.
         *
         * @return true, if 'src' was successfully converted and stored in 
         *         'dst', false otherwise.
         */
        inline static bool Convert(char& dst, const Char src) {
            dst = src;
            return true;
        }

        /**
         * Convert character 'src' to an wide character saved to 'dst'.
         *
         * @param dst The variable receiving the converted character.
         * @param src The character to be converted.
         *
         * @return true, if 'src' was successfully converted and stored in 
         *         'dst', false otherwise.
         */
        inline static bool Convert(wchar_t& dst, const Char src) {
#ifdef _WIN32
            return (::MultiByteToWideChar(CP_ACP, 0, &src, 1, &dst, 1) > 0);
#else /* _WIN32 */
            return Convert(&dst, 1, &src);
#endif /* _WIN32 */
        }

        /**
         * Answer whether the character 'c' is an alphabetic letter.
         *
         * @param c A character.
         *
         * @return true if 'c' is an alphabetic letter, false otherwise.
         */
        inline static bool IsAlpha(const Char c) {
            // Explicit compare prevents C4800.
            return (::isalpha(static_cast<const unsigned char>(c)) != 0);
        }

        /**
         * Answer whether the character 'c' is a digit.
         *
         * @param c A character.
         *
         * @return true if 'c' is a digit, false otherwise.
         */
        inline static bool IsDigit(const Char c) {
            // Explicit compare prevents C4800.
            return (::isdigit(static_cast<const unsigned char>(c)) != 0);
        }

        /**
         * Answer whether the character 'c' is lower case.
         *
         * @param c A character.
         *
         * @return true if 'c' is lower case, false otherwise.
         */
        inline static bool IsLowerCase(const Char c) {
            // Explicit compare prevents C4800.
            return (::islower(static_cast<const unsigned char>(c)) != 0);
        }

        /**
         * Answer whether the character 'c' is a whitespace character.
         *
         * @param c A character.
         *
         * @return true if 'c' is a whitespace, false otherwise.
         */
        inline static bool IsSpace(const Char c) {
            // Explicit compare prevents C4800.
            return (::isspace(static_cast<const unsigned char>(c)) != 0);
        }

        /**
         * Answer whether the character 'c' is upper case.
         *
         * @param c A character.
         *
         * @return true if 'c' is upper case, false otherwise.
         */
        inline static bool IsUpperCase(const Char c) {
            // Explicit compare prevents C4800.
            return (::isupper(static_cast<const unsigned char>(c)) != 0);
        }

        /**
         * Converts the string str to the returned boolean value.
         *
         * @param str The input string.
         *
         * @return The parsed boolean value.
         *
         * @throw IllegalParamException if str is NULL.
         * @throw FormatException if the string could not be parsed to an
         *        boolean value.
         */
        static bool ParseBool(const Char *str);

        /**
         * Converts the string str to the returned floating point value.
         *
         * @param str The input string.
         *
         * @return The parsed floating point value.
         *
         * @throw IllegalParamException if str is NULL.
         * @throw FormatException if the string could not be parsed to an
         *        floating point value.
         */
        static double ParseDouble(const Char *str);

        /**
         * Converts the string str to the returned integer value.
         *
         * @param str The input string.
         *
         * @return The parsed integer value.
         *
         * @throw IllegalParamException if str is NULL.
         * @throw FormatException if the string could not be parsed to an
         *        integer value.
         */
        static int ParseInt(const Char *str);

        /**
         * Convert 'src' to an ANSI character.
         *
         * @param src The character to be converted.
         * @param defChar The character value that will be returned if any 
         *                error occures.
         */
        inline static char ToANSI(const Char src, const char defChar = '?') {
            return src;
        }

        /**
         * Answer the lower case version of 'c'.
         *
         * @param c A character.
         *
         * @return The lower case version of 'c'.
         */
        inline static Char ToLower(const Char c) {
            return static_cast<Char>(::tolower(
                static_cast<const unsigned char>(c)));
        }

        /**
         * Convert 'src' to an Unicode character.
         *
         * @param src The character to be converted.
         * @param defChar The character value that will be returned if any 
         *                error occures.
         */
        inline static wchar_t ToUnicode(const Char src, 
                const wchar_t defChar = L'?') {
            wchar_t o;
            return Convert(o, src) ? o : defChar;
        }

        /**
         * Answer the upper case version of 'c'.
         *
         * @param c A character.
         *
         * @return The upper case version of 'c'.
         */
        inline static Char ToUpper(const Char c) {
            return static_cast<Char>(::toupper(
                static_cast<const unsigned char>(c)));
        }

    protected:

        /** Forbidden Ctor. */
        inline CharTraits(void) {
            throw UnsupportedOperationException("CharTraits<char>", __FILE__, 
                __LINE__);
        }

        /**
         * Convert 'src' to an ANSI string and store it to 'dst'. If the 
         * string is longer than 'cnt' characters, nothing is written and
         * the method returns false.
         *
         * @param dst The buffer to receive the converted string.
         * @param cnt The number of characters that can be stored in 'dst'.
         * @param src The zero-terminated string to be converted.
         *
         * @return true, if 'src' was successfully converted and stored in 
         *         'dst', false otherwise.
         */
        inline static bool Convert(char *dst, const Size cnt, 
                const Char *src) {
            ASSERT(dst != NULL);
            ASSERT(src != NULL);
            StringCopy(dst, src);
            return true;
        }

        /**
         * Convert 'src' to a wide string and store it to 'dst'. 'cnt' is the 
         * size of 'dst'. 'src' must contain at least 'cnt' - 1 valid
         * characters. The caller provides the memory and remains its owner.
         *
         * @param dst The buffer to receive the converted string.
         * @param cnt The number of characters that can be stored in 'dst'.
         * @param src The zero-terminated string to be converted. It must
         *            contain at least 'cnt' - 1 valid characters.
         *
         * @return true, if 'src' was successfully converted and stored in 
         *         'dst', false otherwise.
         */
        inline static bool Convert(wchar_t *dst, const Size cnt,
                const Char *src) {
            ASSERT(dst != NULL);
            ASSERT(src != NULL);

#ifdef _WIN32
            return (::MultiByteToWideChar(CP_ACP, 0, src, cnt, dst, cnt) > 0);
#else /* _WIN32 */
            // TODO: Linux version surely is no inline-candidate
            iconv_t hIconv = INVALID_ICONV_T;
            char *in = reinterpret_cast<char *>(const_cast<Char *>(src));
            char *out = reinterpret_cast<char *>(dst);
            size_t lenIn = cnt * sizeof(Char);
            size_t lenOut = cnt * sizeof(wchar_t);
            bool retval = false;
            //char *li = NULL;

            
            //if (strlen(li = nl_langinfo(CODESET)) < 1) {
            //	return false;
            //}

            if ((hIconv = ::iconv_open(ICONV_CODE_WCHAR, ICONV_CODE_CHAR))
                    != INVALID_ICONV_T) {
                if (::iconv(hIconv, &in, &lenIn, &out, &lenOut) 
                        != static_cast<size_t>(-1)) {
                    retval = true;
                }

                ::iconv_close(hIconv);
            }

            return retval;
#endif /* _WIN32 */
        }

        /**
         * Prints the formatted string 'fmt' to the buffer 'dst' having at least
         * 'cnt' characters.
         *
         * If 'dst' is a NULL pointer or 'cnt' is 0, the method just counts how
         * large 'dst' should be for 'fmt' and the specified parameters.
         *
         * The number of characters returned does not include the terminating
         * zero character.
         *
         * In case of an error, e. g. if 'dst' is too small or 'fmt' is a NULL
         * pointer, the method returns -1;
         *
         * If 'dst' is not NULL and 'cnt' is greater than zero, the method 
         * ensures that the resulting string in 'dst' is zero terminated, 
         * regardless of its content.
         *
         * @param dst    The buffer receiving the formatted string. This buffer
         *               must be able to hold at least 'cnt' characters or must
         *               be NULL.
         * @param cnt    The number of characters that fit into 'dst'.
         * @param fmt    The format string.
         * @param argptr The variable argument list.
         *
         * @return The number of characters written, not including the trailing
         *         zero, or -1 in case of an error like 'dst' being too small 
         *         or 'fmt' being a NULL pointer.
         */
        static Size Format(Char *dst, const Size cnt, const Char *fmt, 
            va_list argptr);

        /**
         * Convert all characters in 'str' to lower case. The currently active 
         * locale is used for conversion.
         *
         * If 'dst' is a NULL pointer or 'cnt' is 0, the method just counts how
         * large 'dst' should be for 'fmt' and the specified parameters.
         *
         * The number of characters returned does not include the terminating
         * zero character.
         *
         * In case of an error, e. g. if 'dst' is too small or 'fmt' is a NULL
         * pointer, the method returns -1;
         *
         * If 'dst' is not NULL and 'cnt' is greater than zero, the method 
         * ensures that the resulting string in 'dst' is zero terminated, 
         * regardless of its content.
         *
         * @param dst Receives the converted string.
         * @param cnt The size of 'dst' in characters.
         * @param str A zero-terminated string. Must not be NULL.
         *
         * @return The number of characters written to 'dst' or -1 in case of an
         *         error.
         */
        static Size ToLower(Char *dst, const Size cnt, const Char *str);

        /**
         * Convert all characters in 'str' to upper case. The currently
         * active locale is used for conversion.
         *
         * If 'dst' is a NULL pointer or 'cnt' is 0, the method just counts how
         * large 'dst' should be for 'fmt' and the specified parameters.
         *
         * The number of characters returned does not include the terminating
         * zero character.
         *
         * In case of an error, e. g. if 'dst' is too small or 'fmt' is a NULL
         * pointer, the method returns -1;
         *
         * If 'dst' is not NULL and 'cnt' is greater than zero, the method 
         * ensures that the resulting string in 'dst' is zero terminated, 
         * regardless of its content.
         *
         * @param dst Receives the converted string.
         * @param cnt The size of 'dst' in characters.
         * @param str A zero-terminated string. Must not be NULL.
         *
         * @return The number of characters written to 'dst' or -1 in case of an
         *         error.
         */
        static Size ToUpper(Char *dst, const Size cnt, const Char *str);

        /* Declare our friends. */
        template<class T> friend class String;
        template<class S, class T, INT32 B> friend class StringConverter;
        friend class vislib::sys::Log;

    }; /* end class CharTraits<char> */



    /**
     * Specialisation for wide character traits.
     *
     * @author Christoph Mueller
     */
    template<> class CharTraits<WCHAR> : public CharTraitsBase<WCHAR> {

    public:

        /**
         * Answer the type identifier of the characters.
         *
         * @return The type identifier.
         */
        inline static vislib::CharType CharType(void) {
            return UNICODE_CHAR;
        }

        /**
         * Convert character 'src' to an ANSI character saved to 'dst'.
         *
         * @param dst The variable receiving the converted character.
         * @param src The character to be converted.
         *
         * @return true, if 'src' was successfully converted and stored in 
         *         'dst', false otherwise.
         */
        inline static bool Convert(char& dst, const Char src) {
#ifdef _WIN32
            return (::WideCharToMultiByte(CP_ACP, 0, &src, 1, &dst, 1, NULL,
                NULL) > 0);
#else /* _WIN32 */
            return Convert(&dst, 1, &src);
#endif /* _WIN32 */
        }

        /**
         * Convert character 'src' to an wide character saved to 'dst'.
         *
         * @param dst The variable receiving the converted character.
         * @param src The character to be converted.
         *
         * @return true, if 'src' was successfully converted and stored in 
         *         'dst', false otherwise.
         */
        inline static bool Convert(wchar_t& dst, const Char src) {
            dst = src;
            return true;
        }

        /**
         * Answer whether the character 'c' is an alphabetic letter.
         *
         * @param c A character.
         *
         * @return true if 'c' is an alphabetic letter, false otherwise.
         */
        inline static bool IsAlpha(const Char c) {
            return (::iswalpha(c) != 0);
        }

        /**
         * Answer whether the character 'c' is a digit.
         *
         * @param c A character.
         *
         * @return true, if 'c' is a digit, false otherwise.
         */
        inline static bool IsDigit(const Char c) {
            return (::iswdigit(c) != 0);    // Explicit compare prevents C4800.
        }

        /**
         * Answer whether the character 'c' is lower case.
         *
         * @param c A character.
         *
         * @return true if 'c' is lower case, false otherwise.
         */
        inline static bool IsLowerCase(const Char c) {
            return (::iswlower(c) != 0);    // Explicit compare prevents C4800.
        }

        /**
         * Answer whether the character 'c' is a whitespace character.
         *
         * @param c A character.
         *
         * @return true, if 'c' is a whitespace, false otherwise.
         */
        inline static bool IsSpace(const Char c) {
            return (::iswspace(c) != 0);    // Explicit compare prevents C4800.
        }

        /**
         * Answer whether the character 'c' is upper case.
         *
         * @param c A character.
         *
         * @return true if 'c' is upper case, false otherwise.
         */
        inline static bool IsUpperCase(const Char c) {
            return (::iswupper(c) != 0);    // Explicit compare prevents C4800.
        }

        /**
         * Converts the string str to the returned boolean value.
         *
         * @param str The input string.
         *
         * @return The parsed boolean value.
         *
         * @throw IllegalParamException if str is NULL.
         * @throw FormatException if the string could not be parsed to an
         *        boolean value.
         */
        static bool ParseBool(const Char *str);

        /**
         * Converts the string str to the returned floating point value.
         *
         * @param str The input string.
         *
         * @return The parsed floating point value.
         *
         * @throw IllegalParamException if str is NULL.
         * @throw FormatException if the string could not be parsed to an
         *        floating point value.
         */
        static double ParseDouble(const Char *str);

        /**
         * Converts the string str to the returned integer value.
         *
         * @param str The input string.
         *
         * @return The parsed integer value.
         *
         * @throw IllegalParamException if str is NULL.
         * @throw FormatException if the string could not be parsed to an
         *        integer value.
         */
        static int ParseInt(const Char *str);

        /**
         * Convert 'src' to an ANSI character.
         *
         * @param src The character to be converted.
         * @param defChar The character value that will be returned if any 
         *                error occures.
         */
        inline static char ToANSI(const Char src, char defChar = '?') {
            char o;
            return Convert(o, src) ? o : defChar;
        }

        /**
         * Answer the lower case version of 'c'.
         *
         * @param c A character.
         *
         * @return The lower case version of 'c'.
         */
        inline static Char ToLower(const Char c) {
            return ::towlower(c);
        }

        /**
         * Convert 'src' to an ANSI character.
         *
         * @param src The character to be converted.
         * @param defChar The character value that will be returned if any 
         *                error occures.
         */
        inline static wchar_t ToUnicode(const Char src, wchar_t defChar = '?') {
            return src;
        }

        /**
         * Answer the upper case version of 'c'.
         *
         * @param c A character.
         *
         * @return The upper case version of 'c'.
         */
        inline static Char ToUpper(const Char c) {
            return ::towupper(c);
        }

    protected:

        /** Forbidden Ctor. */
        inline CharTraits(void) {
            throw UnsupportedOperationException("CharTraits<WCHAR>", __FILE__, 
                __LINE__);
        }

        /**
         * Convert 'src' to an ANSI string and store it to 'dst'.  'cnt' is the
         * size of 'dst'. 'src' must contain at least 'cnt' - 1 valid
         * characters. The caller provides the memory and remains its owner.
         *
         * @param dst The buffer to receive the converted string.
         * @param cnt The number of characters that can be stored in 'dst'.
         * @param src The zero-terminated string to be converted. It must
         *            contain at least 'cnt' - 1 valid characters.
         *
         * @return true, if 'src' was successfully converted and stored in 
         *         'dst', false otherwise.
         */
        inline static bool Convert(char *dst, const Size cnt,
                const Char *src) {
            ASSERT(dst != NULL);
            ASSERT(src != NULL);

#ifdef _WIN32
            return (::WideCharToMultiByte(CP_ACP, 0, src, cnt, dst, cnt, NULL, 
                NULL) > 0);
#else /* _WIN32 */
            iconv_t hIconv = INVALID_ICONV_T;
            char *in = reinterpret_cast<char *>(const_cast<Char *>(src));
            char *out = reinterpret_cast<char *>(dst);
            size_t lenIn = cnt * sizeof(Char);
            size_t lenOut = cnt * sizeof(wchar_t);
            bool retval = false;
            //char *li = NULL;

            //if (strlen(li = nl_langinfo(CODESET)) < 1) {
            //	return false;
            //}

            if ((hIconv = ::iconv_open(ICONV_CODE_CHAR, ICONV_CODE_WCHAR))
                    != INVALID_ICONV_T) {
                if (::iconv(hIconv, &in, &lenIn, &out, &lenOut) 
                        != static_cast<size_t>(-1)) {
                    retval = true;
                }

                ::iconv_close(hIconv);
            }

            return retval;
#endif /* _WIN32 */
        }

        /**
         * Convert 'src' to a wide string and store it to 'dst'. If the 
         * string is longer than 'cnt' characters, nothing is written and
         * the method returns false.
         *
         * @param dst The buffer to receive the converted string.
         * @param cnt The number of characters that can be stored in 'dst'.
         * @param src The zero-terminated string to be converted.
         *
         * @return true, if 'src' was successfully converted and stored in 
         *         'dst', false otherwise.
         */
        inline static bool Convert(wchar_t *dst, const Size cnt, 
                const Char *src) {
            ASSERT(dst != NULL);
            ASSERT(src != NULL);
            StringCopy(dst, src);
            return true;
        }

        /**
         * Prints the formatted string 'fmt' to the buffer 'dst' having at least
         * 'cnt' characters.
         *
         * If 'dst' is a NULL pointer or 'cnt' is 0, the method just counts how
         * large 'dst' should be for 'fmt' and the specified parameters.
         *
         * In case of an error, e. g. if 'dst' is too small or 'fmt' is a NULL
         * pointer, the method returns -1;
         *
         * If 'dst' is not NULL and 'cnt' is greater than zero, the method 
         * ensures that the resulting string in 'dst' is zero terminated, 
         * regardless of its content.
         *
         * Note: Use %hs for printing ANSI strings!
         *
         * @param dst    The buffer receiving the formatted string. This buffer
         *               must be able to hold at least 'cnt' characters or must
         *               be NULL.
         * @param cnt    The number of characters that fit into 'dst'.
         * @param fmt    The format string.
         * @param argptr The variable argument list.
         *
         * @return The number of characters written, not including the trailing
         *         zero, or -1 in case of an error like 'dst' being too small 
         *         or 'fmt' being a NULL pointer.
         */
        static Size Format(Char *dst, const Size cnt, const Char *fmt, 
                va_list argptr);

        /**
         * Convert all characters in 'str' to lower case. The currently active 
         * locale is used for conversion.
         *
         * If 'dst' is a NULL pointer or 'cnt' is 0, the method just counts how
         * large 'dst' should be for 'fmt' and the specified parameters.
         *
         * The number of characters returned does not include the terminating
         * zero character.
         *
         * In case of an error, e. g. if 'dst' is too small or 'fmt' is a NULL
         * pointer, the method returns -1;
         *
         * If 'dst' is not NULL and 'cnt' is greater than zero, the method 
         * ensures that the resulting string in 'dst' is zero terminated, 
         * regardless of its content.
         *
         * @param dst Receives the converted string.
         * @param cnt The size of 'dst' in characters.
         * @param str A zero-terminated string. Must not be NULL.
         *
         * @return The number of characters written to 'dst' or -1 in case of an
         *         error.
         */
        static Size ToLower(Char *dst, const Size cnt, const Char *str);

        /**
         * Convert all characters in 'str' to lower case. The currently active 
         * locale is used for conversion.
         *
         * If 'dst' is a NULL pointer or 'cnt' is 0, the method just counts how
         * large 'dst' should be for 'fmt' and the specified parameters.
         *
         * The number of characters returned does not include the terminating
         * zero character.
         *
         * In case of an error, e. g. if 'dst' is too small or 'fmt' is a NULL
         * pointer, the method returns -1;
         *
         * If 'dst' is not NULL and 'cnt' is greater than zero, the method 
         * ensures that the resulting string in 'dst' is zero terminated, 
         * regardless of its content.
         *
         * @param dst Receives the converted string.
         * @param cnt The size of 'dst' in characters.
         * @param str A zero-terminated string. Must not be NULL.
         *
         * @return The number of characters written to 'dst' or -1 in case of an
         *         error.
         */
        static Size ToUpper(Char *dst, const Size cnt, const Char *str);

        /* Declare our friends. */
        template<class T> friend class String;
        template<class S, class T, INT32 B> friend class StringConverter;
        friend class vislib::sys::Log;

    }; /* end class CharTraits<WCHAR> */


    /* Typedef for template specialisations. */
    typedef CharTraits<char> CharTraitsA;
    typedef CharTraits<WCHAR> CharTraitsW;


    /* Typedef for TCHAR CharTraits. */
#if defined(UNICODE) || defined(_UNICODE)
    typedef CharTraitsW TCharTraits;
#else /* defined(UNICODE) || defined(_UNICODE) */
    typedef CharTraitsA TCharTraits;
#endif /* defined(UNICODE) || defined(_UNICODE) */
}

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CHARTRAITS_H_INCLUDED */
