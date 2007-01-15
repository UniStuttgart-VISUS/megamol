/*
 * CharTraits.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 * Copyright (C) 2005 - 2006 by Christoph Mueller. All rights reserved.
 */

#ifndef VISLIB_CHARTRAITS_H_INCLUDED
#define VISLIB_CHARTRAITS_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <cwctype>

#include "vislib/assert.h"
#include "vislib/memutils.h"
#include "vislib/types.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/FormatException.h"
#include "vislib/IllegalParamException.h"


namespace vislib {

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
     * @author Christoph Mueller
     */
    template<class T> class CharTraits {
    
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
            return (str != NULL) ? CharTraits::StringLength(str) : 0;
        }

    protected:

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

        /**
         * Answer whether 'lhs' and 'rhs' designate equal zero-terminated
         * strings. Both must not be NULL pointers.
         *
         * @param lhs The left hand side operand.
         * @param rhs The right hand side operand.
         *
         * @return true, if 'lhs' and 'rhs' are equal strings, false otherwise.
         */
        inline static bool StringEqual(const Char *lhs, const Char *rhs) {
            ASSERT(lhs != NULL);
            ASSERT(rhs != NULL);

            if (lhs == rhs) {
                /* 'lhs' is a shallow copy of 'rhs'. */
                return true;
            }

            while ((*lhs == *rhs) && rhs) {
                lhs++;
                rhs++;
            }

            return (*lhs == *rhs);
        }

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

        /** Forbidden Ctor. */
        inline CharTraits(void) {
            throw UnsupportedOperationException(
                "vislib::CharTraits<T>::CharTraits", __FILE__, __LINE__);
        }

        /** Empty character sequence constant. */
        static const Char EMPTY_STRING[];

    }; /* end class CharTraits */


    /*
     * vislib::CharTraits<T>::EMPTY_STRING
     */
    template<class T> 
    const typename CharTraits<T>::Char CharTraits<T>::EMPTY_STRING[]
        = { static_cast<Char>(0) };


    /**
     * Specialisation for ANSI character traits.
     *
     * @author Christoph Mueller
     */
    class CharTraitsA : public CharTraits<char> {

    public:

        /**
         * Answer whether the character 'c' is a digit.
         *
         * @param c A character.
         *
         * @return true, if 'c' is a digit, false otherwise.
         */
        inline static bool IsDigit(const Char c) {
            return (::isdigit(c) != 0);     // Explicit compare prevents C4800.
        }

        /**
         * Answer whether the character 'c' is a whitespace character.
         *
         * @param c A character.
         *
         * @return true, if 'c' is a whitespace, false otherwise.
         */
        inline static bool IsSpace(const Char c) {
            return (::isspace(c) != 0);     // Explicit compare prevents C4800.
        }

        /**
         * Answer the lower case version of 'c'.
         *
         * @param c A character.
         *
         * @return The lower case version of 'c'.
         */
        inline static Char ToLower(const Char c) {
            return static_cast<Char>(::tolower(c));
        }

        /**
         * Answer the upper case version of 'c'.
         *
         * @param c A character.
         *
         * @return The upper case version of 'c'.
         */
        inline static Char ToUpper(const Char c) {
            return static_cast<Char>(::toupper(c));
        }

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
        inline static int ParseInt(const Char *str) {
            int retval;
            if (str == NULL) {
                throw IllegalParamException("str", __FILE__, __LINE__);
            }
            if (
#if (_MSC_VER >= 1400)
                sscanf_s
#else  /*(_MSC_VER >= 1400) */
                sscanf
#endif /*(_MSC_VER >= 1400) */
                (str, "%d", &retval) != 1) {
                throw FormatException("Cannot convert String to Integer",
                    __FILE__, __LINE__);
            }
            return retval;
        }

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
        inline static double ParseDouble(const Char *str) {
            double retval;
            if (str == NULL) {
                throw IllegalParamException("str", __FILE__, __LINE__);
            }
            if (
#if (_MSC_VER >= 1400)
                sscanf_s
#else  /*(_MSC_VER >= 1400) */
                sscanf
#endif /*(_MSC_VER >= 1400) */
                (str, "%lf", &retval) != 1) {
                throw FormatException("Cannot convert String to Double", 
                    __FILE__, __LINE__);
            }
            return retval;
        }

    protected:

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
            int lenIn = StringLength(src) + 1;	// Count only once.
            int lenOut = ::MultiByteToWideChar(CP_ACP, 0, src, lenIn, NULL, 0);
                dst[cnt - 1] = 0;
                return (::MultiByteToWideChar(CP_ACP, 0, src, lenIn, dst, 
                    lenOut) > 0);
            } else {
                return false;
            }
#else /* _WIN32 */
            return (::swprintf(dst, cnt, L"%hs", src) == (cnt - 1));
#endif /* _WIN32 */
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
        inline static Size Format(Char *dst, const Size cnt, const Char *fmt, 
                va_list argptr) {
            int retval = -1;

#ifdef _WIN32
            if ((dst == NULL) || (cnt <= 0)) {
                /* Answer the prospective size of the string. */
                retval = ::_vscprintf(fmt, argptr);

            } else {
#if (_MSC_VER >= 1400)
                retval = ::_vsnprintf_s(dst, cnt, cnt, fmt, argptr);
#else /* (_MSC_VER >= 1400) */
                retval = ::_vsnprintf(dst, cnt, fmt, argptr);
#endif /* (_MSC_VER >= 1400) */
            } /* end if ((dst == NULL) || (cnt <= 0)) */

#else /* _WIN32 */
            retval = ::vsnprintf(dst, cnt, fmt, argptr);

            if ((dst != NULL) && (cnt > 0) && (retval > cnt - 1)) {
                retval = -1;
            }
#endif /* _WIN32 */ 

            /* Ensure string being terminated. */
            if ((dst != NULL) && (cnt > 0)) {
                dst[cnt - 1] = 0;
            }
            return static_cast<Size>(retval);
        }

// TODO: Problem with locale.
//      /**
//       * Convert all characters in 'str' to lower case.
//       *
//       * @param str A zero-terminated string. Must not be NULL.
//       */
//      inline static void ToLower(Char *str) {
//          ASSERT(str != NULL);
//#ifdef _WIN32
//          ::_strlwr(str);
//#else /* _WIN32 */
//          assert(false);
//#endif /* _WIN32 */
//      }

//      /**
//       * Convert all characters in 'str' to upper case.
//       *
//       * @param str A zero-terminated string. Must not be NULL.
//       */
//      inline static void ToUpper(Char *str) {
//          ASSERT(str != NULL);
//#ifdef _WIN32
//          ::_strupr(str);
//#else /* _WIN32 */
//          assert(false);
//#endif /* _WIN32 */
//      }

        /** Forbidden Ctor. */
        inline CharTraitsA(void) {
            throw UnsupportedOperationException(
                "vislib::CharTraitsA::CharTraitsA", __FILE__, __LINE__);
        }

        /* Declare our friends. */
        template<class T> friend class String;
        template<class S, class T, INT32 B> friend class StringConverter;

    }; /* end class CharTraitsA */


    /**
     * Specialisation for wide character traits.
     *
     * @author Christoph Mueller
     */
    class CharTraitsW : public CharTraits<WCHAR> {

    public:

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
         * Answer the upper case version of 'c'.
         *
         * @param c A character.
         *
         * @return The upper case version of 'c'.
         */
        inline static Char ToUpper(const Char c) {
            return ::towupper(c);
        }

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
        inline static int ParseInt(const Char *str) {
            int retval;
            if (str == NULL) {
                throw IllegalParamException("str", __FILE__, __LINE__);
            }
            if (
#if (_MSC_VER >= 1400)
                swscanf_s
#else  /*(_MSC_VER >= 1400) */
                swscanf
#endif /*(_MSC_VER >= 1400) */
                (str, L"%d", &retval) != 1) {
                throw FormatException("Cannot convert String to Integer", 
                    __FILE__, __LINE__);
            }
            return retval;
        }

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
        inline static double ParseDouble(const Char *str) {
            double retval;
            if (str == NULL) {
                throw IllegalParamException("str", __FILE__, __LINE__);
            }
            if (
#if (_MSC_VER >= 1400)
                swscanf_s
#else  /*(_MSC_VER >= 1400) */
                swscanf
#endif /*(_MSC_VER >= 1400) */
                (str, L"%lf", &retval) != 1) {
                throw FormatException("Cannot convert String to Double", 
                    __FILE__, __LINE__);
            }
            return retval;
        }

    protected:

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
            int lenIn = StringLength(src)+ 1;	// Count only once.
            int lenOut = ::WideCharToMultiByte(CP_ACP, 0, src, lenIn, NULL, 0, 
                NULL, NULL);
            if ((lenOut > 0) && (static_cast<Size>(lenOut) <= cnt)) {
                return (::WideCharToMultiByte(CP_ACP, 0, src, lenIn, dst, 
                    lenOut, NULL, NULL) > 0);
            } else {
                return false;
            }
#else /* _WIN32 */
            return (::snprintf(dst, cnt, "%ls", src) == (cnt - 1));
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
        inline static Size Format(Char *dst, const Size cnt, const Char *fmt, 
                va_list argptr) {
            int retval = -1;

#ifdef _WIN32
            if ((dst == NULL) || (cnt <= 0)) {
                /* Answer the prospective size of the string. */
                retval = ::_vscwprintf(fmt, argptr);

            } else {
#if (_MSC_VER >= 1400)
                retval = ::_vsnwprintf_s(dst, cnt, cnt, fmt, argptr);
#else /* (_MSC_VER >= 1400) */
                retval = ::_vsnwprintf(dst, cnt, fmt, argptr);
#endif /* (_MSC_VER >= 1400) */
            } /* end if ((dst == NULL) || (cnt <= 0)) */

#else /* _WIN32 */
            // Yes, you can trust your eyes: The char and wide char 
            // implementations under Linux have a completely different 
            // semantic. vswprintf cannot be used for determining the required
            // size as vsprintf can.
            SIZE_T bufferSize, bufferGrow;
            Char *buffer = NULL;

            if ((dst == NULL) || (cnt <= 0)) {
                /* Just count. */
                bufferSize = static_cast<SIZE_T>(1.1
                    * static_cast<float>(::wcslen(fmt)) + 1);
                bufferGrow = static_cast<SIZE_T>(0.5
                    * static_cast<float>(bufferSize));
                buffer = new Char[bufferSize];

                while ((retval = ::vswprintf(buffer, bufferSize, fmt, argptr))
                        == -1) {
                    ARY_SAFE_DELETE(buffer);
                    bufferSize += bufferGrow;
                    buffer = new Char[bufferSize];
                }

                retval = ::wcslen(buffer);
                ARY_SAFE_DELETE(buffer);
                
            } else {
                /* Format the string. */
                retval = ::vswprintf(dst, cnt, fmt, argptr);
            }

#endif /* _WIN32 */ 

            /* Ensure string being terminated. */
            if ((dst != NULL) && (cnt > 0)) {
                dst[cnt - 1] = 0;
            }
            return static_cast<Size>(retval);
        }

//      /**
//       * Convert all characters in 'str' to lower case.
//       *
//       * @param str A zero-terminated string. Must not be NULL.
//       */
//      inline static void ToLower(Char *str) {
//          ASSERT(str != NULL);
//#ifdef _WIN32
//          ::_wcslwr(str);
//#else /* _WIN32 */
//          assert(false);
//#endif /* _WIN32 */
//      }

//      /**
//       * Convert all characters in 'str' to upper case.
//       *
//       * @param str A zero-terminated string. Must not be NULL.
//       */
//      inline static void ToUpper(Char *str) {
//          ASSERT(str != NULL);
//#ifdef _WIN32
//          ::_wcsupr(str);
//#else /* _WIN32 */
//          assert(false);
//#endif /* _WIN32 */
//      }

        /** Forbidden Ctor. */
        inline CharTraitsW(void) {
            throw UnsupportedOperationException(
                "vislib::CharTraitsW::CharTraitsW", __FILE__, __LINE__);
        }

        /* Declare our friends. */
        template<class T> friend class String;
        template<class S, class T, INT32 B> friend class StringConverter;

    }; /* end class CharTraitsW */

    /* Typedef for TCHAR CharTraits. */
#if defined(UNICODE) || defined(_UNICODE)
    typedef CharTraitsW TCharTraits;
#else /* defined(UNICODE) || defined(_UNICODE) */
    typedef CharTraitsA TCharTraits;
#endif /* defined(UNICODE) || defined(_UNICODE) */
}

#endif /* VISLIB_CHARTRAITS_H_INCLUDED */

