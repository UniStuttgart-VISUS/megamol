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
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <cwctype>

#include "vislib/assert.h"
#include "vislib/types.h"
#include "vislib/UnsupportedOperationException.h"


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
	
    protected:

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
		 * Copy from 'src' to 'dst'. If the method returns true, 'dst' points to
		 * a zero-terminated string that holds a copy of 'src'.
		 *
		 * @param dst The buffer to receive the copy.
		 * @param cnt The size of the buffer in characters, e. g. in bytes for a
		 *            char instantiation or in words for a wchar_t 
		 *            instantiation.
		 * @param src The source string.
		 *
		 * @return true, if the copy succeeded, false in case of an error.
		 */
		inline static bool SafeStringCopy(Char *dst, const Size cnt, 
				const Char *src) {

			if ((dst == NULL) || (cnt < 1)) {
				/* No destination buffer. */
				return false;

			} else if (src == NULL) {
				/* No input, create an empty string. */
				ASSERT(dst != NULL);
				ASSERT(cnt >= 1);

				dst[0] = static_cast<Char>(0);
				return true;

			} else {
				/* Have output space and input data. */
				ASSERT(dst != NULL);
				ASSERT(cnt >= 1);
				ASSERT(src != NULL);

 				size_t dstSize = cnt * CharSize();
				size_t srcSize = StringSize(src);

				if (dstSize >= srcSize) {
					::memcpy(dst, src, srcSize);
					return true;

				} else {
					/* 'dst' buffer is too small. */
					return false;
				}
			}
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
            while (*tmp++ = *src++);
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

	}; /* end class CharTraits */


	/**
	 * Specialisation for ANSI character traits.
	 *
	 * @author Christoph Mueller
	 */
	class CharTraitsA : public CharTraits<CHAR> {

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
			return SafeStringCopy(dst, cnt, src);
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
            // TODO: Could work without test, if we assume that String and
            // String convert always pass a buffer that is large enough.
			Size srcLen = SafeStringLength(src);

			if ((dst != NULL) && (cnt > srcLen) && (src != NULL)) {
#if (_MSC_VER >= 1400)
				::_snwprintf_s(dst, cnt, cnt, L"%hs", src);
#elif defined(_WIN32)
                ::_snwprintf(dst, cnt, L"%hs", src);
#else  /*(_MSC_VER >= 1400) */
				::swprintf(dst, cnt, L"%hs", src);
#endif /*(_MSC_VER >= 1400) */
				return true;
			} else {
				return false;
			}
		}

		/**
		 * Answer whether the character 'c' is a digit.
		 *
		 * @param c A character.
		 *
		 * @return true, if 'c' is a digit, false otherwise.
		 */
		inline static bool IsDigit(const Char c) {
			return (::isdigit(c) != 0);		// Explicit compare prevents C4800.
		}

		/**
		 * Answer whether the character 'c' is a whitespace character.
		 *
		 * @param c A character.
		 *
		 * @return true, if 'c' is a whitespace, false otherwise.
		 */
		inline static bool IsSpace(const Char c) {
			return (::isspace(c) != 0);		// Explicit compare prevents C4800.
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

//		/**
//		 * Convert all characters in 'str' to lower case.
//		 *
//		 * @param str A zero-terminated string. Must not be NULL.
//		 */
//		inline static void ToLower(Char *str) {
//			ASSERT(str != NULL);
//#ifdef _WIN32
//			::_strlwr(str);
//#else /* _WIN32 */
//			assert(false);
//#endif /* _WIN32 */
//		}

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

//		/**
//		 * Convert all characters in 'str' to upper case.
//		 *
//		 * @param str A zero-terminated string. Must not be NULL.
//		 */
//		inline static void ToUpper(Char *str) {
//			ASSERT(str != NULL);
//#ifdef _WIN32
//			::_strupr(str);
//#else /* _WIN32 */
//			assert(false);
//#endif /* _WIN32 */
//		}

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
            // TODO: Could work without test, if we assume that String and
            // String convert always pass a buffer that is large enough.
			Size srcLen = SafeStringLength(src);

			if ((dst != NULL) && (cnt > srcLen) && (src != NULL)) {
#if (_MSC_VER >= 1400)
				::_snprintf_s(dst, cnt, cnt, "%ls", src);
#elif defined(_WIN32)
                ::_snprintf(dst, cnt, "%ls", src);
#else  /*(_MSC_VER >= 1400) */
				::snprintf(dst, cnt, "%ls", src);
#endif /*(_MSC_VER >= 1400) */
				return true;
			} else {
				return false;
			}
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
 			return SafeStringCopy(dst, cnt, src);
		}

		/**
		 * Answer whether the character 'c' is a digit.
		 *
		 * @param c A character.
		 *
		 * @return true, if 'c' is a digit, false otherwise.
		 */
		inline static bool IsDigit(const Char c) {
			return (::iswdigit(c) != 0);	// Explicit compare prevents C4800.
		}

		/**
		 * Answer whether the character 'c' is a whitespace character.
		 *
		 * @param c A character.
		 *
		 * @return true, if 'c' is a whitespace, false otherwise.
		 */
		inline static bool IsSpace(const Char c) {
			return (::iswspace(c) != 0);	// Explicit compare prevents C4800.
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

//		/**
//		 * Convert all characters in 'str' to lower case.
//		 *
//		 * @param str A zero-terminated string. Must not be NULL.
//		 */
//		inline static void ToLower(Char *str) {
//			ASSERT(str != NULL);
//#ifdef _WIN32
//			::_wcslwr(str);
//#else /* _WIN32 */
//			assert(false);
//#endif /* _WIN32 */
//		}

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

//		/**
//		 * Convert all characters in 'str' to upper case.
//		 *
//		 * @param str A zero-terminated string. Must not be NULL.
//		 */
//		inline static void ToUpper(Char *str) {
//			ASSERT(str != NULL);
//#ifdef _WIN32
//			::_wcsupr(str);
//#else /* _WIN32 */
//			assert(false);
//#endif /* _WIN32 */
//		}

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

