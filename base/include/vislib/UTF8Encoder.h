/*
 * UTF8Encoder.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_UTF8ENCODER_H_INCLUDED
#define VISLIB_UTF8ENCODER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/String.h"
#include "vislib/types.h"


namespace vislib {


    /**
     * Encoder class for encoding and decoding UTF8 strings
     */
    class UTF8Encoder {

    public:

        /** typedef for UTF8-String sizes */
        typedef INT32 Size;

        /**
         * Calculates the needed size in Bytes to store the given ANSI string
         * in UTF8 coding, including the terminating zero.
         *
         * Note: This implementation should only be used if you are using 7-bit
         * ANSI strings. If you are using any 8-Bit ANSI characters, the whole
         * string must be converted to a Unicode wide string!
         *
         * @param str The input string. Must be zero terminated.
         *
         * @return The size in Bytes needed to store the given string, or 
         *         negative if the string contains characters not 
         *         representable in UTF8.
         */
        static Size CalcUTF8Size(const char *str);

        /**
         * Calculates the needed size in Bytes to store the given UNICODE 
         * string in UTF8 coding, including the terminating zero.
         *
         * Note: This implementation should only be used if you are using 7-bit
         * ANSI strings. If you are using any 8-Bit ANSI characters, the whole
         * string must be converted to a Unicode wide string!
         *
         * @param str The input string. Must be zero terminated.
         *
         * @return The size in Bytes needed to store the given string, or 
         *         negative if the string contains characters not 
         *         representable in UTF8.
         */
        static Size CalcUTF8Size(const wchar_t *str);

        /**
         * Calculates the needed size in Bytes to store the given ANSI string
         * in UTF8 coding, including the terminating zero.
         *
         * @param str The input string.
         *
         * @return The size in Bytes needed to store the given string, or 
         *         negative if the string contains characters not 
         *         representable in UTF8.
         */
        static inline Size CalcUTF8Size(const vislib::StringA& str) {
            return CalcUTF8Size(str.PeekBuffer());
        }

        /**
         * Calculates the needed size in Bytes to store the given UNICODE 
         * string in UTF8 coding, including the terminating zero.
         *
         * @param str The input string.
         *
         * @return The size in Bytes needed to store the given string, or 
         *         negative if the string contains characters not 
         *         representable in UTF8.
         */
        static inline Size CalcUTF8Size(const vislib::StringW& str) {
            return CalcUTF8Size(str.PeekBuffer());
        }

        /**
         * Answer the string length in character of the given UTF8-String, 
         * excluding the terminating zero. If the string is not a valid
         * UTF8-String, teh return value is negative.
         *
         * @param str The UTF8-String.
         *
         * @return The length of the string, or negative on error.
         */
        static Size StringLength(const char *str);

        /**
         * Answer the string length in character of the given UTF8-String, 
         * excluding the terminating zero. If the string is not a valid
         * UTF8-String, teh return value is negative.
         *
         * @param str The UTF8-String.
         *
         * @return The length of the string, or negative on error.
         */
        static inline Size StringLength(const vislib::StringA& str) {
            return StringLength(str.PeekBuffer());
        }

        /**
         * Encodes the given ANSI String into an UTF8-String.
         *
         * Note: This implementation should only be used if you are using 7-bit
         * ANSI strings. If you are using any 8-Bit ANSI characters, the whole
         * string must be converted to a Unicode wide string!
         *
         * @param outTarget The string receiving the UTF8 output.
         * @param source The input String. Must be zero terminated.
         *
         * @return True if the string was successfully encoded, false 
         *         otherwise. If false is returned the content of outTarget is
         *         undefined.
         */
        static bool Encode(vislib::StringA& outTarget, const char *source);

        /**
         * Encodes the given Unicode String into an UTF8-String.
         *
         * @param outTarget The string receiving the UTF8 output.
         * @param source The input string. Must be zero terminated.
         *
         * @return True if the string was successfully encoded, false 
         *         otherwise. If false is returned the content of outTarget is
         *         undefined.
         */
        static bool Encode(vislib::StringA& outTarget, const wchar_t *source);

        /**
         * Encodes the given ANSI String into an UTF8-String.
         *
         * Note: This implementation should only be used if you are using 7-bit
         * ANSI strings. If you are using any 8-Bit ANSI characters, the whole
         * string must be converted to a Unicode wide string!
         *
         * @param outTarget The string receiving the UTF8 output.
         * @param source The input string.
         *
         * @return True if the string was successfully encoded, false 
         *         otherwise. If false is returned the content of outTarget is
         *         undefined.
         */
        static inline bool Encode(vislib::StringA& outTarget, const vislib::StringA& source) {
            return Encode(outTarget, source.PeekBuffer());
        }

        /**
         * Encodes the given Unicode String into an UTF8-String.
         *
         * @param outTarget The string receiving the UTF8 output.
         * @param source The input string.
         *
         * @return True if the string was successfully encoded, false 
         *         otherwise. If false is returned the content of outTarget is
         *         undefined.
         */
        static inline bool Encode(vislib::StringA& outTarget, const vislib::StringW& source) {
            return Encode(outTarget, source.PeekBuffer());
        }

        /**
         * Decodes the given UTF8-String.
         *
         * @param outTarget The string receiving the decoded ANSI String.
         * @param source The UTF8-String. Must be zero terminated.
         *
         * @return True if the string was successfully decoded, false 
         *         otherwise. If false is returned the content of outTarget is
         *         undefined.
         */
        static bool Decode(vislib::StringA& outTarget, const char *source);

        /**
         * Decodes the given UTF8-String.
         *
         * @param outTarget The string receiving the decoded Unicode String.
         * @param source The UTF8-String. Must be zero terminated.
         *
         * @return True if the string was successfully decoded, false 
         *         otherwise. If false is returned the content of outTarget is
         *         undefined.
         */
        static bool Decode(vislib::StringW& outTarget, const char *source);

        /**
         * Decodes the given UTF8-String.
         *
         * @param outTarget The string receiving the decoded ANSI String.
         * @param source The UTF8-String. 
         *
         * @return True if the string was successfully decoded, false 
         *         otherwise. If false is returned the content of outTarget is
         *         undefined.
         */
        static inline bool Decode(vislib::StringA& outTarget, const vislib::StringA& source) {
            return Decode(outTarget, source.PeekBuffer());
        }

        /**
         * Decodes the given UTF8-String.
         *
         * @param outTarget The string receiving the decoded Unicode String.
         * @param source The UTF8-String. 
         *
         * @return True if the string was successfully decoded, false 
         *         otherwise. If false is returned the content of outTarget is
         *         undefined.
         */
        static inline bool Decode(vislib::StringW& outTarget, const vislib::StringA& source) {
            return Decode(outTarget, source.PeekBuffer());
        }

        /** Dtor. */
        ~UTF8Encoder(void);

    private:

        /** Disallow instances of this class. */
        UTF8Encoder(void);

    };
    
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_UTF8ENCODER_H_INCLUDED */
