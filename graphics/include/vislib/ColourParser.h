/*
 * ColourParser.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_COLOURPARSER_H_INCLUDED
#define VISLIB_COLOURPARSER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/ColourRGBAu8.h"
#include "vislib/FormatException.h"
#include "vislib/String.h"


namespace vislib {
namespace graphics {


    /**
     * Parser for string representations of colours. This supports named
     * colours, HTML colours, byte-arrays, and float-arrays.
     *
     * Named Colours:
     * Syntax:
     *   name case insensitive
     * Example:
     *   AliceBlue
     *
     * HTML Colours:
     * Syntax:
     *   #RGB | #RGBA | #RRGGBB | #RRGGBBAA  (hex representation)
     * Example:
     *   #fff | #08CA | #0080c0 | #ff0000FF
     *
     * Arrays: (floats [0..1], bytes [0..255])
     * Syntax: (Number parsing similar to 'atof'/'atoi')
     *   (R; G; B; A) | (R; G; B)
     * Example:
     *   (0.5; 0.75; 1.0; 1.0) | (255; 128; 0)
     */
    class ColourParser {
    public:

        /**
         * Flags for possible representation types
         */
        enum RepresentationType {
            REPTYPE_BYTE = 0x01,
            REPTYPE_FLOAT = 0x02,
            REPTYPE_NAMED = 0x04,
            REPTYPE_HTML = 0x08,
            REPTYPE_QUANT = 0x10 //< allows significant quantization errors 
                                 // when converting float colour values to
                                 // non-float representations
        };

        /**
         * Parses a string into a colour
         *
         * @param inStr The string to be parsed
         * @param outCol The colour object to receive the parsed colour
         *
         * @return A reference to outCol
         *
         * @throw FormatException if 'inStr' cannot be parsed
         */
        static inline ColourRGBAu8& FromString(const vislib::StringA& inStr,
                ColourRGBAu8& outCol, bool allowQuantization = true) {
            unsigned char r, g, b, a;
            FromString(inStr, r, g, b, a, allowQuantization);
            outCol.Set(r, g, b, a);
            return outCol;
        }

        /**
         * Parses a string into a colour
         *
         * @param inStr The string to be parsed
         * @param outR Variable to receive the red colour component
         * @param outG Variable to receive the green colour component
         * @param outB Variable to receive the blue colour component
         * @param allowQuantization If 'inStr' contains a colour description
         *                          which cannot be represented by uint8
         *                          colours without significant this flag
         *                          controlls the behaviour of the method. If
         *                          set to true the colour values will be
         *                          quantized; if set to false the parsing
         *                          will fail.
         *
         * @throw FormatException if 'inStr' cannot be parsed
         */
        static inline void FromString(const vislib::StringA& inStr,
                unsigned char &outR, unsigned char &outG,
                unsigned char &outB, bool allowQuantization = true) {
            unsigned char dummyA;
            FromString(inStr, outR, outG, outB, dummyA, allowQuantization);
        }

        /**
         * Parses a string into a colour
         *
         * @param inStr The string to be parsed
         * @param outR Variable to receive the red colour component
         * @param outG Variable to receive the green colour component
         * @param outB Variable to receive the blue colour component
         * @param outA Variable to receive the alpha component
         * @param allowQuantization If 'inStr' contains a colour description
         *                          which cannot be represented by uint8
         *                          colours without significant this flag
         *                          controlls the behaviour of the method. If
         *                          set to true the colour values will be
         *                          quantized; if set to false the parsing
         *                          will fail.
         *
         * @throw FormatException if 'inStr' cannot be parsed
         */
        static void FromString(const vislib::StringA& inStr,
            unsigned char &outR, unsigned char &outG, unsigned char &outB,
            unsigned char &outA, bool allowQuantization = true);

        /**
         * Parses a string into a colour
         *
         * @param inStr The string to be parsed
         * @param outR Variable to receive the red colour component
         * @param outG Variable to receive the green colour component
         * @param outB Variable to receive the blue colour component
         *
         * @throw FormatException if 'inStr' cannot be parsed
         */
        static inline void FromString(const vislib::StringA& inStr, float &outR,
                float &outG, float &outB) {
            float dummyA;
            FromString(inStr, outR, outG, outB, dummyA);
        }

        /**
         * Parses a string into a colour
         *
         * @param inStr The string to be parsed
         * @param outR Variable to receive the red colour component
         * @param outG Variable to receive the green colour component
         * @param outB Variable to receive the blue colour component
         * @param outA Variable to receive the alpha component
         *
         * @throw FormatException if 'inStr' cannot be parsed
         */
        static void FromString(const vislib::StringA& inStr, float &outR,
            float &outG, float &outB, float &outA);

        /**
         * Generates a string representation for a specified colour.
         *
         * @param outStr The string to receive the output
         * @param inCol The colour to be converted
         * @param repType The allowed representation types
         *
         * @return A reference to outStr
         *
         * @throw FormatException if the colour cannot be represented in any
         *        of the allowed representation types
         */
        static inline vislib::StringA& ToString(vislib::StringA& outStr,
                const ColourRGBAu8& inCol, int repType
                = REPTYPE_BYTE | REPTYPE_NAMED | REPTYPE_HTML) {
            return ToString(outStr, inCol.R(), inCol.G(), inCol.B(),
                inCol.A(), repType);
        }

        /**
         * Generates a string representation for a specified colour.
         *
         * @param outStr The string to receive the output
         * @param inR The red colour component
         * @param inG The green colour component
         * @param inB The blue colour component
         * @param repType The allowed representation types
         *
         * @return A reference to outStr
         *
         * @throw FormatException if the colour cannot be represented in any
         *        of the allowed representation types
         */
        static inline vislib::StringA& ToString(vislib::StringA& outStr,
                unsigned char inR, unsigned char inG, unsigned char inB,
                int repType = REPTYPE_BYTE | REPTYPE_NAMED | REPTYPE_HTML) {
            return ToString(outStr, inR, inG, inB, 255, repType);
        }

        /**
         * Generates a string representation for a specified colour.
         *
         * @param outStr The string to receive the output
         * @param inR The red colour component
         * @param inG The green colour component
         * @param inB The blue colour component
         * @param inA The alpha component
         * @param repType The allowed representation types
         *
         * @return A reference to outStr
         *
         * @throw FormatException if the colour cannot be represented in any
         *        of the allowed representation types
         */
        static vislib::StringA& ToString(vislib::StringA& outStr,
            unsigned char inR, unsigned char inG, unsigned char inB,
            unsigned char inA, int repType = REPTYPE_BYTE | REPTYPE_NAMED
            | REPTYPE_HTML);

        /**
         * Generates a string representation for a specified colour.
         *
         * @param outStr The string to receive the output
         * @param inR The red colour component
         * @param inG The green colour component
         * @param inB The blue colour component
         * @param repType The allowed representation types
         *
         * @return A reference to outStr
         *
         * @throw FormatException if the colour cannot be represented in any
         *        of the allowed representation types
         */
        static inline vislib::StringA& ToString(vislib::StringA& outStr,
                float inR, float inG, float inB, int repType = REPTYPE_BYTE
                | REPTYPE_NAMED | REPTYPE_HTML | REPTYPE_FLOAT) {
            return ToString(outStr, inR, inG, inB, 1.0f, repType);
        }

        /**
         * Generates a string representation for a specified colour.
         *
         * @param outStr The string to receive the output
         * @param inR The red colour component
         * @param inG The green colour component
         * @param inB The blue colour component
         * @param inA The alpha component
         * @param repType The allowed representation types
         *
         * @return A reference to outStr
         *
         * @throw FormatException if the colour cannot be represented in any
         *        of the allowed representation types
         */
        static vislib::StringA& ToString(vislib::StringA& outStr, float inR,
            float inG, float inB, float inA, int repType
            = REPTYPE_BYTE | REPTYPE_NAMED | REPTYPE_HTML | REPTYPE_FLOAT);

    private:

        /**
         * Answers the numeric value of a hex character (0-15) or 255 in case
         * of an error.
         *
         * @param c The character to be interpreted
         *
         * @return The numeric value of 'c'
         */
        static unsigned char hexToNum(const char& c);

        /**
         * Tries to parse an array from inStr
         *
         * @param inStr The input string
         * @param outR Variable to receive the red colour component
         * @param outG Variable to receive the green colour component
         * @param outB Variable to receive the blue colour component
         * @param outA Variable to receive the alpha component
         *
         * @return True if parsed successfully
         */
        static bool parseArray(const vislib::StringA& inStr,
            float &outR, float &outG, float &outB, float &outA);

        /**
         * Parses a colour from an html representation
         *
         * @param inStr The string in HTML representation
         *
         * @return The parsed colour
         *
         * @throw FormatException if the string could not be parsed
         */
        static ColourRGBAu8 parseHTML(const vislib::StringA& inStr);

        /** Forbidden ctor. */
        ColourParser(void);

        /** Forbidden dtor. */
        ~ColourParser(void);

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_COLOURPARSER_H_INCLUDED */

