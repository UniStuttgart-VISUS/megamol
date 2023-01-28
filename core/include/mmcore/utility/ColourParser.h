/*
 * ColourParser.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_COLOURPARSER_H_INCLUDED
#define MEGAMOLCORE_COLOURPARSER_H_INCLUDED
#pragma once

#include "vislib/String.h"


namespace megamol::core::utility {

/**
 * Utility class for converting colour data to and from strings
 */
class ColourParser {
public:
    /**
     * Converts an rgb colour to a string representation
     *
     * @param r The red colour component (0..1)
     * @param g The green colour component (0..1)
     * @param b The blue colour component (0..1)
     *
     * @return The string representation of this colour
     */
    static inline vislib::TString ToString(float r, float g, float b) {
        return ToString(r, g, b, 1.0f);
    }

    /**
     * Converts an rgb colour to a string representation
     *
     * @param r The red colour component (0..1)
     * @param g The green colour component (0..1)
     * @param b The blue colour component (0..1)
     * @param a The alpha transparency value (0..1)
     *
     * @return The string representation of this colour
     */
    static inline vislib::TString ToString(float r, float g, float b, float a) {
#if defined(UNICODE) || defined(_UNICODE)
        return ToStringW(r, g, b, a);
#else
        return ToStringA(r, g, b, a);
#endif
    }

    /**
     * Converts an rgb colour to an ANSI string representation
     *
     * @param r The red colour component (0..1)
     * @param g The green colour component (0..1)
     * @param b The blue colour component (0..1)
     *
     * @return The string representation of this colour
     */
    static inline vislib::StringA ToStringA(float r, float g, float b) {
        return ToStringA(r, g, b, 1.0f);
    }

    /**
     * Converts an rgb colour to an ANSI string representation
     *
     * @param r The red colour component (0..1)
     * @param g The green colour component (0..1)
     * @param b The blue colour component (0..1)
     * @param a The alpha transparency value (0..1)
     *
     * @return The string representation of this colour
     */
    static vislib::StringA ToStringA(float r, float g, float b, float a);

    /**
     * Converts an rgb colour to a unicode string representation
     *
     * @param r The red colour component (0..1)
     * @param g The green colour component (0..1)
     * @param b The blue colour component (0..1)
     *
     * @return The string representation of this colour
     */
    static inline vislib::StringW ToStringW(float r, float g, float b) {
        return vislib::StringW(ToStringA(r, g, b, 1.0f));
    }

    /**
     * Converts an rgb colour to a unicode string representation
     *
     * @param r The red colour component (0..1)
     * @param g The green colour component (0..1)
     * @param b The blue colour component (0..1)
     * @param a The alpha transparency value (0..1)
     *
     * @return The string representation of this colour
     */
    static inline vislib::StringW ToStringW(float r, float g, float b, float a) {
        return vislib::StringW(ToStringA(r, g, b, a));
    }

    /**
     * Converts an ANSI string representation to an rgb colour
     *
     * @param str The string to be parsed
     * @param outR The variable receiving the red colour component (0..1)
     * @param outG The variable receiving the green colour component (0..1)
     * @param outB The variable receiving the blue colour component (0..1)
     *
     * @return 'true' on success, 'false' on failure
     */
    static inline bool FromString(const vislib::StringA& str, float& outR, float& outG, float& outB) {
        float a;
        return FromString(str, outR, outG, outB, a);
    }

    /**
     * Converts an ANSI string representation to an rgb colour
     *
     * @param str The string to be parsed
     * @param outR The variable receiving the red colour component (0..1)
     * @param outG The variable receiving the green colour component (0..1)
     * @param outB The variable receiving the blue colour component (0..1)
     * @param outA The variable receiving the alpha transparency (0..1)
     *
     * @return 'true' on success, 'false' on failure
     */
    static bool FromString(const vislib::StringA& str, float& outR, float& outG, float& outB, float& outA);

    /**
     * Converts an ANSI string representation to an rgb colour
     *
     * @param str The string to be parsed
     * @param outColLen the number of components in the return vector
     * @param outCol a character vector receiving the parsed color components (0..255)
     *
     * @return 'true' on success, 'false' on failure
     */
    static bool FromString(const vislib::StringA& str, unsigned int outColLen, unsigned char* outCol);

    /**
     * Converts an ANSI string representation to an rgb colour
     *
     * @param str The string to be parsed
     * @param outColLen the number of components in the return vector
     * @param outCol a float vector receiving the parsed color components (0..1)
     *
     * @return 'true' on success, 'false' on failure
     */
    static bool FromString(const vislib::StringA& str, unsigned int outColLen, float* outCol);

    /**
     * Converts a unicode string representation to an rgb colour
     *
     * @param str The string to be parsed
     * @param outR The variable receiving the red colour component (0..1)
     * @param outG The variable receiving the green colour component (0..1)
     * @param outB The variable receiving the blue colour component (0..1)
     *
     * @return 'true' on success, 'false' on failure
     */
    static inline bool FromString(const vislib::StringW& str, float& outR, float& outG, float& outB) {
        vislib::StringA strA(str);
        float a;
        return FromString(strA, outR, outG, outB, a);
    }

    /**
     * Converts a unicode string representation to an rgb colour
     *
     * @param str The string to be parsed
     * @param outR The variable receiving the red colour component (0..1)
     * @param outG The variable receiving the green colour component (0..1)
     * @param outB The variable receiving the blue colour component (0..1)
     * @param outA The variable receiving the alpha transparency (0..1)
     *
     * @return 'true' on success, 'false' on failure
     */
    static inline bool FromString(const vislib::StringW& str, float& outR, float& outG, float& outB, float& outA) {
        vislib::StringA strA(str);
        return FromString(strA, outR, outG, outB, outA);
    }

    /**
     * Converts a unicode string representation to an rgb colour
     *
     * @param str The string to be parsed
     * @param outColLen the number of components in the return vector
     * @param outCol a character vector receiving the parsed color components (0..255)
     *
     * @return 'true' on success, 'false' on failure
     */
    static inline bool FromString(const vislib::StringW& str, unsigned int outColLen, unsigned char* outCol) {
        vislib::StringA strA(str);
        return FromString(strA, outColLen, outCol);
    }

    /**
     * Converts a unicode string representation to an rgb colour
     *
     * @param str The string to be parsed
     * @param outColLen the number of components in the return vector
     * @param outCol a float vector receiving the parsed color components (0..1)
     *
     * @return 'true' on success, 'false' on failure
     */
    static inline bool FromString(const vislib::StringW& str, unsigned int outColLen, float* outCol) {
        vislib::StringA strA(str);
        return FromString(strA, outColLen, outCol);
    }

private:
    /**
     * Forbidden ctor
     */
    ColourParser();

    /**
     * Forbidden dtor
     */
    ~ColourParser();

    /**
     * Answers the numeric value of a hex character (0-15) or 255 in case
     * of an error.
     *
     * @param c The character to be interpreted
     *
     * @return The numeric value of 'c'
     */
    static inline unsigned char hexToNum(const char& c);
};


} // namespace megamol::core::utility

#endif /* MEGAMOLCORE_COLOURPARSER_H_INCLUDED */
