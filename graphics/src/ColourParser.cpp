/*
 * ColourParser.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/ColourParser.h"
#include "vislib/assert.h"
#include "vislib/mathfunctions.h"
#include "vislib/NamedColours.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::graphics::ColourParser::FromString
 */
void vislib::graphics::ColourParser::FromString(const vislib::StringA& inStr,
        unsigned char &outR, unsigned char &outG, unsigned char &outB,
        unsigned char &outA, bool allowQuantization) {
    vislib::StringA sb(inStr);
    sb.TrimSpaces();
    if (sb.IsEmpty()) {
        throw vislib::FormatException("inStr was empty", __FILE__, __LINE__);
    }

    // try parse named colour
    try {
        ColourRGBAu8 c = NamedColours::GetColourByName(sb);
        outR = c.R();
        outG = c.G();
        outB = c.B();
        outA = c.A();
        return;
    } catch(...) {
    }

    // try parse HTML colour
    if (sb[0] == '#') {
        ColourRGBAu8 c = parseHTML(sb);
        outR = c.R();
        outG = c.G();
        outB = c.B();
        outA = c.A();
        return;
    }

    // try parse array
    float r, g, b, a;
    if (parseArray(inStr, r, g, b, a)) {
        r *= 255.0f;
        g *= 255.0f;
        b *= 255.0f;
        a *= 255.0f;
        int ri = static_cast<int>(r + 0.5f);
        int gi = static_cast<int>(g + 0.5f);
        int bi = static_cast<int>(b + 0.5f);
        int ai = static_cast<int>(a + 0.5f);
        const float eps = 0.0001f;

        if ((vislib::math::IsEqual(static_cast<float>(ri), r, eps)
            && vislib::math::IsEqual(static_cast<float>(gi), g, eps)
            && vislib::math::IsEqual(static_cast<float>(bi), b, eps)
            && vislib::math::IsEqual(static_cast<float>(ai), a, eps))
                || allowQuantization) {
            outR = ri;
            outG = gi;
            outB = bi;
            outA = ai;
            return;
        }
    }

    throw vislib::FormatException("Unexpected format", __FILE__, __LINE__);
}


/*
 * vislib::graphics::ColourParser::FromString
 */
void vislib::graphics::ColourParser::FromString(const vislib::StringA& inStr,
        float &outR, float &outG, float &outB, float &outA) {
    vislib::StringA sb(inStr);
    sb.TrimSpaces();
    if (sb.IsEmpty()) {
        throw vislib::FormatException("inStr was empty", __FILE__, __LINE__);
    }

    // try parse named colour
    try {
        ColourRGBAu8 c = NamedColours::GetColourByName(sb);
        outR = static_cast<float>(c.R()) / 255.0f;
        outG = static_cast<float>(c.G()) / 255.0f;
        outB = static_cast<float>(c.B()) / 255.0f;
        outA = static_cast<float>(c.A()) / 255.0f;
        return;
    } catch(...) {
    }

    // try parse HTML colour
    if (sb[0] == '#') {
        ColourRGBAu8 c = parseHTML(sb);
        outR = static_cast<float>(c.R()) / 255.0f;
        outG = static_cast<float>(c.G()) / 255.0f;
        outB = static_cast<float>(c.B()) / 255.0f;
        outA = static_cast<float>(c.A()) / 255.0f;
        return;
    }

    // try parse array
    float r, g, b, a;
    if (parseArray(inStr, r, g, b, a)) {
        outR = r;
        outG = g;
        outB = b;
        outA = a;
        return;
    }

    throw vislib::FormatException("Unexpected format", __FILE__, __LINE__);
}


/*
 * vislib::graphics::ColourParser::ToString
 */
vislib::StringA& vislib::graphics::ColourParser::ToString(
        vislib::StringA& outStr, unsigned char inR, unsigned char inG,
        unsigned char inB, unsigned char inA, int repType) {

    if ((repType & REPTYPE_NAMED) == REPTYPE_NAMED) {
        const char *name = NamedColours::GetNameByColour(
            ColourRGBAu8(inR, inG, inB, inA), false);
        if (name != NULL) {
            outStr = name;
            return outStr;
        }
    }

    if ((repType & REPTYPE_BYTE) == REPTYPE_BYTE) {
        if (inA == 255) {
            outStr.Format("(%u; %u; %u)", inR, inG, inB);
        } else {
            outStr.Format("(%u; %u; %u; %u)", inR, inG, inB, inA);
        }
        return outStr;
    }

    if ((repType & REPTYPE_HTML) == REPTYPE_HTML) {
        if (inA == 255) {
            outStr.Format("#%.2x%.2x%.2x", inR, inG, inB);
        } else {
             // non-standard, but i don't care
            outStr.Format("#%.2x%.2x%.2x%.2x", inR, inG, inB, inA);
        }
        return outStr;
    }

    if ((repType & REPTYPE_FLOAT) == REPTYPE_FLOAT) {
        if (inA == 255) {
            outStr.Format("(%f; %f; %f)",
                static_cast<float>(inR) / 255.0f,
                static_cast<float>(inG) / 255.0f,
                static_cast<float>(inB) / 255.0f);
        } else {
            outStr.Format("(%f; %f; %f; %f)",
                static_cast<float>(inR) / 255.0f,
                static_cast<float>(inG) / 255.0f,
                static_cast<float>(inB) / 255.0f,
                static_cast<float>(inA) / 255.0f);
        }
        return outStr;
    }

    throw vislib::FormatException(
        "Cannot generate string in requested representation type",
        __FILE__, __LINE__);
    return outStr;
}


/*
 * vislib::graphics::ColourParser::ToString
 */
vislib::StringA& vislib::graphics::ColourParser::ToString(
        vislib::StringA& outStr, float inR, float inG, float inB, float inA,
        int repType) {

    const float eps = 0.0001f;
    int ri = static_cast<int>(inR * 255.0f + 0.5f);
    int gi = static_cast<int>(inG * 255.0f + 0.5f);
    int bi = static_cast<int>(inB * 255.0f + 0.5f);
    int ai = static_cast<int>(inA * 255.0f + 0.5f);
    if ((vislib::math::IsEqual(static_cast<float>(ri) / 255.0f, inR, eps)
        && vislib::math::IsEqual(static_cast<float>(gi) / 255.0f, inG, eps)
        && vislib::math::IsEqual(static_cast<float>(bi) / 255.0f, inB, eps)
        && vislib::math::IsEqual(static_cast<float>(ai) / 255.0f, inA, eps))
            || ((repType & REPTYPE_QUANT) == REPTYPE_QUANT)) {

        if ((repType & REPTYPE_NAMED) == REPTYPE_NAMED) {
            const char *name = NamedColours::GetNameByColour(
                ColourRGBAu8(ri, gi, bi, ai), false);
            if (name != NULL) {
                outStr = name;
                return outStr;
            }
        }

        if ((repType & REPTYPE_BYTE) == REPTYPE_BYTE) {
            if (ai == 255) {
                outStr.Format("(%u; %u; %u)", ri, gi, bi);
            } else {
                outStr.Format("(%u; %u; %u; %u)", ri, gi, bi, ai);
            }
            return outStr;
        }

        if ((repType & REPTYPE_HTML) == REPTYPE_HTML) {
            if (ai == 255) {
                outStr.Format("#%.2x%.2x%.2x", ri, gi, bi);
            } else {
                 // non-standard, but i don't care
                outStr.Format("#%.2x%.2x%.2x%.2x", ri, gi, bi, ai);
            }
            return outStr;
        }

    }

    if ((repType & REPTYPE_FLOAT) == REPTYPE_FLOAT) {
        if (vislib::math::IsEqual(inA, 1.0f)) {
            outStr.Format("(%f; %f; %f)", inR, inG, inB);
        } else {
            outStr.Format("(%f; %f; %f; %f)", inR, inG, inB, inA);
        }
        return outStr;
    }

    throw vislib::FormatException(
        "Cannot generate string in requested representation type",
        __FILE__, __LINE__);
    return outStr;
}


/*
 * vislib::graphics::ColourParser::hexToNum
 */
unsigned char vislib::graphics::ColourParser::hexToNum(const char& c) {
    if ((c >= '0') && (c <= '9')) return c - '0';
    if ((c >= 'a') && (c <= 'f')) return 10 + c - 'a';
    if ((c >= 'A') && (c <= 'F')) return 10 + c - 'A';
    return 255;
}


/*
 * vislib::graphics::ColourParser::parseArray
 */
bool vislib::graphics::ColourParser::parseArray(const vislib::StringA& inStr,
        float &outR, float &outG, float &outB, float &outA) {
    ASSERT(!inStr.IsEmpty());

    // remove any parentheses
    if (inStr[0] == '(') {
        vislib::StringA::Size len = inStr.Length();
        if ((len > 2) && (inStr[len - 1] == ')')) {
            return parseArray(inStr.Substring(1, len - 2),
                outR, outG, outB, outA);
        } else {
            return false;
        }
    } else if (inStr[0] == '{') {
        vislib::StringA::Size len = inStr.Length();
        if ((len > 2) && (inStr[len - 1] == '}')) {
            return parseArray(inStr.Substring(1, len - 2),
                outR, outG, outB, outA);
        } else {
            return false;
        }
    }

    // detect separators
    SIZE_T scsc = inStr.Count(';');
    SIZE_T csc = inStr.Count(',');

    vislib::StringA rStr, gStr, bStr, aStr;
    if ((scsc == 3) || (scsc == 2)) {
        // separated by semicolon
        vislib::StringA::Size p1 = inStr.Find(';');
        if (p1 == vislib::StringA::INVALID_POS) return false;
        rStr = inStr.Substring(0, p1);
        rStr.TrimSpaces();
        p1++;
        vislib::StringA::Size p2 = inStr.Find(';', p1);
        if (p2 == vislib::StringA::INVALID_POS) return false;
        gStr = inStr.Substring(p1, p2 - p1);
        gStr.TrimSpaces();

        if (scsc == 3) {
            p2++;
            p1 = inStr.Find(';', p2);
            if (p1 == vislib::StringA::INVALID_POS) return false;
            bStr = inStr.Substring(p2, p1 - p2);
            bStr.TrimSpaces();
            aStr = inStr.Substring(p1 + 1);
            aStr.TrimSpaces();

        } else {
            bStr = inStr.Substring(p2 + 1);
            bStr.TrimSpaces();
            aStr.Clear();
        }

    } else if (scsc == 0) {
        // semicolon cannot be part of int or float
        if ((csc == 3) || (csc == 2)) {
            // if float, they use dots (EN)
            vislib::StringA::Size p1 = inStr.Find(',');
            if (p1 == vislib::StringA::INVALID_POS) return false;
            rStr = inStr.Substring(0, p1);
            rStr.TrimSpaces();
            p1++;
            vislib::StringA::Size p2 = inStr.Find(',', p1);
            if (p2 == vislib::StringA::INVALID_POS) return false;
            gStr = inStr.Substring(p1, p2 - p1);
            gStr.TrimSpaces();

            if (scsc == 3) {
                p2++;
                p1 = inStr.Find(',', p2);
                if (p1 == vislib::StringA::INVALID_POS) return false;
                bStr = inStr.Substring(p2, p1 - p2);
                bStr.TrimSpaces();
                aStr = inStr.Substring(p1 + 1);
                aStr.TrimSpaces();

            } else {
                bStr = inStr.Substring(p2 + 1);
                bStr.TrimSpaces();
                aStr.Clear();
            }

        } else {
            // colons for separation AND fraction :-/
            // (Activate super-power-fake heuristic)
            vislib::StringA::Size p1 = inStr.Find(", ");

            if (p1 != vislib::StringA::INVALID_POS) {
                vislib::StringA::Size p2 = inStr.Find(", ", p1 + 1);
                if (p2 != vislib::StringA::INVALID_POS) {
                    vislib::StringA::Size p3 = inStr.Find(", ", p2 + 1);
                    rStr = inStr.Substring(0, p1);
                    rStr.TrimSpaces();
                    gStr = inStr.Substring(p1 + 1, p2 - (p1 + 1));
                    gStr.TrimSpaces();
                    if (p3 != vislib::StringA::INVALID_POS) {
                        bStr = inStr.Substring(p2 + 1, p3 - (p2 + 1));
                        bStr.TrimSpaces();
                        aStr = inStr.Substring(p3 + 1);
                        aStr.TrimSpaces();
                    } else {
                        bStr = inStr.Substring(p2 + 1);
                        bStr.TrimSpaces();
                        aStr.Clear();
                    }
                } else p1 = vislib::StringA::INVALID_POS;
            }

            if (p1 == vislib::StringA::INVALID_POS) {
                if (csc == 5) {
                    vislib::StringA::Size p1 = inStr.Find(",");
                    if (p1 == vislib::StringA::INVALID_POS) return false;
                    p1 = inStr.Find(",", p1 + 1);
                    if (p1 == vislib::StringA::INVALID_POS) return false;
                    vislib::StringA::Size p2 = inStr.Find(",", p1 + 1);
                    if (p2 == vislib::StringA::INVALID_POS) return false;
                    p2 = inStr.Find(",", p2 + 1);
                    if (p2 == vislib::StringA::INVALID_POS) return false;
                    rStr = inStr.Substring(0, p1);
                    rStr.TrimSpaces();
                    gStr = inStr.Substring(p1 + 1, p2 - (p1 + 1));
                    gStr.TrimSpaces();
                    bStr = inStr.Substring(p2 + 1);
                    bStr.TrimSpaces();
                    aStr.Clear();

                } else if (csc == 7) {
                    vislib::StringA::Size p1 = inStr.Find(",");
                    if (p1 == vislib::StringA::INVALID_POS) return false;
                    p1 = inStr.Find(",", p1 + 1);
                    if (p1 == vislib::StringA::INVALID_POS) return false;
                    vislib::StringA::Size p2 = inStr.Find(",", p1 + 1);
                    if (p2 == vislib::StringA::INVALID_POS) return false;
                    p2 = inStr.Find(",", p2 + 1);
                    if (p2 == vislib::StringA::INVALID_POS) return false;
                    vislib::StringA::Size p3 = inStr.Find(",", p2 + 1);
                    if (p3 == vislib::StringA::INVALID_POS) return false;
                    p3 = inStr.Find(",", p3 + 1);
                    if (p3 == vislib::StringA::INVALID_POS) return false;
                    rStr = inStr.Substring(0, p1);
                    rStr.TrimSpaces();
                    gStr = inStr.Substring(p1 + 1, p2 - (p1 + 1));
                    gStr.TrimSpaces();
                    bStr = inStr.Substring(p2 + 1, p3 - (p2 + 1));
                    bStr.TrimSpaces();
                    aStr = inStr.Substring(p3 + 1);
                    aStr.TrimSpaces();
                }
            }
        }

    } else {
        return false; // no separators at all
    }

    if (rStr.IsEmpty() || gStr.IsEmpty() || bStr.IsEmpty()) return false;

    bool isFloat = false;

    if ((rStr.Contains('.') || gStr.Contains('.') || bStr.Contains('.')) && (aStr.IsEmpty() || aStr.Contains('.'))) isFloat = true;
    if (rStr.Contains(',')) {
        rStr.Replace(',', '.');
        isFloat = true;
    }
    if (gStr.Contains(',')) {
        gStr.Replace(',', '.');
        isFloat = true;
    }
    if (bStr.Contains(',')) {
        bStr.Replace(',', '.');
        isFloat = true;
    }
    if (!aStr.IsEmpty() && aStr.Contains(',')) {
        aStr.Replace(',', '.');
        isFloat = true;
    }

    float r, g, b, a;
    try {
        r = static_cast<float>(vislib::CharTraitsA::ParseDouble(rStr));
    } catch(...) {
        return false;
    }
    try {
        g = static_cast<float>(vislib::CharTraitsA::ParseDouble(gStr));
    } catch(...) {
        return false;
    }
    try {
        b = static_cast<float>(vislib::CharTraitsA::ParseDouble(bStr));
    } catch(...) {
        return false;
    }
    if (aStr.IsEmpty()) {
        a = isFloat ? 1.0f : 255.0f;
    } else {
        try {
            a = static_cast<float>(vislib::CharTraitsA::ParseDouble(aStr));
        } catch(...) {
            return false;
        }
    }

    if (!isFloat) {
        r /= 255.0f;
        g /= 255.0f;
        b /= 255.0f;
        a /= 255.0f;
    }

    outR = r;
    outG = g;
    outB = b;
    outA = a;

    return true;
}


/*
 * vislib::graphics::ColourParser::parseHTML
 */
vislib::graphics::ColourRGBAu8 vislib::graphics::ColourParser::parseHTML(
        const vislib::StringA& inStr) {
    vislib::graphics::ColourRGBAu8 col;
    unsigned char c, b;
    ASSERT(!inStr.IsEmpty() && (inStr[0] == '#'));

    if (inStr.Length() == 4) { // short RGB
        c = hexToNum(inStr[1]);
        if (c < 16) {
            col.SetR(c * 16 + c);
            c = hexToNum(inStr[2]);
        }
        if (c < 16) {
            col.SetG(c * 16 + c);
            c = hexToNum(inStr[3]);
        }
        if (c < 16) {
            col.SetB(c * 16 + c);
            col.SetA(255);
        }

    } else if (inStr.Length() == 5) { // short RGBA
        c = hexToNum(inStr[1]);
        if (c < 16) {
            col.SetR(c * 16 + c);
            c = hexToNum(inStr[2]);
        }
        if (c < 16) {
            col.SetG(c * 16 + c);
            c = hexToNum(inStr[3]);
        }
        if (c < 16) {
            col.SetB(c * 16 + c);
            c = hexToNum(inStr[4]);
            if (c > 15) c = 15;
            col.SetA(c * 16 + c);
        }

    } else if (inStr.Length() == 7) { // normal RGB
        c = hexToNum(inStr[1]);
        if (c < 16) {
            b = c * 16;
            c = hexToNum(inStr[2]);
        }
        if (c < 16) {
            col.SetR(b + c);
            c = hexToNum(inStr[3]);
        }
        if (c < 16) {
            b = c * 16;
            c = hexToNum(inStr[4]);
        }
        if (c < 16) {
            col.SetG(b + c);
            c = hexToNum(inStr[5]);
        }
        if (c < 16) {
            b = c * 16;
            c = hexToNum(inStr[6]);
        }
        if (c < 16) {
            col.SetB(b + c);
            col.SetA(255);
        }

    } else if (inStr.Length() == 9) { // normal RGBA
        c = hexToNum(inStr[1]);
        if (c < 16) {
            b = c * 16;
            c = hexToNum(inStr[2]);
        }
        if (c < 16) {
            col.SetR(b + c);
            c = hexToNum(inStr[3]);
        }
        if (c < 16) {
            b = c * 16;
            c = hexToNum(inStr[4]);
        }
        if (c < 16) {
            col.SetG(b + c);
            c = hexToNum(inStr[5]);
        }
        if (c < 16) {
            b = c * 16;
            c = hexToNum(inStr[6]);
        }
        if (c < 16) {
            col.SetB(b + c);
            c = hexToNum(inStr[5]);
            if (c < 16) {
                b = c * 16;
                c = hexToNum(inStr[6]);
                if (c > 16) {
                    b = 255;
                    c = 0;
                }
            } else {
                b = 255;
                c = 0;
            }
            col.SetA(b + c);
        }

    } else {
        throw vislib::FormatException(
            "Illegal length of HTML input", __FILE__, __LINE__);
    }

    if (c > 15) {
        throw vislib::FormatException(
            "Illegal character in HTML input", __FILE__, __LINE__);
    }

    return col;
}


/*
 * vislib::graphics::ColourParser::ColourParser
 */
vislib::graphics::ColourParser::ColourParser(void) {
    throw vislib::UnsupportedOperationException("ColourParser::ctor",
        __FILE__, __LINE__);
}


/*
 * vislib::graphics::ColourParser::~ColourParser
 */
vislib::graphics::ColourParser::~ColourParser(void) {
    throw vislib::UnsupportedOperationException("ColourParser::dtor",
        __FILE__, __LINE__);
}
