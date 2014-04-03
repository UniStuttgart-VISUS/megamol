/*
 * ColourParser.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/ColourParser.h"
#include "the/assert.h"
#include "vislib/mathfunctions.h"
#include "vislib/NamedColours.h"
#include "the/not_supported_exception.h"
#include "the/string.h"
#include "the/text/string_builder.h"
#include "the/assert.h"


/*
 * vislib::graphics::ColourParser::FromString
 */
void vislib::graphics::ColourParser::FromString(const the::astring& inStr,
        unsigned char &outR, unsigned char &outG, unsigned char &outB,
        unsigned char &outA, bool allowQuantization) {
    the::astring sb(inStr);
    the::text::string_utility::trim(sb);
    if (sb.empty()) {
        throw the::format_exception("inStr was empty", __FILE__, __LINE__);
    }

    // try parse named colour
    try {
        ColourRGBAu8 c = NamedColours::GetColourByName(sb.c_str());
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

    throw the::format_exception("Unexpected format", __FILE__, __LINE__);
}


/*
 * vislib::graphics::ColourParser::FromString
 */
void vislib::graphics::ColourParser::FromString(const the::astring& inStr,
        float &outR, float &outG, float &outB, float &outA) {
    the::astring sb(inStr);
    the::text::string_utility::trim(sb);
    if (sb.empty()) {
        throw the::format_exception("inStr was empty", __FILE__, __LINE__);
    }

    // try parse named colour
    try {
        ColourRGBAu8 c = NamedColours::GetColourByName(sb.c_str());
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

    throw the::format_exception("Unexpected format", __FILE__, __LINE__);
}


/*
 * vislib::graphics::ColourParser::ToString
 */
the::astring& vislib::graphics::ColourParser::ToString(
        the::astring& outStr, unsigned char inR, unsigned char inG,
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
            the::text::astring_builder::format_to(outStr, "(%u; %u; %u)", inR, inG, inB);
        } else {
            the::text::astring_builder::format_to(outStr, "(%u; %u; %u; %u)", inR, inG, inB, inA);
        }
        return outStr;
    }

    if ((repType & REPTYPE_HTML) == REPTYPE_HTML) {
        if (inA == 255) {
            the::text::astring_builder::format_to(outStr, "#%.2x%.2x%.2x", inR, inG, inB);
        } else {
             // non-standard, but i don't care
            the::text::astring_builder::format_to(outStr, "#%.2x%.2x%.2x%.2x", inR, inG, inB, inA);
        }
        return outStr;
    }

    if ((repType & REPTYPE_FLOAT) == REPTYPE_FLOAT) {
        if (inA == 255) {
            the::text::astring_builder::format_to(outStr, "(%f; %f; %f)",
                static_cast<float>(inR) / 255.0f,
                static_cast<float>(inG) / 255.0f,
                static_cast<float>(inB) / 255.0f);
        } else {
            the::text::astring_builder::format_to(outStr, "(%f; %f; %f; %f)",
                static_cast<float>(inR) / 255.0f,
                static_cast<float>(inG) / 255.0f,
                static_cast<float>(inB) / 255.0f,
                static_cast<float>(inA) / 255.0f);
        }
        return outStr;
    }

    throw the::format_exception(
        "Cannot generate string in requested representation type",
        __FILE__, __LINE__);
    return outStr;
}


/*
 * vislib::graphics::ColourParser::ToString
 */
the::astring& vislib::graphics::ColourParser::ToString(
        the::astring& outStr, float inR, float inG, float inB, float inA,
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
                the::text::astring_builder::format_to(outStr, "(%u; %u; %u)", ri, gi, bi);
            } else {
                the::text::astring_builder::format_to(outStr, "(%u; %u; %u; %u)", ri, gi, bi, ai);
            }
            return outStr;
        }

        if ((repType & REPTYPE_HTML) == REPTYPE_HTML) {
            if (ai == 255) {
                the::text::astring_builder::format_to(outStr, "#%.2x%.2x%.2x", ri, gi, bi);
            } else {
                 // non-standard, but i don't care
                the::text::astring_builder::format_to(outStr, "#%.2x%.2x%.2x%.2x", ri, gi, bi, ai);
            }
            return outStr;
        }

    }

    if ((repType & REPTYPE_FLOAT) == REPTYPE_FLOAT) {
        if (vislib::math::IsEqual(inA, 1.0f)) {
            the::text::astring_builder::format_to(outStr, "(%f; %f; %f)", inR, inG, inB);
        } else {
            the::text::astring_builder::format_to(outStr, "(%f; %f; %f; %f)", inR, inG, inB, inA);
        }
        return outStr;
    }

    throw the::format_exception(
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
bool vislib::graphics::ColourParser::parseArray(const the::astring& inStr,
        float &outR, float &outG, float &outB, float &outA) {
    THE_ASSERT(!inStr.empty());

    // remove any parentheses
    if (inStr[0] == '(') {
        the::astring::size_type len = inStr.size();
        if ((len > 2) && (inStr[len - 1] == ')')) {
            return parseArray(inStr.substr(1, len - 2),
                outR, outG, outB, outA);
        } else {
            return false;
        }
    } else if (inStr[0] == '{') {
        the::astring::size_type len = inStr.size();
        if ((len > 2) && (inStr[len - 1] == '}')) {
            return parseArray(inStr.substr(1, len - 2),
                outR, outG, outB, outA);
        } else {
            return false;
        }
    }

    // detect separators
    size_t scsc = the::text::string_utility::count(inStr, ';');
    size_t csc = the::text::string_utility::count(inStr, ',');

    the::astring rStr, gStr, bStr, aStr;
    if ((scsc == 3) || (scsc == 2)) {
        // separated by semicolon
        the::astring::size_type p1 = inStr.find(';');
        if (p1 == the::astring::npos) return false;
        rStr = inStr.substr(0, p1);
        the::text::string_utility::trim(rStr);
        p1++;
        the::astring::size_type p2 = inStr.find(';', p1);
        if (p2 == the::astring::npos) return false;
        gStr = inStr.substr(p1, p2 - p1);
        the::text::string_utility::trim(gStr);

        if (scsc == 3) {
            p2++;
            p1 = inStr.find(';', p2);
            if (p1 == the::astring::npos) return false;
            bStr = inStr.substr(p2, p1 - p2);
            the::text::string_utility::trim(bStr);
            aStr = inStr.substr(p1 + 1);
            the::text::string_utility::trim(aStr);

        } else {
            bStr = inStr.substr(p2 + 1);
            the::text::string_utility::trim(bStr);
            aStr.clear();
        }

    } else if (scsc == 0) {
        // semicolon cannot be part of int or float
        if ((csc == 3) || (csc == 2)) {
            // if float, they use dots (EN)
            the::astring::size_type p1 = inStr.find(',');
            if (p1 == the::astring::npos) return false;
            rStr = inStr.substr(0, p1);
            the::text::string_utility::trim(rStr);
            p1++;
            the::astring::size_type p2 = inStr.find(',', p1);
            if (p2 == the::astring::npos) return false;
            gStr = inStr.substr(p1, p2 - p1);
            the::text::string_utility::trim(gStr);

            if (scsc == 3) {
                p2++;
                p1 = inStr.find(',', p2);
                if (p1 == the::astring::npos) return false;
                bStr = inStr.substr(p2, p1 - p2);
                the::text::string_utility::trim(bStr);
                aStr = inStr.substr(p1 + 1);
                the::text::string_utility::trim(aStr);

            } else {
                bStr = inStr.substr(p2 + 1);
                the::text::string_utility::trim(bStr);
                aStr.clear();
            }

        } else {
            // colons for separation AND fraction :-/
            // (Activate super-power-fake heuristic)
            the::astring::size_type p1 = inStr.find(", ");

            if (p1 != the::astring::npos) {
                the::astring::size_type p2 = inStr.find(", ", p1 + 1);
                if (p2 != the::astring::npos) {
                    the::astring::size_type p3 = inStr.find(", ", p2 + 1);
                    rStr = inStr.substr(0, p1);
                    the::text::string_utility::trim(rStr);
                    gStr = inStr.substr(p1 + 1, p2 - (p1 + 1));
                    the::text::string_utility::trim(gStr);
                    if (p3 != the::astring::npos) {
                        bStr = inStr.substr(p2 + 1, p3 - (p2 + 1));
                        the::text::string_utility::trim(bStr);
                        aStr = inStr.substr(p3 + 1);
                        the::text::string_utility::trim(aStr);
                    } else {
                        bStr = inStr.substr(p2 + 1);
                        the::text::string_utility::trim(bStr);
                        aStr.clear();
                    }
                } else p1 = the::astring::npos;
            }

            if (p1 == the::astring::npos) {
                if (csc == 5) {
                    the::astring::size_type p1 = inStr.find(",");
                    if (p1 == the::astring::npos) return false;
                    p1 = inStr.find(",", p1 + 1);
                    if (p1 == the::astring::npos) return false;
                    the::astring::size_type p2 = inStr.find(",", p1 + 1);
                    if (p2 == the::astring::npos) return false;
                    p2 = inStr.find(",", p2 + 1);
                    if (p2 == the::astring::npos) return false;
                    rStr = inStr.substr(0, p1);
                    the::text::string_utility::trim(rStr);
                    gStr = inStr.substr(p1 + 1, p2 - (p1 + 1));
                    the::text::string_utility::trim(gStr);
                    bStr = inStr.substr(p2 + 1);
                    the::text::string_utility::trim(bStr);
                    aStr.clear();

                } else if (csc == 7) {
                    the::astring::size_type p1 = inStr.find(",");
                    if (p1 == the::astring::npos) return false;
                    p1 = inStr.find(",", p1 + 1);
                    if (p1 == the::astring::npos) return false;
                    the::astring::size_type p2 = inStr.find(",", p1 + 1);
                    if (p2 == the::astring::npos) return false;
                    p2 = inStr.find(",", p2 + 1);
                    if (p2 == the::astring::npos) return false;
                    the::astring::size_type p3 = inStr.find(",", p2 + 1);
                    if (p3 == the::astring::npos) return false;
                    p3 = inStr.find(",", p3 + 1);
                    if (p3 == the::astring::npos) return false;
                    rStr = inStr.substr(0, p1);
                    the::text::string_utility::trim(rStr);
                    gStr = inStr.substr(p1 + 1, p2 - (p1 + 1));
                    the::text::string_utility::trim(gStr);
                    bStr = inStr.substr(p2 + 1, p3 - (p2 + 1));
                    the::text::string_utility::trim(bStr);
                    aStr = inStr.substr(p3 + 1);
                    the::text::string_utility::trim(aStr);
                }
            }
        }

    } else {
        return false; // no separators at all
    }

    if (rStr.empty() || gStr.empty() || bStr.empty()) return false;

    bool isFloat = false;

    if ((the::text::string_utility::contains(rStr, '.') || the::text::string_utility::contains(gStr, '.') || the::text::string_utility::contains(bStr, '.')) && (aStr.empty() || the::text::string_utility::contains(aStr, '.'))) isFloat = true;
    if (the::text::string_utility::contains(rStr, ',')) {
        the::text::string_utility::replace(rStr, ',', '.');
        isFloat = true;
    }
    if (the::text::string_utility::contains(gStr, ',')) {
        the::text::string_utility::replace(gStr, ',', '.');
        isFloat = true;
    }
    if (the::text::string_utility::contains(bStr, ',')) {
        the::text::string_utility::replace(bStr, ',', '.');
        isFloat = true;
    }
    if (!aStr.empty() && the::text::string_utility::contains(aStr, ',')) {
        the::text::string_utility::replace(aStr, ',', '.');
        isFloat = true;
    }

    float r, g, b, a;
    try {
        r = static_cast<float>(the::text::string_utility::parse_double(rStr));
    } catch(...) {
        return false;
    }
    try {
        g = static_cast<float>(the::text::string_utility::parse_double(gStr));
    } catch(...) {
        return false;
    }
    try {
        b = static_cast<float>(the::text::string_utility::parse_double(bStr));
    } catch(...) {
        return false;
    }
    if (aStr.empty()) {
        a = isFloat ? 1.0f : 255.0f;
    } else {
        try {
            a = static_cast<float>(the::text::string_utility::parse_double(aStr));
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
        const the::astring& inStr) {
    vislib::graphics::ColourRGBAu8 col;
    unsigned char c, b;
    THE_ASSERT(!inStr.empty() && (inStr[0] == '#'));

    if (inStr.size() == 4) { // short RGB
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

    } else if (inStr.size() == 5) { // short RGBA
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

    } else if (inStr.size() == 7) { // normal RGB
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

    } else if (inStr.size() == 9) { // normal RGBA
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
        throw the::format_exception(
            "Illegal length of HTML input", __FILE__, __LINE__);
    }

    if (c > 15) {
        throw the::format_exception(
            "Illegal character in HTML input", __FILE__, __LINE__);
    }

    return col;
}


/*
 * vislib::graphics::ColourParser::ColourParser
 */
vislib::graphics::ColourParser::ColourParser(void) {
    throw the::not_supported_exception("ColourParser::ctor",
        __FILE__, __LINE__);
}


/*
 * vislib::graphics::ColourParser::~ColourParser
 */
vislib::graphics::ColourParser::~ColourParser(void) {
    throw the::not_supported_exception("ColourParser::dtor",
        __FILE__, __LINE__);
}
