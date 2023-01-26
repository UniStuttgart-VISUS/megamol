/*
 * ColourParser.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/utility/ColourParser.h"
#include "vislib/String.h"
#include "vislib/StringTokeniser.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/types.h"

using namespace megamol::core;


namespace megamol {
namespace core {
namespace utility {

/**
 * Named colour
 */
typedef struct _namedcolour_t {

    /** Name of the colour */
    const char* name;

    /** The colour rgb code: 0x00RRGGBB */
    DWORD colour;

} NamedColour;

/*
 * namedColours
 */
static NamedColour namedColours[] = {
    /* .Net named colours */
    {"AliceBlue", 0x00F0F8FF}, {"AntiqueWhite", 0x00FAEBD7}, {"Aqua", 0x0000FFFF}, {"Aquamarine", 0x007FFFD4},
    {"Azure", 0x00F0FFFF}, {"Beige", 0x00F5F5DC}, {"Bisque", 0x00FFE4C4}, {"Black", 0x00000000},
    {"BlanchedAlmond", 0x00FFEBCD}, {"Blue", 0x000000FF}, {"BlueViolet", 0x008A2BE2}, {"Brown", 0x00A52A2A},
    {"BurlyWood", 0x00DEB887}, {"CadetBlue", 0x005F9EA0}, {"Chartreuse", 0x007FFF00}, {"Chocolate", 0x00D2691E},
    {"Coral", 0x00FF7F50}, {"CornflowerBlue", 0x006495ED}, {"Cornsilk", 0x00FFF8DC}, {"Crimson", 0x00DC143C},
    {"Cyan", 0x0000FFFF}, {"DarkBlue", 0x0000008B}, {"DarkCyan", 0x00008B8B}, {"DarkGoldenrod", 0x00B8860B},
    {"DarkGray", 0x00A9A9A9}, {"DarkGrey", 0x00A9A9A9}, {"DarkGreen", 0x00006400}, {"DarkKhaki", 0x00BDB76B},
    {"DarkMagenta", 0x008B008B}, {"DarkOliveGreen", 0x00556B2F}, {"DarkOrange", 0x00FF8C00}, {"DarkOrchid", 0x009932CC},
    {"DarkRed", 0x008B0000}, {"DarkSalmon", 0x00E9967A}, {"DarkSeaGreen", 0x008FBC8B}, {"DarkSlateBlue", 0x00483D8B},
    {"DarkSlateGray", 0x002F4F4F}, {"DarkSlateGrey", 0x002F4F4F}, {"DarkTurquoise", 0x0000CED1},
    {"DarkViolet", 0x009400D3}, {"DeepPink", 0x00FF1493}, {"DeepSkyBlue", 0x0000BFFF}, {"DimGray", 0x00696969},
    {"DimGrey", 0x00696969}, {"DodgerBlue", 0x001E90FF}, {"Firebrick", 0x00B22222}, {"FloralWhite", 0x00FFFAF0},
    {"ForestGreen", 0x00228B22}, {"Fuchsia", 0x00FF00FF}, {"Gainsboro", 0x00DCDCDC}, {"GhostWhite", 0x00F8F8FF},
    {"Gold", 0x00FFD700}, {"Goldenrod", 0x00DAA520}, {"Gray", 0x00808080}, {"Grey", 0x00808080}, {"Green", 0x00008000},
    {"GreenYellow", 0x00ADFF2F}, {"Honeydew", 0x00F0FFF0}, {"HotPink", 0x00FF69B4}, {"IndianRed", 0x00CD5C5C},
    {"Indigo", 0x004B0082}, {"Ivory", 0x00FFFFF0}, {"Khaki", 0x00F0E68C}, {"Lavender", 0x00E6E6FA},
    {"LavenderBlush", 0x00FFF0F5}, {"LawnGreen", 0x007CFC00}, {"LemonChiffon", 0x00FFFACD}, {"LightBlue", 0x00ADD8E6},
    {"LightCoral", 0x00F08080}, {"LightCyan", 0x00E0FFFF}, {"LightGoldenrodYellow", 0x00FAFAD2},
    {"LightGray", 0x00D3D3D3}, {"LightGrey", 0x00D3D3D3}, {"LightGreen", 0x0090EE90}, {"LightPink", 0x00FFB6C1},
    {"LightSalmon", 0x00FFA07A}, {"LightSeaGreen", 0x0020B2AA}, {"LightSkyBlue", 0x0087CEFA},
    {"LightSlateGray", 0x00778899}, {"LightSlateGrey", 0x00778899}, {"LightSteelBlue", 0x00B0C4DE},
    {"LightYellow", 0x00FFFFE0}, {"Lime", 0x0000FF00}, {"LimeGreen", 0x0032CD32}, {"Linen", 0x00FAF0E6},
    {"Magenta", 0x00FF00FF}, {"Maroon", 0x00800000}, {"MediumAquamarine", 0x0066CDAA}, {"MediumBlue", 0x000000CD},
    {"MediumOrchid", 0x00BA55D3}, {"MediumPurple", 0x009370DB}, {"MediumSeaGreen", 0x003CB371},
    {"MediumSlateBlue", 0x007B68EE}, {"MediumSpringGreen", 0x0000FA9A}, {"MediumTurquoise", 0x0048D1CC},
    {"MediumVioletRed", 0x00C71585}, {"MidnightBlue", 0x00191970}, {"MintCream", 0x00F5FFFA}, {"MistyRose", 0x00FFE4E1},
    {"Moccasin", 0x00FFE4B5}, {"NavajoWhite", 0x00FFDEAD}, {"Navy", 0x00000080}, {"OldLace", 0x00FDF5E6},
    {"Olive", 0x00808000}, {"OliveDrab", 0x006B8E23}, {"Orange", 0x00FFA500}, {"OrangeRed", 0x00FF4500},
    {"Orchid", 0x00DA70D6}, {"PaleGoldenrod", 0x00EEE8AA}, {"PaleGreen", 0x0098FB98}, {"PaleTurquoise", 0x00AFEEEE},
    {"PaleVioletRed", 0x00DB7093}, {"PapayaWhip", 0x00FFEFD5}, {"PeachPuff", 0x00FFDAB9}, {"Peru", 0x00CD853F},
    {"Pink", 0x00FFC0CB}, {"Plum", 0x00DDA0DD}, {"PowderBlue", 0x00B0E0E6}, {"Purple", 0x00800080}, {"Red", 0x00FF0000},
    {"RosyBrown", 0x00BC8F8F}, {"RoyalBlue", 0x004169E1}, {"SaddleBrown", 0x008B4513}, {"Salmon", 0x00FA8072},
    {"SandyBrown", 0x00F4A460}, {"SeaGreen", 0x002E8B57}, {"SeaShell", 0x00FFF5EE}, {"Sienna", 0x00A0522D},
    {"Silver", 0x00C0C0C0}, {"SkyBlue", 0x0087CEEB}, {"SlateBlue", 0x006A5ACD}, {"SlateGray", 0x00708090},
    {"SlateGrey", 0x00708090}, {"Snow", 0x00FFFAFA}, {"SpringGreen", 0x0000FF7F}, {"SteelBlue", 0x004682B4},
    {"Tan", 0x00D2B48C}, {"Teal", 0x00008080}, {"Thistle", 0x00D8BFD8}, {"Tomato", 0x00FF6347},
    {"Transparent", 0x00FFFFFF}, {"Turquoise", 0x0040E0D0}, {"Violet", 0x00EE82EE}, {"Wheat", 0x00F5DEB3},
    {"White", 0x00FFFFFF}, {"WhiteSmoke", 0x00F5F5F5}, {"Yellow", 0x00FFFF00}, {"YellowGreen", 0x009ACD32},
    /* additional named colours */
    {"MegaMolBlue", 0x0080C0FF},
    /* end of list guard */
    {NULL, 0x00000000}};

} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */


/*
 * utility::ColourParser::ToStringA
 */
vislib::StringA utility::ColourParser::ToStringA(float r, float g, float b, float a) {
    vislib::StringA rv;
    if ((r > -vislib::math::FLOAT_EPSILON) && (r < 1.0f + vislib::math::FLOAT_EPSILON) &&
        (g > -vislib::math::FLOAT_EPSILON) && (g < 1.0f + vislib::math::FLOAT_EPSILON) &&
        (b > -vislib::math::FLOAT_EPSILON) && (b < 1.0f + vislib::math::FLOAT_EPSILON) &&
        (a > -vislib::math::FLOAT_EPSILON) && (a < 1.0f + vislib::math::FLOAT_EPSILON)) {
        // colour can be represented with bytes
        unsigned char r8 = static_cast<unsigned char>(vislib::math::Clamp(static_cast<int>(r * 255.0f + 0.5f), 0, 255));
        unsigned char g8 = static_cast<unsigned char>(vislib::math::Clamp(static_cast<int>(g * 255.0f + 0.5f), 0, 255));
        unsigned char b8 = static_cast<unsigned char>(vislib::math::Clamp(static_cast<int>(b * 255.0f + 0.5f), 0, 255));
        unsigned char a8 = static_cast<unsigned char>(vislib::math::Clamp(static_cast<int>(a * 255.0f + 0.5f), 0, 255));

        if (a8 == 255) {
            // rgb colour

            // named colour:
            //DWORD cc = (static_cast<DWORD>(r8) << 16)
            //    + (static_cast<DWORD>(g8) << 8) + static_cast<DWORD>(b8);
            //for (int i = 0; namedColours[i].name != NULL; i++) {
            //    if (namedColours[i].colour == cc) {
            //        return namedColours[i].name;
            //    }
            //}

            // html colour
            rv.Format("#%.2x%.2x%.2x", r8, g8, b8);
        } else {
            // rgba colour
            rv.Format("#%.2x%.2x%.2x%.2x", r8, g8, b8, a8);
        }
    } else {
        // hdr colour
        if (vislib::math::IsEqual(a, 1.0f)) {
            rv.Format("Colour(%f; %f; %f)", r, g, b);
        } else {
            rv.Format("Colour(%f; %f; %f; %f)", r, g, b, a);
        }
    }
    return rv;
}


/*
 * utility::ColourParser::FromString
 */
bool utility::ColourParser::FromString(const vislib::StringA& str, float& outR, float& outG, float& outB, float& outA) {
    vislib::StringA s(str);
    s.TrimSpaces();
    if (s.IsEmpty()) {
        return false;
    }

    if (s[0] == '#') {
        // byte colour
        int cnt = 0;
        int len = s.Length();
        for (int i = 1; i < len; i++) {
            if (((s[i] >= '0') && (s[i] <= '9')) || ((s[i] >= 'a') && (s[i] <= 'f')) ||
                ((s[i] >= 'A') && (s[i] <= 'F'))) {
                cnt++;
            } else {
                break;
            }
        }
        if (cnt >= 8) {
            // rgba
            outR = static_cast<float>((hexToNum(s[1]) << 4) + hexToNum(s[2])) / 255.0f;
            outG = static_cast<float>((hexToNum(s[3]) << 4) + hexToNum(s[4])) / 255.0f;
            outB = static_cast<float>((hexToNum(s[5]) << 4) + hexToNum(s[6])) / 255.0f;
            outA = static_cast<float>((hexToNum(s[7]) << 4) + hexToNum(s[8])) / 255.0f;
            return true;

        } else if (cnt >= 6) {
            // rgb
            outR = static_cast<float>((hexToNum(s[1]) << 4) + hexToNum(s[2])) / 255.0f;
            outG = static_cast<float>((hexToNum(s[3]) << 4) + hexToNum(s[4])) / 255.0f;
            outB = static_cast<float>((hexToNum(s[5]) << 4) + hexToNum(s[6])) / 255.0f;
            outA = 1.0f;
            return true;

        } else if (cnt >= 4) {
            // short rgba
            outR = static_cast<float>(hexToNum(s[1])) / 15.0f;
            outG = static_cast<float>(hexToNum(s[2])) / 15.0f;
            outB = static_cast<float>(hexToNum(s[3])) / 15.0f;
            outA = static_cast<float>(hexToNum(s[4])) / 15.0f;
            return true;

        } else if (cnt == 3) {
            // short rgb
            outR = static_cast<float>(hexToNum(s[1])) / 15.0f;
            outG = static_cast<float>(hexToNum(s[2])) / 15.0f;
            outB = static_cast<float>(hexToNum(s[3])) / 15.0f;
            outA = 1.0f;
            return true;
        }

    } else if (s.StartsWithInsensitive("Colour(")) {
        // hdr colour
        vislib::StringA::Size pos = s.Find(')');
        if (pos == vislib::StringA::INVALID_POS) {
            pos = s.Length();
        }
        vislib::Array<vislib::StringA> tokens = vislib::StringTokeniserA::Split(s.Substring(7, pos - 7), ";", true);
        try {
            if (tokens.Count() >= 4) {
                // rgba
                outR = static_cast<float>(vislib::CharTraitsA::ParseDouble(tokens[0]));
                outG = static_cast<float>(vislib::CharTraitsA::ParseDouble(tokens[1]));
                outB = static_cast<float>(vislib::CharTraitsA::ParseDouble(tokens[2]));
                outA = static_cast<float>(vislib::CharTraitsA::ParseDouble(tokens[3]));
                return true;

            } else if (tokens.Count() >= 3) {
                // rgb
                outR = static_cast<float>(vislib::CharTraitsA::ParseDouble(tokens[0]));
                outG = static_cast<float>(vislib::CharTraitsA::ParseDouble(tokens[1]));
                outB = static_cast<float>(vislib::CharTraitsA::ParseDouble(tokens[2]));
                outA = 1.0f;
                return true;
            }

        } catch (...) {}

    } else {
        // named colour ?
        for (int i = 0; namedColours[i].name != NULL; i++) {
            if (s.Equals(namedColours[i].name, false)) {
                outR = static_cast<float>((namedColours[i].colour >> 16) % 256) / 255.0f;
                outG = static_cast<float>((namedColours[i].colour >> 8) % 256) / 255.0f;
                outB = static_cast<float>(namedColours[i].colour % 256) / 255.0f;
                outA = 1.0f;
                return true;
            }
        }
    }

    return false;
}


/*
 * utility::ColourParser::FromString
 */
bool utility::ColourParser::FromString(const vislib::StringA& str, unsigned int outColLen, unsigned char* outCol) {
    float r, g, b, a;
    bool ret = FromString(str, r, g, b, a);

    if (outColLen > 0) {
        outCol[0] = static_cast<unsigned char>(vislib::math::Clamp(r, 0.0f, 1.0f) * 255.0f);
        if (outColLen > 1) {
            outCol[1] = static_cast<unsigned char>(vislib::math::Clamp(g, 0.0f, 1.0f) * 255.0f);
            if (outColLen > 2) {
                outCol[2] = static_cast<unsigned char>(vislib::math::Clamp(b, 0.0f, 1.0f) * 255.0f);
                if (outColLen > 3)
                    outCol[3] = static_cast<unsigned char>(vislib::math::Clamp(a, 0.0f, 1.0f) * 255.0f);
            }
        }
    }

    return ret;
}


/*
 * utility::ColourParser::FromString
 */
bool utility::ColourParser::FromString(const vislib::StringA& str, unsigned int outColLen, float* outCol) {
    float r, g, b, a;
    bool ret = FromString(str, r, g, b, a);

    if (outColLen > 0) {
        outCol[0] = vislib::math::Clamp(r, 0.0f, 1.0f);
        if (outColLen > 1) {
            outCol[1] = vislib::math::Clamp(g, 0.0f, 1.0f);
            if (outColLen > 2) {
                outCol[2] = vislib::math::Clamp(b, 0.0f, 1.0f);
                if (outColLen > 3)
                    outCol[3] = vislib::math::Clamp(a, 0.0f, 1.0f);
            }
        }
    }
    return ret;
}


/*
 * utility::ColourParser::ColourParser
 */
utility::ColourParser::ColourParser() {
    throw vislib::UnsupportedOperationException("ColourParser::Ctor", __FILE__, __LINE__);
}


/*
 * utility::ColourParser::~ColourParser
 */
#ifdef _WIN32
#pragma warning(disable : 4722 4297)
#endif
utility::ColourParser::~ColourParser() {
    throw vislib::UnsupportedOperationException("ColourParser::Dtor", __FILE__, __LINE__);
}
#ifdef _WIN32
#pragma warning(default : 4722 4297)
#endif


/*
 * utility::ColourParser::hexToNum
 */
unsigned char utility::ColourParser::hexToNum(const char& c) {
    if ((c >= '0') && (c <= '9')) {
        return c - '0';
    }
    if ((c >= 'a') && (c <= 'f')) {
        return 10 + c - 'a';
    }
    if ((c >= 'A') && (c <= 'F')) {
        return 10 + c - 'A';
    }
    return 255;
}
