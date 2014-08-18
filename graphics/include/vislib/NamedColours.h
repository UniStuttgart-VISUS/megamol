/*
 * NamedColours.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_NAMEDCOLOURS_H_INCLUDED
#define VISLIB_NAMEDCOLOURS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/ColourRGBAu8.h"
#include "vislib/NoSuchElementException.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/String.h"
#include "vislib/types.h"


namespace vislib {
namespace graphics {


    /**
     * Static utility class for named colours
     */
    class NamedColours {
    public:

#if 1 /* REGION Colour constants */

        /** Colour constant for AliceBlue */
        static ColourRGBAu8 AliceBlue;

        /** Colour constant for AntiqueWhite */
        static ColourRGBAu8 AntiqueWhite;

        /** Colour constant for Aqua */
        static ColourRGBAu8 Aqua;

        /** Colour constant for Aquamarine */
        static ColourRGBAu8 Aquamarine;

        /** Colour constant for Azure */
        static ColourRGBAu8 Azure;

        /** Colour constant for Beige */
        static ColourRGBAu8 Beige;

        /** Colour constant for Bisque */
        static ColourRGBAu8 Bisque;

        /** Colour constant for Black */
        static ColourRGBAu8 Black;

        /** Colour constant for BlanchedAlmond */
        static ColourRGBAu8 BlanchedAlmond;

        /** Colour constant for Blue */
        static ColourRGBAu8 Blue;

        /** Colour constant for BlueViolet */
        static ColourRGBAu8 BlueViolet;

        /** Colour constant for Brown */
        static ColourRGBAu8 Brown;

        /** Colour constant for BurlyWood */
        static ColourRGBAu8 BurlyWood;

        /** Colour constant for CadetBlue */
        static ColourRGBAu8 CadetBlue;

        /** Colour constant for Chartreuse */
        static ColourRGBAu8 Chartreuse;

        /** Colour constant for Chocolate */
        static ColourRGBAu8 Chocolate;

        /** Colour constant for Coral */
        static ColourRGBAu8 Coral;

        /** Colour constant for CornflowerBlue */
        static ColourRGBAu8 CornflowerBlue;

        /** Colour constant for Cornsilk */
        static ColourRGBAu8 Cornsilk;

        /** Colour constant for Crimson */
        static ColourRGBAu8 Crimson;

        /** Colour constant for Cyan */
        static ColourRGBAu8 Cyan;

        /** Colour constant for DarkBlue */
        static ColourRGBAu8 DarkBlue;

        /** Colour constant for DarkCyan */
        static ColourRGBAu8 DarkCyan;

        /** Colour constant for DarkGoldenrod */
        static ColourRGBAu8 DarkGoldenrod;

        /** Colour constant for DarkGray */
        static ColourRGBAu8 DarkGray;

        /** Colour constant for DarkGrey */
        static ColourRGBAu8 DarkGrey;

        /** Colour constant for DarkGreen */
        static ColourRGBAu8 DarkGreen;

        /** Colour constant for DarkKhaki */
        static ColourRGBAu8 DarkKhaki;

        /** Colour constant for DarkMagenta */
        static ColourRGBAu8 DarkMagenta;

        /** Colour constant for DarkOliveGreen */
        static ColourRGBAu8 DarkOliveGreen;

        /** Colour constant for DarkOrange */
        static ColourRGBAu8 DarkOrange;

        /** Colour constant for DarkOrchid */
        static ColourRGBAu8 DarkOrchid;

        /** Colour constant for DarkRed */
        static ColourRGBAu8 DarkRed;

        /** Colour constant for DarkSalmon */
        static ColourRGBAu8 DarkSalmon;

        /** Colour constant for DarkSeaGreen */
        static ColourRGBAu8 DarkSeaGreen;

        /** Colour constant for DarkSlateBlue */
        static ColourRGBAu8 DarkSlateBlue;

        /** Colour constant for DarkSlateGray */
        static ColourRGBAu8 DarkSlateGray;

        /** Colour constant for DarkSlateGrey */
        static ColourRGBAu8 DarkSlateGrey;

        /** Colour constant for DarkTurquoise */
        static ColourRGBAu8 DarkTurquoise;

        /** Colour constant for DarkViolet */
        static ColourRGBAu8 DarkViolet;

        /** Colour constant for DeepPink */
        static ColourRGBAu8 DeepPink;

        /** Colour constant for DeepSkyBlue */
        static ColourRGBAu8 DeepSkyBlue;

        /** Colour constant for DimGray */
        static ColourRGBAu8 DimGray;

        /** Colour constant for DimGrey */
        static ColourRGBAu8 DimGrey;

        /** Colour constant for DodgerBlue */
        static ColourRGBAu8 DodgerBlue;

        /** Colour constant for Firebrick */
        static ColourRGBAu8 Firebrick;

        /** Colour constant for FloralWhite */
        static ColourRGBAu8 FloralWhite;

        /** Colour constant for ForestGreen */
        static ColourRGBAu8 ForestGreen;

        /** Colour constant for Fuchsia */
        static ColourRGBAu8 Fuchsia;

        /** Colour constant for Gainsboro */
        static ColourRGBAu8 Gainsboro;

        /** Colour constant for GhostWhite */
        static ColourRGBAu8 GhostWhite;

        /** Colour constant for Gold */
        static ColourRGBAu8 Gold;

        /** Colour constant for Goldenrod */
        static ColourRGBAu8 Goldenrod;

        /** Colour constant for Gray */
        static ColourRGBAu8 Gray;

        /** Colour constant for Grey */
        static ColourRGBAu8 Grey;

        /** Colour constant for Green */
        static ColourRGBAu8 Green;

        /** Colour constant for GreenYellow */
        static ColourRGBAu8 GreenYellow;

        /** Colour constant for Honeydew */
        static ColourRGBAu8 Honeydew;

        /** Colour constant for HotPink */
        static ColourRGBAu8 HotPink;

        /** Colour constant for IndianRed */
        static ColourRGBAu8 IndianRed;

        /** Colour constant for Indigo */
        static ColourRGBAu8 Indigo;

        /** Colour constant for Ivory */
        static ColourRGBAu8 Ivory;

        /** Colour constant for Khaki */
        static ColourRGBAu8 Khaki;

        /** Colour constant for Lavender */
        static ColourRGBAu8 Lavender;

        /** Colour constant for LavenderBlush */
        static ColourRGBAu8 LavenderBlush;

        /** Colour constant for LawnGreen */
        static ColourRGBAu8 LawnGreen;

        /** Colour constant for LemonChiffon */
        static ColourRGBAu8 LemonChiffon;

        /** Colour constant for LightBlue */
        static ColourRGBAu8 LightBlue;

        /** Colour constant for LightCoral */
        static ColourRGBAu8 LightCoral;

        /** Colour constant for LightCyan */
        static ColourRGBAu8 LightCyan;

        /** Colour constant for LightGoldenrodYellow */
        static ColourRGBAu8 LightGoldenrodYellow;

        /** Colour constant for LightGray */
        static ColourRGBAu8 LightGray;

        /** Colour constant for LightGrey */
        static ColourRGBAu8 LightGrey;

        /** Colour constant for LightGreen */
        static ColourRGBAu8 LightGreen;

        /** Colour constant for LightPink */
        static ColourRGBAu8 LightPink;

        /** Colour constant for LightSalmon */
        static ColourRGBAu8 LightSalmon;

        /** Colour constant for LightSeaGreen */
        static ColourRGBAu8 LightSeaGreen;

        /** Colour constant for LightSkyBlue */
        static ColourRGBAu8 LightSkyBlue;

        /** Colour constant for LightSlateGray */
        static ColourRGBAu8 LightSlateGray;

        /** Colour constant for LightSlateGrey */
        static ColourRGBAu8 LightSlateGrey;

        /** Colour constant for LightSteelBlue */
        static ColourRGBAu8 LightSteelBlue;

        /** Colour constant for LightYellow */
        static ColourRGBAu8 LightYellow;

        /** Colour constant for Lime */
        static ColourRGBAu8 Lime;

        /** Colour constant for LimeGreen */
        static ColourRGBAu8 LimeGreen;

        /** Colour constant for Linen */
        static ColourRGBAu8 Linen;

        /** Colour constant for Magenta */
        static ColourRGBAu8 Magenta;

        /** Colour constant for Maroon */
        static ColourRGBAu8 Maroon;

        /** Colour constant for MediumAquamarine */
        static ColourRGBAu8 MediumAquamarine;

        /** Colour constant for MediumBlue */
        static ColourRGBAu8 MediumBlue;

        /** Colour constant for MediumOrchid */
        static ColourRGBAu8 MediumOrchid;

        /** Colour constant for MediumPurple */
        static ColourRGBAu8 MediumPurple;

        /** Colour constant for MediumSeaGreen */
        static ColourRGBAu8 MediumSeaGreen;

        /** Colour constant for MediumSlateBlue */
        static ColourRGBAu8 MediumSlateBlue;

        /** Colour constant for MediumSpringGreen */
        static ColourRGBAu8 MediumSpringGreen;

        /** Colour constant for MediumTurquoise */
        static ColourRGBAu8 MediumTurquoise;

        /** Colour constant for MediumVioletRed */
        static ColourRGBAu8 MediumVioletRed;

        /** Colour constant for MidnightBlue */
        static ColourRGBAu8 MidnightBlue;

        /** Colour constant for MintCream */
        static ColourRGBAu8 MintCream;

        /** Colour constant for MistyRose */
        static ColourRGBAu8 MistyRose;

        /** Colour constant for Moccasin */
        static ColourRGBAu8 Moccasin;

        /** Colour constant for NavajoWhite */
        static ColourRGBAu8 NavajoWhite;

        /** Colour constant for Navy */
        static ColourRGBAu8 Navy;

        /** Colour constant for OldLace */
        static ColourRGBAu8 OldLace;

        /** Colour constant for Olive */
        static ColourRGBAu8 Olive;

        /** Colour constant for OliveDrab */
        static ColourRGBAu8 OliveDrab;

        /** Colour constant for Orange */
        static ColourRGBAu8 Orange;

        /** Colour constant for OrangeRed */
        static ColourRGBAu8 OrangeRed;

        /** Colour constant for Orchid */
        static ColourRGBAu8 Orchid;

        /** Colour constant for PaleGoldenrod */
        static ColourRGBAu8 PaleGoldenrod;

        /** Colour constant for PaleGreen */
        static ColourRGBAu8 PaleGreen;

        /** Colour constant for PaleTurquoise */
        static ColourRGBAu8 PaleTurquoise;

        /** Colour constant for PaleVioletRed */
        static ColourRGBAu8 PaleVioletRed;

        /** Colour constant for PapayaWhip */
        static ColourRGBAu8 PapayaWhip;

        /** Colour constant for PeachPuff */
        static ColourRGBAu8 PeachPuff;

        /** Colour constant for Peru */
        static ColourRGBAu8 Peru;

        /** Colour constant for Pink */
        static ColourRGBAu8 Pink;

        /** Colour constant for Plum */
        static ColourRGBAu8 Plum;

        /** Colour constant for PowderBlue */
        static ColourRGBAu8 PowderBlue;

        /** Colour constant for Purple */
        static ColourRGBAu8 Purple;

        /** Colour constant for Red */
        static ColourRGBAu8 Red;

        /** Colour constant for RosyBrown */
        static ColourRGBAu8 RosyBrown;

        /** Colour constant for RoyalBlue */
        static ColourRGBAu8 RoyalBlue;

        /** Colour constant for SaddleBrown */
        static ColourRGBAu8 SaddleBrown;

        /** Colour constant for Salmon */
        static ColourRGBAu8 Salmon;

        /** Colour constant for SandyBrown */
        static ColourRGBAu8 SandyBrown;

        /** Colour constant for SeaGreen */
        static ColourRGBAu8 SeaGreen;

        /** Colour constant for SeaShell */
        static ColourRGBAu8 SeaShell;

        /** Colour constant for Sienna */
        static ColourRGBAu8 Sienna;

        /** Colour constant for Silver */
        static ColourRGBAu8 Silver;

        /** Colour constant for SkyBlue */
        static ColourRGBAu8 SkyBlue;

        /** Colour constant for SlateBlue */
        static ColourRGBAu8 SlateBlue;

        /** Colour constant for SlateGray */
        static ColourRGBAu8 SlateGray;

        /** Colour constant for SlateGrey */
        static ColourRGBAu8 SlateGrey;

        /** Colour constant for Snow */
        static ColourRGBAu8 Snow;

        /** Colour constant for SpringGreen */
        static ColourRGBAu8 SpringGreen;

        /** Colour constant for SteelBlue */
        static ColourRGBAu8 SteelBlue;

        /** Colour constant for Tan */
        static ColourRGBAu8 Tan;

        /** Colour constant for Teal */
        static ColourRGBAu8 Teal;

        /** Colour constant for Thistle */
        static ColourRGBAu8 Thistle;

        /** Colour constant for Tomato */
        static ColourRGBAu8 Tomato;

        /** Colour constant for Transparent */
        static ColourRGBAu8 Transparent;

        /** Colour constant for Turquoise */
        static ColourRGBAu8 Turquoise;

        /** Colour constant for Violet */
        static ColourRGBAu8 Violet;

        /** Colour constant for Wheat */
        static ColourRGBAu8 Wheat;

        /** Colour constant for White */
        static ColourRGBAu8 White;

        /** Colour constant for WhiteSmoke */
        static ColourRGBAu8 WhiteSmoke;

        /** Colour constant for Yellow */
        static ColourRGBAu8 Yellow;

        /** Colour constant for YellowGreen */
        static ColourRGBAu8 YellowGreen;

        /** Colour constant for MegaMolBlue */
        static ColourRGBAu8 MegaMolBlue;

#endif /* REGION */

        /**
         * Answer the number of named colours
         *
         * @return The number of named colours
         */
        static SIZE_T CountNamedColours(void);

        /**
         * Answer the idx-th named colour
         *
         * @param idx The zero-based index
         *
         * @return The idx-th named colour
         *
         * @throw OutOfRangeException if idx < 0 or idx >= CountNamedColours
         */
        static const ColourRGBAu8& GetColourByIndex(SIZE_T idx);

        /**
         * Answer the colour of the specified name
         *
         * @param name The name of the colour to be returned
         *
         * @return The colour of the specified name
         *
         * @throw NoSuchElementException if 'name' is not a named colour name
         */
        static const ColourRGBAu8& GetColourByName(const char *name);

        /**
         * Answer the idx-th named colour name
         *
         * @param idx The zero-based index
         *
         * @return The idx-th named colour name
         *
         * @throw OutOfRangeException if idx < 0 or idx >= CountNamedColours
         */
        static const char *GetNameByIndex(SIZE_T idx);

        /**
         * Answer the name of the colour
         *
         * @param col The colour to test
         * @param throwException If set to 'true' a NoSuchElementException is
         *                       thrown if 'col' is not a named colour.
         *
         * @return ASCII string of the colour name or NULL if 'col' is not a
         *         named colour. Do not free the memory of the return value.
         *
         * @throw NoSuchElementException if 'col' is not a named colour and
         *        'throwException' is 'true'.
         */
        static const char *GetNameByColour(const ColourRGBAu8& col,
            bool throwException = true);

        /**
         * Answer whether or not 'col' is a named colour
         *
         * @param col The colour to test
         *
         * @return True if 'col' is a named colour
         */
        static inline bool IsNamedColour(const ColourRGBAu8& col) {
            return (GetNameByColour(col, false) != NULL);
        }

    private:

        /**
         * Utility struct for named colours
         */
        typedef struct _namedcolourindex_t {

            /** The colours name */
            const vislib::StringA name;

            /** The named colour */
            const ColourRGBAu8 colour;

        } NamedColourIndex;

        /** The size of the index of named colours */
        static SIZE_T count;

        /** The index of named colours */
        static NamedColourIndex index[];

        /**
         * Creates a colour from hex-html value
         *
         * @param hex The hex-html colour value
         *
         * @return The colour
         */
        static ColourRGBAu8 colFromHex(DWORD hex);

        /** Forbidden ctor. */
        NamedColours(void);

        /** Forbidden dtor. */
        ~NamedColours(void);

    };
    
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_NAMEDCOLOURS_H_INCLUDED */

