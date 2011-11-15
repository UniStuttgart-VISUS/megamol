/*
 * Color.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MMPROTEINPLUGIN_COLOR_H_INCLUDED
#define MMPROTEINPLUGIN_COLOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "MolecularDataCall.h"
#include "CallProteinData.h"
#include "CallProteinMovementData.h"
#include <string>

namespace megamol {
namespace protein {

    class Color {

    public:

        /** The names of the coloring modes */
        enum ColoringMode {
            ELEMENT     = 0,
            STRUCTURE   = 1,
            RAINBOW     = 2,
            BFACTOR     = 3,
            CHARGE      = 4,
            OCCUPANCY   = 5,
            CHAIN       = 6,
            MOLECULE    = 7,
            RESIDUE     = 8,
            CHAINBOW    = 9,
            AMINOACID   = 10,
            VALUE       = 11,
            CHAIN_ID    = 12,
            MOVEMENT    = 13
        };

        /**
         * Fill amino acid color table.
         *
         * @param aminoAcidColorTable The amino acid color table.
         */
        static void FillAminoAcidColorTable(
            vislib::Array<vislib::math::Vector<float, 3> >
              &aminoAcidColorTable);

        /**
         * Get the coloring mode at a certain index of a given data call.
         *
         * @param mol The data call.
         * @param idx The index.
         *
         * @return The coloring mode.
         */
        static Color::ColoringMode GetModeByIndex(const MolecularDataCall *mol,
            unsigned int idx);

        /**
         * Get the coloring mode at a certain index of a given data call.
         *
         * @param mol The data call.
         * @param idx The index.
         *
         * @return The coloring mode.
         */
        static Color::ColoringMode GetModeByIndex(const CallProteinData *prot,
            unsigned int idx);

        /**
         * Get the corresponding name of a given coloring mode.
         *
         * @param col The coloring mode.
         *
         * @return The name.
         */
        static std::string GetName(Color::ColoringMode col);

        /**
         * Get the number of coloring modes used by a given data call.
         *
         * @param mol The data call.
         *
         * @return The number of coloring modes.
         */
        static unsigned int GetNumOfColoringModes(const MolecularDataCall *mol) {
            return 9;
        }

        /**
         * Get the number of coloring modes used by a given data call.
         *
         * @param prot The data call.
         *
         * @return The number of coloring modes.
         */
        static unsigned int GetNumOfColoringModes(const CallProteinData *prot) {
            return 7;
        }

        /**
         * Make color table for all atoms by linearly interpolating between two
         * given coloring modes.
         *
         * The color table is only computed if it is empty or if the
         * recomputation is forced by parameter.
         *
         * @param mol                 The data interface.
         * @param cm0                 The first coloring mode.
         * @param cm1                 The second coloring mode.
         * @param weight0             The weight for the first coloring mode.
         * @param weight1             The weight for the second coloring mode.
         * @param atomColorTable      The atom color table.
         * @param colorLookupTable    The color lookup table.
         * @param rainbowColors       The rainbow color lookup table.
         * @param minGradColor        The minimum value for gradient coloring.
         * @param midGradColor        The middle value for gradient coloring.
         * @param maxGradColor        The maximum value for gradient coloring.
         * @param forceRecompute      Force recomputation of the color table.
         */
        static void MakeColorTable(const MolecularDataCall *mol,
            ColoringMode cm0,
            ColoringMode cm1,
            float weight0,
            float weight1,
            vislib::Array<float> &atomColorTable,
            vislib::Array<vislib::math::Vector<float, 3> > &colorLookupTable,
            vislib::Array<vislib::math::Vector<float, 3> > &rainbowColors,
            vislib::TString minGradColor,
            vislib::TString midGradColor,
            vislib::TString maxGradColor,
            bool forceRecompute = false);


        /**
         * Make color table for all atoms acoording to the current coloring
         * mode.
         * The color table is only computed if it is empty or if the
         * recomputation is forced by parameter.
         *
         * @param mol                 The data interface.
         * @param currentColoringMode The current coloring mode.
         * @param atomColorTable      The atom color table.
         * @param colorLookupTable    The color lookup table.
         * @param rainbowColors       The rainbow color lookup table.
         * @param minGradColor        The minimum value for gradient coloring.
         * @param midGradColor        The middle value for gradient coloring.
         * @param maxGradColor        The maximum value for gradient coloring.
         * @param forceRecompute      Force recomputation of the color table.
         */
        static void MakeColorTable(const MolecularDataCall *mol,
            ColoringMode currentColoringMode,
            vislib::Array<float> &atomColorTable,
            vislib::Array<vislib::math::Vector<float, 3> > &colorLookupTable,
            vislib::Array<vislib::math::Vector<float, 3> > &rainbowColors,
            vislib::TString minGradColor,
            vislib::TString midGradColor,
            vislib::TString maxGradColor,
            bool forceRecompute = false);


        /**
         * Make color table for all atoms acoording to the current coloring
         * mode.
         * The color table is only computed if it is empty or if the
         * recomputation is forced by parameter.
         *
         * @param prot                The data interface.
         * @param currentColoringMode The current coloring mode.
         * @param protAtomColorTable  The atom color table.
         * @param aminoAcidColorTable The amino acid color table.
         * @param rainbowColors       The rainbow color lookup table.
         * @param forceRecompute      Force recomputation of the color table.
         */
        static void MakeColorTable( const CallProteinData *prot,
            ColoringMode currentColoringMode,
            vislib::Array<float> &atomColorTable,
            vislib::Array<vislib::math::Vector<float, 3> > &aminoAcidColorTable,
            vislib::Array<vislib::math::Vector<float, 3> > &rainbowColors,
            bool forceRecompute = false,
            vislib::math::Vector<float, 3> minValueColor
                    = vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f),
            vislib::math::Vector<float, 3> meanValueColor
                    = vislib::math::Vector<float, 3>(0.5f, 0.5f, 0.5f),
            vislib::math::Vector<float, 3> maxValueColor
                    = vislib::math::Vector<float, 3>(1.0f, 1.0f, 1.0f));

        /**
         * Make color table for all atoms acoording to the current coloring
         * mode.
         * The color table is only computed if it is empty or if the
         * recomputation is forced by parameter.
         *
         * @param prot                The data interface.
         * @param currentColoringMode The current coloring mode.
         * @param colMax              colMax
         * @param colMid              colMid
         * @param colMin              colMin
         * @param col                 col
         * @param protAtomColorTable  The atom color table.
         * @param aminoAcidColorTable The amino acid color table.
         * @param rainbowColors       The rainbow color lookup table.
         * @param forceRecompute      Force recomputation of the color table.
         */
        static void MakeColorTable( const CallProteinMovementData *prot,
            ColoringMode currentColoringMode,
            vislib::Array<float> &protAtomColorTable,
            vislib::Array<vislib::math::Vector<float, 3> > &aminoAcidColorTable,
            vislib::Array<vislib::math::Vector<float, 3> > &rainbowColors,
            vislib::math::Vector<int, 3> colMax,
            vislib::math::Vector<int, 3> colMid,
            vislib::math::Vector<int, 3> colMin,
            vislib::math::Vector<int, 3> col,
            bool forceRecompute = false);

         /**
         * Creates a rainbow color table with 'num' entries.
         *
         * @param num            The number of color entries.
         * @param rainbowColors  The rainbow color lookup table.
         */
        static void MakeRainbowColorTable( unsigned int num,
            vislib::Array<vislib::math::Vector<float, 3> > &rainbowColors);

        /**
         * Read color table from file.
         *
         * @param filename          The filename of the color table file.
         * @param colorLookupTable  The color lookup table.
         */
        static void ReadColorTableFromFile( vislib::StringA filename,
            vislib::Array<vislib::math::Vector<float, 3> > &colorLookupTable);

    };

} /* end namespace protein */
} /* end namespace megaMol */

#endif /* MMPROTEINPLUGIN_COLOR_H_INCLUDED */
