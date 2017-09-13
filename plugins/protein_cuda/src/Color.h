/*
 * Color.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MMPROTEINCUDAPLUGIN_COLOR_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_COLOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "protein_calls/MolecularDataCall.h"
#include "protein_calls/BindingSiteCall.h"
#include <string>

namespace megamol {
namespace protein_cuda {

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
            MOVEMENT    = 13,
            BINDINGSITE = 14
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
		static Color::ColoringMode GetModeByIndex(const megamol::protein_calls::MolecularDataCall *mol,
            unsigned int idx);

        /**
         * Get the coloring mode at a certain index of a given data call.
         *
         * @param mol The molecular data call.
         * @param bs  The binding site data call.
         * @param idx The index.
         *
         * @return The coloring mode.
         */
		static Color::ColoringMode GetModeByIndex(const megamol::protein_calls::MolecularDataCall *mol,
            const protein_calls::BindingSiteCall *bs, unsigned int idx);

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
		static unsigned int GetNumOfColoringModes(const megamol::protein_calls::MolecularDataCall *mol) {
            return 9;
        }

        /**
         * Get the number of coloring modes used by a given data call.
         *
         * @param mol The data call.
         *
         * @return The number of coloring modes.
         */
		static unsigned int GetNumOfColoringModes(const megamol::protein_calls::MolecularDataCall *mol, const protein_calls::BindingSiteCall *bs) {
            return 10;
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
         * @param bs                  The binding site data call.
         */
		static void MakeColorTable(const megamol::protein_calls::MolecularDataCall *mol,
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
            bool forceRecompute = false,
			const protein_calls::BindingSiteCall *bs = 0);


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
         * @param bs                  The binding site data call.
         */
		static void MakeColorTable(const megamol::protein_calls::MolecularDataCall *mol,
            ColoringMode currentColoringMode,
            vislib::Array<float> &atomColorTable,
            vislib::Array<vislib::math::Vector<float, 3> > &colorLookupTable,
            vislib::Array<vislib::math::Vector<float, 3> > &rainbowColors,
            vislib::TString minGradColor,
            vislib::TString midGradColor,
            vislib::TString maxGradColor,
            bool forceRecompute = false,
			const protein_calls::BindingSiteCall *bs = 0);

		/**
         * Make color table for all atoms acoording to compare two different
		 * proteins
         * The color table is only computed if it is empty or if the
         * recomputation is forced by parameter.
         *
         * @param mol1                The first data interface. 
		 *                            This one is rendered
		 * @param mol2				  The second data interface.
         * @param currentColoringMode The current coloring mode.
         * @param atomColorTable      The atom color table.
         * @param colorLookupTable    The color lookup table.
         * @param rainbowColors       The rainbow color lookup table.
         * @param minGradColor        The minimum value for gradient coloring.
         * @param midGradColor        The middle value for gradient coloring.
         * @param maxGradColor        The maximum value for gradient coloring.
         * @param forceRecompute      Force recomputation of the color table.
         * @param bs                  The binding site data call.
         */
		static void MakeComparisonColorTable(const megamol::protein_calls::MolecularDataCall *mol1,
			const megamol::protein_calls::MolecularDataCall *mol2,
			ColoringMode currentColoringMode,
			vislib::Array<float> &atomColorTable,
			vislib::Array<vislib::math::Vector<float, 3> > &colorLookupTable,
			vislib::Array<vislib::math::Vector<float, 3> > &rainbowColors,
			vislib::TString minGradColor,
			vislib::TString midGradColor,
			vislib::TString maxGradColor,
			bool forceRecompute = false,
			const protein_calls::BindingSiteCall *bs = 0);

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

} /* end namespace protein_cuda */
} /* end namespace megaMol */

#endif /* MMPROTEINCUDAPLUGIN_COLOR_H_INCLUDED */
