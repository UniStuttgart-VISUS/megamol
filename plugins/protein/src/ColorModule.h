//
// ColorModule.h
//
// Copyright (C) 2014 by University of Stuttgart (VISUS).
// All rights reserved.
//

#ifndef MMPROTEINPLUGIN_COLORMODULE_H_INCLUDED
#define MMPROTEINPLUGIN_COLORMODULE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "protein_calls/BindingSiteCall.h"
#include "CallColor.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "protein_calls/MolecularDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Renderer3DModule.h"

namespace megamol {
namespace protein {

class ColorModule : public core::Module {

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

	enum ComparisonMode {
		ZERO_TO_MAX    = 0,
		ZERO_TO_VALUE  = 1,
		MIN_TO_MAX     = 2,
		MIN_TO_VALUE   = 3,
		VALUE_TO_MAX   = 4,
		VALUE_TO_VALUE = 5
	};

	enum ComparisonColoringMode {
		SINGLE_COLOR = 0,
		TWO_COLORS = 1,
		COLOR_GRADIENT = 2
	};

     /**
      * Answer the name of this module.
      *
      * @return The name of this module.
      */
     static const char *ClassName(void) {
         return "ColorModule";
     }


     /**
      * Answer a human readable description of this module.
      *
      * @return A human readable description of this module.
      */
     static const char *Description(void) {
         return "Offers protein coloring.";
     }
	 
	 /**
      * Answers whether this module is available on the current system.
      *
      * @return 'true' if the module is available, 'false' otherwise.
      */
     static bool IsAvailable(void) {
         return true;
     }


     /** Ctor. */
     ColorModule(void);


     /** Dtor. */
     virtual ~ColorModule(void);


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
      * @param atomColorTable      The atom color table.
      * @param colorLookupTable    The color lookup table.
      * @param rainbowColors       The rainbow color lookup table.
      * @param minGradColor        The minimum value for gradient coloring.
      * @param midGradColor        The middle value for gradient coloring.
      * @param maxGradColor        The maximum value for gradient coloring.
      * @param forceRecompute      Force recomputation of the color table.
      * @param bs                  The binding site data call.
      */
	 void MakeColorTable(const megamol::protein_calls::MolecularDataCall *mol,
		ColoringMode cm0,
        ColoringMode cm1,
        vislib::Array<float> &atomColorTable,
        vislib::Array<vislib::math::Vector<float, 3> > &colorLookupTable,
        vislib::Array<vislib::math::Vector<float, 3> > &rainbowColors,
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
	 void MakeColorTable(const megamol::protein_calls::MolecularDataCall *mol,
		ColoringMode currentColoringMode,
        vislib::Array<float> &atomColorTable,
        vislib::Array<vislib::math::Vector<float, 3> > &colorLookupTable,
        vislib::Array<vislib::math::Vector<float, 3> > &rainbowColors,
        bool forceRecompute = false,
		const protein_calls::BindingSiteCall *bs = 0);

	 /**
      * Make color table for all atoms acoording to compare two different
	  * proteins
      * The color table is only computed if it is empty or if the
      * recomputation is forced by parameter.
      *
      * @param mol                The data interface. 
      * @param currentColoringMode The current coloring mode.
      * @param atomColorTable      The atom color table.
      * @param colorLookupTable    The color lookup table.
      * @param rainbowColors       The rainbow color lookup table.
	  *	@param frameID			   The frame ID
      * @param forceRecompute      Force recomputation of the color table.
      * @param bs                  The binding site data call.
      */
	 void MakeComparisonColorTable(megamol::protein_calls::MolecularDataCall *mol1,
		ColoringMode currentColoringMode,
		vislib::Array<float> &atomColorTable,
		vislib::Array<vislib::math::Vector<float, 3> > &colorLookupTable,
		vislib::Array<vislib::math::Vector<float, 3> > &rainbowColors,
		unsigned int frameID,
		bool forceRecompute = false,
		const protein_calls::BindingSiteCall *bs = 0);

	 /**
      * Fill amino acid color table.
      *
      * @param aminoAcidColorTable The amino acid color table.
      */
     void FillAminoAcidColorTable(
		vislib::Array<vislib::math::Vector<float, 3> >
        &aminoAcidColorTable);

	 /**
      * Get the corresponding name of a given coloring mode.
      *
      * @param col The coloring mode.
      *
      * @return The name.
      */
     std::string getName(ColorModule::ColoringMode col);

	 /**
      * Get the corresponding name of a given comparison mode.
      *
      * @param col The comparison mode.
      *
      * @return The name.
      */
	 std::string getName(ColorModule::ComparisonMode com);

	 /**
      * Get the corresponding name of a given comparison coloring mode.
      *
      * @param col The comparison coloring mode.
      *
      * @return The name.
      */
	 std::string getName(ColorModule::ComparisonColoringMode col);
	 

	 /**
      * Creates a rainbow color table with 'num' entries.
      *
      * @param num            The number of color entries.
      * @param rainbowColors  The rainbow color lookup table.
      */
     void MakeRainbowColorTable( unsigned int num,
		vislib::Array<vislib::math::Vector<float, 3> > &rainbowColors);

     /**
      * Read color table from file.
      *
      * @param filename          The filename of the color table file.
      * @param colorLookupTable  The color lookup table.
      */
     void ReadColorTableFromFile( vislib::StringA filename,
		vislib::Array<vislib::math::Vector<float, 3> > &colorLookupTable);
protected:
	/**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
     virtual bool create(void);


     /**
      * Implementation of 'release'.
      */
     virtual void release(void);

	 /**
      * Call callback to get the data
      *
      * @param c The calling call
      *
      * @return True on success
      */
     bool getColor(core::Call& call);

     /**
      * Call callback to get the extent of the data
      *
      * @param c The calling call
      *
      * @return True on success
      */
     bool getExtents(core::Call& call);

	 /**
      * Get the coloring mode at a certain index.
      *
      * @param idx The index.
      *
      * @return The coloring mode.
      */
     ColorModule::ColoringMode getModeByIndex(unsigned int idx);
	 
	 /**
      * Get the comparison mode at a certain index.
      *
      * @param idx The index.
      *
      * @return The comparison mode.
      */
	 ColorModule::ComparisonMode getComparisonModeByIndex(unsigned int idx);

	 /**
      * Get the comparison coloring mode at a certain index.
      *
      * @param idx The index.
      *
      * @return The comparison coloring mode.
      */
	 ColorModule::ComparisonColoringMode getComparisonColoringModeByIndex(unsigned int idx);


	 /**
	  *	Get the Number of coloring modes.
	  *
	  * @return The number of coloring modes.
	  */
	 unsigned int getNumOfColoringModes() {
		return 10;
	 };

	 /**
	  *	Get the Number of comparison modes.
	  *
	  * @return The number of comparison modes.
	  */
	 unsigned int getNumOfComparisonModes() {
		return 6;
	 };

	 /**
	  *	Get the Number of comparison coloring modes.
	  *
	  * @return The number of comparison coloring modes.
	  */
	 unsigned int getNumOfComparisonColoringModes() {
		return 4;
	 };

	 /**
	  *	Checks, if some params are changed by the user
	  *
	  *	@return true, if a param was changed, false otherwise
	  */
	 bool updateParams();

private:
	// caller slot
    megamol::core::CallerSlot molDataCallerSlot;

	/** parameter slot for color table filename */
	megamol::core::param::ParamSlot colorTableFileParam;
	/** parameter slot for minimum gradient color */
	megamol::core::param::ParamSlot minGradColorParam;
	/** parameter slot for middle gradient color */
	megamol::core::param::ParamSlot midGradColorParam;
	/** parameter slot for maximum gradient color */
	megamol::core::param::ParamSlot maxGradColorParam;
	/** parameter slot for the comparison color */
	megamol::core::param::ParamSlot colorParam;
	/** parameter slot for coloring mode */
	megamol::core::param::ParamSlot coloringMode0Param;
	/** parameter slot for coloring mode */
	megamol::core::param::ParamSlot coloringMode1Param;
	/** parameter slot for comparison mode */
	megamol::core::param::ParamSlot comparisonModeParam;
	/** parameter slot for the comparison color */
	megamol::core::param::ParamSlot comparisonColorParam;
	/** parameter slot for the weighting factor */
	megamol::core::param::ParamSlot weightingParam;
	/** parameter slot for the distance the coloring starts*/
	megamol::core::param::ParamSlot minDistanceParam;
	/** parameter slot for the distance the coloring stops*/
	megamol::core::param::ParamSlot maxDistanceParam;

	/** callee slot to return the computed colors */
	megamol::core::CalleeSlot colorOutSlot;

	/** The color lookup table (for chains, amino acids,...) */
	vislib::Array<vislib::math::Vector<float, 3> > colorLookupTable;

	/** The color lookup table which stores the rainbow colors */
	vislib::Array<vislib::math::Vector<float, 3> > rainbowColors;

	// state variables
	//ColoringMode currentColoringMode0;
	//ColoringMode currentColoringMode1;
	//ComparisonMode currentComparisonMode;
	//ComparisonColoringMode currentComparisonColoringMode;

	float weight0;
	float weight1;
};

} // end namespace protein
} // end namespace megamol

#endif // MMPROTEINPLUGIN_COLORMODULE_H_INCLUDED