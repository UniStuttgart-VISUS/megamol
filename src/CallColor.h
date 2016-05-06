//
// CallColor.h
//
// Copyright (C) 2014 by University of Stuttgart (VISUS).
// All rights reserved.
//

#ifndef MMPROTEINPLUGIN_CALLCOLOR_H_INCLUDED
#define MMPROTEINPLUGIN_CALLCOLOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "protein_calls/MolecularDataCall.h"
#include "protein_calls/BindingSiteCall.h"

namespace megamol {
namespace protein {

class CallColor : public core::Call {

public:
	/**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "CallColor";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Call for sending color information";
    }

    /**
      * Answer the number of functions used for this call.
      *
      * @return The number of functions used for this call.
      */
     static unsigned int FunctionCount(void) {
         return 2;
     }

     /**
      * Answer the name of the function used for this call.
      *
      * @param idx The index of the function to return it's name.
      *
      * @return The name of the requested function.
      */
     static const char * FunctionName(unsigned int idx) {
         switch (idx) {
             case 0: return "GetColor";
			 case 1: return "GetExtents";
			 default: return NULL;
         }
     }

	 /**
	  *	Struct to store information for a single c-alpha atom
	  *	containing position and relevant atom indices
	  */
	 struct cAlpha {
		/** position of the c-alpha atom*/
		vislib::math::Point<float, 3> p;
		/** index of the c-alpha atom*/
		unsigned int index;
		/** smallest index of an atom in the same aminoacid as the c-alpha atom*/
		unsigned int firstAtomIdx;
		/** highest index of an atom in the same aminoacid as the c-alpha atom*/
		unsigned int lastAtomIdx;

		/** Ctor*/
		cAlpha(float x, float y, float z, 
			unsigned int idx, unsigned int fAI,
			unsigned int lAI): p(x,y,z), index(idx),
			firstAtomIdx(fAI), lastAtomIdx(lAI) {};
	 };

	 /** Ctor */
	 CallColor() { 
		 this->numEntries = 100;
		 this->comparisonEnabled = false;
		 this->forceRecompute = true;
		 this->isWeighted = true;
	 }

	 /**
	  *	Sets the protein that the color table is computed for
	  *
	  *	@param mol1 The MolecularDataCall of the protein
	  */
	 void SetColoringTarget(megamol::protein_calls::MolecularDataCall *mol1) {
		 this->mol1 = mol1;
	 }

	 /**
	  *	Returns the MolecularDataCall that contains the
	  * protein that should be colored.
	  *
	  * @return The MolecularDataCall containing the protein.
	  */
	 megamol::protein_calls::MolecularDataCall* GetColoringTarget() {
		 return this->mol1;
	 }

	 /**
	  *	Sets the BindingSiteCall.
	  *
	  *	@param bs The BindingSiteCall
	  */
	 void SetBindingSiteCall(const protein_calls::BindingSiteCall *bs) {
		 this->bs = bs;
	 }

	 /**
	  *	Returns the BindingSiteCall
	  *
	  *	@return The BindingSiteCall
	  */
	 const protein_calls::BindingSiteCall* GetBindingSiteCall() {
		 return this->bs;
	 }

	 /**
	  *	Sets the forceRecompute flag.
	  *	
	  *	@param fr true, if the color table should be recomputed
	  */
	 void SetForceRecompute(bool fr) {
		 this->forceRecompute = fr;
	 }

	 /**
	  *	Returns, if the recomputation of the color table should
	  * be forced.
	  *
	  *	@return true, if the recomputation should be forced, false otherwise
	  */
	 bool GetForceRecompute() {
		 return forceRecompute;
	 }

	 /**
	  *	Sets the weighted flag.
	  *
	  *	@param w true, if the colortable should weight between two coloring modes
	  */
	 void SetWeighted(bool w) {
		 this->isWeighted = w;
	 }

	 /**
	  *	Returns, if the color table computation should weight between
	  * two different coloring modes.
	  *
	  *	@return true, if the computation weights between two modes, false otherwise
	  */
	 bool GetWeighted() {
		 return this->isWeighted;
	 }

	 /**
	  *	Sets the color table for the protein. 
	  *
	  *	@param atomColorTable The color table
	  */
	 void SetAtomColorTable(vislib::Array<float> *atomColorTable) {
		 this->atomColorTable = atomColorTable;
	 }

	 /**
	  *	Returns the color table used to color the protein.
	  *
	  * @return The atom color table
	  */
	 vislib::Array<float>* GetAtomColorTable() {
		 return this->atomColorTable;
	 }

	 /**
	  *	Sets the color lookup table.
	  *
	  *	@param clt The color lookup table
	  */
	 void SetColorLookupTable(vislib::Array<vislib::math::Vector<float, 3> > *clt) {
		 this->colorLookupTable = clt;
	 }

	 /**
	  *	Returns the color lookup table.
	  *
	  *	@return The color lookup table
	  */
	 vislib::Array<vislib::math::Vector<float, 3> >* GetColorLookupTable() {
		 return this->colorLookupTable;
	 }

	 /**
	  *	Sets the rainbow color table.
	  *
	  *	@param rct The rainbow color table
	  */
	 void SetRainbowColorTable(vislib::Array<vislib::math::Vector<float, 3> > *rct) {
		 this->rainbowColors = rct;
	 }

	 /**
	  *	Returns the rainbow color table.
	  *
	  *	@return The rainbow color table
	  */
	 vislib::Array<vislib::math::Vector<float, 3> >* GetRainbowColorTable() {
		 return this->rainbowColors;
	 }

	 /**
	  *	Sets the number of entries the rainbow color table should have.
	  *
	  *	@param num The number of entries of the rainbow color table
	  */
	 void SetNumEntries(unsigned int num) {
		 this->numEntries = num;
	 }

	 /** 
	  *	Returns the number of entries of the rainbow color table.
	  *
	  *	@return The number of entries of the rainbow color table
	  */
	 unsigned int GetNumEntries() {
		 return this->numEntries;
	 }

	 /**
	  *	Enables or disables comparison of proteins.
	  *
	  *	@param ce true to enable, false do disable
	  */
	 void SetComparisonEnabled(bool ce) {
		 this->comparisonEnabled = ce;
	 }

	 /**
	  *	Returns if the comparison of proteins is enabled
	  *
	  *	@return true, if comparison is enabled, false otherwise
	  */
	 bool GetComparisonEnabled() {
		 return this->comparisonEnabled;
	 }

	 /**
	  *	Sets the dirty flag for the color module
	  *
	  *	@param d true, if the module is dirty, false otherwise
	  */
	 void SetDirty(bool d) {
		 this->isDirty = d;
	 }

	 /**
	  *	Returns if the dirty flag is set
	  *
	  *	@param true, if the dirty flag is set, false otherwise
	  */
	 bool IsDirty() {
		 return this->isDirty;
	 }

	 /**
	  *	Sets the frame ID for the MolecularDataCall
	  *
	  *	@param fid The frame id
	  */
	 void SetFrameID(unsigned int fid) {
		 this->frameID = fid;
	 }

	 /**
	  *	Returns the frame ID of the MolecularDataCall
	  *
	  *	@return The frame ID
	  */
	 unsigned int GetFrameID() {
		 return this->frameID; 
	 }

private:

	/** The color table for the given protein.*/
	vislib::Array<float> * atomColorTable;

	/** The color lookup table.*/
	vislib::Array<vislib::math::Vector<float, 3> > * colorLookupTable;

	/** The rainbow colors.*/
	vislib::Array<vislib::math::Vector< float, 3> > * rainbowColors;

	/** The protein that the color table is computed for*/
	megamol::protein_calls::MolecularDataCall * mol1;
	
	/** The binding site call*/
	const protein_calls::BindingSiteCall * bs;

	/** Should the color table be recomputed?*/
	bool forceRecompute;
	
	/** Do we want a weighted color table?*/
	bool isWeighted;

	/** Number of entries the rainbow color table should have*/
	unsigned int numEntries;

	/** The current frame id for the MolecularDataCall*/
	unsigned int frameID;

	/** Is the comparison of two atoms enabled?*/
	bool comparisonEnabled;

	/** Is the color-module dirty?*/
	bool isDirty;
};

	/** Description class typedef */
	typedef megamol::core::factories::CallAutoDescription<CallColor> CallColorDescription;

} // end namespace protein
} // end namespace megamol

#endif // MMPROTEINPLUGIN_CALLCOLOR_H_INCLUDED