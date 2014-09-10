/*
 * ColorModule.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include <stdafx.h>

#define _USE_MATH_DEFINES 1

#include "ColorModule.h"
#include "RMS.h"
#include "param/EnumParam.h"
#include "param/FloatParam.h"
#include "param/StringParam.h"
#include "utility/ColourParser.h"
#include "vislib/ASCIIFileBuffer.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include <iostream>
#include <vector>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;

/*
 * ColorModule::ColorModule
 */
ColorModule::ColorModule(void) : core::Module(),
	colorOutSlot("colorOut", "The slot providing the computed colors"),
	colorTableFileParam( "colorTableFilename", "The filename of the color table."),
	minGradColorParam("minGradColor", "The color for the minimum value for gradient coloring"),
	midGradColorParam("midGradColor", "The color for the middle value for gradient coloring"),
	maxGradColorParam("maxGradColor", "The color for the maximum value for gradient coloring"),
	coloringMode0Param("coloringMode0", "The first coloring mode."),
	coloringMode1Param("coloringMode1", "The second coloring mode."),
	colorParam("comparison::comparisonColor", "The color that should be used for comparison"),
	comparisonModeParam("comparison::comparisonMode", "The comparison mode"),
	comparisonColorParam("comparison::comparisonColoringMode", "The comparison coloring mode"),
	weightingParam("weighting", "The color weighting factor"),
	minDistanceParam("comparison::minDistance","The lower end of the colored distance interval"),
	maxDistanceParam("comparison::maxDistance","The upper end of the colored distance interval"),
	molDataCallerSlot("getdataCompare","Connects the protein rendering with the protein data storage of the comparison base protein") {

	vislib::StringA filename("colors.txt");
	ReadColorTableFromFile(filename, this->colorLookupTable);
	this->colorTableFileParam.SetParameter(new param::StringParam(A2T(filename)));
	this->MakeSlotAvailable(&this->colorTableFileParam);

	// the color for the minimum value (gradient coloring)
    this->minGradColorParam.SetParameter(new param::StringParam( "#0000ff"));
    this->MakeSlotAvailable( &this->minGradColorParam);

    // the color for the middle value (gradient coloring)
    this->midGradColorParam.SetParameter(new param::StringParam( "#00ff00"));
    this->MakeSlotAvailable( &this->midGradColorParam);

    // the color for the maximum value (gradient coloring)
    this->maxGradColorParam.SetParameter(new param::StringParam( "#ff0000"));
    this->MakeSlotAvailable( &this->maxGradColorParam);

	this->colorOutSlot.SetCallback(CallColor::ClassName(), CallColor::FunctionName(0), &ColorModule::getColor);
	this->colorOutSlot.SetCallback(CallColor::ClassName(), CallColor::FunctionName(1), &ColorModule::getExtents);
	this->MakeSlotAvailable(&this->colorOutSlot);

	this->colorParam.SetParameter(new param::StringParam("#ff0000"));
	this->MakeSlotAvailable( &this->colorParam);

	this->currentColoringMode0 = ColoringMode::CHAIN;
	this->currentColoringMode1 = ColoringMode::STRUCTURE;
	param::EnumParam *cm0 = new param::EnumParam ( int(this->currentColoringMode0));
	param::EnumParam *cm1 = new param::EnumParam ( int(this->currentColoringMode1));
	unsigned int cCnt;
	ColoringMode cMode;
	for( cCnt = 0; cCnt < getNumOfColoringModes(); ++cCnt) {
		cMode = getModeByIndex(cCnt);
		cm0->SetTypePair(cMode, getName(cMode).c_str());
		cm1->SetTypePair(cMode, getName(cMode).c_str());
	}
	this->coloringMode0Param << cm0;
	this->coloringMode1Param << cm1;
	this->MakeSlotAvailable(&this->coloringMode0Param);
	this->MakeSlotAvailable(&this->coloringMode1Param);

	this->currentComparisonMode = ComparisonMode::ZERO_TO_MAX;
	param::EnumParam *com = new param::EnumParam ( int(this->currentComparisonMode));
	ComparisonMode comMode;
	for( cCnt = 0; cCnt < getNumOfColoringModes(); ++cCnt) {
		comMode = getComparisonModeByIndex(cCnt);
		com->SetTypePair(comMode, getName(comMode).c_str());
	}
	this->comparisonModeParam << com;
	this->MakeSlotAvailable(&this->comparisonModeParam);

	this->currentComparisonColoringMode = ComparisonColoringMode::SINGLE_COLOR;
	param::EnumParam *ccm = new param::EnumParam ( int(this->currentComparisonColoringMode));
	ComparisonColoringMode ccMode;
	for( cCnt = 0; cCnt < getNumOfColoringModes(); ++cCnt) {
		ccMode = getComparisonColoringModeByIndex(cCnt);
		ccm->SetTypePair(ccMode, getName(ccMode).c_str());
	}
	this->comparisonColorParam << ccm;
	this->MakeSlotAvailable(&this->comparisonColorParam);

	this->weightingParam.SetParameter(new param::FloatParam(0.5f, 0.0f, 1.0f));
	this->MakeSlotAvailable(&this->weightingParam);

	this->weight0 = 0.5;
	this->weight1 = 0.5;

	this->minDistanceParam.SetParameter(new param::FloatParam(0.0, 0.0));
	this->MakeSlotAvailable(&this->minDistanceParam);

	this->maxDistanceParam.SetParameter(new param::FloatParam(1.0, 0.0));
	this->MakeSlotAvailable(&this->maxDistanceParam);

	this->molDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->molDataCallerSlot);
}

bool ColorModule::updateParams() {
	bool retVal = false;

	if(weightingParam.IsDirty()) {
		weight0 = weightingParam.Param<param::FloatParam>()->Value();
		weight1 = 1.0f - weight0;
		weightingParam.ResetDirty();
		retVal = true;
	}

	if(coloringMode0Param.IsDirty()) {
		this->currentColoringMode0 = static_cast<ColorModule::ColoringMode>(int(this->coloringMode0Param.Param<param::EnumParam>()->Value()));
		this->coloringMode0Param.ResetDirty();
		retVal = true;
	}

	if(coloringMode1Param.IsDirty()) {
		this->currentColoringMode1 = static_cast<ColorModule::ColoringMode>(int(this->coloringMode1Param.Param<param::EnumParam>()->Value()));
		this->coloringMode1Param.ResetDirty();
		retVal = true;
	}

	if(comparisonModeParam.IsDirty()) {
		this->currentComparisonMode = static_cast<ColorModule::ComparisonMode>(int(this->comparisonModeParam.Param<param::EnumParam>()->Value()));
		this->comparisonModeParam.ResetDirty();
		retVal = true;
	}

	if(comparisonColorParam.IsDirty()) {
		this->currentComparisonColoringMode = static_cast<ColorModule::ComparisonColoringMode>(int(this->comparisonColorParam.Param<param::EnumParam>()->Value()));
		this->comparisonColorParam.ResetDirty();
		retVal = true;
	}

	// check other slots
	if(colorTableFileParam.IsDirty() || minGradColorParam.IsDirty() || midGradColorParam.IsDirty() ||
		maxGradColorParam.IsDirty() || colorParam.IsDirty() || minDistanceParam.IsDirty() || 
		maxDistanceParam.IsDirty()) {
		retVal = true; 
	}

	return retVal;
}

/*
 * ColorModule::~ColorModule
 */
ColorModule::~ColorModule(void) {
	this->Release();
}

/*
 * ColorModule::create
 */
bool ColorModule::create(void) {
	return true;
}

/*
 * ColorModule::release
 */
void ColorModule::release(void) {

}

/*
 *	ColorModule::getData
 */
bool ColorModule::getColor(core::Call& call) {
	updateParams();

	CallColor * col = dynamic_cast<CallColor*>(&call);
	if(col == NULL) return false;

	MolecularDataCall * mol2 = molDataCallerSlot.CallAs<MolecularDataCall>();

	if(mol2 != NULL) {
		if(!(*mol2)(1)) return false; // Get Extents
		mol2->SetFrameID(static_cast<int>(col->GetFrameID()));
		if(!(*mol2)(MolecularDataCall::CallForGetData)) return false; // Get Data
	}

	vislib::Array<float>* atomColorTable = col->GetAtomColorTable();

	vislib::Array<vislib::math::Vector<float, 3> >* rainbowColorTable = col->GetRainbowColorTable();
	vislib::Array<vislib::math::Vector<float, 3> >* cLT = col->GetColorLookupTable();

	ReadColorTableFromFile(this->colorTableFileParam.Param<param::StringParam>()->Value(),
		*cLT);

	MakeRainbowColorTable(col->GetNumEntries(), 
		*rainbowColorTable);
	
	if(!col->GetComparisonEnabled()) {
		if(col->GetWeighted()) {
			MakeColorTable(col->GetColoringTarget(),
				this->currentColoringMode0,
				this->currentColoringMode1,
				*atomColorTable,
				*cLT,
				*rainbowColorTable,
				col->GetForceRecompute(),
				col->GetBindingSiteCall());

			col->SetAtomColorTable(atomColorTable);
		} else {
			MakeColorTable(col->GetColoringTarget(),
				this->currentColoringMode0,
				*atomColorTable,
				*cLT,
				*rainbowColorTable,
				col->GetForceRecompute(),
				col->GetBindingSiteCall());

			col->SetAtomColorTable(atomColorTable);
		}
	}

	if(col->GetComparisonEnabled()) {
		MakeComparisonColorTable(col->GetColoringTarget(), 
			this->currentColoringMode0,
			*atomColorTable,
			*cLT,
			*rainbowColorTable,
			col->GetFrameID(),
			col->GetForceRecompute(),
			col->GetBindingSiteCall());

		col->SetAtomColorTable(atomColorTable);
	}

	col->SetRainbowColorTable(rainbowColorTable);
	col->SetColorLookupTable(cLT);

	return true;
}

/*
 * ColorModule::getExtent
 */
bool ColorModule::getExtents(core::Call& call) {
	bool ret = updateParams();

	CallColor * col = dynamic_cast<CallColor*>(&call);
	if(col == NULL) return false;

	if(ret)
		col->SetDirty(true);

	return true;
}

/*
 * ColorModule::ReadColorTableFromFile
 */
void ColorModule::ReadColorTableFromFile(vislib::StringA filename,
	vislib::Array<vislib::math::Vector<float, 3> > &colorLookupTable) {

	// file buffer variable
    vislib::sys::ASCIIFileBuffer file;
    // delete old color table
    colorLookupTable.SetCount( 0);
    // try to load the color table file
    if( file.LoadFile( filename) ) {
        float r, g, b;
        colorLookupTable.AssertCapacity( file.Count());
        // get colors from file
        for( unsigned int cnt = 0; cnt < file.Count(); ++cnt ) {
        	vislib::StringA lineStr( file.Line(cnt) );
            if( lineStr.Length() > 0 && utility::ColourParser::FromString( lineStr, r, g, b) ) {
                colorLookupTable.Add( vislib::math::Vector<float, 3>
                  ( r, g, b));
            }
        }
    }
    // if the file could not be loaded or contained no valid colors
    if( colorLookupTable.Count() == 0 ) {
        // set default color table
        colorLookupTable.SetCount( 25);
        colorLookupTable[0].Set( 0.70f, 0.8f, 0.4f);
        colorLookupTable[1].Set( 1.0f, 0.0f, 0.0f);
        colorLookupTable[2].Set( 1.0f, 1.0f, 0.0f);
        colorLookupTable[3].Set( 0.0f, 1.0f, 0.0f);
        colorLookupTable[4].Set( 0.0f, 1.0f, 1.0f);
        colorLookupTable[5].Set( 0.0f, 0.0f, 1.0f);
        colorLookupTable[6].Set( 1.0f, 0.0f, 1.0f);
        colorLookupTable[7].Set( 0.5f, 0.0f, 0.0f);
        colorLookupTable[8].Set( 0.5f, 0.5f, 0.0f);
        colorLookupTable[9].Set( 0.0f, 0.5f, 0.0f);
        colorLookupTable[10].Set( 0.00f, 0.50f, 0.50f);
        colorLookupTable[11].Set( 0.00f, 0.00f, 0.50f);
        colorLookupTable[12].Set( 0.50f, 0.00f, 0.50f);
        colorLookupTable[13].Set( 1.00f, 0.50f, 0.00f);
        colorLookupTable[14].Set( 0.00f, 0.50f, 1.00f);
        colorLookupTable[15].Set( 1.00f, 0.50f, 1.00f);
        colorLookupTable[16].Set( 0.50f, 0.25f, 0.00f);
        colorLookupTable[17].Set( 1.00f, 1.00f, 0.50f);
        colorLookupTable[18].Set( 0.50f, 1.00f, 0.50f);
        colorLookupTable[19].Set( 0.75f, 1.00f, 0.00f);
        colorLookupTable[20].Set( 0.50f, 0.00f, 0.75f);
        colorLookupTable[21].Set( 1.00f, 0.50f, 0.50f);
        colorLookupTable[22].Set( 0.75f, 1.00f, 0.75f);
        colorLookupTable[23].Set( 0.75f, 0.75f, 0.50f);
        colorLookupTable[24].Set( 1.00f, 0.75f, 0.50f);
    }
}

/*
 * ColorModule::FillAminoAcidColorTable
 */
void ColorModule::FillAminoAcidColorTable(
	vislib::Array<vislib::math::Vector<float, 3> > &aminoAcidColorTable) {

	aminoAcidColorTable.Clear();
    aminoAcidColorTable.SetCount( 25);
    aminoAcidColorTable[0].Set( 0.5f, 0.5f, 0.5f);
    aminoAcidColorTable[1].Set( 1.0f, 0.0f, 0.0f);
    aminoAcidColorTable[2].Set( 1.0f, 1.0f, 0.0f);
    aminoAcidColorTable[3].Set( 0.0f, 1.0f, 0.0f);
    aminoAcidColorTable[4].Set( 0.0f, 1.0f, 1.0f);
    aminoAcidColorTable[5].Set( 0.0f, 0.0f, 1.0f);
    aminoAcidColorTable[6].Set( 1.0f, 0.0f, 1.0f);
    aminoAcidColorTable[7].Set( 0.5f, 0.0f, 0.0f);
    aminoAcidColorTable[8].Set( 0.5f, 0.5f, 0.0f);
    aminoAcidColorTable[9].Set( 0.0f, 0.5f, 0.0f);
    aminoAcidColorTable[10].Set( 0.0f, 0.5f, 0.5f);
    aminoAcidColorTable[11].Set( 0.0f, 0.0f, 0.5f);
    aminoAcidColorTable[12].Set( 0.5f, 0.0f, 0.5f);
    aminoAcidColorTable[13].Set( 1.0f, 0.5f, 0.0f);
    aminoAcidColorTable[14].Set( 0.0f, 0.5f, 1.0f);
    aminoAcidColorTable[15].Set( 1.0f, 0.5f, 1.0f);
    aminoAcidColorTable[16].Set( 0.5f, 0.25f, 0.0f);
    aminoAcidColorTable[17].Set( 1.0f, 1.0f, 0.5f);
    aminoAcidColorTable[18].Set( 0.5f, 1.0f, 0.5f);
    aminoAcidColorTable[19].Set( 0.75f, 1.0f, 0.0f);
    aminoAcidColorTable[20].Set( 0.5f, 0.0f, 0.75f);
    aminoAcidColorTable[21].Set( 1.0f, 0.5f, 0.5f);
    aminoAcidColorTable[22].Set( 0.75f, 1.0f, 0.75f);
    aminoAcidColorTable[23].Set( 0.75f, 0.75f, 0.5f);
    aminoAcidColorTable[24].Set( 1.0f, 0.75f, 0.5f);
}

std::string ColorModule::getName(ColorModule::ColoringMode col) {
	switch(col) {
        case ELEMENT     : return "Element";
        case STRUCTURE   : return "Structure";
        case RAINBOW     : return "Rainbow";
        case BFACTOR     : return "BFactor";
        case CHARGE      : return "Charge";
        case OCCUPANCY   : return "Occupancy";
        case CHAIN       : return "Chain";
        case MOLECULE    : return "Molecule";
        case RESIDUE     : return "Residue";
        case CHAINBOW    : return "Chainbow";
        case AMINOACID   : return "Aminoacid";
        case VALUE       : return "Value";
        case CHAIN_ID    : return "ChainID";
        case MOVEMENT    : return "Movement";
        case BINDINGSITE : return "BindingSite";
        default : return "";
    }
}

std::string ColorModule::getName(ColorModule::ComparisonMode col) {
	switch(col) {
		case ZERO_TO_MAX : return "0 .. max";
		case ZERO_TO_VALUE : return "0 .. y";
		case MIN_TO_MAX : return "min .. max";
		case MIN_TO_VALUE : return "min .. y";
		case VALUE_TO_MAX : return "x .. max";
		case VALUE_TO_VALUE : return "x .. y";
		default : return "";
	}
}

std::string ColorModule::getName(ColorModule::ComparisonColoringMode col) {
	switch(col) {
		case SINGLE_COLOR : return "Single Color";
		case COLOR_GRADIENT : return "Color Gradient";
		case TWO_COLORS : return "Two Colors";
		default : return "";
	}
}

/*
 * ColorModule::GetModeByIndex
 */
ColorModule::ColoringMode ColorModule::getModeByIndex(unsigned int idx) {
    switch(idx) {
        case 0 : return ColoringMode::ELEMENT;
        case 1 : return ColoringMode::RESIDUE;
        case 2 : return ColoringMode::STRUCTURE;
        case 3 : return ColoringMode::BFACTOR;
        case 4 : return ColoringMode::CHARGE;
        case 5 : return ColoringMode::OCCUPANCY;
        case 6 : return ColoringMode::CHAIN;
        case 7 : return ColoringMode::MOLECULE;
        case 8 : return ColoringMode::RAINBOW;
        case 9 : return ColoringMode::BINDINGSITE;
        default : return ColoringMode::ELEMENT;
    }
}

ColorModule::ComparisonMode ColorModule::getComparisonModeByIndex(unsigned int idx) {
	switch(idx) {
		case 0 : return ComparisonMode::ZERO_TO_MAX;
		case 1 : return ComparisonMode::ZERO_TO_VALUE;
		case 2 : return ComparisonMode::MIN_TO_MAX;
		case 3 : return ComparisonMode::MIN_TO_VALUE;
		case 4 : return ComparisonMode::VALUE_TO_MAX;
		case 5 : return ComparisonMode::VALUE_TO_VALUE;
		default : return ComparisonMode::ZERO_TO_MAX;
	}
}

ColorModule::ComparisonColoringMode ColorModule::getComparisonColoringModeByIndex(unsigned int idx) {
	switch(idx) {
		case 0 : return ComparisonColoringMode::SINGLE_COLOR;
		case 1 : return ComparisonColoringMode::TWO_COLORS;
		case 2 : return ComparisonColoringMode::COLOR_GRADIENT;
		default : return ComparisonColoringMode::SINGLE_COLOR;
	}
}

/*
 * Creates a rainbow color table with 'num' entries.
 */
void ColorModule::MakeRainbowColorTable( unsigned int num,
    vislib::Array<vislib::math::Vector<float, 3> > &rainbowColors) {

    unsigned int n = (num/4);
    // the color table should have a minimum size of 16
    if( n < 4 )
        n = 4;
    rainbowColors.Clear();
    rainbowColors.AssertCapacity( num);
    float f = 1.0f/float(n);
    vislib::math::Vector<float,3> color;
    color.Set( 1.0f, 0.0f, 0.0f);
    for( unsigned int i = 0; i < n; i++) {
        color.SetY( vislib::math::Min( color.GetY() + f, 1.0f));
        rainbowColors.Add( color);
    }
    for( unsigned int i = 0; i < n; i++) {
        color.SetX( vislib::math::Max( color.GetX() - f, 0.0f));
        rainbowColors.Add( color);
    }
    for( unsigned int i = 0; i < n; i++) {
        color.SetZ( vislib::math::Min( color.GetZ() + f, 1.0f));
        rainbowColors.Add( color);
    }
    for( unsigned int i = 0; i < n; i++) {
        color.SetY( vislib::math::Max( color.GetY() - f, 0.0f));
        rainbowColors.Add( color);
    }
}

/*
 * ColorModule::MakeColorTable
 */
void ColorModule::MakeColorTable(const MolecularDataCall *mol,
	ColoringMode cm0,
    ColoringMode cm1,
    vislib::Array<float> &atomColorTable,
    vislib::Array<vislib::math::Vector<float, 3> > &colorLookupTable,
    vislib::Array<vislib::math::Vector<float, 3> > &rainbowColors,
    bool forceRecompute,
    const BindingSiteCall *bs) {
	
	// if recomputation is forced: clear current color table
    if(forceRecompute) {
        atomColorTable.Clear();
    }

    // Clamp weights to zero
    if(weight0 < 0.0) weight0 = 0.0;
    if(weight1 < 0.0) weight1 = 0.0;

    // Normalize weights
    weight0 = weight0 / (weight0 + weight1);
    weight1 = weight1 / (weight0 + weight1);

    // only compute color table if necessary
    if(atomColorTable.IsEmpty()) {

        vislib::Array<float> color0;
        vislib::Array<float> color1;

        // reserve memory for all atoms
        atomColorTable.AssertCapacity(mol->AtomCount()*3);

        // Compute first color table
        MakeColorTable(mol, cm0, color0, colorLookupTable, rainbowColors,
            true, bs);

        // Compute second color table
        MakeColorTable(mol, cm1, color1, colorLookupTable, rainbowColors,
            true, bs);

        // Interpolate
        for(unsigned int cnt = 0; cnt < mol->AtomCount()*3; cnt++) {
            atomColorTable.Add(color0[cnt]*weight0+color1[cnt]*weight1);
        }

    }
}

/*
 * ColorModule::MakeColorTable
 */
void ColorModule::MakeColorTable(const MolecularDataCall *mol,
	ColoringMode currentColoringMode,
    vislib::Array<float> &atomColorTable,
    vislib::Array<vislib::math::Vector<float, 3> > &colorLookupTable,
    vislib::Array<vislib::math::Vector<float, 3> > &rainbowColors,
    bool forceRecompute,
    const BindingSiteCall *bs) {
	
	// temporary variables
    unsigned int cnt, idx, cntAtom, cntRes, cntChain, cntMol, cntSecS, atomIdx,
        atomCnt;
    vislib::math::Vector<float, 3> color;
    float r, g, b;

	vislib::TString minGradColor = this->minGradColorParam.Param<param::StringParam>()->Value();
	vislib::TString midGradColor = this->midGradColorParam.Param<param::StringParam>()->Value();
	vislib::TString maxGradColor = this->maxGradColorParam.Param<param::StringParam>()->Value();

    // if recomputation is forced: clear current color table
    if( forceRecompute ) {
        atomColorTable.Clear();
    }
    // reserve memory for all atoms
    atomColorTable.AssertCapacity( mol->AtomCount() * 3 );

    // only compute color table if necessary
    if( atomColorTable.IsEmpty() ) {
        if( currentColoringMode == ELEMENT ) {
            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {

                atomColorTable.Add( float( mol->AtomTypes()[mol->
                    AtomTypeIndices()[cnt]].Colour()[0]) / 255.0f);

                atomColorTable.Add( float( mol->AtomTypes()[mol->
                    AtomTypeIndices()[cnt]].Colour()[1]) / 255.0f);

                atomColorTable.Add( float( mol->AtomTypes()[mol->
                    AtomTypeIndices()[cnt]].Colour()[2]) / 255.0f);
            }
        } // ... END coloring mode ELEMENT
        else if( currentColoringMode == RESIDUE ) {
            unsigned int resTypeIdx;
            // loop over all residues
            for( cntRes = 0; cntRes < mol->ResidueCount(); ++cntRes ) {
                // loop over all atoms of the current residue
                idx = mol->Residues()[cntRes]->FirstAtomIndex();
                cnt = mol->Residues()[cntRes]->AtomCount();
                // get residue type index
                resTypeIdx = mol->Residues()[cntRes]->Type();
                for( cntAtom = idx; cntAtom < idx + cnt; ++cntAtom ) {
                    // Special cases for water/toluol/methanol
                    if (mol->ResidueTypeNames()[resTypeIdx].Equals("SOL")) {
                        // Water
                        atomColorTable.Add(0.3f);
                        atomColorTable.Add(0.3f);
                        atomColorTable.Add(1.0f);
                    } else if(mol->ResidueTypeNames()[resTypeIdx].Equals("MeOH")) {
                        // methanol
                        atomColorTable.Add(1.0f);
                        atomColorTable.Add(0.5f);
                        atomColorTable.Add(0.0f);
                    }else if(mol->ResidueTypeNames()[resTypeIdx].Equals("TOL")) {
                        // toluol
                        atomColorTable.Add(1.0f);
                        atomColorTable.Add(1.0f);
                        atomColorTable.Add(1.0f);
                    }
                    else {
                    atomColorTable.Add( colorLookupTable[resTypeIdx%
                        colorLookupTable.Count()].X());
                    atomColorTable.Add( colorLookupTable[resTypeIdx%
                        colorLookupTable.Count()].Y());
                    atomColorTable.Add( colorLookupTable[resTypeIdx%
                        colorLookupTable.Count()].Z());
                    }
                }
            }
        } // ... END coloring mode RESIDUE
        else if( currentColoringMode == STRUCTURE ) {
            utility::ColourParser::FromString( "#00ff00", r, g, b);
            vislib::math::Vector<float, 3> colNone( r, g, b);
            utility::ColourParser::FromString( "#ff0000", r, g, b);
            vislib::math::Vector<float, 3> colHelix( r, g, b);
            utility::ColourParser::FromString( "#0000ff", r, g, b);
            vislib::math::Vector<float, 3> colSheet( r, g, b);
            utility::ColourParser::FromString( "#888888", r, g, b);
            vislib::math::Vector<float, 3> colRCoil( r, g, b);
            // loop over all atoms and fill the table with the default color
            for( cntAtom = 0; cntAtom < mol->AtomCount(); ++cntAtom ) {
                atomColorTable.Add( colNone.X());
                atomColorTable.Add( colNone.Y());
                atomColorTable.Add( colNone.Z());
            }
            // write colors for sec structure elements
            MolecularDataCall::SecStructure::ElementType elemType;

            for( cntSecS = 0; cntSecS < mol->SecondaryStructureCount();
                 ++cntSecS ) {

                idx = mol->SecondaryStructures()[cntSecS].FirstAminoAcidIndex();
                cnt = idx +
                    mol->SecondaryStructures()[cntSecS].AminoAcidCount();
                elemType = mol->SecondaryStructures()[cntSecS].Type();

                for( cntRes = idx; cntRes < cnt; ++cntRes ) {
                    atomIdx = mol->Residues()[cntRes]->FirstAtomIndex();
                    atomCnt = atomIdx + mol->Residues()[cntRes]->AtomCount();
                    for( cntAtom = atomIdx; cntAtom < atomCnt; ++cntAtom ) {
                        if( elemType ==
                            MolecularDataCall::SecStructure::TYPE_HELIX ) {
                            atomColorTable[3*cntAtom+0] = colHelix.X();
                            atomColorTable[3*cntAtom+1] = colHelix.Y();
                            atomColorTable[3*cntAtom+2] = colHelix.Z();
                        } else if( elemType ==
                            MolecularDataCall::SecStructure::TYPE_SHEET ) {
                            atomColorTable[3*cntAtom+0] = colSheet.X();
                            atomColorTable[3*cntAtom+1] = colSheet.Y();
                            atomColorTable[3*cntAtom+2] = colSheet.Z();
                        } else if( elemType ==
                            MolecularDataCall::SecStructure::TYPE_COIL ) {
                            atomColorTable[3*cntAtom+0] = colRCoil.X();
                            atomColorTable[3*cntAtom+1] = colRCoil.Y();
                            atomColorTable[3*cntAtom+2] = colRCoil.Z();
                        }
                    }
                }
            }
        } // ... END coloring mode STRUCTURE
        else if( currentColoringMode == BFACTOR ) {
            float r, g, b;
            // get min color
            utility::ColourParser::FromString(
               minGradColor,
                r, g, b);
            vislib::math::Vector<float, 3> colMin( r, g, b);
            // get mid color
            utility::ColourParser::FromString(
                midGradColor,
                r, g, b);
            vislib::math::Vector<float, 3> colMid( r, g, b);
            // get max color
            utility::ColourParser::FromString(
                maxGradColor,
                r, g, b);
            vislib::math::Vector<float, 3> colMax( r, g, b);
            // temp color variable
            vislib::math::Vector<float, 3> col;

            float min( mol->MinimumBFactor());
            float max( mol->MaximumBFactor());
            float mid( ( max - min)/2.0f + min );
            float val;

            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                if( min == max ) {
                    atomColorTable.Add( colMid.GetX() );
                    atomColorTable.Add( colMid.GetY() );
                    atomColorTable.Add( colMid.GetZ() );
                    continue;
                }

                val = mol->AtomBFactors()[cnt];
                // below middle value --> blend between min and mid color
                if( val < mid ) {
                    col = colMin + ( ( colMid - colMin ) / ( mid - min) ) *
                        ( val - min );
                    atomColorTable.Add( col.GetX() );
                    atomColorTable.Add( col.GetY() );
                    atomColorTable.Add( col.GetZ() );
                }
                // above middle value --> blend between max and mid color
                else if( val > mid ) {
                    col = colMid + ( ( colMax - colMid ) / ( max - mid) ) *
                        ( val - mid );
                    atomColorTable.Add( col.GetX() );
                    atomColorTable.Add( col.GetY() );
                    atomColorTable.Add( col.GetZ() );
                }
                // middle value --> assign mid color
                else {
                    atomColorTable.Add( colMid.GetX() );
                    atomColorTable.Add( colMid.GetY() );
                    atomColorTable.Add( colMid.GetZ() );
                }
            }
        } // ... END coloring mode BFACTOR
        else if( currentColoringMode == CHARGE ) {
            float r, g, b;
            // get min color
            utility::ColourParser::FromString(
                minGradColor,
                r, g, b);
            vislib::math::Vector<float, 3> colMin( r, g, b);
            // get mid color
            utility::ColourParser::FromString(
                midGradColor,
                r, g, b);
            vislib::math::Vector<float, 3> colMid( r, g, b);
            // get max color
            utility::ColourParser::FromString(
                maxGradColor,
                r, g, b);
            vislib::math::Vector<float, 3> colMax( r, g, b);
            // temp color variable
            vislib::math::Vector<float, 3> col;

            float min( mol->MinimumCharge());
            float max( mol->MaximumCharge());
            float mid( ( max - min)/2.0f + min );
            float val;

            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                if( min == max ) {
                    atomColorTable.Add( colMid.GetX() );
                    atomColorTable.Add( colMid.GetY() );
                    atomColorTable.Add( colMid.GetZ() );
                    continue;
                }

                val = mol->AtomCharges()[cnt];
                // below middle value --> blend between min and mid color
                if( val < mid ) {
                    col = colMin + ( ( colMid - colMin ) / ( mid - min) ) *
                        ( val - min );
                    atomColorTable.Add( col.GetX() );
                    atomColorTable.Add( col.GetY() );
                    atomColorTable.Add( col.GetZ() );
                }
                // above middle value --> blend between max and mid color
                else if( val > mid ) {
                    col = colMid + ( ( colMax - colMid ) / ( max - mid) ) *
                        ( val - mid );
                    atomColorTable.Add( col.GetX() );
                    atomColorTable.Add( col.GetY() );
                    atomColorTable.Add( col.GetZ() );
                }
                // middle value --> assign mid color
                else {
                    atomColorTable.Add( colMid.GetX() );
                    atomColorTable.Add( colMid.GetY() );
                    atomColorTable.Add( colMid.GetZ() );
                }
            }
        } // ... END coloring mode CHARGE
        else if( currentColoringMode == OCCUPANCY ) {
            float r, g, b;
            // get min color
            utility::ColourParser::FromString(
                minGradColor,
                r, g, b);
            vislib::math::Vector<float, 3> colMin( r, g, b);
            // get mid color
            utility::ColourParser::FromString(
                midGradColor,
                r, g, b);
            vislib::math::Vector<float, 3> colMid( r, g, b);
            // get max color
            utility::ColourParser::FromString(
                maxGradColor,
                r, g, b);
            vislib::math::Vector<float, 3> colMax( r, g, b);
            // temp color variable
            vislib::math::Vector<float, 3> col;

            float min( mol->MinimumOccupancy());
            float max( mol->MaximumOccupancy());
            float mid( ( max - min)/2.0f + min );
            float val;

            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                if( min == max ) {
                    atomColorTable.Add( colMid.GetX() );
                    atomColorTable.Add( colMid.GetY() );
                    atomColorTable.Add( colMid.GetZ() );
                    continue;
                }

                val = mol->AtomOccupancies()[cnt];
                // below middle value --> blend between min and mid color
                if( val < mid ) {
                    col = colMin + ( ( colMid - colMin ) / ( mid - min) ) *
                        ( val - min );
                    atomColorTable.Add( col.GetX() );
                    atomColorTable.Add( col.GetY() );
                    atomColorTable.Add( col.GetZ() );
                }
                // above middle value --> blend between max and mid color
                else if( val > mid ) {
                    col = colMid + ( ( colMax - colMid ) / ( max - mid) ) *
                        ( val - mid );
                    atomColorTable.Add( col.GetX() );
                    atomColorTable.Add( col.GetY() );
                    atomColorTable.Add( col.GetZ() );
                }
                // middle value --> assign mid color
                else {
                    atomColorTable.Add( colMid.GetX() );
                    atomColorTable.Add( colMid.GetY() );
                    atomColorTable.Add( colMid.GetZ() );
                }
            }
        } // ... END coloring mode OCCUPANCY
        else if( currentColoringMode == CHAIN ) {
            // get the last atom of the last res of the last mol of the first chain
            cntChain = 0;
            cntMol = mol->Chains()[cntChain].MoleculeCount() - 1;
            cntRes = mol->Molecules()[cntMol].FirstResidueIndex()
                + mol->Molecules()[cntMol].ResidueCount() - 1;
            cntAtom = mol->Residues()[cntRes]->FirstAtomIndex()
                + mol->Residues()[cntRes]->AtomCount() - 1;
            // get the first color
            idx = 0;
            color = colorLookupTable[idx%colorLookupTable.Count()];
            // loop over all atoms
            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                // check, if the last atom of the current chain is reached
                if( cnt > cntAtom ) {
                    // get the last atom of the last res of the last mol of the next chain
                    cntChain++;
                    cntMol = mol->Chains()[cntChain].FirstMoleculeIndex()
                        + mol->Chains()[cntChain].MoleculeCount() - 1;
                    cntRes = mol->Molecules()[cntMol].FirstResidueIndex()
                        + mol->Molecules()[cntMol].ResidueCount() - 1;
                    cntAtom = mol->Residues()[cntRes]->FirstAtomIndex()
                        + mol->Residues()[cntRes]->AtomCount() - 1;
                    // get the next color
                    idx++;
                    color = colorLookupTable[idx%colorLookupTable.Count()];

                }
                atomColorTable.Add( color.X());
                atomColorTable.Add( color.Y());
                atomColorTable.Add( color.Z());
            }
        } // ... END coloring mode CHAIN
        else if( currentColoringMode == MOLECULE ) {
            // get the last atom of the last res of the first mol
            cntMol = 0;
            cntRes = mol->Molecules()[cntMol].FirstResidueIndex()
                + mol->Molecules()[cntMol].ResidueCount() - 1;
            cntAtom = mol->Residues()[cntRes]->FirstAtomIndex()
                + mol->Residues()[cntRes]->AtomCount() - 1;
            // get the first color
            idx = 0;
            color = colorLookupTable[idx%colorLookupTable.Count()];
            // loop over all atoms
            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                // check, if the last atom of the current chain is reached
                if( cnt > cntAtom ) {
                    // get the last atom of the last res of the next mol
                    cntMol++;
                    cntRes = mol->Molecules()[cntMol].FirstResidueIndex()
                        + mol->Molecules()[cntMol].ResidueCount() - 1;
                    cntAtom = mol->Residues()[cntRes]->FirstAtomIndex()
                        + mol->Residues()[cntRes]->AtomCount() - 1;
                    // get the next color
                    idx++;
                    color = colorLookupTable[idx%colorLookupTable.Count()];

                }
                atomColorTable.Add( color.X());
                atomColorTable.Add( color.Y());
                atomColorTable.Add( color.Z());
            }
        } // ... END coloring mode MOLECULE
        else if( currentColoringMode == BINDINGSITE ) {
            // initialize all colors as white
            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                atomColorTable.Add( 1.0f);
                atomColorTable.Add( 1.0f);
                atomColorTable.Add( 1.0f);
            }
            // search for binding sites if BindingSiteCall is available
            if( bs ) {
                // temporary variables
                unsigned int firstMol;
                unsigned int firstRes;
                unsigned int firstAtom;
                unsigned int atomIdx;
                for( unsigned int cCnt = 0; cCnt < mol->ChainCount(); cCnt++ ) {
                    firstMol = mol->Chains()[cCnt].FirstMoleculeIndex();
                    for( unsigned int mCnt = firstMol; mCnt < firstMol + mol->Chains()[cCnt].MoleculeCount(); mCnt++ ) {
                        firstRes = mol->Molecules()[mCnt].FirstResidueIndex();
                        for( unsigned int rCnt = 0; rCnt < mol->Molecules()[mCnt].ResidueCount(); rCnt++ ) {
                            // try to match binding sites
                            vislib::Pair<char, unsigned int> bsRes;
                            // loop over all binding sites
                            for( unsigned int bsCnt = 0; bsCnt < bs->GetBindingSiteCount(); bsCnt++ ) {
                                for( unsigned int bsResCnt = 0; bsResCnt < bs->GetBindingSite(bsCnt)->Count(); bsResCnt++ ) {
                                    bsRes = bs->GetBindingSite(bsCnt)->operator[](bsResCnt);
                                    if( mol->Chains()[cCnt].Name() == bsRes.First() &&
                                        mol->Residues()[firstRes+rCnt]->OriginalResIndex() == bsRes.Second() &&
                                        mol->ResidueTypeNames()[mol->Residues()[firstRes+rCnt]->Type()] == bs->GetBindingSiteResNames(bsCnt)->operator[](bsResCnt) ) {
                                            // TODO loop over all atoms and add the color
                                        firstAtom = mol->Residues()[firstRes+rCnt]->FirstAtomIndex();
                                        for( unsigned int aCnt = 0; aCnt < mol->Residues()[firstRes+rCnt]->AtomCount(); aCnt++ ) {
                                            atomIdx = firstAtom + aCnt;
                                            atomColorTable[3 * atomIdx + 0] = bs->GetBindingSiteColor(bsCnt).X();
                                            atomColorTable[3 * atomIdx + 1] = bs->GetBindingSiteColor(bsCnt).Y();
                                            atomColorTable[3 * atomIdx + 2] = bs->GetBindingSiteColor(bsCnt).Z();
                                        }
                                    }
                                }
                            }
                        } // residues
                    } // molecules
                } // chains
            } // BindingSiteCall available
        } // ... END coloring mode BINDINGSITE
        else if( currentColoringMode == RAINBOW ) {
            for( cnt = 0; cnt < mol->AtomCount(); ++cnt ) {
                idx = int( ( float( cnt) / float( mol->AtomCount())) *
                    float( rainbowColors.Count()));
                color = rainbowColors[idx];
                atomColorTable.Add( color.GetX());
                atomColorTable.Add( color.GetY());
                atomColorTable.Add( color.GetZ());
            }
        } // ... END coloring mode RAINBOW
    }
}

void ColorModule::MakeComparisonColorTable(MolecularDataCall *mol1,
	ColoringMode currentColoringMode,
	vislib::Array<float> &atomColorTable,
	vislib::Array<vislib::math::Vector<float, 3> > &colorLookupTable,
	vislib::Array<vislib::math::Vector<float, 3> > &rainbowColors,
	unsigned int frameID,
	bool forceRecompute,
	const BindingSiteCall *bs) {

	// temporary variables
    unsigned int cntAtom;
    vislib::math::Vector<float, 3> color;
    //float r, g, b;

    // if recomputation is forced: clear current color table
    if( forceRecompute ) {
        atomColorTable.Clear();
    }

	// reserve memory for all atoms
    atomColorTable.AssertCapacity( mol1->AtomCount() * 3 );

	// fill table with the default color
	for( cntAtom = 0; cntAtom < mol1->AtomCount(); ++cntAtom ) {
		atomColorTable.Add(0.0);
		atomColorTable.Add(0.0);
		atomColorTable.Add(0.0);
	}

	MolecularDataCall *mol2 = this->molDataCallerSlot.CallAs<MolecularDataCall>();

	// vectors for the c alpha atoms
	std::vector<CallColor::cAlpha> cAlphas1;
	std::vector<CallColor::cAlpha> cAlphas2;

	// vectors for the c alpha positions (for later rms fitting)
	std::vector<float> pos1;
	std::vector<float> pos2;

	unsigned int ssc = mol1->ResidueCount();
	if(ssc > mol2->ResidueCount())
		ssc = mol2->ResidueCount();
		
	float max = -2.0f;
	float min;
	bool minSet = false;

	// do we have a second data slot?
	// if not, use the next timestep
	if(mol2 != NULL) {
		(*mol2)(1);
		mol2->SetFrameID(static_cast<int>(frameID));
		(*mol2)(MolecularDataCall::CallForGetData);

		// fill c alpha vectors
		for (unsigned int cntRes = 0; cntRes < ssc; cntRes++) {
		
			MolecularDataCall::AminoAcid * aminoacid2;

			if( mol1->Residues()[cntRes]->Identifier() == MolecularDataCall::Residue::AMINOACID
				&& mol2->Residues()[cntRes]->Identifier() == MolecularDataCall::Residue::AMINOACID) {
				aminoacid2 = (MolecularDataCall::AminoAcid*)(mol2->Residues()[cntRes]);
			} else {// TODO check if this is correct
				continue;
			}

			unsigned int ca2 = aminoacid2->CAlphaIndex();

			CallColor::cAlpha calpha2(mol2->AtomPositions()[3*ca2+0],
				mol2->AtomPositions()[3*ca2+1],
				mol2->AtomPositions()[3*ca2+2],
				ca2, 
				mol2->Residues()[cntRes]->FirstAtomIndex(),
				mol2->Residues()[cntRes]->FirstAtomIndex() + 
					mol2->Residues()[cntRes]->AtomCount() - 1);

			cAlphas2.push_back(calpha2);

			for(unsigned int i = 0; i < 3; i++) {
				pos2.push_back(mol2->AtomPositions()[3*ca2+i]);
			}
		}
	} else { // only one data call available, take next timestep

		// request next timestep
		(*mol1)(1);
		
		if(frameID + 1 <= mol1->FrameCount()) {
			mol1->SetFrameID(static_cast<int>(frameID+1));
		} else {
			mol1->SetFrameID(static_cast<int>(0));
		}

		(*mol1)(MolecularDataCall::CallForGetData);

		// fill c alpha vectors
		for (unsigned int cntRes = 0; cntRes < ssc; cntRes++) {
		
			MolecularDataCall::AminoAcid * aminoacid2;

			if( mol1->Residues()[cntRes]->Identifier() == MolecularDataCall::Residue::AMINOACID) {
				aminoacid2 = (MolecularDataCall::AminoAcid*)(mol1->Residues()[cntRes]);
			} else {// TODO check if this is correct
				continue;
			}

			unsigned int ca2 = aminoacid2->CAlphaIndex();

			CallColor::cAlpha calpha2(mol1->AtomPositions()[3*ca2+0],
				mol1->AtomPositions()[3*ca2+1],
				mol1->AtomPositions()[3*ca2+2],
				ca2, 
				mol1->Residues()[cntRes]->FirstAtomIndex(),
				mol1->Residues()[cntRes]->FirstAtomIndex() + 
					mol1->Residues()[cntRes]->AtomCount() - 1);

			cAlphas2.push_back(calpha2);

			for(unsigned int i = 0; i < 3; i++) {
				pos2.push_back(mol1->AtomPositions()[3*ca2+i]);
			}
		}

		// request previous timestep
		mol1->SetFrameID(static_cast<int>(frameID));
		(*mol1)(MolecularDataCall::CallForGetData);
	}

	// fill c alpha vector for the first molecule
	for (unsigned int cntRes = 0; cntRes < ssc; cntRes++) {
		
		MolecularDataCall::AminoAcid * aminoacid1;

		if( mol1->Residues()[cntRes]->Identifier() == MolecularDataCall::Residue::AMINOACID) {
			aminoacid1 = (MolecularDataCall::AminoAcid*)(mol1->Residues()[cntRes]);
		} else {// TODO check if this is correct
			continue;
		}

		unsigned int ca1 = aminoacid1->CAlphaIndex();

		CallColor::cAlpha calpha1(mol1->AtomPositions()[3*ca1+0],
			mol1->AtomPositions()[3*ca1+1],
			mol1->AtomPositions()[3*ca1+2],
			ca1, 
			mol1->Residues()[cntRes]->FirstAtomIndex(),
			mol1->Residues()[cntRes]->FirstAtomIndex() + 
				mol1->Residues()[cntRes]->AtomCount() - 1);

		cAlphas1.push_back(calpha1);
			
		for(unsigned int i = 0; i < 3; i++) {
			pos1.push_back(mol1->AtomPositions()[3*ca1+i]);
		}
	}

	// perform rms fitting
	std::vector<float> mass;
	std::vector<int> mask;
	float rotation[3][3], translation[3];

	for(unsigned int i = 0; i < cAlphas1.size(); i++) {
		mass.push_back(1.0f);
		mask.push_back(1);
	}

	int rmsValue;

	if(cAlphas1.size() >= cAlphas2.size()) {
		rmsValue = CalculateRMS( (unsigned int)cAlphas1.size(), true, 2, &mass[0], 
			&mask[0], &pos2[0], &pos1[0], rotation, translation);
	} else {
		rmsValue = CalculateRMS( (unsigned int)cAlphas2.size(), true, 2, &mass[0], 
			&mask[0], &pos2[0], &pos1[0], rotation, translation);
	}


	// compute distances
	for(unsigned int i = 0; i < cAlphas1.size(); i++) {
		float xDist = pos1[3*i+0] - pos2[3*i+0];
		float yDist = pos1[3*i+1] - pos2[3*i+1];
		float zDist = pos1[3*i+2] - pos2[3*i+2];
		
		float absoluteDist = sqrt(xDist*xDist + yDist*yDist + zDist*zDist);

		if(absoluteDist > max)
			max = absoluteDist;
		
		if(!minSet) {
			minSet = true;
			min = absoluteDist;
		} else {
			if(absoluteDist < min)
				min = absoluteDist;
		}

		// set the color for every atom
		for(unsigned int j = cAlphas1[i].firstAtomIdx; j <= cAlphas1[i].lastAtomIdx ; ++j ) {
			atomColorTable[3*j+0] = absoluteDist;
			atomColorTable[3*j+1] = absoluteDist;
			atomColorTable[3*j+2] = absoluteDist;
		}
	}

	switch(this->currentComparisonMode) {
		case ZERO_TO_MAX:
			min = 0.0f; 
			break;
		case ZERO_TO_VALUE:
			min = 0.0f;
			max = maxDistanceParam.Param<param::FloatParam>()->Value();
			break;
		case MIN_TO_MAX:
			break;
		case MIN_TO_VALUE:
			max = maxDistanceParam.Param<param::FloatParam>()->Value();
			break;
		case VALUE_TO_MAX:
			min = minDistanceParam.Param<param::FloatParam>()->Value();
			break;
		case VALUE_TO_VALUE:
			min = minDistanceParam.Param<param::FloatParam>()->Value();
			max = maxDistanceParam.Param<param::FloatParam>()->Value();
			break;
		default: // ZERO_TO_MAX
			min = 0.0f;
	}

	this->minDistanceParam.Param<param::FloatParam>()->SetValue(min);
	this->maxDistanceParam.Param<param::FloatParam>()->SetValue(max);

	float diff = max - min;

	// get colors
	float r,g,b;
	utility::ColourParser::FromString(
		this->minGradColorParam.Param<param::StringParam>()->Value(),
		r,g,b);
	vislib::math::Vector<float, 3> colMin( r, g, b);
	utility::ColourParser::FromString(
		this->midGradColorParam.Param<param::StringParam>()->Value(),
		r,g,b);
	vislib::math::Vector<float, 3> colMid( r, g, b);
	utility::ColourParser::FromString(
		this->maxGradColorParam.Param<param::StringParam>()->Value(),
		r,g,b);
	vislib::math::Vector<float, 3> colMax( r, g, b);

	utility::ColourParser::FromString(
		this->colorParam.Param<param::StringParam>()->Value(),
		r,g,b);
	vislib::math::Vector<float, 3> col( r, g, b);

	if( diff > 0.0000001f ) {
		for( cntAtom = 0; cntAtom < mol1->AtomCount(); ++cntAtom ) {
			switch(this->currentComparisonColoringMode) {
				case(SINGLE_COLOR):
					// shift to interval start
					atomColorTable[3*cntAtom+0] -= min;
					atomColorTable[3*cntAtom+1] -= min;
					atomColorTable[3*cntAtom+2] -= min;
					// apply color and normalize
					atomColorTable[3*cntAtom+0] *= col.GetX() / diff;
					atomColorTable[3*cntAtom+1] *= col.GetY() / diff;
					atomColorTable[3*cntAtom+2] *= col.GetZ() / diff;
					break;
				case(COLOR_GRADIENT):
					// go through every value
					for(int i = 0; i < 3; i++) {
						atomColorTable[3*cntAtom+i] -= min;
						atomColorTable[3*cntAtom+i] /= diff;
						float distanceValue = atomColorTable[3*cntAtom+i];
						if(atomColorTable[3*cntAtom+i] < weight0) {
							float factor = distanceValue / weight0;
							atomColorTable[3*cntAtom+i] = (1.0f - factor) * colMin[i] + factor * colMid[i];
						} else {
							float factor = (distanceValue - weight0) / weight1;
							atomColorTable[3*cntAtom+i] = (1.0f - factor) * colMid[i] + factor * colMax[i];
						}
					}
					break;
				case(TWO_COLORS):
					for(int i = 0; i < 3; i++) {
						atomColorTable[3*cntAtom+i] -= min;
						atomColorTable[3*cntAtom+i] /= diff;
						float distanceValue = atomColorTable[3*cntAtom+i];
						atomColorTable[3*cntAtom+i] = (1.0f - distanceValue) * colMin[i] + distanceValue * colMax[i];
					}
					break;
				default: // SINGLE_COLOR
					atomColorTable[3*cntAtom+0] -= min;
					atomColorTable[3*cntAtom+1] -= min;
					atomColorTable[3*cntAtom+2] -= min;
					atomColorTable[3*cntAtom+0] *= col.GetX() / diff;
					atomColorTable[3*cntAtom+1] *= col.GetY() / diff;
					atomColorTable[3*cntAtom+2] *= col.GetZ() / diff;
			}

			// set everything out of range to the range borders
			if(atomColorTable[3*cntAtom+0] < 0.0)
				atomColorTable[3*cntAtom+0] = 0.0;
			if(atomColorTable[3*cntAtom+1] < 0.0)
				atomColorTable[3*cntAtom+1] = 0.0;
			if(atomColorTable[3*cntAtom+2] < 0.0)
				atomColorTable[3*cntAtom+2] = 0.0;

			if(atomColorTable[3*cntAtom+0] > 1.0)
				atomColorTable[3*cntAtom+0] = 1.0;
			if(atomColorTable[3*cntAtom+1] > 1.0)
				atomColorTable[3*cntAtom+1] = 1.0;
			if(atomColorTable[3*cntAtom+2] > 1.0)
				atomColorTable[3*cntAtom+2] = 1.0;
		}
	}
}