/*
 * UncertaintyDataCall.cpp
 *
 * Author: Matthias Braun
 * Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 *
 * This module is based on the source code of "UncertaintyDataCall" in megamol protein_calls plugin (svn revision 17).
 *
 */


#include "stdafx.h"
#include "UncertaintyDataCall.h"


using namespace megamol;
using namespace megamol::protein_uncertainty;


/*
* UncertaintyDataCall::CallForGetData
*/
const unsigned int UncertaintyDataCall::CallForGetData = 0;


/*
* UncertaintyDataCall::UncertaintyDataCall
*/
UncertaintyDataCall::UncertaintyDataCall(void) : megamol::core::Call(),
                                                 secStructAssignment(NULL), sortedSecStructUncertainty(NULL), 
                                                 secStructUncertainty(NULL), pdbIndex(NULL), 
                                                 chainID(NULL), residueFlag(NULL), aminoAcidName(NULL), pdbID(NULL),
												 recalcUncertainty(NULL), pdbAssignmentHelix(NULL), pdbAssignmentSheet(NULL){
                                                     
}


/*
* UncertaintyDataCall::~UncertaintyDataCall
*/
UncertaintyDataCall::~UncertaintyDataCall(void) {
    this->secStructAssignment = NULL;
    this->pdbIndex = NULL;
    this->chainID = NULL;
    this->residueFlag = NULL;
    this->aminoAcidName = NULL;
    this->secStructUncertainty = NULL;
    this->sortedSecStructUncertainty = NULL;
	this->recalcUncertainty = NULL;
	this->pdbID = NULL;
	this->pdbAssignmentHelix = NULL;
	this->pdbAssignmentSheet = NULL;

}


/*
* UncertaintyDataCall::secStructureColor
*
* Source: https://wiki.selfhtml.org/wiki/Grafik/Farbpaletten
*/
vislib::math::Vector<float, 4> UncertaintyDataCall::GetSecStructColor(UncertaintyDataCall::secStructure s) {

    vislib::math::Vector<float, 4> color;
    color.Set(1.0f, 1.0f, 1.0f, 1.0f);

    switch (s) {                                                     //  R     G     B     A
    case (UncertaintyDataCall::secStructure::H_ALPHA_HELIX) : color.Set(1.0f, 0.0f, 0.0f, 1.0f); break; // HSL: H=  , S=  , L=  
    case (UncertaintyDataCall::secStructure::G_310_HELIX) :   color.Set(1.0f, 0.5f, 0.0f, 1.0f); break; // HSL: H=  , S=  , L=  
    case (UncertaintyDataCall::secStructure::I_PI_HELIX) :    color.Set(1.0f, 1.0f, 0.0f, 1.0f); break; // HSL: H=  , S=  , L=  
    case (UncertaintyDataCall::secStructure::E_EXT_STRAND) :  color.Set(0.0f, 0.0f, 1.0f, 1.0f); break; // HSL: H=  , S=  , L=  
    case (UncertaintyDataCall::secStructure::T_H_TURN) :      color.Set(0.5f, 1.0f, 0.0f, 1.0f); break; // HSL: H=  , S=  , L=  
    case (UncertaintyDataCall::secStructure::B_BRIDGE) :      color.Set(0.0f, 0.5f, 1.0f, 1.0f); break; // HSL: H=  , S=  , L=   
    case (UncertaintyDataCall::secStructure::S_BEND) :        color.Set(0.0f, 0.5f, 0.0f, 1.0f); break; // HSL: H=  , S=  , L=  
    case (UncertaintyDataCall::secStructure::C_COIL) :        color.Set(0.4f, 0.2f, 0.4f, 1.0f); break; // HSL: H=  , S=  , L=  
    case (UncertaintyDataCall::secStructure::NOTDEFINED) :    color.Set(0.0f, 0.0f, 0.0f, 1.0f); break; // HSL: H=  , S=  , L=  
    default: break;
    }
    
    return color;
}


/*
* UncertaintyDataCall::secStructureDesc
*/
vislib::StringA UncertaintyDataCall::GetSecStructDesc(UncertaintyDataCall::secStructure s) {

    vislib::StringA tmpStr = "No description";

    switch (s) {
    case (UncertaintyDataCall::secStructure::H_ALPHA_HELIX) : tmpStr = "H - Alpha Helix"; break;
    case (UncertaintyDataCall::secStructure::G_310_HELIX) :   tmpStr = "G - 3-10 Helix"; break;
    case (UncertaintyDataCall::secStructure::I_PI_HELIX) :    tmpStr = "I - Pi Helix"; break;
    case (UncertaintyDataCall::secStructure::E_EXT_STRAND) :  tmpStr = "E - Strand"; break;
    case (UncertaintyDataCall::secStructure::T_H_TURN) :      tmpStr = "T - Turn"; break;
    case (UncertaintyDataCall::secStructure::B_BRIDGE) :      tmpStr = "B - Bridge"; break;
    case (UncertaintyDataCall::secStructure::S_BEND) :        tmpStr = "S - Bend"; break;
    case (UncertaintyDataCall::secStructure::C_COIL) :        tmpStr = "C - Random Coil"; break;
    case (UncertaintyDataCall::secStructure::NOTDEFINED) :    tmpStr = "Not defined"; break;
    default: break;
    }

    return tmpStr;
}


/*
* Check if the residue is an amino acid.
*/
char UncertaintyDataCall::AminoacidThreeToOneLetterCode(vislib::StringA resName) {

	if (resName.Equals("ALA")) return 'A';
	else if (resName.Equals("ARG")) return 'R';
	else if (resName.Equals("ASN")) return 'N';
	else if (resName.Equals("ASP")) return 'D';
	else if (resName.Equals("CYS")) return 'C';
	else if (resName.Equals("GLN")) return 'Q';
	else if (resName.Equals("GLU")) return 'E';
	else if (resName.Equals("GLY")) return 'G';
	else if (resName.Equals("HIS")) return 'H';
	else if (resName.Equals("ILE")) return 'I';
	else if (resName.Equals("LEU")) return 'L';
	else if (resName.Equals("LYS")) return 'K';
	else if (resName.Equals("MET")) return 'M';
	else if (resName.Equals("PHE")) return 'F';
	else if (resName.Equals("PRO")) return 'P';
	else if (resName.Equals("SER")) return 'S';
	else if (resName.Equals("THR")) return 'T';
	else if (resName.Equals("TRP")) return 'W';
	else if (resName.Equals("TYR")) return 'Y';
	else if (resName.Equals("VAL")) return 'V';
	else if (resName.Equals("ASH")) return 'D';
	else if (resName.Equals("CYX")) return 'C';
	else if (resName.Equals("CYM")) return 'C';
	else if (resName.Equals("GLH")) return 'E';
	else if (resName.Equals("HID")) return 'H';
	else if (resName.Equals("HIE")) return 'H';
	else if (resName.Equals("HIP")) return 'H';
	else if (resName.Equals("MSE")) return 'M';
	else if (resName.Equals("LYN")) return 'K';
	else if (resName.Equals("TYM")) return 'Y';
	else return '?';

}
