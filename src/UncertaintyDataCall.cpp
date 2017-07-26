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
                                                 sortedSecStructAssignment(NULL), secStructUncertainty(NULL), 
                                                 pdbIndex(NULL), chainID(NULL), residueFlag(NULL), aminoAcidName(NULL), pdbID(NULL),
												 recalcUncertainty(NULL), pdbAssignmentHelix(NULL), pdbAssignmentSheet(NULL),
                                                 strideStructThreshold(NULL), dsspStructEnergy(NULL), isTimeAccumulation(false), timestepNumber(0) {
                                                     
}


/*
* UncertaintyDataCall::~UncertaintyDataCall
*/
UncertaintyDataCall::~UncertaintyDataCall(void) {
    this->pdbIndex = NULL;
    this->chainID = NULL;
    this->residueFlag = NULL;
    this->aminoAcidName = NULL;
    this->secStructUncertainty = NULL;
    this->sortedSecStructAssignment = NULL;
	this->recalcUncertainty = NULL;
	this->pdbID = NULL;
	this->pdbAssignmentHelix = NULL;
	this->pdbAssignmentSheet = NULL;
    this->strideStructThreshold = NULL;
    this->dsspStructEnergy = NULL;
}


/*
* UncertaintyDataCall::secStructureColor
*
* Source: https://wiki.selfhtml.org/wiki/Grafik/Farbpaletten
*/
vislib::math::Vector<float, 4> UncertaintyDataCall::GetSecStructColor(UncertaintyDataCall::secStructure s) {

    vislib::math::Vector<float, 4> color;
    color.Set(1.0f, 1.0f, 1.0f, 1.0f);

	// colorbrewer
	// 11-class Paired
	/*  166,206,227
		31,120,180
		178,223,138
		51,160,44
		251,154,153
		227,26,28
		253,191,111
		255,127,0
		202,178,214
		106,61,154
		255,255,15*/

    switch (s) {  
                                                                     //  R                 G               B                A
	case (UncertaintyDataCall::secStructure::G_310_HELIX) :   color.Set(251.0f / 255.0f, 154.0f / 255.0f, 153.0f / 255.0f, 1.0f); break; // HSL: H=340°, S=1.00, L=0.40  | 310
	case (UncertaintyDataCall::secStructure::T_H_TURN) :      color.Set( 51.0f / 255.0f, 160.0f / 255.0f,  44.0f / 255.0f, 1.0f); break; // HSL: H=290°, S=1.00, L=0.30  | alpha
    case (UncertaintyDataCall::secStructure::H_ALPHA_HELIX) : color.Set(227.0f / 255.0f,  26.0f / 255.0f,  28.0f / 255.0f, 1.0f); break; // HSL: H=250°, S=1.00, L=0.40  | PI                                                               
	case (UncertaintyDataCall::secStructure::I_PI_HELIX) :    color.Set(255.0f / 255.0f, 127.0f / 255.0f,   0.0f / 255.0f, 1.0f); break; // HSL: H=200°, S=1.00, L=0.50  | turn
	case (UncertaintyDataCall::secStructure::S_BEND) :        color.Set(178.0f / 255.0f, 223.0f / 255.0f, 138.0f / 255.0f, 1.0f); break; // HSL: H=160°, S=1.00, L=0.40  | bend
	case (UncertaintyDataCall::secStructure::C_COIL) :        color.Set(153.0f / 255.0f, 153.0f / 255.0f, 153.0f / 255.0f, 1.0f); break; // HSL: H=120°, S=1.00, L=0.15  | coil
	case (UncertaintyDataCall::secStructure::B_BRIDGE) :      color.Set(166.0f / 255.0f, 206.0f / 255.0f, 227.0f / 255.0f, 1.0f); break; // HSL: H= 90°, S=1.00, L=0.50  | bridge
	case (UncertaintyDataCall::secStructure::E_EXT_STRAND) :  color.Set( 31.0f / 255.0f, 120.0f / 255.0f, 180.0f / 255.0f, 1.0f); break; // HSL: H= 60°, S=1.00, L=0.40  | strand
	case (UncertaintyDataCall::secStructure::NOTDEFINED) :    color.Set(  0.0f / 255.0f,   0.0f / 255.0f,   0.0f / 255.0f, 1.0f); break; // HSL: H=  0°, S=1.00, L=0.00  | -
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
    case (UncertaintyDataCall::secStructure::H_ALPHA_HELIX) : tmpStr = "Alpha-Helix (H)"; break;
    case (UncertaintyDataCall::secStructure::G_310_HELIX) :   tmpStr = "310-Helix (G)"; break;
    case (UncertaintyDataCall::secStructure::I_PI_HELIX) :    tmpStr = "Pi-Helix (I)"; break;
    case (UncertaintyDataCall::secStructure::E_EXT_STRAND) :  tmpStr = "Strand (E)"; break;
    case (UncertaintyDataCall::secStructure::T_H_TURN) :      tmpStr = "Turn (T)"; break;
    case (UncertaintyDataCall::secStructure::B_BRIDGE) :      tmpStr = "Bridge (B)"; break;
    case (UncertaintyDataCall::secStructure::S_BEND) :        tmpStr = "Bend (S)"; break;
    case (UncertaintyDataCall::secStructure::C_COIL) :        tmpStr = "Random Coil (C)"; break;
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
