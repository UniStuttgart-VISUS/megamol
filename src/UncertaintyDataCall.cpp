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
                                                 chainID(NULL), residueFlag(NULL), aminoAcidName(NULL), pdbID(NULL) {
                                                     
    this->recalcUncertainty  = false;
    this->pdbID              = "";
    this->pdbAssignmentHelix = UncertaintyDataCall::pdbAssMethod::PROMOTIF;
    this->pdbAssignmentSheet = UncertaintyDataCall::pdbAssMethod::PROMOTIF;
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
}


/*
* UncertaintyDataCall::secStructureColor
*/
// https://wiki.selfhtml.org/wiki/Grafik/Farbpaletten
vislib::math::Vector<float, 4> UncertaintyDataCall::GetSecStructColor(UncertaintyDataCall::secStructure s) {

    vislib::math::Vector<float, 4> color;
    color.Set(1.0f, 1.0f, 1.0f, 1.0f);

    switch (s) {
    case (UncertaintyDataCall::secStructure::H_ALPHA_HELIX) : color.Set(1.0f, 0.0f, 0.0f, 1.0f); break;
    case (UncertaintyDataCall::secStructure::G_310_HELIX) :   color.Set(1.0f, 0.5f, 0.0f, 1.0f); break;
    case (UncertaintyDataCall::secStructure::I_PI_HELIX) :    color.Set(1.0f, 1.0f, 0.0f, 1.0f); break;
    case (UncertaintyDataCall::secStructure::E_EXT_STRAND) :  color.Set(0.0f, 0.0f, 1.0f, 1.0f); break;
    case (UncertaintyDataCall::secStructure::T_H_TURN) :      color.Set(0.5f, 1.0f, 0.0f, 1.0f); break;
    case (UncertaintyDataCall::secStructure::B_BRIDGE) :      color.Set(0.0f, 0.5f, 1.0f, 1.0f); break;
    case (UncertaintyDataCall::secStructure::S_BEND) :        color.Set(0.0f, 1.0f, 0.0f, 1.0f); break;
    case (UncertaintyDataCall::secStructure::C_COIL) :        color.Set(0.4f, 0.4f, 0.4f, 1.0f); break;
    case (UncertaintyDataCall::secStructure::NOTDEFINED) :    color.Set(0.1f, 0.1f, 0.1f, 1.0f); break;
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
