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
* UncertaintyDataCall::UncertaintyDataCall
*/
const unsigned int UncertaintyDataCall::CallForGetData = 0;


UncertaintyDataCall::UncertaintyDataCall(void) : megamol::core::Call(),
                                                 dsspSecStructure(NULL), strideSecStructure(NULL), pdbSecStructure(NULL),
                                                 indexAminoAcidchainID(NULL), secStructUncertainty(NULL){
}


UncertaintyDataCall::~UncertaintyDataCall(void) {
    this->dsspSecStructure = NULL;
    this->strideSecStructure = NULL;
    this->pdbSecStructure = NULL,
    this->indexAminoAcidchainID = NULL;
    this->secStructUncertainty = NULL;
}

vislib::Pair<UncertaintyDataCall::secStructure, float> UncertaintyDataCall::mostLikelySecStructure(unsigned int i) {
    // temp varaibles
    vislib::math::Vector<float, 4> tmpSecStruct = this->secStructUncertainty->operator[](i);
    vislib::Pair<secStructure, float> maxUncer(secStructure::NOTDEFINED, 0.0f);

    if (!this->secStructUncertainty)
        return maxUncer;
    else if (this->secStructUncertainty->Count() <= i)
        return maxUncer;
    else {

        maxUncer.Second() = tmpSecStruct[secStructure::NOTDEFINED];

        if (maxUncer.Second() < tmpSecStruct[secStructure::HELIX]) {
            maxUncer.First() = secStructure::HELIX; 
            maxUncer.Second() = tmpSecStruct[secStructure::HELIX];
        }   
        if (maxUncer.Second() < tmpSecStruct[secStructure::STRAND]) {
            maxUncer.First() = secStructure::STRAND;
            maxUncer.Second() = tmpSecStruct[secStructure::STRAND];
        }
        if (maxUncer.Second() < tmpSecStruct[secStructure::COIL]) {
            maxUncer.First() = secStructure::COIL;
            maxUncer.Second() = tmpSecStruct[secStructure::COIL];
        }

        return maxUncer;
    }
}

