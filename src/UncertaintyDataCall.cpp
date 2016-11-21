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
                                                 dsspSecStructure(NULL), strideSecStructure(NULL), pdbSecStructure(NULL),
                                                 sortedSecStructUncertainty(NULL), secStructUncertainty(NULL), pdbIndex(NULL), 
                                                 chainID(NULL), missingFlag(NULL), aminoAcidName(NULL), pdbID(NULL) {
}


/*
* UncertaintyDataCall::~UncertaintyDataCall
*/
UncertaintyDataCall::~UncertaintyDataCall(void) {
    this->dsspSecStructure = NULL;
    this->strideSecStructure = NULL;
    this->pdbSecStructure = NULL,
    this->pdbIndex = NULL;
    this->chainID = NULL;
    this->missingFlag = NULL;
    this->aminoAcidName = NULL;
    this->secStructUncertainty = NULL;
    this->sortedSecStructUncertainty = NULL;
}

