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
                                                 indexAminoAcidchainID(NULL) {

}


UncertaintyDataCall::~UncertaintyDataCall(void) {
    this->dsspSecStructure = NULL;
    this->strideSecStructure = NULL;
    this->pdbSecStructure = NULL,
    this->indexAminoAcidchainID = NULL;

}
