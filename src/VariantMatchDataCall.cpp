//
// VariantMatchDataCall.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Jun 15, 2013
//     Author: scharnkn
//

#include "stdafx.h"
#include "VariantMatchDataCall.h"

using namespace megamol;
using namespace megamol::protein;

const unsigned int VariantMatchDataCall::CallForGetData = 0;

/*
 * VariantMatchDataCall::VariantMatchDataCall
 */
VariantMatchDataCall::VariantMatchDataCall(void) : Call(), variantCnt(0),
        labels(NULL), match(NULL) {

}


/*
 * VariantMatchDataCall::~VariantMatchDataCall
 */
VariantMatchDataCall::~VariantMatchDataCall(void) {

}




