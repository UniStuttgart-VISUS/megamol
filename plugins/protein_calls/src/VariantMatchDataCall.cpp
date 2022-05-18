//
// VariantMatchDataCall.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Jun 15, 2013
//     Author: scharnkn
//

#include "protein_calls/VariantMatchDataCall.h"
#include "stdafx.h"

using namespace megamol;
using namespace megamol::protein_calls;

const unsigned int megamol::protein_calls::VariantMatchDataCall::CallForGetData = 0;

/*
 * VariantMatchDataCall::VariantMatchDataCall
 */
VariantMatchDataCall::VariantMatchDataCall(void) : Call(), variantCnt(0), labels(NULL), match(NULL) {}


/*
 * VariantMatchDataCall::~VariantMatchDataCall
 */
VariantMatchDataCall::~VariantMatchDataCall(void) {}
