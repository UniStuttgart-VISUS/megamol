/*
 * RamachandranDataCall.cpp
 *
 * Author: Karsten Schatz
 * Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */
#include "stdafx.h"
#include "RamachandranDataCall.h"

using namespace megamol;
using namespace megamol::protein_uncertainty;

const unsigned int RamachandranDataCall::CallForGetData = 0;

/*
 * RamachandranDataCall::RamachandranDataCall
 */
RamachandranDataCall::RamachandranDataCall(void) : core::Call(), selectedAminoAcid(-1), angleVector(nullptr), pointStateVector(nullptr), probabilityVector(nullptr), time(0.0f) {
}

/*
 * RamachandranDataCall::~RamachandranDataCall
 */
RamachandranDataCall::~RamachandranDataCall(void) {
}