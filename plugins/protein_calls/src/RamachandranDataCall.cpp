/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */
#include "protein_calls/RamachandranDataCall.h"

using namespace megamol;
using namespace megamol::protein_calls;

const unsigned int RamachandranDataCall::CallForGetData = 0;

/*
 * RamachandranDataCall::RamachandranDataCall
 */
RamachandranDataCall::RamachandranDataCall()
        : core::Call()
        , selectedAminoAcid(-1)
        , angleVector(nullptr)
        , pointStateVector(nullptr)
        , probabilityVector(nullptr)
        , time(0.0f) {}

/*
 * RamachandranDataCall::~RamachandranDataCall
 */
RamachandranDataCall::~RamachandranDataCall() {}
