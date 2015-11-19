/*
 * ParticleRelistCall.cpp
 *
 * Copyright (C) 2015 by MegaMol Team (TU Dresden)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/moldyn/ParticleRelistCall.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::moldyn;

ParticleRelistCall::ParticleRelistCall(void) : AbstractGetDataCall(), tarListCount(0), srcPartCount(0), srcParticleTarLists(nullptr) {
    // intentionally empty
}

ParticleRelistCall::~ParticleRelistCall(void) {
    tarListCount = 0;
    srcPartCount = 0;
    srcParticleTarLists = nullptr; // we don't own the memory, so we don't delete
}
