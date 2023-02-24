/*
 * ParticleRelistCall.cpp
 *
 * Copyright (C) 2015 by MegaMol Team (TU Dresden)
 * Alle Rechte vorbehalten.
 */
#include "geometry_calls/ParticleRelistCall.h"


namespace megamol::geocalls {

ParticleRelistCall::ParticleRelistCall()
        : AbstractGetData3DCall()
        , tarListCount(0)
        , srcPartCount(0)
        , srcParticleTarLists(nullptr) {
    // intentionally empty
}

ParticleRelistCall::~ParticleRelistCall() {
    tarListCount = 0;
    srcPartCount = 0;
    srcParticleTarLists = nullptr; // we don't own the memory, so we don't delete
}
} // namespace megamol::geocalls
