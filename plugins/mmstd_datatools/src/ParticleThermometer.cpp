/*
 * ParticleThermometer.cpp
 *
 * Copyright (C) 2017 by MegaMol team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "ParticleThermometer.h"
#include "mmcore/param/BoolParam.h"
#include "nanoflann.hpp"
#include <cstdint>
#include <algorithm>
#include <cfloat>
#include <cassert>

using namespace megamol;
using namespace megamol::stdplugin;


/*
 * datatools::ParticleThermometer::ParticleThermometer
 */
datatools::ParticleThermometer::ParticleThermometer(void)
        : cyclXSlot("cyclX", "Considers cyclic boundary conditions in X direction"),
        cyclYSlot("cyclY", "Considers cyclic boundary conditions in Y direction"),
        cyclZSlot("cyclZ", "Considdrs cyclic boundary conditions in Z direction"),
        datahash(0), time(0), newColors(), minCol(0.0f), maxCol(1.0f) {

    this->cyclXSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclXSlot);

    this->cyclYSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclYSlot);

    this->cyclZSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclZSlot);
}


/*
 * datatools::ParticleColorSignedDistance::~ParticleColorSignedDistance
 */
datatools::ParticleThermometer::~ParticleThermometer(void) {
    this->Release();
}

/*
* datatools::ParticleThermometer::create
*/
bool datatools::ParticleThermometer::create(void) {
    return true;
}


/*
* datatools::ParticleThermometer::release
*/
void datatools::ParticleThermometer::release(void) {
}