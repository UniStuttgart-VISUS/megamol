/*
 * AbstractQuartzModule.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "AbstractQuartzModule.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol::demos_gl {


/*
 * AbstractQuartzModule::AbstractQuartzModule
 */
AbstractQuartzModule::AbstractQuartzModule()
        : dataInSlot("datain", "slot to get the data")
        , typesInSlot("typesin", "solt to get the types data")
        , typesDataHash(0) {

    this->dataInSlot.SetCompatibleCall<core::factories::CallAutoDescription<ParticleGridDataCall>>();

    this->typesInSlot.SetCompatibleCall<core::factories::CallAutoDescription<CrystalDataCall>>();
}


/*
 * AbstractQuartzModule::~AbstractQuartzModule
 */
AbstractQuartzModule::~AbstractQuartzModule() {}


/*
 * AbstractQuartzModule::getParticleData
 */
ParticleGridDataCall* AbstractQuartzModule::getParticleData() {
    ParticleGridDataCall* pgdc = this->dataInSlot.CallAs<ParticleGridDataCall>();
    if (pgdc != NULL) {
        if (!(*pgdc)(ParticleGridDataCall::CallForGetData)) {
            pgdc = NULL;
        }
    }
    return pgdc;
}


/*
 * AbstractQuartzModule::getCrystaliteData
 */
CrystalDataCall* AbstractQuartzModule::getCrystaliteData() {
    CrystalDataCall* tdc = this->typesInSlot.CallAs<CrystalDataCall>();
    if (tdc != NULL) {
        if (!(*tdc)(CrystalDataCall::CallForGetData)) {
            tdc = NULL;
        }
    }
    return tdc;
}

} // namespace megamol::demos_gl
