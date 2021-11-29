/*
 * AbstractQuartzModule.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "AbstractQuartzModule.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "stdafx.h"

namespace megamol {
namespace demos_gl {


/*
 * AbstractQuartzModule::AbstractQuartzModule
 */
AbstractQuartzModule::AbstractQuartzModule(void)
        : dataInSlot("datain", "slot to get the data")
        , typesInSlot("typesin", "solt to get the types data")
        , typesDataHash(0) {

    this->dataInSlot.SetCompatibleCall<core::factories::CallAutoDescription<ParticleGridDataCall>>();

    this->typesInSlot.SetCompatibleCall<core::factories::CallAutoDescription<CrystalDataCall>>();
}


/*
 * AbstractQuartzModule::~AbstractQuartzModule
 */
AbstractQuartzModule::~AbstractQuartzModule(void) {}


/*
 * AbstractQuartzModule::getParticleData
 */
ParticleGridDataCall* AbstractQuartzModule::getParticleData(void) {
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
CrystalDataCall* AbstractQuartzModule::getCrystaliteData(void) {
    CrystalDataCall* tdc = this->typesInSlot.CallAs<CrystalDataCall>();
    if (tdc != NULL) {
        if (!(*tdc)(CrystalDataCall::CallForGetData)) {
            tdc = NULL;
        }
    }
    return tdc;
}

} // namespace demos_gl
} /* end namespace megamol */
