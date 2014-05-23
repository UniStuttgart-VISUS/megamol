/*
 * DynDensityGradientEstimator.cpp
 *
 *  Created on: May 22, 2014
 *      Author: scharnkn@visus.uni-stuttgart.de
 */

#include "stdafx.h"
#include "DynDensityGradientEstimator.h"
#include "moldyn/MultiParticleDataCall.h"
#include "moldyn/DirectionalParticleDataCall.h"

using namespace megamol;
using namespace megamol::core;

/*
 * misc::DynDensityGradientEstimator::DynDensityGradientEstimator
 */
misc::DynDensityGradientEstimator::DynDensityGradientEstimator(void)
        : Module(), getPartDataSlot("getPartData", "..."), putDirDataSlot(
                "putDirData", "...") {

    this->getPartDataSlot.SetCompatibleCall<
            moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getPartDataSlot);

    this->putDirDataSlot.SetCallback("DirectionalParticleDataCall", "GetData",
            &DynDensityGradientEstimator::getData);
    this->putDirDataSlot.SetCallback("DirectionalParticleDataCall", "GetExtent",
            &DynDensityGradientEstimator::getExtent);
    this->MakeSlotAvailable(&this->putDirDataSlot);

}

/*
 * misc::DynDensityGradientEstimator::~DynDensityGradientEstimator
 */
misc::DynDensityGradientEstimator::~DynDensityGradientEstimator(void) {
    this->Release();
}

/*
 * misc::DynDensityGradientEstimator::create
 */
bool misc::DynDensityGradientEstimator::create(void) {
    return true;
}

/*
 * misc::DynDensityGradientEstimator::release
 */
void misc::DynDensityGradientEstimator::release(void) {

}

/*
 * misc::DynDensityGradientEstimator::getExtent
 */
bool misc::DynDensityGradientEstimator::getExtent(Call& call) {

    // Get a pointer to the incoming data call.
    moldyn::DirectionalParticleDataCall *callIn = dynamic_cast<moldyn::DirectionalParticleDataCall*>(&call);
    if(callIn == NULL) return false;

    // Get a pointer to the outgoing data call.
    moldyn::MultiParticleDataCall *callOut = this->getPartDataSlot.CallAs<moldyn::MultiParticleDataCall>();
    if(callOut == NULL) return false;

    // Get extend.
    if(!(*callOut)(1)) return false;

    // Set extend.
    callIn->AccessBoundingBoxes().Clear();
    callIn->SetExtent(callOut->FrameCount(), callOut->AccessBoundingBoxes());

    return true;
}

/*
 * misc::DynDensityGradientEstimator::getData
 */
bool misc::DynDensityGradientEstimator::getData(Call& caller) {
    return true;
}
