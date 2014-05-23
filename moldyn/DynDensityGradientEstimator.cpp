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
 * moldyn::DynDensityGradientEstimator::DynDensityGradientEstimator
 */
moldyn::DynDensityGradientEstimator::DynDensityGradientEstimator(void)
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
 * moldyn::DynDensityGradientEstimator::~DynDensityGradientEstimator
 */
moldyn::DynDensityGradientEstimator::~DynDensityGradientEstimator(void) {
    this->Release();
}

/*
 * moldyn::DynDensityGradientEstimator::create
 */
bool moldyn::DynDensityGradientEstimator::create(void) {
    return true;
}

/*
 * moldyn::DynDensityGradientEstimator::release
 */
void moldyn::DynDensityGradientEstimator::release(void) {

}

/*
 * moldyn::DynDensityGradientEstimator::getExtent
 */
bool moldyn::DynDensityGradientEstimator::getExtent(Call& call) {

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
 * moldyn::DynDensityGradientEstimator::getData
 */
bool moldyn::DynDensityGradientEstimator::getData(Call& caller) {
    return true;
}
