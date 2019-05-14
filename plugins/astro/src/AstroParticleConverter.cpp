/*
 * AstroParticleConverter.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart)
 * All rights reserved.
 */

#include "stdafx.h"
#include "AstroParticleConverter.h"

using namespace megamol;
using namespace megamol::astro;
using namespace megamol::core;
using namespace megamol::core::moldyn;

/*
 * AstroParticleConverter::AstroParticleConverter
 */
AstroParticleConverter::AstroParticleConverter(void)
    : Module()
    , sphereDataSlot("sphereData", "Output slot for the resulting sphere data")
    , astroDataSlot("astroData", "Input slot for astronomical data")
    , lastDataHash(0)
    , hashOffset(0)
    , colmin(0.0f)
    , colmax(1.0f) {

    this->astroDataSlot.SetCompatibleCall<AstroDataCallDescription>();
    this->MakeSlotAvailable(&this->astroDataSlot);

    this->sphereDataSlot.SetCallback(
        MultiParticleDataCall::ClassName(), MultiParticleDataCall::FunctionName(0), &AstroParticleConverter::getData);
    this->sphereDataSlot.SetCallback(
        MultiParticleDataCall::ClassName(), MultiParticleDataCall::FunctionName(1), &AstroParticleConverter::getExtent);
    this->MakeSlotAvailable(&this->sphereDataSlot);
}

/*
 * AstroParticleConverter::~AstroParticleConverter
 */
AstroParticleConverter::~AstroParticleConverter(void) { this->Release(); }

/*
 * AstroParticleConverter::create
 */
bool AstroParticleConverter::create(void) {
    // intentionally empty
    return true;
}

/*
 * AstroParticleConverter::release
 */
void AstroParticleConverter::release(void) {
    // intentionally empty
}

/*
 * AstroParticleConverter::getData
 */
bool AstroParticleConverter::getData(Call& call) {
    MultiParticleDataCall* mpdc = dynamic_cast<MultiParticleDataCall*>(&call);
    if (mpdc == nullptr) return false;

    AstroDataCall* ast = this->astroDataSlot.CallAs<AstroDataCall>();
    if (ast == nullptr) return false;

    ast->SetFrameID(mpdc->FrameID(), mpdc->IsFrameForced());

    if ((*ast)(AstroDataCall::CallForGetData)) {
        if (this->lastDataHash != ast->DataHash()) {
            this->lastDataHash = ast->DataHash();
            this->colmin =
                *std::min_element(ast->GetDensity()->begin(), ast->GetDensity()->end());
            this->colmax =
                *std::max_element(ast->GetDensity()->begin(), ast->GetDensity()->end());
        }
        auto particleCount = ast->GetParticleCount();
        mpdc->SetDataHash(this->lastDataHash + this->hashOffset);
        mpdc->SetParticleListCount(1);
        MultiParticleDataCall::Particles& p = mpdc->AccessParticles(0);
        p.SetCount(particleCount);
        if (p.GetCount() > 0) {
            p.SetVertexData(MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ, ast->GetPositions()->data());
            p.SetColourData(
                MultiParticleDataCall::Particles::COLDATA_FLOAT_I, ast->GetDensity()->data());
            p.SetColourMapIndexValues(this->colmin, this->colmax);
        }

        return true;
    }

    return false;
}

/*
 * AstroParticleConverter::getExtent
 */
bool AstroParticleConverter::getExtent(Call& call) {
    MultiParticleDataCall* mpdc = dynamic_cast<MultiParticleDataCall*>(&call);
    if (mpdc == nullptr) return false;

    AstroDataCall* ast = this->astroDataSlot.CallAs<AstroDataCall>();
    if (ast == nullptr) return false;

    if ((*ast)(AstroDataCall::CallForGetExtent)) {
        mpdc->SetFrameCount(ast->FrameCount());
        mpdc->AccessBoundingBoxes() = ast->AccessBoundingBoxes();
        return true;
    }
    return false;
}
