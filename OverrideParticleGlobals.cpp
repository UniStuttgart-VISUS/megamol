/*
 * OverrideParticleGlobals.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "OverrideParticleGlobals.h"
#include "param/IntParam.h"
#include "param/BoolParam.h"
#include "param/FloatParam.h"
#include "param/StringParam.h"
#include "vislib/ColourParser.h"
#include <cstdint>

using namespace megamol;
using namespace megamol::stdplugin;


/*
 * datatools::OverrideParticleGlobals::OverrideParticleGlobals
 */
datatools::OverrideParticleGlobals::OverrideParticleGlobals(void)
        : AbstractParticleManipulator("outData", "indata"),
        overrideAllListSlot("overrideAllLists", "Activates overriding the selected values for all particle lists"),
        overrideListSlot("list", "The particle list to override the values of"),
        overrideRadiusSlot("overrideRadius", "Activates overriding the radius"),
        radiusSlot("radius", "The new radius value"),
        overrideColorSlot("overrideColor", "Activates overriding the color"),
        colorSlot("color", "The new color value") {

    this->overrideAllListSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->overrideAllListSlot);

    this->overrideListSlot.SetParameter(new core::param::IntParam(0, 0));
    this->MakeSlotAvailable(&this->overrideListSlot);

    this->overrideRadiusSlot.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->overrideRadiusSlot);

    this->radiusSlot.SetParameter(new core::param::FloatParam(0.5f, 0.0f));
    this->MakeSlotAvailable(&this->radiusSlot);

    this->overrideColorSlot.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->overrideColorSlot);

    this->colorSlot.SetParameter(new core::param::StringParam("white"));
    this->MakeSlotAvailable(&this->colorSlot);
}


/*
 * datatools::OverrideParticleGlobals::~OverrideParticleGlobals
 */
datatools::OverrideParticleGlobals::~OverrideParticleGlobals(void) {
    this->Release();
}


/*
 * datatools::OverrideParticleGlobals::manipulateData
 */
bool datatools::OverrideParticleGlobals::manipulateData(
        megamol::core::moldyn::MultiParticleDataCall& outData,
        megamol::core::moldyn::MultiParticleDataCall& inData) {
    using megamol::core::moldyn::MultiParticleDataCall;

    bool overrideAll = this->overrideAllListSlot.Param<core::param::BoolParam>()->Value();
    int listId = this->overrideListSlot.Param<core::param::IntParam>()->Value();
    bool overrideRadius = this->overrideRadiusSlot.Param<core::param::BoolParam>()->Value();
    float radius = this->radiusSlot.Param<core::param::FloatParam>()->Value();
    bool overrideColor = this->overrideColorSlot.Param<core::param::BoolParam>()->Value();
    uint8_t color[4];
    try {
        vislib::graphics::ColourParser::FromString(
            this->colorSlot.Param<core::param::StringParam>()->Value(),
            color[0], color[1], color[2], color[3]);
    } catch(...) {
        ::memset(color, 0, 4);
    }

    outData = inData; // also transfers the unlocker to 'outData'

    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData

    if (!overrideColor && !overrideRadius) return true;

    unsigned int plc = outData.GetParticleListCount();
    unsigned int i = 0;
    if (!overrideAll) {
        i = static_cast<unsigned int>(listId);
        plc = std::min<unsigned int>(plc, i + 1);
    }
    for (; i < plc; i++) {
        MultiParticleDataCall::Particles& p = outData.AccessParticles(i);

        if (overrideRadius) {
            p.SetGlobalRadius(radius);
            if (p.GetVertexDataType() == MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR) {
                p.SetVertexData(MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ,
                    p.GetVertexData(), std::max<unsigned int>(16, p.GetVertexDataStride()));
            }
        }

        if (overrideColor) {
            p.SetGlobalColour(color[0], color[1], color[2], color[3]);
            p.SetColourData(MultiParticleDataCall::Particles::COLDATA_NONE, nullptr);
        }
    }

    return true;
}
