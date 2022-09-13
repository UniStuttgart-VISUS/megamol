/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd_gl/renderer/TimeMultiplier.h"

#include <cmath>

#include <glm/glm.hpp>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/FloatParam.h"

using namespace megamol::mmstd_gl;

/*
 * TimeMultiplier::TimeMultiplier
 */
TimeMultiplier::TimeMultiplier()
        : RendererModule<CallRender3DGL, ModuleGL>()
        , multiplierSlot("multiplier", "time dilation factor") {

    this->multiplierSlot.SetParameter(new core::param::FloatParam(1.0f));
    this->MakeSlotAvailable(&this->multiplierSlot);

    this->MakeSlotAvailable(&this->chainRenderSlot);
    this->MakeSlotAvailable(&this->renderSlot);
}

/*
 * TimeMultiplier::~TimeMultiplier
 */
TimeMultiplier::~TimeMultiplier() {
    this->Release();
}

/*
 * TimeMultiplier::create
 */
bool TimeMultiplier::create() {
    return true;
}

/*
 * TimeMultiplier::release
 */
void TimeMultiplier::release() {

}

/*
 * TimeMultiplier::GetExtents
 */
bool TimeMultiplier::GetExtents(CallRender3DGL& call) {

    CallRender3DGL* chainedCall = this->chainRenderSlot.CallAs<CallRender3DGL>();
    if (chainedCall != nullptr) {
        const auto multiplier = this->multiplierSlot.Param<core::param::FloatParam>()->Value();

        *chainedCall = call;

        const auto origTime = call.Time();
        chainedCall->SetTime(origTime * multiplier);

        if ((*chainedCall)(core::view::AbstractCallRender::FnGetExtents)) {
            call = *chainedCall;
            call.SetTime(origTime);
            auto adjustedLength =
                static_cast<unsigned int>(std::floor(static_cast<float>(chainedCall->TimeFramesCount()) / multiplier));
            call.SetTimeFramesCount(std::min(chainedCall->TimeFramesCount(), adjustedLength));
            return true;
        }
        return false;
    }
    return true;
}

/*
 * TimeMultiplier::Render
 */
bool TimeMultiplier::Render(CallRender3DGL& call) {
    CallRender3DGL* chainedCall = this->chainRenderSlot.CallAs<CallRender3DGL>();
    if (chainedCall == nullptr) {
        return true;
    }

    bool renderRes = true;

    *chainedCall = call;

    const auto multiplier = this->multiplierSlot.Param<core::param::FloatParam>()->Value();
    const auto origTime = call.Time();
    chainedCall->SetTime(origTime * multiplier);

    renderRes &= (*chainedCall)(core::view::AbstractCallRender::FnRender);

    return renderRes;
}
