/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd_gl/renderer/TimeDivisor.h"

#include <cmath>

#include <glm/glm.hpp>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/FloatParam.h"

using namespace megamol::mmstd_gl;

/*
 * TimeDivisor::TimeDivisor
 */
TimeDivisor::TimeDivisor()
        : RendererModule<CallRender3DGL, ModuleGL>()
        , divisorSlot("divisor", "time reduction factor") {

    this->divisorSlot.SetParameter(new core::param::FloatParam(1.0f));
    this->MakeSlotAvailable(&this->divisorSlot);

    this->MakeSlotAvailable(&this->chainRenderSlot);
    this->MakeSlotAvailable(&this->renderSlot);
}

/*
 * TimeDivisor::~TimeDivisor
 */
TimeDivisor::~TimeDivisor() {
    this->Release();
}

/*
 * TimeDivisor::create
 */
bool TimeDivisor::create() {
    return true;
}

/*
 * TimeDivisor::release
 */
void TimeDivisor::release() {

}

/*
 * TimeDivisor::GetExtents
 */
bool TimeDivisor::GetExtents(CallRender3DGL& call) {

    CallRender3DGL* chainedCall = this->chainRenderSlot.CallAs<CallRender3DGL>();
    if (chainedCall != nullptr) {
        *chainedCall = call;

        const auto divisor = this->divisorSlot.Param<core::param::FloatParam>()->Value();
        const auto origTime = call.Time();
        chainedCall->SetTime(std::floor(origTime / divisor));

        if ((*chainedCall)(core::view::AbstractCallRender::FnGetExtents)) {
            call = *chainedCall;
            call.SetTime(origTime);
            return true;
        }
    }
    return true;
}

/*
 * TimeDivisor::Render
 */
bool TimeDivisor::Render(CallRender3DGL& call) {
    CallRender3DGL* chainedCall = this->chainRenderSlot.CallAs<CallRender3DGL>();
    if (chainedCall == nullptr) {
        return true;
    }

    bool renderRes = true;

    *chainedCall = call;

    const auto divisor = this->divisorSlot.Param<core::param::FloatParam>()->Value();
    const auto origTime = call.Time();
    chainedCall->SetTime(std::floor(origTime / divisor));

    renderRes &= (*chainedCall)(core::view::AbstractCallRender::FnRender);

    return renderRes;
}
