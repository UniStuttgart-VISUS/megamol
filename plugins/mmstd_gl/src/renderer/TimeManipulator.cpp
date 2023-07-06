/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd_gl/renderer/TimeManipulator.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include <cmath>
#include <glm/glm.hpp>

using namespace megamol::mmstd_gl;

/*
 * TimeManipulator::TimeManipulator
 */
TimeManipulator::TimeManipulator()
        : RendererModule<CallRender3DGL, ModuleGL>()
        , multiplierSlot("multiplier", "time dilation factor")
        , overrideLengthSlot("override length", "switchable")
        , resultingLengthSlot("reported length", "pretend the graph behind this module has this temporal extent") {

    this->multiplierSlot.SetParameter(new core::param::FloatParam(1.0f));
    this->MakeSlotAvailable(&this->multiplierSlot);

    this->overrideLengthSlot.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->overrideLengthSlot);

    this->resultingLengthSlot.SetParameter(new core::param::IntParam(1));
    this->MakeSlotAvailable(&this->resultingLengthSlot);

    this->MakeSlotAvailable(&this->chainRenderSlot);
    this->MakeSlotAvailable(&this->renderSlot);
}

/*
 * TimeManipulator::~TimeManipulator
 */
TimeManipulator::~TimeManipulator() {
    this->Release();
}

/*
 * TimeManipulator::create
 */
bool TimeManipulator::create() {
    return true;
}

/*
 * TimeManipulator::release
 */
void TimeManipulator::release() {}

/*
 * TimeMultiplier::GetExtents
 */
bool TimeManipulator::GetExtents(CallRender3DGL& call) {

    CallRender3DGL* chainedCall = this->chainRenderSlot.CallAs<CallRender3DGL>();
    if (chainedCall != nullptr) {
        const auto multiplier = this->multiplierSlot.Param<core::param::FloatParam>()->Value();

        *chainedCall = call;

        const auto origTime = call.Time();
        chainedCall->SetTime(origTime * multiplier);

        if ((*chainedCall)(core::view::AbstractCallRender::FnGetExtents)) {
            call = *chainedCall;
            call.SetTime(origTime);
            if (this->overrideLengthSlot.Param<core::param::BoolParam>()->Value()) {
                call.SetTimeFramesCount(this->resultingLengthSlot.Param<core::param::IntParam>()->Value());
            } else {
                auto adjustedLength = static_cast<unsigned int>(
                    std::floor(static_cast<float>(chainedCall->TimeFramesCount()) / multiplier));
                call.SetTimeFramesCount(std::min(chainedCall->TimeFramesCount(), adjustedLength));
            }
            return true;
        }
        return false;
    } else {
        if (this->overrideLengthSlot.Param<core::param::BoolParam>()->Value()) {
            call.SetTimeFramesCount(this->resultingLengthSlot.Param<core::param::IntParam>()->Value());
        }
    }
    return true;
}

/*
 * TimeManipulator::Render
 */
bool TimeManipulator::Render(CallRender3DGL& call) {
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
