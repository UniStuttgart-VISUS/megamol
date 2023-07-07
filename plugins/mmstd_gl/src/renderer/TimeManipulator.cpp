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
#include <imgui.h>
#include <imgui_internal.h>

using namespace megamol::mmstd_gl;

/*
 * TimeManipulator::TimeManipulator
 */
TimeManipulator::TimeManipulator()
        : RendererModule<CallRender3DGL, ModuleGL>()
        , multiplierSlot("multiplier", "time dilation factor")
        , overrideLengthSlot("override length", "switchable")
        , resultingLengthSlot("reported length", "pretend the graph behind this module has this temporal extent")
        , pinTimeSlot("pin time", "switchable")
        , pinnedTimeSlot("pinned time", "the outgoing time, overwrites incoming changes")
        , showDebugSlot("show debug", "show info on incoming/outgoing times") {

    this->multiplierSlot.SetParameter(new core::param::FloatParam(1.0f));
    this->MakeSlotAvailable(&this->multiplierSlot);

    this->overrideLengthSlot.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->overrideLengthSlot);

    this->resultingLengthSlot.SetParameter(new core::param::IntParam(1));
    this->MakeSlotAvailable(&this->resultingLengthSlot);

    this->pinTimeSlot.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&pinTimeSlot);

    this->pinnedTimeSlot.SetParameter(new core::param::FloatParam(1.0f));
    this->MakeSlotAvailable(&pinnedTimeSlot);

    this->showDebugSlot.SetParameter(new core::param::BoolParam(false));
    this->MakeSlotAvailable(&this->showDebugSlot);

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

bool TimeManipulator::ManipulateTime(CallRender3DGL& call, CallRender3DGL* chainedCall, uint32_t idx) {
    auto res = false;

    if (chainedCall != nullptr) {
        const auto multiplier = this->multiplierSlot.Param<core::param::FloatParam>()->Value();

        *chainedCall = call;

        incomingTime = call.Time();
        if (this->pinTimeSlot.Param<core::param::BoolParam>()->Value()) {
            outgoingTime = this->pinnedTimeSlot.Param<core::param::FloatParam>()->Value();
        } else {
            outgoingTime = incomingTime * multiplier;
        }
        chainedCall->SetTime(outgoingTime);

        if ((*chainedCall)(idx)) {
            call = *chainedCall;
            call.SetTime(incomingTime);
            if (this->overrideLengthSlot.Param<core::param::BoolParam>()->Value()) {
                reportedFrameCount = this->resultingLengthSlot.Param<core::param::IntParam>()->Value();
                call.SetTimeFramesCount(reportedFrameCount);
            } else {
                auto adjustedLength = static_cast<unsigned int>(
                    std::floor(static_cast<float>(chainedCall->TimeFramesCount()) / multiplier));
                reportedFrameCount = std::min(chainedCall->TimeFramesCount(), adjustedLength);
                call.SetTimeFramesCount(reportedFrameCount);
            }
            res = true;
        } else {
            res = false;
        }
    } else {
        if (this->overrideLengthSlot.Param<core::param::BoolParam>()->Value()) {
            reportedFrameCount = this->resultingLengthSlot.Param<core::param::IntParam>()->Value();
            call.SetTimeFramesCount(reportedFrameCount);
        }
    }
    return res;
}


/*
 * TimeMultiplier::GetExtents
 */
bool TimeManipulator::GetExtents(CallRender3DGL& call) {
    auto res = false;

    CallRender3DGL* chainedCall = this->chainRenderSlot.CallAs<CallRender3DGL>();

    res = this->ManipulateTime(call, chainedCall, core::view::AbstractCallRender::FnGetExtents);

    bool valid_imgui_scope =
        ((ImGui::GetCurrentContext() != nullptr) ? (ImGui::GetCurrentContext()->WithinFrameScope) : (false));
    if (valid_imgui_scope && showDebugSlot.Param<core::param::BoolParam>()->Value()) {

        const auto current_frame = frontend_resources.get<frontend_resources::FrameStatistics>().rendered_frames_count;
        if (current_frame == lastDrawnFrame)
            return true;
        lastDrawnFrame = current_frame;

        if (ImGui::Begin(this->Name(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::LabelText("incoming time", "%f", incomingTime);
            ImGui::LabelText("outgoing time", "%f", outgoingTime);
            ImGui::LabelText("reported frame count", "%u", reportedFrameCount);
        }
        ImGui::End();
    }

    return res;
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

    //*chainedCall = call;

    //const auto multiplier = this->multiplierSlot.Param<core::param::FloatParam>()->Value();
    //const auto origTime = call.Time();
    //chainedCall->SetTime(origTime * multiplier);

    //renderRes &= (*chainedCall)(core::view::AbstractCallRender::FnRender);
    renderRes &= this->ManipulateTime(call, chainedCall, core::view::AbstractCallRender::FnRender);

    return renderRes;
}
