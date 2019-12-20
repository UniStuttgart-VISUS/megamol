/*
* TimeLineRenderer.cpp
*
* Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "TimeLineRenderer.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::view;
using namespace megamol::core::utility;
using namespace megamol::cinematic;

using namespace vislib;


TimeLineRenderer::TimeLineRenderer(void) : view::Renderer2DModule()
	, keyframeKeeperSlot("keyframeData", "Connects to the KeyframeKeeper")
    , moveRightFrameParam("gotoRightFrame", "Move to right animation time frame.")
    , moveLeftFrameParam("gotoLeftFrame", "Move to left animation time frame.")
    , resetPanScaleParam("resetAxes", "Reset shifted and scaled time axes.")
    , axes()
    , utils()
    , texture(0)
    , yAxisParam(ActiveParam::SIMULATION_TIME)
    , dragDropKeyframe()
    , dragDropActive(false)
    , axisDragDropMode(0)
    , axisScaleMode(0)
    , keyframeMarkHeight(1.0f)
    , rulerMarkHeight(1.0f)
    , viewport(1.0f, 1.0f)
    , fps(24)
    , mouseX(0.0f)
    , mouseY(0.0f)
    , lastMouseX(0.0f)
    , lastMouseY(0.0f)
    , mouseButton(MouseButton::BUTTON_LEFT)
    , mouseAction(MouseButtonAction::RELEASE) {

    this->keyframeKeeperSlot.SetCompatibleCall<CallKeyframeKeeperDescription>();
    this->MakeSlotAvailable(&this->keyframeKeeperSlot);

    // init parameters
    this->moveRightFrameParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_RIGHT, core::view::Modifier::CTRL));
    this->MakeSlotAvailable(&this->moveRightFrameParam);

    this->moveLeftFrameParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_LEFT, core::view::Modifier::CTRL));
    this->MakeSlotAvailable(&this->moveLeftFrameParam);

    this->resetPanScaleParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_P, core::view::Modifier::CTRL));
    this->MakeSlotAvailable(&this->resetPanScaleParam);

    for (size_t i = 0; i < Axis::COUNT; ++i) {
        this->axes[i].startPos = { 0.0f, 0.0f };
        this->axes[i].endPos = { 0.0f, 0.0f };
        this->axes[i].length = 0.0f;
        this->axes[i].maxValue = 1.0f;
        this->axes[i].segmSize = 0.0f;
        this->axes[i].segmValue = 0.0f;
        this->axes[i].scaleFactor = 0.0f;
        this->axes[i].scaleOffset = 0.0f;
        this->axes[i].scaleDelta = 0.0f;
        this->axes[i].valueFractionLength = 0.0f;
        this->axes[i].rulerPos = 0.0f;
        this->axes[i].formatStr = "%.5f";
    }
}


TimeLineRenderer::~TimeLineRenderer(void) {

	this->Release();
}


bool TimeLineRenderer::create(void) {
	
    // Initialise render utils
    if (!this->utils.Initialise(this->GetCoreInstance())) {
        vislib::sys::Log::DefaultLog.WriteError("[TIMELINE RENDERER] [create] Couldn't initialize render utils.");
        return false;
    }

    // Load texture
    vislib::StringA shortfilename = "arrow.png";
    auto fullfilename = megamol::core::utility::ResourceWrapper::getFileName(this->GetCoreInstance()->Configuration(), shortfilename);
    if (!this->utils.LoadTextureFromFile(std::wstring(fullfilename.PeekBuffer()), this->texture)) {
        vislib::sys::Log::DefaultLog.WriteError("[TIMELINE RENDERER] [create] Couldn't load marker texture.");
        return false;
    }

    return true;
}


void TimeLineRenderer::release(void) {

}


bool TimeLineRenderer::GetExtents(view::CallRender2D& call) {

    glm::vec2 currentViewport;
    currentViewport.x = static_cast<float>(call.GetViewport().GetSize().GetWidth());
    currentViewport.y = static_cast<float>(call.GetViewport().GetSize().GetHeight());
    call.SetBoundingBox(call.GetViewport());

    if (currentViewport != this->viewport) {
        this->viewport = currentViewport;

        // Set axes position depending on font size
        vislib::StringA tmpStr;
        if (this->axes[Axis::Y].maxValue > this->axes[Axis::X].maxValue) {
            tmpStr.Format("%.6f ", this->axes[Axis::Y].maxValue);
        } else {
            tmpStr.Format("%.6f ", this->axes[Axis::X].maxValue);
        }

        float strHeight = this->utils.GetTextLineHeight();
        float strWidth = this->utils.GetTextLineWidth(std::string(tmpStr.PeekBuffer()));
        this->rulerMarkHeight = strHeight / 2.0f;
        this->keyframeMarkHeight = strHeight*1.5f;

        this->axes[Axis::X].startPos = this->axes[Axis::Y].startPos = glm::vec2(strWidth + strHeight * 1.5f, strHeight*2.5f);
        this->axes[Axis::X].endPos = glm::vec2(this->viewport.x - strWidth, strHeight * 2.5f);
        this->axes[Axis::Y].endPos = glm::vec2(strWidth + strHeight * 1.5f, this->viewport.y - (this->keyframeMarkHeight * 1.1f) - strHeight);
        for (size_t i = 0; i < Axis::COUNT; ++i) {
            this->axes[i].length = glm::length(this->axes[i].endPos - this->axes[i].startPos);
            this->axes[i].scaleFactor = 1.0f;
        }

        return this->recalcAxesData();
    }

    return true;
}


bool TimeLineRenderer::Render(view::CallRender2D& call) {

    // Get update data from keyframe keeper
    auto ccc = this->keyframeKeeperSlot.CallAs<CallKeyframeKeeper>();
    if (!ccc) return false;
    if (!(*ccc)(CallKeyframeKeeper::CallForGetUpdatedKeyframeData)) return false;
     auto keyframes = ccc->GetKeyframes();
    if (keyframes == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn("[TIMELINE RENDERER] [Render] Pointer to keyframe array is nullptr.");
        return false;
    }

    // Get maximum value for x axis (always animation time)
    if (this->axes[Axis::X].maxValue != ccc->GetTotalAnimTime()) {
        this->axes[Axis::X].maxValue = ccc->GetTotalAnimTime();
        this->recalcAxesData();
    }

    // Get max value for y axis depending on chosen parameter
    float yAxisMaxValue = 0.0f;
    switch (this->yAxisParam) {
        case (ActiveParam::SIMULATION_TIME): yAxisMaxValue = ccc->GetTotalSimTime(); break;
        default: break;
    }
    if (this->axes[Axis::Y].maxValue != yAxisMaxValue) {
        this->axes[Axis::Y].maxValue = yAxisMaxValue;
        this->recalcAxesData();
    }

    // Get fps
    this->fps = ccc->GetFps();

    // Update parameters
    if (this->moveRightFrameParam.IsDirty()) {
        this->moveRightFrameParam.ResetDirty();
        // Set selected animation time to right animation time frame
        float at = ccc->GetSelectedKeyframe().GetAnimTime();
        float fpsFrac = 1.0f / (float)(this->fps);
        float t = std::round(at / fpsFrac) * fpsFrac;
        t += fpsFrac;
        if (std::abs(t - std::round(t)) < (fpsFrac / 2.0)) {
            t = std::round(t);
        }
        t = (t > this->axes[Axis::X].maxValue) ? (this->axes[Axis::X].maxValue) : (t);
        ccc->SetSelectedKeyframeTime(t);
        if (!(*ccc)(CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime)) return false;
    }
    if (this->moveLeftFrameParam.IsDirty()) {
        this->moveLeftFrameParam.ResetDirty();
        // Set selected animation time to left animation time frame
        float at = ccc->GetSelectedKeyframe().GetAnimTime();
        float fpsFrac = 1.0f / (float)(this->fps);
        float t = std::round(at / fpsFrac) * fpsFrac;
        t -= fpsFrac;
        if (std::abs(t - std::round(t)) < (fpsFrac / 2.0)) {
            t = std::round(t);
        }
        t = (t < 0.0f) ? (0.0f) : (t);
        ccc->SetSelectedKeyframeTime(t);
        if (!(*ccc)(CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime)) return false;
    }

    if (this->resetPanScaleParam.IsDirty()) {
        this->resetPanScaleParam.ResetDirty();
        for (size_t i = 0; i < Axis::COUNT; ++i) {
            this->axes[i].scaleFactor = 1.0f;
            this->axes[i].scaleOffset = 0.0f;
        }
        this->recalcAxesData();
    }

    // Init rendering
    float start_x, start_y, end_x, end_y, x, y;
    glm::vec3 start, end;
    glm::vec4 color;
    glm::vec3 cam_view = { 0.0f, 0.0f, -1.0f };
    glm::vec3 cam_pos = { 0.0f, 0.0f, 1.0f };
    glm::vec3 origin = { this->axes[Axis::X].startPos.x, this->axes[Axis::X].startPos.y, 0.0f };
    float yAxisValue = 0.0f;
    auto cbc = call.GetBackgroundColour();
    glm::vec4 back_color = glm::vec4(static_cast<float>(cbc[0]) / 255.0f, static_cast<float>(cbc[1]) / 255.0f, static_cast<float>(cbc[2]) / 255.0f, 1.0f);
    this->utils.SetBackgroundColor(back_color);
    glm::mat4 ortho = glm::ortho(0.0f, this->viewport.x, 0.0f, this->viewport.y, -1.0f, 1.0f);
    auto skf = ccc->GetSelectedKeyframe();

    // Push rulers ------------------------------------------------------------
    color = this->utils.Color(CinematicUtils::Colors::FOREGROUND);
    // Draw x axis ruler lines
    start = glm::vec3(this->axes[Axis::X].startPos.x - this->rulerMarkHeight, this->axes[Axis::X].startPos.y, 0.0f);
    end = glm::vec3(this->axes[Axis::X].endPos.x + this->rulerMarkHeight, this->axes[Axis::X].endPos.y, 0.0f);
    this->utils.PushLinePrimitive(start, end, 2.5f, cam_view, cam_pos, color);
    float loop_max = this->axes[Axis::X].length + (this->axes[Axis::X].segmSize / 2.0f);
    for (float f = this->axes[Axis::X].scaleOffset; f <= loop_max; f = f + this->axes[Axis::X].segmSize) {
        if (f >= 0.0f) {
            start = origin + glm::vec3(f, 0.0f, 0.0f);
            end = origin + glm::vec3(f, -this->rulerMarkHeight, 0.0f);
            this->utils.PushLinePrimitive(start, end, 2.5f, cam_view, cam_pos, color);
        }
    }
    // Push y axis ruler lines
    start = glm::vec3(this->axes[Axis::X].startPos.x, this->axes[Axis::X].startPos.y - this->rulerMarkHeight, 0.0f);
    end = glm::vec3(this->axes[Axis::Y].endPos.x, this->axes[Axis::Y].endPos.y + this->rulerMarkHeight, 0.0f);
    this->utils.PushLinePrimitive(start, end, 2.5f, cam_view, cam_pos, color);
    loop_max = this->axes[Axis::Y].length + (this->axes[Axis::Y].segmSize / 2.0f);
    for (float f = this->axes[Axis::Y].scaleOffset; f <= loop_max; f = f + this->axes[Axis::Y].segmSize) {
        if (f >= 0.0f) {
            start = origin + glm::vec3(-this->rulerMarkHeight, f, 0.0f);
            end = origin + glm::vec3(0.0f, f, 0.0f);
            this->utils.PushLinePrimitive(start, end, 2.5f, cam_view, cam_pos, color);
        }
    }

    // Push line strip between keyframes --------------------------------------
    if (keyframes->size() > 0) {
        color = this->utils.Color(CinematicUtils::Colors::KEYFRAME_SPLINE);
        // First vertex
        start_x = this->axes[Axis::X].scaleOffset;
        float yAxisValue = 0.0f;
        switch (this->yAxisParam) {
            case (ActiveParam::SIMULATION_TIME): yAxisValue = (*keyframes).front().GetSimTime(); break;
            default: break;
        }
        start_y = this->axes[Axis::Y].scaleOffset + yAxisValue * this->axes[Axis::Y].maxValue * this->axes[Axis::Y].valueFractionLength;
        for (unsigned int i = 0; i < keyframes->size(); i++) {
            end_x = this->axes[Axis::X].scaleOffset + (*keyframes)[i].GetAnimTime() * this->axes[Axis::X].valueFractionLength;
            yAxisValue = 0.0f;
            switch (this->yAxisParam) {
                case (ActiveParam::SIMULATION_TIME): yAxisValue = (*keyframes)[i].GetSimTime(); break;
                default: break;
            }
            end_y = this->axes[Axis::Y].scaleOffset  + yAxisValue * this->axes[Axis::Y].maxValue  * this->axes[Axis::Y].valueFractionLength;
            start = origin + glm::vec3(start_x, start_y, 0.0f);
            end = origin + glm::vec3(end_x, end_y, 0.0f);
            this->utils.PushLinePrimitive(start, end, 2.0f, cam_view, cam_pos, color);
            start_x = end_x;
            start_y = end_y;
        }
        // Last vertex
        end_x = this->axes[Axis::X].scaleOffset + this->axes[Axis::X].maxValue * this->axes[Axis::X].valueFractionLength;
        yAxisValue = 0.0f;
        switch (this->yAxisParam) {
            case (ActiveParam::SIMULATION_TIME): yAxisValue = (*keyframes).back().GetSimTime(); break;
            default: break;
        }
        end_y = this->axes[Axis::Y].scaleOffset + yAxisValue * this->axes[Axis::Y].maxValue * this->axes[Axis::Y].valueFractionLength;
        start = origin + glm::vec3(start_x, start_y, 0.0f);
        end = origin + glm::vec3(end_x, end_y, 0.0f);
        this->utils.PushLinePrimitive(start, end, 2.0f, cam_view, cam_pos, color);
    }

    // Push frame marker lines ------------------------------------------------
    float frameFrac = this->axes[Axis::X].length / ((float)(this->fps) * (this->axes[Axis::X].maxValue)) * this->axes[Axis::X].scaleFactor;
    loop_max = this->axes[Axis::X].length + (frameFrac / 2.0f);
    for (float f = this->axes[Axis::X].scaleOffset; f <= loop_max; f = (f + frameFrac)) {
        if (f >= 0.0f) {
            start = origin + glm::vec3(f, 0.0f, 0.0f);
            end = origin + glm::vec3(f, this->rulerMarkHeight, 0.0f);
            this->utils.PushLinePrimitive(start, end, 1.0f, cam_view, cam_pos, this->utils.Color(CinematicUtils::Colors::FRAME_MARKER));
        }
    }

    // Push markers for all existing keyframes --------------------------------
    for (unsigned int i = 0; i < keyframes->size(); i++) {
        x = this->axes[Axis::X].scaleOffset + (*keyframes)[i].GetAnimTime() * this->axes[Axis::X].valueFractionLength;
        yAxisValue = 0.0f;
        switch (this->yAxisParam) {
        case (ActiveParam::SIMULATION_TIME): yAxisValue = (*keyframes)[i].GetSimTime(); break;
        default: break;
        }
        y = this->axes[Axis::Y].scaleOffset + yAxisValue * this->axes[Axis::Y].maxValue * this->axes[Axis::Y].valueFractionLength;
        if (((x >= 0.0f) && (x <= this->axes[Axis::X].length)) && ((y >= 0.0f) && (y <= this->axes[Axis::Y].length))) {
            color = this->utils.Color(CinematicUtils::Colors::KEYFRAME);
            if ((*keyframes)[i] == skf) {
                color = this->utils.Color(CinematicUtils::Colors::KEYFRAME_SELECTED);
            }
            this->pushMarkerTexture(this->axes[Axis::X].startPos.x + x, this->axes[Axis::X].startPos.y + y, this->keyframeMarkHeight, color);
        }
    }

    // Push marker and lines for interpolated selected keyframe ---------------
    x = this->axes[Axis::X].scaleOffset + skf.GetAnimTime() * this->axes[Axis::X].valueFractionLength;
    yAxisValue = 0.0f;
    switch (this->yAxisParam) {
        case (ActiveParam::SIMULATION_TIME): yAxisValue = skf.GetSimTime(); break;
        default: break;
    }
    y = this->axes[Axis::Y].scaleOffset + yAxisValue * this->axes[Axis::Y].maxValue  * this->axes[Axis::Y].valueFractionLength;
    if (((x >= 0.0f) && (x <= this->axes[Axis::X].length)) && ((y >= 0.0f) && (y <= this->axes[Axis::Y].length))) {
        color = this->utils.Color(CinematicUtils::Colors::KEYFRAME_SELECTED);
        this->pushMarkerTexture(this->axes[Axis::X].startPos.x + x, this->axes[Axis::X].startPos.y + y, (this->keyframeMarkHeight*0.75f), color);
        start = origin + glm::vec3(x, 0.0f, 0.0f);
        end = origin + glm::vec3(x, y, 0.0f);
        this->utils.PushLinePrimitive(start, end, 1.0f, cam_view, cam_pos, color);
        start = origin + glm::vec3(0.0f, y, 0.0f);
        end = origin + glm::vec3(x, y, 0.0f);
        this->utils.PushLinePrimitive(start, end, 1.0f, cam_view, cam_pos, color);
    }

    // Push marker for dragged keyframe ---------------------------------------
    if (this->dragDropActive) {
        x = this->axes[Axis::X].scaleOffset + this->dragDropKeyframe.GetAnimTime() * this->axes[Axis::X].valueFractionLength;
        yAxisValue = 0.0f;
        switch (this->yAxisParam) {
        case (ActiveParam::SIMULATION_TIME): yAxisValue = this->dragDropKeyframe.GetSimTime(); break;
        default: break;
        }
        y = this->axes[Axis::Y].scaleOffset + yAxisValue * this->axes[Axis::Y].maxValue  * this->axes[Axis::Y].valueFractionLength;
        if (((x >= 0.0f) && (x <= this->axes[Axis::X].length)) && ((y >= 0.0f) && (y <= this->axes[Axis::Y].length))) {
            this->pushMarkerTexture(this->axes[Axis::X].startPos.x + x, this->axes[Axis::X].startPos.y + y, this->keyframeMarkHeight, this->utils.Color(CinematicUtils::Colors::KEYFRAME_DRAGGED));
        }
    }

    // Push text --------------------------------------------------------------
    vislib::StringA tmpStr;
    float strHeight = this->utils.GetTextLineHeight();
    // X axis time steps
    float timeStep = 0.0f;
    tmpStr.Format(this->axes[Axis::X].formatStr.c_str(), this->axes[Axis::X].maxValue);
    float strWidth = this->utils.GetTextLineWidth(std::string(tmpStr.PeekBuffer()));
    for (float f = this->axes[Axis::X].scaleOffset; f < this->axes[Axis::X].length + (this->axes[Axis::X].segmSize / 10.0f); f = f + this->axes[Axis::X].segmSize) {
        if (f >= 0.0f) {
            tmpStr.Format(this->axes[Axis::X].formatStr.c_str(), timeStep);
            this->utils.PushText(std::string(tmpStr.PeekBuffer()), this->axes[Axis::X].startPos.x + f - strWidth / 2.0f, this->axes[Axis::X].startPos.y - this->rulerMarkHeight, 0.0f);
        }
        timeStep += this->axes[Axis::X].segmValue;
    }
    // Y axis time steps
    timeStep = 0.0f;
    tmpStr.Format(this->axes[Axis::Y].formatStr.c_str(), this->axes[Axis::Y].maxValue);
    strWidth = this->utils.GetTextLineWidth(std::string(tmpStr.PeekBuffer()));
    float tmpStrWidth = strWidth;
    for (float f = this->axes[Axis::Y].scaleOffset; f < this->axes[Axis::Y].length + (this->axes[Axis::Y].segmSize / 10.0f); f = f + this->axes[Axis::Y].segmSize) {
        if (f >= 0.0f) {
            tmpStr.Format(this->axes[Axis::Y].formatStr.c_str(), timeStep);
            this->utils.PushText(std::string(tmpStr.PeekBuffer()), this->axes[Axis::X].startPos.x - this->rulerMarkHeight - strWidth, this->axes[Axis::X].startPos.y + strHeight / 2.0f + f, 0.0f);
        }
        timeStep += this->axes[Axis::Y].segmValue;
    }
    // Axis captions
    std::string caption = "Animation Time and Frames ";
    strWidth = this->utils.GetTextLineWidth(caption);
    this->utils.PushText(caption, this->axes[Axis::X].startPos.x + this->axes[Axis::X].length / 2.0f - strWidth / 2.0f, this->axes[Axis::X].startPos.y - this->utils.GetTextLineHeight() - this->rulerMarkHeight, 0.0f);
    caption = " ";
    switch (this->yAxisParam) {
        case (ActiveParam::SIMULATION_TIME): caption = "Simulation Time "; break;
        default: break;
    }
    strWidth = this->utils.GetTextLineWidth(caption);
    this->utils.SetTextRotation(90.0f, 0.0f, 0.0f, 1.0f);
    this->utils.PushText(caption, this->axes[Axis::X].startPos.y + this->axes[Axis::Y].length / 2.0f - strWidth / 2.0f, (-1.0f)*this->axes[Axis::X].startPos.x + tmpStrWidth + this->rulerMarkHeight + 1.5f*strHeight, 0.0f);
    this->utils.SetTextRotation(0.0f, 0.0f, 0.0f, 0.0f);

    // Push menu --------------------------------------------------------------
    auto activeKeyframe = (this->dragDropActive) ? (this->dragDropKeyframe) : (skf);
    std::stringstream stream;
    stream << std::fixed << std::setprecision(3) <<
        " Animation Time: " << activeKeyframe.GetAnimTime() <<
        " | Animation Frame: " << std::floor(activeKeyframe.GetAnimTime() * static_cast<float>(this->fps));
    switch (this->yAxisParam) {
        case (ActiveParam::SIMULATION_TIME): stream << " | Simulation Time: " << (activeKeyframe.GetSimTime() * this->axes[Axis::Y].maxValue) << " "; break;
        default: break;
    }
    std::string leftLabel = " TIMELINE ";
    std::string midLabel = stream.str();
    std::string rightLabel = "";
    this->utils.PushMenu(leftLabel, midLabel, rightLabel, this->viewport.x, this->viewport.y);

    // Draw all ---------------------------------------------------------------
    this->utils.DrawAll(ortho, this->viewport);

	return true;
}


void TimeLineRenderer::pushMarkerTexture(float pos_x, float pos_y, float size, glm::vec4 color) {

    // Push texture markers
    glm::vec3 pos_bottom_left  = { pos_x - (size / 2.0f), pos_y, 0.0f };
    glm::vec3 pos_upper_left   = { pos_x - (size / 2.0f), pos_y + size, 0.0f };
    glm::vec3 pos_upper_right  = { pos_x + (size / 2.0f), pos_y + size, 0.0f };
    glm::vec3 pos_bottom_right = { pos_x + (size / 2.0f), pos_y, 0.0f };
    this->utils.Push2DColorTexture(this->texture, pos_bottom_left, pos_upper_left, pos_upper_right, pos_bottom_right, true, color);
}


bool TimeLineRenderer::recalcAxesData(void) {

    vislib::StringA tmpStr;

    // Check for too small viewport
    if ((this->axes[Axis::X].startPos.x >= this->axes[Axis::X].endPos.x) ||
        (this->axes[Axis::Y].startPos.y >= this->axes[Axis::Y].endPos.y)) {
        vislib::sys::Log::DefaultLog.WriteWarn("[TIMELINE RENDERER] [GetExtents] Viewport is too small to calculate proper dimensions of time line diagram.");
        return false;
    }

    for (size_t i = 0; i < Axis::COUNT; ++i) {

        if (this->axes[i].maxValue <= 0.0f) {
            vislib::sys::Log::DefaultLog.WriteError("[TIMELINE RENDERER] [recalcAxesData] Invalid max value %f of axis %d", this->axes[i].maxValue, i);
            return false;
        }

        float powersOfTen = 1.0f;
        float tmpTime = this->axes[i].maxValue;
        while (tmpTime > 1.0f) {
            tmpTime /= 10.0f;
            powersOfTen *= 10.0f;
        }
        this->axes[i].segmValue = powersOfTen;

        unsigned int animPot = 0;
        unsigned int refine = 1;

        while (refine != 0) {
            float div = 5.0f;
            if (refine % 2 == 1) {
                div = 2.0f;
            }
            refine++;
            this->axes[i].segmValue /= div;

            if (this->axes[i].segmValue < 3.0f) {
                animPot++;
            }

            float maxSegmSize = 0.0;
            switch (i) {
                case (Axis::X): {
                    std::stringstream stream;
                    stream << std::fixed << std::setprecision(animPot) << this->axes[i].maxValue;
                    maxSegmSize = this->utils.GetTextLineWidth(stream.str()) * 1.25f;
                } break;
                case (Axis::Y): {
                    maxSegmSize = this->utils.GetTextLineHeight() * 1.25f;
                } break;
                default: break;
            }

            this->axes[i].segmSize = this->axes[i].length / this->axes[i].maxValue * this->axes[i].segmValue * this->axes[i].scaleFactor;

            if (this->axes[i].segmSize < maxSegmSize) {
                this->axes[i].segmValue *= div;
                this->axes[i].segmSize = this->axes[i].length / this->axes[i].maxValue * this->axes[i].segmValue * this->axes[i].scaleFactor;
                if (animPot > 0) {
                    animPot--;
                }
                if (refine % 2 == 0) {
                    refine = 0;
                }
            }
        }
        std::stringstream stream;
        stream << std::fixed << "%." << animPot << "f";
        this->axes[i].formatStr = stream.str();

        this->axes[i].valueFractionLength = this->axes[i].length / this->axes[i].maxValue * this->axes[i].scaleFactor;
        this->axes[i].scaleOffset = this->axes[i].rulerPos - (this->axes[i].scaleDelta * this->axes[i].scaleFactor);
        this->axes[i].scaleOffset = (this->axes[i].scaleOffset > 0.0f) ? (0.0f) : (this->axes[i].scaleOffset);

        // hard offset reset if scaling factor is less than one
        if (this->axes[i].scaleFactor <= 1.0f) {
            this->axes[i].scaleOffset = 0.0f;
        }
    }

    return true;
}


bool TimeLineRenderer::OnMouseButton(megamol::core::view::MouseButton button, megamol::core::view::MouseButtonAction action, megamol::core::view::Modifiers mods) {

    auto ccc = this->keyframeKeeperSlot.CallAs<CallKeyframeKeeper>();
    if (ccc == nullptr) return false;
    if (!(*ccc)(CallKeyframeKeeper::CallForGetUpdatedKeyframeData)) return false;
    auto keyframes = ccc->GetKeyframes();
    if (keyframes == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn("[TIMELINE RENDERER] [OnMouseButton] Pointer to keyframe array is nullptr.");
        return false;
    }

    auto down = (action == MouseButtonAction::PRESS);
    this->mouseAction = action;
    this->mouseButton = button;
    float yAxisValue;

    // LEFT-CLICK --- keyframe selection
    if (button == MouseButton::BUTTON_LEFT) {
        // Do not snap to keyframe when mouse movement is continuous
        float offset = this->keyframeMarkHeight / 2.0f;
        float xAxisX, yAxisY, posX, posY;
        //Check all keyframes if they are hit
        bool hit = false;
        for (unsigned int i = 0; i < keyframes->size(); i++) {
            xAxisX = this->axes[Axis::X].scaleOffset + (*keyframes)[i].GetAnimTime() * this->axes[Axis::X].valueFractionLength;
            yAxisValue = 0.0f;
            switch (this->yAxisParam) {
                case (ActiveParam::SIMULATION_TIME): yAxisValue = (*keyframes)[i].GetSimTime(); break;
                default: break;
            }
            yAxisY  = this->axes[Axis::Y].scaleOffset  + yAxisValue * this->axes[Axis::Y].maxValue  * this->axes[Axis::Y].valueFractionLength;
            if ((xAxisX >= 0.0f) && (xAxisX <= this->axes[Axis::X].length)) {
                posX = this->axes[Axis::X].startPos.x + xAxisX;
                posY = this->axes[Axis::X].startPos.y + yAxisY;
                if (((this->mouseX < (posX + offset)) && (this->mouseX > (posX - offset))) &&
                    ((this->mouseY < (posY + 2.0*offset)) && (this->mouseY > (posY)))) {
                    // If another keyframe is already hit, check which keyframe is closer to mouse position
                    if (hit) {
                        float deltaX = glm::abs(posX - this->mouseX);
                        xAxisX = this->axes[Axis::X].scaleOffset + ccc->GetSelectedKeyframe().GetAnimTime() * this->axes[Axis::X].valueFractionLength;
                        if ((xAxisX >= 0.0f) && (xAxisX <= this->axes[Axis::X].length)) {
                            posX = this->axes[Axis::X].startPos.x + xAxisX;
                            if (deltaX < glm::abs(posX - this->mouseX)) {
                                ccc->SetSelectedKeyframeTime((*keyframes)[i].GetAnimTime());
                            }
                        }
                    }
                    else {
                        ccc->SetSelectedKeyframeTime((*keyframes)[i].GetAnimTime());
                    }
                    hit = true;
                }
            }
        }
        if (hit) {
            // Set hit keyframe as selected
            if (!(*ccc)(CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime)) return false;
        }
        else {
            // Get interpolated keyframe selection
            if ((this->mouseX >= this->axes[Axis::X].startPos.x) && (this->mouseX <= this->axes[Axis::X].endPos.x)) {
                // Set an interpolated keyframe as selected
                float xt = (((-1.0f)*this->axes[Axis::X].scaleOffset + (this->mouseX - this->axes[Axis::X].startPos.x)) / this->axes[Axis::X].scaleFactor) / this->axes[Axis::X].length * this->axes[Axis::X].maxValue;
                ccc->SetSelectedKeyframeTime(xt);
                if (!(*ccc)(CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime)) return false;
            }
        }
    } // RIGHT-CLICK --- Drag & Drop of keyframe OR pan axes ...
    else if (button == MouseButton::BUTTON_RIGHT) {
        if (down) {
            //Check all keyframes if they are hit
            this->dragDropActive = false;
            float offset = this->keyframeMarkHeight / 2.0f;
            float xAxisX, yAxisY, posX, posY;
            bool hit = false;
            for (unsigned int i = 0; i < keyframes->size(); i++) {
                xAxisX = this->axes[Axis::X].scaleOffset + (*keyframes)[i].GetAnimTime() * this->axes[Axis::X].valueFractionLength;
                yAxisValue = 0.0f;
                switch (this->yAxisParam) {
                    case (ActiveParam::SIMULATION_TIME): yAxisValue = (*keyframes)[i].GetSimTime(); break;
                    default: break;
                }
                yAxisY = this->axes[Axis::Y].scaleOffset + yAxisValue * this->axes[Axis::Y].maxValue  * this->axes[Axis::Y].valueFractionLength;
                if ((xAxisX >= 0.0f) && (xAxisX <= this->axes[Axis::X].length)) {
                    posX = this->axes[Axis::X].startPos.x + xAxisX;
                    posY = this->axes[Axis::X].startPos.y + yAxisY;
                    if (((this->mouseX < (posX + offset)) && (this->mouseX > (posX - offset))) &&
                        ((this->mouseY < (posY + 2.0*offset)) && (this->mouseY > (posY)))) {
                        // If another keyframe is already hit, check which keyframe is closer to mouse position
                        if (hit) {
                            float deltaX = glm::abs(posX - this->mouseX);
                            xAxisX = this->axes[Axis::X].scaleOffset + ccc->GetSelectedKeyframe().GetAnimTime() * this->axes[Axis::X].valueFractionLength;
                            if ((xAxisX >= 0.0f) && (xAxisX <= this->axes[Axis::X].length)) {
                                posX = this->axes[Axis::X].startPos.x + xAxisX;
                                if (deltaX < glm::abs(posX - this->mouseX)) {
                                    this->dragDropKeyframe = (*keyframes)[i];
                                    ccc->SetSelectedKeyframeTime((*keyframes)[i].GetAnimTime());
                                }
                            }
                        }
                        else {
                            this->dragDropKeyframe = (*keyframes)[i];
                            ccc->SetSelectedKeyframeTime((*keyframes)[i].GetAnimTime());
                        }
                        hit = true;
                    }
                }
            }

            if (hit) {
                // Store hit keyframe locally
                this->dragDropActive = true;
                this->axisDragDropMode = 0;
                if (!(*ccc)(CallKeyframeKeeper::CallForSetDragKeyframe)) return false;
            }
            this->lastMouseX = this->mouseX;
            this->lastMouseY = this->mouseY;
        }
        else {
            // Drop currently dragged keyframe
            if (this->dragDropActive) {
                float xt = this->dragDropKeyframe.GetAnimTime();
                yAxisValue = 0.0f;
                switch (this->yAxisParam) {
                    case (ActiveParam::SIMULATION_TIME): yAxisValue = this->dragDropKeyframe.GetSimTime(); break;
                    default: break;
                }
                float yt = yAxisValue;
                if (this->axisDragDropMode == 1) { // x axis
                    xt = this->dragDropKeyframe.GetAnimTime() + ((this->mouseX - this->lastMouseX) / this->axes[Axis::X].scaleFactor) / this->axes[Axis::X].length * this->axes[Axis::X].maxValue;
                    if (this->mouseX <= this->axes[Axis::X].startPos.x) {
                        xt = 0.0f;
                    }
                    if (this->mouseX >= this->axes[Axis::X].endPos.x) {
                        xt = this->axes[Axis::X].maxValue;
                    }
                    yt = yAxisValue;
                }
                else if (this->axisDragDropMode == 2) { // y axis
                    yt = yAxisValue + ((this->mouseY - this->lastMouseY) / this->axes[Axis::Y].scaleFactor) / this->axes[Axis::Y].length;
                    if (this->mouseY < this->axes[Axis::X].startPos.y) {
                        yt = 0.0f;
                    }
                    if (this->mouseY > this->axes[Axis::Y].endPos.y) {
                        yt = 1.0f;
                    }
                    xt = this->dragDropKeyframe.GetAnimTime();
                }
                ccc->SetDropTimes(xt, yt);
                if (!(*ccc)(CallKeyframeKeeper::CallForSetDropKeyframe)) return false;

                this->dragDropActive = false;
                this->axisDragDropMode = 0;
            }
        }
    } // MIDDLE-CLICK --- Axis scaling
    else if (button == MouseButton::BUTTON_MIDDLE) {
        if (down) {
            // Just save current mouse position
            this->axisScaleMode  = 0;
            this->lastMouseX = this->mouseX;
            this->lastMouseY = this->mouseY;

            this->axes[Axis::X].rulerPos = glm::clamp(this->mouseX - this->axes[Axis::X].startPos.x, 0.0f, this->axes[Axis::X].length);
            this->axes[Axis::Y].rulerPos  = glm::clamp(this->mouseY - this->axes[Axis::X].startPos.y, 0.0f, this->axes[Axis::Y].length);

            this->axes[Axis::Y].scaleDelta = (this->axes[Axis::Y].rulerPos - this->axes[Axis::Y].scaleOffset) / this->axes[Axis::Y].scaleFactor;
            this->axes[Axis::X].scaleDelta = (this->axes[Axis::X].rulerPos - this->axes[Axis::X].scaleOffset) / this->axes[Axis::X].scaleFactor;
        }
    }

    return true;
}


bool TimeLineRenderer::OnMouseMove(double x, double y) {

    auto ccc = this->keyframeKeeperSlot.CallAs<CallKeyframeKeeper>();
    if (ccc == nullptr) return false;
    if (!(*ccc)(CallKeyframeKeeper::CallForGetUpdatedKeyframeData)) return false;

    bool down = (this->mouseAction == MouseButtonAction::PRESS);
    float yAxisValue;

    // Store current mouse position
    this->mouseX = (float)static_cast<int>(x);
    this->mouseY = (float)static_cast<int>(y);

    // LEFT-CLICK --- keyframe selection
    if (this->mouseButton == MouseButton::BUTTON_LEFT) {
        if (down) {
            // Get interpolated keyframe selection
            if ((this->mouseX >= this->axes[Axis::X].startPos.x) && (this->mouseX <= this->axes[Axis::X].endPos.x)) {
                // Set an interpolated keyframe as selected
                float xt = (((-1.0f)*this->axes[Axis::X].scaleOffset + (this->mouseX - this->axes[Axis::X].startPos.x)) / this->axes[Axis::X].scaleFactor) / this->axes[Axis::X].length * this->axes[Axis::X].maxValue;
                ccc->SetSelectedKeyframeTime(xt);
                if (!(*ccc)(CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime)) return false;
            }
        }
    } // RIGHT-CLICK --- Drag & Drop of keyframe OR pan axes ...
    else if (this->mouseButton == MouseButton::BUTTON_RIGHT) {
        if (down) {
            // Update time of dragged keyframe. Only for locally stored dragged keyframe -> just for drawing
            if (this->dragDropActive) {
                if (this->axisDragDropMode == 0) { // first time after activation of dragging a keyframe
                    if (glm::abs(this->mouseX - this->lastMouseX) > glm::abs(this->mouseY - this->lastMouseY)) {
                        this->axisDragDropMode = 1;
                    }
                    else {
                        this->axisDragDropMode = 2;
                    }
                }

                if (this->axisDragDropMode == 1) { // x axis
                    float xt = this->dragDropKeyframe.GetAnimTime() + ((this->mouseX - this->lastMouseX) / this->axes[Axis::X].scaleFactor) / this->axes[Axis::X].length * this->axes[Axis::X].maxValue;
                    if (this->mouseX < this->axes[Axis::X].startPos.x) {
                        xt = 0.0f;
                    }
                    if (this->mouseX > this->axes[Axis::X].endPos.x) {
                        xt = this->axes[Axis::X].maxValue;
                    }
                    this->dragDropKeyframe.SetAnimTime(xt);
                }
                else if (this->axisDragDropMode == 2) { // y axis
                    float yAxisValue = 0.0f;
                    switch (this->yAxisParam) {
                        case (ActiveParam::SIMULATION_TIME): yAxisValue = this->dragDropKeyframe.GetSimTime(); break;
                        default: break;
                    }
                    float yt = yAxisValue + ((this->mouseY - this->lastMouseY) / this->axes[Axis::Y].scaleFactor) / this->axes[Axis::Y].length;
                    if (this->mouseY < this->axes[Axis::X].startPos.y) {
                        yt = 0.0f;
                    }
                    if (this->mouseY > this->axes[Axis::Y].endPos.y) {
                        yt = 1.0f;
                    }
                    switch (this->yAxisParam) {
                        case (ActiveParam::SIMULATION_TIME): this->dragDropKeyframe.SetSimTime(yt); break;
                        default: break;
                    }
                }
            }
            else {
                // Pan axes ...
                float panFac = 0.5f;
                this->axes[Axis::X].scaleOffset += (this->mouseX - this->lastMouseX) * panFac;
                this->axes[Axis::Y].scaleOffset  += (this->mouseY - this->lastMouseY) * panFac;

                // Limit pan
                if (this->axes[Axis::X].scaleOffset >= 0.0f) {
                    this->axes[Axis::X].scaleOffset = 0.0f;
                }
                else if ((this->axes[Axis::X].scaleOffset + (this->axes[Axis::X].maxValue * this->axes[Axis::X].valueFractionLength)) < this->axes[Axis::X].length) {
                    this->axes[Axis::X].scaleOffset = this->axes[Axis::X].length - (this->axes[Axis::X].maxValue * this->axes[Axis::X].valueFractionLength);
                }
                if (this->axes[Axis::Y].scaleOffset >= 0.0f) {
                    this->axes[Axis::Y].scaleOffset = 0.0f;
                }
                else if ((this->axes[Axis::Y].scaleOffset + (this->axes[Axis::Y].maxValue * this->axes[Axis::Y].valueFractionLength)) < this->axes[Axis::Y].length) {
                    this->axes[Axis::Y].scaleOffset = this->axes[Axis::Y].length - (this->axes[Axis::Y].maxValue * this->axes[Axis::Y].valueFractionLength);
                }

            }
            this->lastMouseX = this->mouseX;
            this->lastMouseY = this->mouseY;
        }
    } // MIDDLE-CLICK --- Axis scaling
    else if (this->mouseButton == MouseButton::BUTTON_MIDDLE) {
        if (down) {
            float sensitivityX = 0.01f;
            float sensitivityY = 0.03f;
            float diffX = (this->mouseX - this->lastMouseX);
            float diffY = (this->mouseY - this->lastMouseY);

            if (this->axisScaleMode == 0) { // first time after activation of dragging a keyframe
                if (glm::abs(diffX) > glm::abs(diffY)) {
                    this->axisScaleMode = 1;
                }
                else {
                    this->axisScaleMode = 2;
                }
            }

            if (this->axisScaleMode == 1) { // x axis

                this->axes[Axis::X].scaleFactor += diffX * sensitivityX;
                //vislib::sys::Log::DefaultLog.WriteInfo("[axes[Axis::X].scaleFactor] %f", this->axes[Axis::X].scaleFactor);

                this->axes[Axis::X].scaleFactor = (this->axes[Axis::X].scaleFactor < 1.0f) ? (1.0f) : (this->axes[Axis::X].scaleFactor);
                this->recalcAxesData();
            }
            else if (this->axisScaleMode == 2) { // y axis

                this->axes[Axis::Y].scaleFactor += diffY * sensitivityY;
                //vislib::sys::Log::DefaultLog.WriteInfo("[axes[Axis::Y].scaleFactor] %f", this->axes[Axis::Y].scaleFactor);

                this->axes[Axis::Y].scaleFactor = (this->axes[Axis::Y].scaleFactor < 1.0f) ? (1.0f) : (this->axes[Axis::Y].scaleFactor);
                this->recalcAxesData();
            }
            this->lastMouseX = this->mouseX;
            this->lastMouseY = this->mouseY;
        }
    }

    return true;
}
