/*
 * TimeLineRenderer.cpp
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "TimeLineRenderer.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/utility/log/Log.h"

#include <iomanip>


using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::view;
using namespace megamol::core::utility;
using namespace megamol::cinematic_gl;

using namespace vislib;

#define CCTLR_Z_BACK (0.0f)
#define CCTLR_Z_MIDDLE (0.25f)
#define CCTLR_Z_FRONT (0.75f)

TimeLineRenderer::TimeLineRenderer()
        : mmstd_gl::Renderer2DModuleGL()
        , keyframeKeeperSlot("keyframeData", "Connects to the KeyframeKeeper")
        , moveRightFrameParam("gotoRightFrame", "Move to right animation time frame.")
        , moveLeftFrameParam("gotoLeftFrame", "Move to left animation time frame.")
        , resetPanScaleParam("resetAxes", "Reset shifted and scaled time axes.")
        , axes()
        , utils()
        , texture_id(0)
        , yAxisParam(ActiveParam::SIMULATION_TIME)
        , dragDropKeyframe()
        , dragDropActive(false)
        , axisDragDropMode(0)
        , axisScaleMode(0)
        , keyframeMarkSize(1.0f)
        , rulerMarkHeight(1.0f)
        , viewport(1.0f, 1.0f)
        , fps(24)
        , mouseX(0.0f)
        , mouseY(0.0f)
        , lastMouseX(0.0f)
        , lastMouseY(0.0f)
        , mouseButton(MouseButton::BUTTON_LEFT)
        , mouseAction(MouseButtonAction::RELEASE)
        , lineHeight(1.0f) {

    this->keyframeKeeperSlot.SetCompatibleCall<cinematic::CallKeyframeKeeperDescription>();
    this->MakeSlotAvailable(&this->keyframeKeeperSlot);

    // init parameters
    this->moveRightFrameParam.SetParameter(
        new param::ButtonParam(core::view::Key::KEY_RIGHT, core::view::Modifier::SHIFT));
    this->MakeSlotAvailable(&this->moveRightFrameParam);

    this->moveLeftFrameParam.SetParameter(
        new param::ButtonParam(core::view::Key::KEY_LEFT, core::view::Modifier::SHIFT));
    this->MakeSlotAvailable(&this->moveLeftFrameParam);

    this->resetPanScaleParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_P, core::view::Modifier::SHIFT));
    this->MakeSlotAvailable(&this->resetPanScaleParam);

    for (size_t i = 0; i < Axis::COUNT; ++i) {
        this->axes[i].startPos = {0.0f, 0.0f};
        this->axes[i].endPos = {0.0f, 0.0f};
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


TimeLineRenderer::~TimeLineRenderer() {

    this->Release();
}


bool TimeLineRenderer::create() {

    // Initialise render utils
    if (!this->utils.Initialise(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>())) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[TIMELINE RENDERER] Couldn't initialize render utils. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
        return false;
    }

    // Load texture
    std::string texture_shortfilename = "arrow.png";
    bool loaded_texture = false;
    std::string texture_filepath;
    auto resource_directories =
        frontend_resources.get<megamol::frontend_resources::RuntimeConfig>().resource_directories;
    for (auto& resource_directory : resource_directories) {
        auto found_filepath =
            megamol::core::utility::FileUtils::SearchFileRecursive(resource_directory, texture_shortfilename);
        if (!found_filepath.empty()) {
            texture_filepath = found_filepath;
        }
    }
    loaded_texture = this->utils.LoadTextureFromFile(this->texture_id, texture_filepath);

    if (!loaded_texture) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[TIMELINE RENDERER] Couldn't load marker texture. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    return true;
}


void TimeLineRenderer::release() {}


bool TimeLineRenderer::GetExtents(mmstd_gl::CallRender2DGL& call) {

    call.AccessBoundingBoxes().SetBoundingBox(0.0f, 0.0f, 0.0f, this->viewport.x, this->viewport.y, 0.0f);

    return true;
}


bool TimeLineRenderer::Render(mmstd_gl::CallRender2DGL& call) {

    auto lhsFbo = call.GetFramebuffer();

    // Get camera
    view::Camera camera = call.GetCamera();
    auto cam_pose = camera.getPose();
    glm::vec3 cam_pos = cam_pose.position;
    glm::vec3 cam_view = cam_pose.direction;
    glm::vec3 cam_up = cam_pose.up;
    auto view = camera.getViewMatrix();
    auto proj = camera.getProjectionMatrix();
    glm::mat4 ortho = proj * view;

    // Get viewport
    const float vp_fw = static_cast<float>(lhsFbo->getWidth());
    const float vp_fh = static_cast<float>(lhsFbo->getHeight());
    const glm::vec2 vp_dim = {vp_fw, vp_fh};

    if ((this->viewport != vp_dim) || (this->lineHeight != this->utils.GetTextLineHeight())) {
        this->viewport = vp_dim;
        this->lineHeight = this->utils.GetTextLineHeight();

        // Set axes position depending on font size
        vislib::StringA tmpStr;
        if (this->axes[Axis::Y].maxValue > this->axes[Axis::X].maxValue) {
            tmpStr.Format("%.6f ", this->axes[Axis::Y].maxValue);
        } else {
            tmpStr.Format("%.6f ", this->axes[Axis::X].maxValue);
        }
        float strWidth = this->utils.GetTextLineWidth(std::string(tmpStr.PeekBuffer()));
        this->rulerMarkHeight = this->lineHeight / 2.0f;
        this->keyframeMarkSize = this->lineHeight * 1.5f;
        this->axes[Axis::X].startPos = glm::vec2(strWidth + this->lineHeight * 1.5f, (this->lineHeight * 2.5f));
        this->axes[Axis::Y].startPos = this->axes[Axis::X].startPos;
        this->axes[Axis::X].endPos = glm::vec2(this->viewport.x - strWidth, this->lineHeight * 2.5f);
        this->axes[Axis::Y].endPos = glm::vec2(
            strWidth + this->lineHeight * 1.5f, this->viewport.y - (this->keyframeMarkSize * 1.1f) - this->lineHeight);
        for (size_t i = 0; i < Axis::COUNT; ++i) {
            this->axes[i].length = glm::length(this->axes[i].endPos - this->axes[i].startPos);
            this->axes[i].scaleFactor = 1.0f;
        }

        this->recalcAxesData();
    }

    // Get update data from keyframe keeper
    auto ccc = this->keyframeKeeperSlot.CallAs<cinematic::CallKeyframeKeeper>();
    if (!ccc)
        return false;
    if (!(*ccc)(cinematic::CallKeyframeKeeper::CallForGetUpdatedKeyframeData))
        return false;
    auto keyframes = ccc->GetKeyframes();
    if (keyframes == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[TIMELINE RENDERER] Pointer to keyframe array is nullptr.");
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
    case (ActiveParam::SIMULATION_TIME):
        yAxisMaxValue = ccc->GetTotalSimTime();
        break;
    default:
        break;
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
        if (!(*ccc)(cinematic::CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime))
            return false;
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
        if (!(*ccc)(cinematic::CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime))
            return false;
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
    float yAxisValue = 0.0f;
    auto cbc = call.BackgroundColor();
    glm::vec4 back_color = glm::vec4(static_cast<float>(cbc[0]) / 255.0f, static_cast<float>(cbc[1]) / 255.0f,
        static_cast<float>(cbc[2]) / 255.0f, 1.0f);
    this->utils.SetBackgroundColor(back_color);
    auto skf = ccc->GetSelectedKeyframe();

    // Push rulers ------------------------------------------------------------
    glm::vec3 origin = {this->axes[Axis::X].startPos.x, this->axes[Axis::X].startPos.y, CCTLR_Z_BACK};

    color = this->utils.Color(CinematicUtils::Colors::FOREGROUND);
    // Draw x axis ruler lines
    start =
        glm::vec3(this->axes[Axis::X].startPos.x - this->rulerMarkHeight, this->axes[Axis::X].startPos.y, CCTLR_Z_BACK);
    end = glm::vec3(this->axes[Axis::X].endPos.x + this->rulerMarkHeight, this->axes[Axis::X].endPos.y, CCTLR_Z_BACK);
    this->utils.PushLinePrimitive(start, end, 2.5f, cam_view, cam_pos, color);
    float loop_max = this->axes[Axis::X].length + (this->axes[Axis::X].segmSize / 2.0f);
    for (float f = this->axes[Axis::X].scaleOffset; f <= loop_max; f = f + this->axes[Axis::X].segmSize) {
        if (f >= 0.0f) {
            start = origin + glm::vec3(f, 0.0f, CCTLR_Z_BACK);
            end = origin + glm::vec3(f, -this->rulerMarkHeight, CCTLR_Z_BACK);
            this->utils.PushLinePrimitive(start, end, 2.5f, cam_view, cam_pos, color);
        }
    }
    // Push y axis ruler lines
    start =
        glm::vec3(this->axes[Axis::X].startPos.x, this->axes[Axis::X].startPos.y - this->rulerMarkHeight, CCTLR_Z_BACK);
    end = glm::vec3(this->axes[Axis::Y].endPos.x, this->axes[Axis::Y].endPos.y + this->rulerMarkHeight, CCTLR_Z_BACK);
    this->utils.PushLinePrimitive(start, end, 2.5f, cam_view, cam_pos, color);
    loop_max = this->axes[Axis::Y].length + (this->axes[Axis::Y].segmSize / 2.0f);
    for (float f = this->axes[Axis::Y].scaleOffset; f <= loop_max; f = f + this->axes[Axis::Y].segmSize) {
        if (f >= 0.0f) {
            start = origin + glm::vec3(-this->rulerMarkHeight, f, CCTLR_Z_BACK);
            end = origin + glm::vec3(0.0f, f, CCTLR_Z_BACK);
            this->utils.PushLinePrimitive(start, end, 2.5f, cam_view, cam_pos, color);
        }
    }

    // Push line strip between keyframes --------------------------------------
    origin = {this->axes[Axis::X].startPos.x, this->axes[Axis::X].startPos.y, CCTLR_Z_MIDDLE};

    if (keyframes->size() > 0) {
        color = this->utils.Color(CinematicUtils::Colors::KEYFRAME_SPLINE);
        // First vertex
        start_x = this->axes[Axis::X].scaleOffset;
        float yAxisValue = 0.0f;
        switch (this->yAxisParam) {
        case (ActiveParam::SIMULATION_TIME):
            yAxisValue = (*keyframes).front().GetSimTime();
            break;
        default:
            break;
        }
        start_y = this->axes[Axis::Y].scaleOffset +
                  yAxisValue * this->axes[Axis::Y].maxValue * this->axes[Axis::Y].valueFractionLength;
        for (unsigned int i = 0; i < keyframes->size(); i++) {
            end_x = this->axes[Axis::X].scaleOffset +
                    (*keyframes)[i].GetAnimTime() * this->axes[Axis::X].valueFractionLength;
            yAxisValue = 0.0f;
            switch (this->yAxisParam) {
            case (ActiveParam::SIMULATION_TIME):
                yAxisValue = (*keyframes)[i].GetSimTime();
                break;
            default:
                break;
            }
            end_y = this->axes[Axis::Y].scaleOffset +
                    yAxisValue * this->axes[Axis::Y].maxValue * this->axes[Axis::Y].valueFractionLength;
            start = origin + glm::vec3(start_x, start_y, CCTLR_Z_MIDDLE);
            end = origin + glm::vec3(end_x, end_y, CCTLR_Z_MIDDLE);
            this->utils.PushLinePrimitive(start, end, 2.0f, cam_view, cam_pos, color);
            start_x = end_x;
            start_y = end_y;
        }
        // Last vertex
        end_x =
            this->axes[Axis::X].scaleOffset + this->axes[Axis::X].maxValue * this->axes[Axis::X].valueFractionLength;
        yAxisValue = 0.0f;
        switch (this->yAxisParam) {
        case (ActiveParam::SIMULATION_TIME):
            yAxisValue = (*keyframes).back().GetSimTime();
            break;
        default:
            break;
        }
        end_y = this->axes[Axis::Y].scaleOffset +
                yAxisValue * this->axes[Axis::Y].maxValue * this->axes[Axis::Y].valueFractionLength;
        start = origin + glm::vec3(start_x, start_y, CCTLR_Z_MIDDLE);
        end = origin + glm::vec3(end_x, end_y, CCTLR_Z_MIDDLE);
        this->utils.PushLinePrimitive(start, end, 2.0f, cam_view, cam_pos, color);
    }

    // Push frame marker lines ------------------------------------------------
    float frameFrac = this->axes[Axis::X].length / ((float)(this->fps) * (this->axes[Axis::X].maxValue)) *
                      this->axes[Axis::X].scaleFactor;
    loop_max = this->axes[Axis::X].length + (frameFrac / 2.0f);
    for (float f = this->axes[Axis::X].scaleOffset; f <= loop_max; f = (f + frameFrac)) {
        if (f >= 0.0f) {
            start = origin + glm::vec3(f, 0.0f, CCTLR_Z_MIDDLE);
            end = origin + glm::vec3(f, this->rulerMarkHeight, CCTLR_Z_MIDDLE);
            this->utils.PushLinePrimitive(
                start, end, 1.0f, cam_view, cam_pos, this->utils.Color(CinematicUtils::Colors::FRAME_MARKER));
        }
    }

    // Push markers for all existing keyframes --------------------------------
    for (unsigned int i = 0; i < keyframes->size(); i++) {
        x = this->axes[Axis::X].scaleOffset + (*keyframes)[i].GetAnimTime() * this->axes[Axis::X].valueFractionLength;
        yAxisValue = 0.0f;
        switch (this->yAxisParam) {
        case (ActiveParam::SIMULATION_TIME):
            yAxisValue = (*keyframes)[i].GetSimTime();
            break;
        default:
            break;
        }
        y = this->axes[Axis::Y].scaleOffset +
            yAxisValue * this->axes[Axis::Y].maxValue * this->axes[Axis::Y].valueFractionLength;
        if (((x >= 0.0f) && (x <= this->axes[Axis::X].length)) && ((y >= 0.0f) && (y <= this->axes[Axis::Y].length))) {
            color = this->utils.Color(CinematicUtils::Colors::KEYFRAME);
            if ((*keyframes)[i] == skf) {
                color = this->utils.Color(CinematicUtils::Colors::KEYFRAME_SELECTED);
            }
            this->pushMarkerTexture(
                this->axes[Axis::X].startPos.x + x, this->axes[Axis::X].startPos.y + y, this->keyframeMarkSize, color);
        }
    }

    // Push marker and lines for interpolated selected keyframe ---------------
    x = this->axes[Axis::X].scaleOffset + skf.GetAnimTime() * this->axes[Axis::X].valueFractionLength;
    yAxisValue = 0.0f;
    switch (this->yAxisParam) {
    case (ActiveParam::SIMULATION_TIME):
        yAxisValue = skf.GetSimTime();
        break;
    default:
        break;
    }
    y = this->axes[Axis::Y].scaleOffset +
        yAxisValue * this->axes[Axis::Y].maxValue * this->axes[Axis::Y].valueFractionLength;
    if (((x >= 0.0f) && (x <= this->axes[Axis::X].length)) && ((y >= 0.0f) && (y <= this->axes[Axis::Y].length))) {
        color = this->utils.Color(CinematicUtils::Colors::KEYFRAME_SELECTED);
        this->pushMarkerTexture(this->axes[Axis::X].startPos.x + x, this->axes[Axis::X].startPos.y + y,
            (this->keyframeMarkSize * 0.75f), color);
        start = origin + glm::vec3(x, 0.0f, CCTLR_Z_MIDDLE);
        end = origin + glm::vec3(x, y, CCTLR_Z_MIDDLE);
        this->utils.PushLinePrimitive(start, end, 1.0f, cam_view, cam_pos, color);
        start = origin + glm::vec3(0.0f, y, CCTLR_Z_MIDDLE);
        end = origin + glm::vec3(x, y, CCTLR_Z_MIDDLE);
        this->utils.PushLinePrimitive(start, end, 1.0f, cam_view, cam_pos, color);
    }

    // Push marker for dragged keyframe ---------------------------------------
    if (this->dragDropActive) {
        x = this->axes[Axis::X].scaleOffset +
            this->dragDropKeyframe.GetAnimTime() * this->axes[Axis::X].valueFractionLength;
        yAxisValue = 0.0f;
        switch (this->yAxisParam) {
        case (ActiveParam::SIMULATION_TIME):
            yAxisValue = this->dragDropKeyframe.GetSimTime();
            break;
        default:
            break;
        }
        y = this->axes[Axis::Y].scaleOffset +
            yAxisValue * this->axes[Axis::Y].maxValue * this->axes[Axis::Y].valueFractionLength;
        if (((x >= 0.0f) && (x <= this->axes[Axis::X].length)) && ((y >= 0.0f) && (y <= this->axes[Axis::Y].length))) {
            this->pushMarkerTexture(this->axes[Axis::X].startPos.x + x, this->axes[Axis::X].startPos.y + y,
                this->keyframeMarkSize, this->utils.Color(CinematicUtils::Colors::KEYFRAME_DRAGGED));
        }
    }

    // Push text --------------------------------------------------------------
    vislib::StringA tmpStr;
    // X axis time steps
    float timeStep = 0.0f;
    tmpStr.Format(this->axes[Axis::X].formatStr.c_str(), this->axes[Axis::X].maxValue);
    float strWidth = this->utils.GetTextLineWidth(std::string(tmpStr.PeekBuffer()));
    for (float f = this->axes[Axis::X].scaleOffset;
         f < this->axes[Axis::X].length + (this->axes[Axis::X].segmSize / 10.0f);
         f = f + this->axes[Axis::X].segmSize) {
        if (f >= 0.0f) {
            tmpStr.Format(this->axes[Axis::X].formatStr.c_str(), timeStep);
            this->utils.Push2DText(ortho, std::string(tmpStr.PeekBuffer()),
                this->axes[Axis::X].startPos.x + f - strWidth / 2.0f,    // x
                this->axes[Axis::X].startPos.y - this->rulerMarkHeight); // y
        }
        timeStep += this->axes[Axis::X].segmValue;
    }
    // Y axis time steps
    timeStep = 0.0f;
    tmpStr.Format(this->axes[Axis::Y].formatStr.c_str(), this->axes[Axis::Y].maxValue);
    strWidth = this->utils.GetTextLineWidth(std::string(tmpStr.PeekBuffer()));
    float tmpStrWidth = strWidth;
    for (float f = this->axes[Axis::Y].scaleOffset;
         f < this->axes[Axis::Y].length + (this->axes[Axis::Y].segmSize / 10.0f);
         f = f + this->axes[Axis::Y].segmSize) {
        if (f >= 0.0f) {
            tmpStr.Format(this->axes[Axis::Y].formatStr.c_str(), timeStep);
            this->utils.Push2DText(ortho, std::string(tmpStr.PeekBuffer()),
                this->axes[Axis::X].startPos.x - this->rulerMarkHeight - strWidth, // x
                this->axes[Axis::X].startPos.y + this->lineHeight / 2.0f + f);     // y
        }
        timeStep += this->axes[Axis::Y].segmValue;
    }
    // Axis captions
    std::string caption = "Animation Time and Frames ";
    strWidth = this->utils.GetTextLineWidth(caption);
    this->utils.Push2DText(ortho, caption,
        this->axes[Axis::X].startPos.x + this->axes[Axis::X].length / 2.0f - strWidth / 2.0f, // x
        this->axes[Axis::X].startPos.y - this->lineHeight - this->rulerMarkHeight);           // y
    caption = " ";
    switch (this->yAxisParam) {
    case (ActiveParam::SIMULATION_TIME):
        caption = "Simulation Time ";
        break;
    default:
        break;
    }
    /// TODO Fix SDFFont text rotation...
    /*
    strWidth = this->utils.GetTextLineWidth(caption);
    this->utils.SetTextRotation(70.0f, cam_view);
    this->utils.Push2DText(ortho, caption,
        this->axes[Axis::X].startPos.y + this->axes[Axis::Y].length / 2.0f - strWidth / 2.0f, // x
        this->axes[Axis::X].startPos.x + tmpStrWidth + this->rulerMarkHeight + 1.5f * this->lineHeight); // y
    this->utils.ResetTextRotation();
    */
    // TEMP
    strWidth = this->utils.GetTextLineWidth(caption);
    this->utils.Push2DText(ortho, caption, this->axes[Axis::X].startPos.x - strWidth / 2.0f,
        this->axes[Axis::Y].endPos.y + this->lineHeight + this->rulerMarkHeight);

    // Push menu --------------------------------------------------------------
    auto activeKeyframe = (this->dragDropActive) ? (this->dragDropKeyframe) : (skf);
    std::stringstream stream;
    stream << std::fixed << std::setprecision(3) << " Animation Time: " << activeKeyframe.GetAnimTime()
           << " | Animation Frame: " << std::floor(activeKeyframe.GetAnimTime() * static_cast<float>(this->fps));
    switch (this->yAxisParam) {
    case (ActiveParam::SIMULATION_TIME):
        stream << " | Simulation Time: " << (activeKeyframe.GetSimTime() * this->axes[Axis::Y].maxValue) << " ";
        break;
    default:
        break;
    }
    std::string leftLabel = " TIMELINE ";
    std::string midLabel = stream.str();
    std::string rightLabel = "";
    this->utils.PushMenu(ortho, leftLabel, midLabel, rightLabel, this->viewport, 0.0f);

    // Draw all ---------------------------------------------------------------
    this->utils.DrawAll(ortho, this->viewport);

    return true;
}

void TimeLineRenderer::pushMarkerTexture(float pos_x, float pos_y, float size, glm::vec4 color) {

    // Push texture markers
    glm::vec3 pos_bottom_left = {pos_x - (size / 2.0f), pos_y, CCTLR_Z_FRONT};
    glm::vec3 pos_upper_left = {pos_x - (size / 2.0f), pos_y + size, CCTLR_Z_FRONT};
    glm::vec3 pos_upper_right = {pos_x + (size / 2.0f), pos_y + size, CCTLR_Z_FRONT};
    glm::vec3 pos_bottom_right = {pos_x + (size / 2.0f), pos_y, CCTLR_Z_FRONT};
    this->utils.Push2DColorTexture(
        this->texture_id, pos_bottom_left, pos_upper_left, pos_upper_right, pos_bottom_right, true, color);
}


bool TimeLineRenderer::recalcAxesData() {

    for (size_t i = 0; i < Axis::COUNT; ++i) {
        if (this->axes[i].maxValue <= 0.0f) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[TIMELINE RENDERER] Invalid max value %f of axis %d. [%s, %s, line %d]", this->axes[i].maxValue, i,
                __FILE__, __FUNCTION__, __LINE__);
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
                maxSegmSize = this->lineHeight * 1.25f;
            } break;
            default:
                break;
            }

            this->axes[i].segmSize =
                this->axes[i].length / this->axes[i].maxValue * this->axes[i].segmValue * this->axes[i].scaleFactor;

            if (maxSegmSize == 0) {
                break;
            } else if (this->axes[i].segmSize < maxSegmSize) {
                this->axes[i].segmValue *= div;
                this->axes[i].segmSize =
                    this->axes[i].length / this->axes[i].maxValue * this->axes[i].segmValue * this->axes[i].scaleFactor;
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


bool TimeLineRenderer::OnMouseButton(megamol::core::view::MouseButton button,
    megamol::core::view::MouseButtonAction action, megamol::core::view::Modifiers mods) {

    auto ccc = this->keyframeKeeperSlot.CallAs<cinematic::CallKeyframeKeeper>();
    if (ccc == nullptr)
        return false;
    if (!(*ccc)(cinematic::CallKeyframeKeeper::CallForGetUpdatedKeyframeData))
        return false;
    auto keyframes = ccc->GetKeyframes();
    if (keyframes == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[TIMELINE RENDERER] Pointer to keyframe array is nullptr.");
        return false;
    }

    auto down = (action == MouseButtonAction::PRESS);
    this->mouseAction = action;
    this->mouseButton = button;
    float yAxisValue;

    // LEFT-CLICK --- keyframe selection
    if (button == MouseButton::BUTTON_LEFT) {
        // Do not snap to keyframe when mouse movement is continuous
        float offset = this->keyframeMarkSize / 2.0f;
        float xAxisX, yAxisY, posX, posY;
        //Check all keyframes if they are hit
        bool hit = false;
        for (unsigned int i = 0; i < keyframes->size(); i++) {
            xAxisX = this->axes[Axis::X].scaleOffset +
                     (*keyframes)[i].GetAnimTime() * this->axes[Axis::X].valueFractionLength;
            yAxisValue = 0.0f;
            switch (this->yAxisParam) {
            case (ActiveParam::SIMULATION_TIME):
                yAxisValue = (*keyframes)[i].GetSimTime();
                break;
            default:
                break;
            }
            yAxisY = this->axes[Axis::Y].scaleOffset +
                     yAxisValue * this->axes[Axis::Y].maxValue * this->axes[Axis::Y].valueFractionLength;
            if ((xAxisX >= 0.0f) && (xAxisX <= this->axes[Axis::X].length)) {
                posX = this->axes[Axis::X].startPos.x + xAxisX;
                posY = this->axes[Axis::X].startPos.y + yAxisY;
                if (((this->mouseX < (posX + offset)) && (this->mouseX > (posX - offset))) &&
                    ((this->mouseY < (posY + 2.0 * offset)) && (this->mouseY > (posY)))) {
                    // If another keyframe is already hit, check which keyframe is closer to mouse position
                    if (hit) {
                        float deltaX = glm::abs(posX - this->mouseX);
                        xAxisX = this->axes[Axis::X].scaleOffset +
                                 ccc->GetSelectedKeyframe().GetAnimTime() * this->axes[Axis::X].valueFractionLength;
                        if ((xAxisX >= 0.0f) && (xAxisX <= this->axes[Axis::X].length)) {
                            posX = this->axes[Axis::X].startPos.x + xAxisX;
                            if (deltaX < glm::abs(posX - this->mouseX)) {
                                ccc->SetSelectedKeyframeTime((*keyframes)[i].GetAnimTime());
                            }
                        }
                    } else {
                        ccc->SetSelectedKeyframeTime((*keyframes)[i].GetAnimTime());
                    }
                    hit = true;
                }
            }
        }
        if (hit) {
            // Set hit keyframe as selected
            if (!(*ccc)(cinematic::CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime))
                return false;
        } else {
            // Get interpolated keyframe selection
            if ((this->mouseX >= this->axes[Axis::X].startPos.x) && (this->mouseX <= this->axes[Axis::X].endPos.x)) {
                // Set an interpolated keyframe as selected
                float xt =
                    (((-1.0f) * this->axes[Axis::X].scaleOffset + (this->mouseX - this->axes[Axis::X].startPos.x)) /
                        this->axes[Axis::X].scaleFactor) /
                    this->axes[Axis::X].length * this->axes[Axis::X].maxValue;
                ccc->SetSelectedKeyframeTime(xt);
                if (!(*ccc)(cinematic::CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime))
                    return false;
            }
        }
    } // RIGHT-CLICK --- Drag & Drop of keyframe OR pan axes ...
    else if (button == MouseButton::BUTTON_RIGHT) {
        if (down) {
            //Check all keyframes if they are hit
            this->dragDropActive = false;
            float offset = this->keyframeMarkSize / 2.0f;
            float xAxisX, yAxisY, posX, posY;
            bool hit = false;
            for (unsigned int i = 0; i < keyframes->size(); i++) {
                xAxisX = this->axes[Axis::X].scaleOffset +
                         (*keyframes)[i].GetAnimTime() * this->axes[Axis::X].valueFractionLength;
                yAxisValue = 0.0f;
                switch (this->yAxisParam) {
                case (ActiveParam::SIMULATION_TIME):
                    yAxisValue = (*keyframes)[i].GetSimTime();
                    break;
                default:
                    break;
                }
                yAxisY = this->axes[Axis::Y].scaleOffset +
                         yAxisValue * this->axes[Axis::Y].maxValue * this->axes[Axis::Y].valueFractionLength;
                if ((xAxisX >= 0.0f) && (xAxisX <= this->axes[Axis::X].length)) {
                    posX = this->axes[Axis::X].startPos.x + xAxisX;
                    posY = this->axes[Axis::X].startPos.y + yAxisY;
                    if (((this->mouseX < (posX + offset)) && (this->mouseX > (posX - offset))) &&
                        ((this->mouseY < (posY + 2.0 * offset)) && (this->mouseY > (posY)))) {
                        // If another keyframe is already hit, check which keyframe is closer to mouse position
                        if (hit) {
                            float deltaX = glm::abs(posX - this->mouseX);
                            xAxisX = this->axes[Axis::X].scaleOffset +
                                     ccc->GetSelectedKeyframe().GetAnimTime() * this->axes[Axis::X].valueFractionLength;
                            if ((xAxisX >= 0.0f) && (xAxisX <= this->axes[Axis::X].length)) {
                                posX = this->axes[Axis::X].startPos.x + xAxisX;
                                if (deltaX < glm::abs(posX - this->mouseX)) {
                                    this->dragDropKeyframe = (*keyframes)[i];
                                    ccc->SetSelectedKeyframeTime((*keyframes)[i].GetAnimTime());
                                }
                            }
                        } else {
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
                if (!(*ccc)(cinematic::CallKeyframeKeeper::CallForSetDragKeyframe))
                    return false;
            }
            this->lastMouseX = this->mouseX;
            this->lastMouseY = this->mouseY;
        } else {
            // Drop currently dragged keyframe
            if (this->dragDropActive) {
                float xt = this->dragDropKeyframe.GetAnimTime();
                yAxisValue = 0.0f;
                switch (this->yAxisParam) {
                case (ActiveParam::SIMULATION_TIME):
                    yAxisValue = this->dragDropKeyframe.GetSimTime();
                    break;
                default:
                    break;
                }
                float yt = yAxisValue;
                if (this->axisDragDropMode == 1) { // x axis
                    xt = this->dragDropKeyframe.GetAnimTime() +
                         ((this->mouseX - this->lastMouseX) / this->axes[Axis::X].scaleFactor) /
                             this->axes[Axis::X].length * this->axes[Axis::X].maxValue;
                    if (this->mouseX <= this->axes[Axis::X].startPos.x) {
                        xt = 0.0f;
                    }
                    if (this->mouseX >= this->axes[Axis::X].endPos.x) {
                        xt = this->axes[Axis::X].maxValue;
                    }
                    yt = yAxisValue;
                } else if (this->axisDragDropMode == 2) { // y axis
                    yt = yAxisValue + ((this->mouseY - this->lastMouseY) / this->axes[Axis::Y].scaleFactor) /
                                          this->axes[Axis::Y].length;
                    if (this->mouseY < this->axes[Axis::X].startPos.y) {
                        yt = 0.0f;
                    }
                    if (this->mouseY > this->axes[Axis::Y].endPos.y) {
                        yt = 1.0f;
                    }
                    xt = this->dragDropKeyframe.GetAnimTime();
                }
                ccc->SetDropTimes(xt, yt);
                if (!(*ccc)(cinematic::CallKeyframeKeeper::CallForSetDropKeyframe))
                    return false;

                this->dragDropActive = false;
                this->axisDragDropMode = 0;
            }
        }
    } // MIDDLE-CLICK --- Axis scaling
    else if (button == MouseButton::BUTTON_MIDDLE) {
        if (down) {
            // Just save current mouse position
            this->axisScaleMode = 0;
            this->lastMouseX = this->mouseX;
            this->lastMouseY = this->mouseY;

            this->axes[Axis::X].rulerPos =
                glm::clamp(this->mouseX - this->axes[Axis::X].startPos.x, 0.0f, this->axes[Axis::X].length);
            this->axes[Axis::Y].rulerPos =
                glm::clamp(this->mouseY - this->axes[Axis::X].startPos.y, 0.0f, this->axes[Axis::Y].length);

            this->axes[Axis::Y].scaleDelta =
                (this->axes[Axis::Y].rulerPos - this->axes[Axis::Y].scaleOffset) / this->axes[Axis::Y].scaleFactor;
            this->axes[Axis::X].scaleDelta =
                (this->axes[Axis::X].rulerPos - this->axes[Axis::X].scaleOffset) / this->axes[Axis::X].scaleFactor;
        }
    }

    return true;
}


bool TimeLineRenderer::OnMouseMove(double x, double y) {

    auto ccc = this->keyframeKeeperSlot.CallAs<cinematic::CallKeyframeKeeper>();
    if (ccc == nullptr)
        return false;
    if (!(*ccc)(cinematic::CallKeyframeKeeper::CallForGetUpdatedKeyframeData))
        return false;

    bool down = (this->mouseAction == MouseButtonAction::PRESS);
    float yAxisValue;

    // Store current mouse position
    this->mouseX = (float)static_cast<int>(x);
    this->mouseY = this->viewport.y - (float)static_cast<int>(y);

    // LEFT-CLICK --- keyframe selection
    if (this->mouseButton == MouseButton::BUTTON_LEFT) {
        if (down) {
            // Get interpolated keyframe selection
            if ((this->mouseX >= this->axes[Axis::X].startPos.x) && (this->mouseX <= this->axes[Axis::X].endPos.x)) {
                // Set an interpolated keyframe as selected
                float xt =
                    (((-1.0f) * this->axes[Axis::X].scaleOffset + (this->mouseX - this->axes[Axis::X].startPos.x)) /
                        this->axes[Axis::X].scaleFactor) /
                    this->axes[Axis::X].length * this->axes[Axis::X].maxValue;
                ccc->SetSelectedKeyframeTime(xt);
                if (!(*ccc)(cinematic::CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime))
                    return false;
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
                    } else {
                        this->axisDragDropMode = 2;
                    }
                }

                if (this->axisDragDropMode == 1) { // x axis
                    float xt = this->dragDropKeyframe.GetAnimTime() +
                               ((this->mouseX - this->lastMouseX) / this->axes[Axis::X].scaleFactor) /
                                   this->axes[Axis::X].length * this->axes[Axis::X].maxValue;
                    if (this->mouseX < this->axes[Axis::X].startPos.x) {
                        xt = 0.0f;
                    }
                    if (this->mouseX > this->axes[Axis::X].endPos.x) {
                        xt = this->axes[Axis::X].maxValue;
                    }
                    this->dragDropKeyframe.SetAnimTime(xt);
                } else if (this->axisDragDropMode == 2) { // y axis
                    float yAxisValue = 0.0f;
                    switch (this->yAxisParam) {
                    case (ActiveParam::SIMULATION_TIME):
                        yAxisValue = this->dragDropKeyframe.GetSimTime();
                        break;
                    default:
                        break;
                    }
                    float yt = yAxisValue + ((this->mouseY - this->lastMouseY) / this->axes[Axis::Y].scaleFactor) /
                                                this->axes[Axis::Y].length;
                    if (this->mouseY < this->axes[Axis::X].startPos.y) {
                        yt = 0.0f;
                    }
                    if (this->mouseY > this->axes[Axis::Y].endPos.y) {
                        yt = 1.0f;
                    }
                    switch (this->yAxisParam) {
                    case (ActiveParam::SIMULATION_TIME):
                        this->dragDropKeyframe.SetSimTime(yt);
                        break;
                    default:
                        break;
                    }
                }
            } else {
                // Pan axes ...
                float panFac = 0.4f;
                this->axes[Axis::X].scaleOffset += (this->mouseX - this->lastMouseX) * panFac;
                this->axes[Axis::Y].scaleOffset += (this->mouseY - this->lastMouseY) * panFac;

                // Limit pan
                if (this->axes[Axis::X].scaleOffset >= 0.0f) {
                    this->axes[Axis::X].scaleOffset = 0.0f;
                } else if ((this->axes[Axis::X].scaleOffset +
                               (this->axes[Axis::X].maxValue * this->axes[Axis::X].valueFractionLength)) <
                           this->axes[Axis::X].length) {
                    this->axes[Axis::X].scaleOffset =
                        this->axes[Axis::X].length -
                        (this->axes[Axis::X].maxValue * this->axes[Axis::X].valueFractionLength);
                }
                if (this->axes[Axis::Y].scaleOffset >= 0.0f) {
                    this->axes[Axis::Y].scaleOffset = 0.0f;
                } else if ((this->axes[Axis::Y].scaleOffset +
                               (this->axes[Axis::Y].maxValue * this->axes[Axis::Y].valueFractionLength)) <
                           this->axes[Axis::Y].length) {
                    this->axes[Axis::Y].scaleOffset =
                        this->axes[Axis::Y].length -
                        (this->axes[Axis::Y].maxValue * this->axes[Axis::Y].valueFractionLength);
                }
            }
            this->lastMouseX = this->mouseX;
            this->lastMouseY = this->mouseY;
        }
    } // MIDDLE-CLICK --- Axis scaling
    else if (this->mouseButton == MouseButton::BUTTON_MIDDLE) {
        if (down) {
            float sensitivityX = 0.01f;
            float sensitivityY = 0.01f;
            float diffX = (this->mouseX - this->lastMouseX);
            float diffY = (this->mouseY - this->lastMouseY);

            if (this->axisScaleMode == 0) { // first time after activation of dragging a keyframe
                if (glm::abs(diffX) > glm::abs(diffY)) {
                    this->axisScaleMode = 1;
                } else {
                    this->axisScaleMode = 2;
                }
            }

            if (this->axisScaleMode == 1) { // x axis

                this->axes[Axis::X].scaleFactor += diffX * sensitivityX;
                //megamol::core::utility::log::Log::DefaultLog.WriteInfo("[axes[Axis::X].scaleFactor] %f", this->axes[Axis::X].scaleFactor);

                this->axes[Axis::X].scaleFactor =
                    (this->axes[Axis::X].scaleFactor < 1.0f) ? (1.0f) : (this->axes[Axis::X].scaleFactor);
                this->recalcAxesData();
            } else if (this->axisScaleMode == 2) { // y axis

                this->axes[Axis::Y].scaleFactor += diffY * sensitivityY;
                //megamol::core::utility::log::Log::DefaultLog.WriteInfo("[axes[Axis::Y].scaleFactor] %f", this->axes[Axis::Y].scaleFactor);

                this->axes[Axis::Y].scaleFactor =
                    (this->axes[Axis::Y].scaleFactor < 1.0f) ? (1.0f) : (this->axes[Axis::Y].scaleFactor);
                this->recalcAxesData();
            }
            this->lastMouseX = this->mouseX;
            this->lastMouseY = this->mouseY;
        }
    }

    return true;
}
