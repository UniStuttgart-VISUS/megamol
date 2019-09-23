/*
* TimeLineRenderer.cpp
*
* Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"

#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/view/Renderer2DModule.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/EnumParam.h"

#include "vislib/String.h"
#include "vislib/graphics/gl/OpenGLTexture2D.h"
#include "vislib/sys/Log.h"
#include "vislib/graphics/gl/SimpleFont.h"
#include "vislib/graphics/BitmapImage.h"

#include "TimeLineRenderer.h"
#include "CallKeyframeKeeper.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::view;
using namespace megamol::core::utility;
using namespace megamol::cinematic;

using namespace vislib;


TimeLineRenderer::TimeLineRenderer(void) : view::Renderer2DModule(),

	keyframeKeeperSlot("getkeyframes", "Connects to the KeyframeKeeper"),
    rulerFontParam("fontSize", "The font size."),
    moveRightFrameParam("gotoRightFrame", "Move to right animation time frame."),
    moveLeftFrameParam("gotoLeftFrame", "Move to left animation time frame."),
    resetPanScaleParam("resetAxes", "Reset shifted and scaled time axes."),
    utils(),
    theFont(megamol::core::utility::SDFFont::FontName::ROBOTO_SANS),
    texture(0),
    axisStartPos(),
    animAxisEndPos(0.0f, 0.0f),
    animAxisLen(0.0f),
    animSegmSize(0.0f),
    animTotalTime(1.0f),
    animSegmValue(0.0f),
    animScaleFac(0.0f),
    animScaleOffset(0.0f),
    animLenTimeFrac(0.0f),
    animScalePos(0.0f),
    animScaleDelta(0.0f),
    animFormatStr("%.5f "),
    simAxisEndPos(0.0f, 0.0f),
    simAxisLen(0.0f),
    simSegmSize(0.0f),
    simTotalTime(1.0f),
    simSegmValue(0.0f),
    simScaleFac(0.0f),
    simScaleOffset(0.0f),
    simLenTimeFrac(0.0f),
    simScalePos(0.0f),
    simScaleDelta(0.0f),
    simFormatStr("%.5f "),
    scaleAxis(0),
    dragDropKeyframe(),
    dragDropActive(false),
    dragDropAxis(0),
    fontSize(22.0f),
    keyfMarkSize(1.0f),
    rulerMarkSize(1.0f),
    fps(24),
    viewport(1.0f, 1.0f),
    mouseX(0.0f),
    mouseY(0.0f),
    lastMouseX(0.0f),
    lastMouseY(0.0f),
    mouseButton(MouseButton::BUTTON_LEFT),
    mouseAction(MouseButtonAction::RELEASE)
{

    this->keyframeKeeperSlot.SetCompatibleCall<CallKeyframeKeeperDescription>();
    this->MakeSlotAvailable(&this->keyframeKeeperSlot);

    // init parameters
    this->rulerFontParam.SetParameter(new param::FloatParam(this->fontSize, 0.000001f));
    this->MakeSlotAvailable(&this->rulerFontParam);

    this->moveRightFrameParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_RIGHT, core::view::Modifier::CTRL));
    this->MakeSlotAvailable(&this->moveRightFrameParam);

    this->moveLeftFrameParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_LEFT, core::view::Modifier::CTRL));
    this->MakeSlotAvailable(&this->moveLeftFrameParam);

    this->resetPanScaleParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_P, core::view::Modifier::CTRL));
    this->MakeSlotAvailable(&this->resetPanScaleParam);
}


TimeLineRenderer::~TimeLineRenderer(void) {

	this->Release();
}


bool TimeLineRenderer::create(void) {
	
    // Initialise font
    if (!this->theFont.Initialise(this->GetCoreInstance())) {
        vislib::sys::Log::DefaultLog.WriteError("[TIMELINE RENDERER] [create] Couldn't initialize the font.");
        return false;
    }

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

    // nothing to do here ...
}


bool TimeLineRenderer::GetExtents(view::CallRender2D& call) {

	core::view::CallRender2D *cr = dynamic_cast<core::view::CallRender2D*>(&call);
	if (cr == nullptr) return false;

    cr->SetBoundingBox(cr->GetViewport());

    glm::vec2 currentViewport;
    currentViewport.x = static_cast<float>(cr->GetViewport().GetSize().GetWidth());
    currentViewport.y = static_cast<float>(cr->GetViewport().GetSize().GetHeight());

    // if viewport changes ....
    if (currentViewport != this->viewport) {
    
        // Set time line position depending on font size
        vislib::StringA tmpStr;
        if (this->simTotalTime > this->animTotalTime) {
            tmpStr.Format("%.5f ", this->simTotalTime);
        } else {
            tmpStr.Format("%.5f ", this->animTotalTime);
        }

        float strHeight = this->theFont.LineHeight(this->fontSize);
        float strWidth = this->theFont.LineWidth(this->fontSize, tmpStr);
        this->rulerMarkSize = strHeight / 2.0f;
        this->keyfMarkSize = strHeight*1.5f;

        this->axisStartPos = glm::vec2(strWidth + strHeight * 1.5f, strHeight*2.5f);
        this->animAxisEndPos = glm::vec2(currentViewport.x - strWidth, strHeight * 2.5f);
        this->simAxisEndPos = glm::vec2(strWidth + strHeight * 1.5f, currentViewport.y - this->keyfMarkSize - CC_MENU_HEIGHT);

        this->animAxisLen  = glm::length(this->animAxisEndPos - this->axisStartPos);
        this->simAxisLen   = glm::length(this->simAxisEndPos - this->axisStartPos);

        // Reset scaling factor
        this->animScaleFac = 1.0f; 
        this->simScaleFac  = 1.0f;

        this->adaptAxis();

        this->viewport = currentViewport;
    }

    return true;
}


void TimeLineRenderer::adaptAxis(void) {

    vislib::StringA tmpStr;
    float strWidth;

    // ANIMATION
    if (this->animTotalTime <= 0.0f) {
        vislib::sys::Log::DefaultLog.WriteError("[TIMELINE RENDERER] [adaptAxis] Invalid total animation time: %f", this->animTotalTime);
        return;
    }

    float powersOfTen = 1.0f;
    float tmpTime = this->animTotalTime;
    while (tmpTime > 1.0f) {
        tmpTime /= 10.0f;
        powersOfTen *= 10.0f;
    }
    this->animSegmValue = powersOfTen; // max value

    unsigned int animPot = 0;
    unsigned int refine = 1;
    while (refine != 0) {

        float div = 5.0f;
        if (refine % 2 == 1) {
            div = 2.0f;
        }
        refine++;
        this->animSegmValue /= div;

        if (this->animSegmValue < 3.0f) {
            animPot++;
        }
        this->animFormatStr.Format("%i", animPot);
        this->animFormatStr.Prepend("%.");
        this->animFormatStr.Append("f ");
        tmpStr.Format(this->animFormatStr.PeekBuffer(), this->animTotalTime);
        strWidth = this->theFont.LineWidth(this->fontSize, tmpStr) * 1.25f;

        this->animSegmSize = this->animAxisLen / this->animTotalTime * this->animSegmValue * this->animScaleFac;

        if (this->animSegmSize < strWidth) {
            this->animSegmValue *= div;
            this->animSegmSize = this->animAxisLen / this->animTotalTime * this->animSegmValue * this->animScaleFac;
            if (animPot > 0) {
                animPot--;
            }
            if (refine % 2 == 0) {
                refine = 0;
            }
        }
    }
    this->animFormatStr.Format("%i", animPot);
    this->animFormatStr.Prepend("%.");
    this->animFormatStr.Append("f ");

    this->animLenTimeFrac = this->animAxisLen / this->animTotalTime * this->animScaleFac;
    this->animScaleOffset = this->animScalePos - (this->animScaleDelta * this->animScaleFac);
    this->animScaleOffset = (this->animScaleOffset > 0.0f) ? (0.0f) : (this->animScaleOffset);

    // hard reset if scaling factor is one
    if (this->animScaleFac <= 1.0f) {
        this->animScaleOffset = 0.0f;
    }

    // SIMULATION
    if (this->simTotalTime <= 0.0f) {
        vislib::sys::Log::DefaultLog.WriteError("[TIMELINE RENDERER] [adaptAxis] Invalid total simulation time: %f", this->simTotalTime);
        return;
    }

    powersOfTen = 1.0f;
    tmpTime = this->simTotalTime;
    while (tmpTime > 1.0f) {
        tmpTime /= 10.0f;
        powersOfTen *= 10.0f;
    }
    this->simSegmValue = powersOfTen;

    refine = 1;
    unsigned int simPot = 0;
    float minSegSize = this->theFont.LineHeight(this->fontSize) * 1.25f;
    while (refine != 0) {

        float div = 5.0f;
        if (refine % 2 == 1) {
            div = 2.0f;
        }
        refine++;
        this->simSegmValue /= div;

        if (this->simSegmValue < 3.0f) {
            simPot++;
        }

        this->simSegmSize = this->simAxisLen / this->simTotalTime * this->simSegmValue * this->simScaleFac;
        if (this->simSegmSize < minSegSize) {
            this->simSegmValue *= div;
            this->simSegmSize = this->simAxisLen / this->simTotalTime * this->simSegmValue * this->simScaleFac;

            if (simPot > 0) {
                simPot--;
            }
            if (refine % 2 == 0) {
                refine = 0;
            }
        }
    }
    this->simFormatStr.Format("%i", simPot);
    this->simFormatStr.Prepend("%.");
    this->simFormatStr.Append("f ");

    this->simLenTimeFrac = this->simAxisLen / this->simTotalTime * this->simScaleFac;
    this->simScaleOffset = this->simScalePos - (this->simScaleDelta * this->simScaleFac);
    this->simScaleOffset = (this->simScaleOffset > 0.0f) ? (0.0f) : (this->simScaleOffset);

    // hard reset if scaling factor is one
    if (this->simScaleFac <= 1.0f) {
        this->simScaleOffset = 0.0f;
    }
}


bool TimeLineRenderer::Render(view::CallRender2D& call) {

    core::view::CallRender2D *cr = dynamic_cast<core::view::CallRender2D*>(&call);
    if (cr == nullptr) return false;

    // Update data in cinematic camera call
    CallKeyframeKeeper *ccc = this->keyframeKeeperSlot.CallAs<CallKeyframeKeeper>();
    if (!ccc) return false;
    if (!(*ccc)(CallKeyframeKeeper::CallForGetUpdatedKeyframeData)) return false;

     auto keyframes = ccc->getKeyframes();
    if (keyframes == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn("[TIMELINE RENDERER] [Render] Pointer to keyframe array is nullptr.");
        return false;
    }

    // Get maximum animation time 
    if (this->animTotalTime != ccc->getTotalAnimTime()) {
        this->animTotalTime = ccc->getTotalAnimTime();
        this->adaptAxis();
    }
    // Get max simulation time
    if (this->simTotalTime != ccc->getTotalSimTime()) {
        this->simTotalTime = ccc->getTotalSimTime();
        this->adaptAxis();
    }
    // Get fps
    this->fps = ccc->getFps();

    // Update parameters
    if (this->rulerFontParam.IsDirty()) {
        this->rulerFontParam.ResetDirty();
        this->fontSize = this->rulerFontParam.Param<param::FloatParam>()->Value();
        // Recalc extends of time line which depends on font size
        this->GetExtents(call);
    }

    if (this->moveRightFrameParam.IsDirty()) {
        this->moveRightFrameParam.ResetDirty();
        // Set selected animation time to right animation time frame
        float at = ccc->getSelectedKeyframe().GetAnimTime();
        float fpsFrac = 1.0f / (float)(this->fps);
        float t = std::round(at / fpsFrac) * fpsFrac;
        t += fpsFrac;
        if (std::abs(t - std::round(t)) < (fpsFrac / 2.0)) {
            t = std::round(t);
        }
        t = (t > this->animTotalTime) ? (this->animTotalTime) : (t);
        ccc->setSelectedKeyframeTime(t);
        if (!(*ccc)(CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime)) return false;
    }

    if (this->moveLeftFrameParam.IsDirty()) {
        this->moveLeftFrameParam.ResetDirty();
        // Set selected animation time to left animation time frame
        float at = ccc->getSelectedKeyframe().GetAnimTime();
        float fpsFrac = 1.0f / (float)(this->fps);
        float t = std::round(at / fpsFrac) * fpsFrac;
        t -= fpsFrac;
        if (std::abs(t - std::round(t)) < (fpsFrac / 2.0)) {
            t = std::round(t);
        }
        t = (t < 0.0f) ? (0.0f) : (t);
        ccc->setSelectedKeyframeTime(t);
        if (!(*ccc)(CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime)) return false;
    }

    if (this->resetPanScaleParam.IsDirty()) {
        this->resetPanScaleParam.ResetDirty();
        this->simScaleFac     = 1.0f;
        this->simScaleOffset  = 0.0f;
        this->animScaleFac    = 1.0f;
        this->animScaleOffset = 0.0f;
        this->adaptAxis();
    }




    // Update render utils
    glm::vec4 bc;
    glGetFloatv(GL_COLOR_CLEAR_VALUE, static_cast<GLfloat*>(glm::value_ptr(bc)));
    this->utils.SetBackgroundColor(bc);
    this->utils.ClearAllQueues();

    // Draw frame markers -----------------------------------------------------
    float frameFrac = this->animAxisLen / ((float)(this->fps) * (this->animTotalTime)) * this->animScaleFac;
    auto color = this->utils.Color(CinematicUtils::Colors::FRAME_MARKER);
    glm::vec3 start, end;
    for (float f = this->animScaleOffset; f <= this->animAxisLen; f = (f + frameFrac)) {
        if (f >= 0.0f) {
            start = glm::vec3((this->axisStartPos.x + f), this->axisStartPos.y, 0.0f);
            end = glm::vec3((this->axisStartPos.x + f), (this->axisStartPos.y + this->rulerMarkSize), 0.0f);
            this->utils.PushLinePrimitive(start, end, 50.0f, glm::vec3(start.x, start.y, 1.0f), color);
            break;
        }
    }






    // Opengl setup -----------------------------------------------------------
    GLfloat tmpLw;
    glGetFloatv(GL_LINE_WIDTH, &tmpLw);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Draw rulers ------------------------------------------------------------
    glLineWidth(2.5f);
    glColor4fv(glm::value_ptr(this->utils.Color(CinematicUtils::Colors::FOREGROUND)));
    glBegin(GL_LINES);
        glVertex2f(this->axisStartPos.x   - this->rulerMarkSize, this->axisStartPos.y);
        glVertex2f(this->animAxisEndPos.x + this->rulerMarkSize, this->animAxisEndPos.y);
        // Draw animation ruler lines
        glVertex2f(this->axisStartPos.x,                     this->axisStartPos.y + this->rulerMarkSize);
        glVertex2f(this->axisStartPos.x,                     this->axisStartPos.y - this->rulerMarkSize);
        glVertex2f(this->axisStartPos.x + this->animAxisLen, this->axisStartPos.y + this->rulerMarkSize);
        glVertex2f(this->axisStartPos.x + this->animAxisLen, this->axisStartPos.y - this->rulerMarkSize);
        for (float f = this->animScaleOffset; f < this->animAxisLen; f = f + this->animSegmSize) {
            if (f >= 0.0f) {
                glVertex2f(this->axisStartPos.x + f, this->axisStartPos.y);
                glVertex2f(this->axisStartPos.x + f, this->axisStartPos.y - this->rulerMarkSize);
            }
        }
        // Draw simulation ruler lines
        glVertex2f(this->axisStartPos.x,  this->axisStartPos.y  - this->rulerMarkSize);
        glVertex2f(this->simAxisEndPos.x, this->simAxisEndPos.y + this->rulerMarkSize);
        glVertex2f(this->simAxisEndPos.x - this->rulerMarkSize, this->simAxisEndPos.y);
        glVertex2f(this->simAxisEndPos.x + this->rulerMarkSize, this->simAxisEndPos.y);
        for (float f = this->simScaleOffset; f < this->simAxisLen; f = f + this->simSegmSize) {
            if (f >= 0.0f) {
                glVertex2f(this->axisStartPos.x - this->rulerMarkSize, this->axisStartPos.y + f);
                glVertex2f(this->axisStartPos.x, this->axisStartPos.y + f);
            }
        }
    glEnd();

    float x, y;
    Keyframe skf = ccc->getSelectedKeyframe();

    // Draw line strip between keyframes --------------------------------------
    if (keyframes->size() > 0) {
        glColor4fv(glm::value_ptr(this->utils.Color(CinematicUtils::Colors::KEYFRAME_SPLINE)));
        glBegin(GL_LINE_STRIP);
            // First vertex
            x = this->animScaleOffset;
            y = this->simScaleOffset + (*keyframes).front().GetSimTime() * this->simTotalTime * this->simLenTimeFrac;
            glVertex2f(this->axisStartPos.x + x, this->axisStartPos.y + y);
            for (unsigned int i = 0; i < keyframes->size(); i++) {
                x = this->animScaleOffset + (*keyframes)[i].GetAnimTime() * this->animLenTimeFrac;
                y = this->simScaleOffset  + (*keyframes)[i].GetSimTime()  * this->simTotalTime  * this->simLenTimeFrac;
                glVertex2f(this->axisStartPos.x + x, this->axisStartPos.y + y);
            }
            // Last vertex
            x = this->animScaleOffset + this->animTotalTime * this->animLenTimeFrac;
            y = this->simScaleOffset + (*keyframes).back().GetSimTime()  * this->simTotalTime * this->simLenTimeFrac;
            glVertex2f(this->axisStartPos.x + x, this->axisStartPos.y + y);
        glEnd();
    }

    // Draw markers for existing keyframes in array ---------------------------
    for (unsigned int i = 0; i < keyframes->size(); i++) {
        x = this->animScaleOffset + (*keyframes)[i].GetAnimTime() * this->animLenTimeFrac;
        y = this->simScaleOffset + (*keyframes)[i].GetSimTime() * this->simTotalTime  * this->simLenTimeFrac;
        if (((x >= 0.0f) && (x <= this->animAxisLen)) && ((y >= 0.0f) && (y <= this->simAxisLen))) {
            auto color = this->utils.Color(CinematicUtils::Colors::KEYFRAME);
            if ((*keyframes)[i] == skf) {
                color = this->utils.Color(CinematicUtils::Colors::KEYFRAME_SELECTED);
            }
            this->pushMarkerTexture(this->axisStartPos.x + x, this->axisStartPos.y + y, color);
        }
    }

    // Draw interpolated selected keyframe marker -----------------------------
    x = this->animScaleOffset + skf.GetAnimTime() * this->animLenTimeFrac;
    y = this->simScaleOffset + skf.GetSimTime() * this->simTotalTime  * this->simLenTimeFrac;
    if (((x >= 0.0f) && (x <= this->animAxisLen)) && ((y >= 0.0f) && (y <= this->simAxisLen))) {
        float tmpMarkerSize = this->keyfMarkSize;
        this->keyfMarkSize = this->keyfMarkSize*0.75f;
        this->pushMarkerTexture(this->axisStartPos.x + x, this->axisStartPos.y + y, this->utils.Color(CinematicUtils::Colors::KEYFRAME_SELECTED));
        this->keyfMarkSize = tmpMarkerSize;
        glLineWidth(1.0f);
        glColor4fv(glm::value_ptr(this->utils.Color(CinematicUtils::Colors::KEYFRAME_SELECTED)));
        glBegin(GL_LINES);
            glVertex2f(this->axisStartPos.x + x, this->axisStartPos.y);
            glVertex2f(this->axisStartPos.x + x, this->axisStartPos.y + y);
            glVertex2f(this->axisStartPos.x, this->axisStartPos.y + y);
            glVertex2f(this->axisStartPos.x + x, this->axisStartPos.y + y);
        glEnd();
    }

    // Draw dragged keyframe --------------------------------------------------
    if (this->dragDropActive) {
        x = this->animScaleOffset + this->dragDropKeyframe.GetAnimTime() * this->animLenTimeFrac;
        y = this->simScaleOffset + this->dragDropKeyframe.GetSimTime() * this->simTotalTime  * this->simLenTimeFrac;
        if (((x >= 0.0f) && (x <= this->animAxisLen)) && ((y >= 0.0f) && (y <= this->simAxisLen))) {
            this->pushMarkerTexture(this->axisStartPos.x + x, this->axisStartPos.y + y, this->utils.Color(CinematicUtils::Colors::KEYFRAME_DRAGGED));
        }
    }

    // Draw ruler captions ----------------------------------------------------
    vislib::StringA tmpStr;
    float strHeight = this->theFont.LineHeight(this->fontSize);
    // animation time steps
    float timeStep = 0.0f;
    tmpStr.Format(this->animFormatStr.PeekBuffer(), this->animTotalTime);
    float strWidth = this->theFont.LineWidth(this->fontSize, tmpStr);
    for (float f = this->animScaleOffset; f < this->animAxisLen + this->animSegmSize / 10.0f; f = f + this->animSegmSize) {
        if (f >= 0.0f) {
            tmpStr.Format(this->animFormatStr.PeekBuffer(), timeStep);
            strWidth = this->theFont.LineWidth(this->fontSize, tmpStr);
            this->theFont.DrawString(glm::value_ptr(this->utils.Color(CinematicUtils::Colors::FONT)), this->axisStartPos.x + f - strWidth / 2.0f, this->axisStartPos.y - this->rulerMarkSize ,
                strWidth, strHeight, this->fontSize, false, tmpStr, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
        }
        timeStep += this->animSegmValue;
    }
    // simulation time steps
    timeStep = 0.0f;
    tmpStr.Format(this->simFormatStr.PeekBuffer(), this->simTotalTime);
    strWidth = this->theFont.LineWidth(this->fontSize, tmpStr);
    float tmpStrWidth = strWidth;
    for (float f = this->simScaleOffset; f < this->simAxisLen + this->simSegmSize / 10.0f; f = f + this->simSegmSize) {
        if (f >= 0.0f) {
            tmpStr.Format(this->simFormatStr.PeekBuffer(), timeStep);
            strWidth = this->theFont.LineWidth(this->fontSize, tmpStr);
            this->theFont.DrawString(glm::value_ptr(this->utils.Color(CinematicUtils::Colors::FONT)), this->axisStartPos.x - this->rulerMarkSize - strWidth, this->axisStartPos.y + strHeight / 2.0f + f,
                strWidth, strHeight, this->fontSize, false, tmpStr, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
        }
        timeStep += this->simSegmValue;
    }

    // axis captions
    tmpStr = "Animation Time and Frames ";
    strWidth = this->theFont.LineWidth(this->fontSize, tmpStr);
    this->theFont.DrawString(glm::value_ptr(this->utils.Color(CinematicUtils::Colors::FONT)), this->axisStartPos.x + this->animAxisLen / 2.0f - strWidth / 2.0f, this->axisStartPos.y - this->theFont.LineHeight(this->fontSize) - this->rulerMarkSize,
        this->fontSize, false, tmpStr, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
    this->theFont.SetRotation(90.0f, 0.0f, 0.0f, 1.0f);
    tmpStr = "Simulation Time ";
    strWidth = this->theFont.LineWidth(this->fontSize, tmpStr);
    this->theFont.DrawString(glm::value_ptr(this->utils.Color(CinematicUtils::Colors::FONT)), this->axisStartPos.y + this->simAxisLen / 2.0f - strWidth / 2.0f, (-1.0f)*this->axisStartPos.x + tmpStrWidth + this->rulerMarkSize + 1.5f*strHeight,
        this->fontSize, false, tmpStr, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
    this->theFont.ResetRotation();

    // DRAW MENU --------------------------------------------------------------

    auto activeKeyframe = skf;
    if (this->dragDropActive) {
        activeKeyframe = this->dragDropKeyframe;
    }
    vislib::StringA leftLabel = " TIMELINE ";
    vislib::StringA midLabel = "";
    midLabel.Format("Animation Time: %.3f | Animation Frame: %.0f | Simulation Time: %.3f ", 
        activeKeyframe.GetAnimTime(), std::floor(activeKeyframe.GetAnimTime() * (float)(this->fps)), (activeKeyframe.GetSimTime() * this->simTotalTime));
    vislib::StringA rightLabel = "";
 
    float lbFontSize = (CC_MENU_HEIGHT);
    float leftLabelWidth = this->theFont.LineWidth(lbFontSize, leftLabel);
    float midleftLabelWidth = this->theFont.LineWidth(lbFontSize, midLabel);
    float rightLabelWidth = this->theFont.LineWidth(lbFontSize, rightLabel);

    // Adapt font size if height of menu text is greater than menu height
    float vpH = static_cast<float>(cr->GetViewport().GetSize().GetHeight());
    float vpW = static_cast<float>(cr->GetViewport().GetSize().GetWidth());

    float vpWhalf = vpW / 2.0f;
    while (((leftLabelWidth + midleftLabelWidth / 2.0f) > vpWhalf) || ((rightLabelWidth + midleftLabelWidth / 2.0f) > vpWhalf)) {
        lbFontSize -= 0.5f;
        leftLabelWidth = this->theFont.LineWidth(lbFontSize, leftLabel);
        midleftLabelWidth = this->theFont.LineWidth(lbFontSize, midLabel);
        rightLabelWidth = this->theFont.LineWidth(lbFontSize, rightLabel);
    }

    // Draw menu background
    float woff = 0.0f; // (vpW*0.005f);
    float hoff = 0.0f; // (vpH*0.005f);
    glColor4fv(glm::value_ptr(this->utils.Color(CinematicUtils::Colors::MENU)));
    glBegin(GL_QUADS);
        glVertex2f(-woff, vpH + hoff);
        glVertex2f(-woff, vpH + hoff - (CC_MENU_HEIGHT));
        glVertex2f(vpW + woff,  vpH + hoff - (CC_MENU_HEIGHT));
        glVertex2f(vpW + woff,  vpH + hoff);
    glEnd();

    // Draw menu labels
    this->utils.SetBackgroundColor(this->utils.Color(CinematicUtils::Colors::MENU));
    float labelPosY = vpH + hoff - (CC_MENU_HEIGHT) / 2.0f + lbFontSize / 2.0f;
    this->theFont.DrawString(glm::value_ptr(this->utils.Color(CinematicUtils::Colors::FONT)), 0.0f, labelPosY, lbFontSize, false, leftLabel, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
    this->theFont.DrawString(glm::value_ptr(this->utils.Color(CinematicUtils::Colors::FONT)), (vpW - midleftLabelWidth) / 2.0f, labelPosY, lbFontSize, false, midLabel, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
    this->theFont.DrawString(glm::value_ptr(this->utils.Color(CinematicUtils::Colors::FONT)), (vpW - rightLabelWidth), labelPosY, lbFontSize, false, rightLabel, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);

    // ------------------------------------------------------------------------

    // Reset opengl 
    glLineWidth(tmpLw);
    glDisable(GL_BLEND);

    // ########################################################################

    glm::mat4 mv;
    glGetFloatv(GL_MODELVIEW_MATRIX, glm::value_ptr(mv));
    glm::mat4 mp;
    glGetFloatv(GL_PROJECTION_MATRIX, glm::value_ptr(mp));
    glm::mat4 mvp = mp * mv;
    this->utils.DrawAllPrimitives(mvp);

	return true;
}


void TimeLineRenderer::pushMarkerTexture(float pos_x, float pos_y, glm::vec4 color) {

    glm::vec3 pos_bottom_left  = { pos_x - (this->keyfMarkSize / 2.0f), pos_y, 0.0f };
    glm::vec3 pos_upper_left   = { pos_x - (this->keyfMarkSize / 2.0f), pos_y + this->keyfMarkSize, 0.0f };
    glm::vec3 pos_upper_right  = { pos_x + (this->keyfMarkSize / 2.0f), pos_y + this->keyfMarkSize, 0.0f };
    glm::vec3 pos_bottom_right = { pos_x + (this->keyfMarkSize / 2.0f), pos_y, 0.0f};

    //this->utils.Push2DTexture(this->texture, pos_bottom_left, pos_upper_left, pos_upper_right, pos_bottom_right, true, color);

    //this->utils.PushQuadPrimitive(pos_bottom_left, pos_upper_left, pos_upper_right, pos_bottom_right, color);

    this->utils.PushLinePrimitive(pos_bottom_left, pos_upper_right, 100.0f, glm::vec3(0.0f, 0.0f, 1000.0f), color);

}


bool TimeLineRenderer::OnMouseButton(megamol::core::view::MouseButton button, megamol::core::view::MouseButtonAction action, megamol::core::view::Modifiers mods) {

    auto down = (action == MouseButtonAction::PRESS);
    this->mouseAction = action;
    this->mouseButton = button;

    CallKeyframeKeeper *ccc = this->keyframeKeeperSlot.CallAs<CallKeyframeKeeper>();
    if (ccc == nullptr) return false;
    if (!(*ccc)(CallKeyframeKeeper::CallForGetUpdatedKeyframeData)) return false;

    auto keyframes = ccc->getKeyframes();
    if (keyframes == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn("[TIMELINE RENDERER] [Mouse Event] Pointer to keyframe array is null.");
        return false;
    }

    // LEFT-CLICK --- keyframe selection
    if (button == MouseButton::BUTTON_LEFT) {
        // Do not snap to keyframe when mouse movement is continuous
        float offset = this->keyfMarkSize / 2.0f;
        float animAxisX, simAxisY, posX, posY;
        //Check all keyframes if they are hit
        bool hit = false;
        for (unsigned int i = 0; i < keyframes->size(); i++) {
            animAxisX = this->animScaleOffset + (*keyframes)[i].GetAnimTime() * this->animLenTimeFrac;
            simAxisY  = this->simScaleOffset  + (*keyframes)[i].GetSimTime() * this->simTotalTime  * this->simLenTimeFrac;
            if ((animAxisX >= 0.0f) && (animAxisX <= this->animAxisLen)) {
                posX = this->axisStartPos.x + animAxisX;
                posY = this->axisStartPos.y + simAxisY;
                if (((this->mouseX < (posX + offset)) && (this->mouseX > (posX - offset))) &&
                    ((this->mouseY < (posY + 2.0*offset)) && (this->mouseY > (posY)))) {
                    // If another keyframe is already hit, check which keyframe is closer to mouse position
                    if (hit) {
                        float deltaX = glm::abs(posX - this->mouseX);
                        animAxisX = this->animScaleOffset + ccc->getSelectedKeyframe().GetAnimTime() * this->animLenTimeFrac;
                        if ((animAxisX >= 0.0f) && (animAxisX <= this->animAxisLen)) {
                            posX = this->axisStartPos.x + animAxisX;
                            if (deltaX < glm::abs(posX - this->mouseX)) {
                                ccc->setSelectedKeyframeTime((*keyframes)[i].GetAnimTime());
                            }
                        }
                    }
                    else {
                        ccc->setSelectedKeyframeTime((*keyframes)[i].GetAnimTime());
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
            if ((this->mouseX >= this->axisStartPos.x) && (this->mouseX <= this->animAxisEndPos.x)) {
                // Set an interpolated keyframe as selected
                float at = (((-1.0f)*this->animScaleOffset + (this->mouseX - this->axisStartPos.x)) / this->animScaleFac) / this->animAxisLen * this->animTotalTime;
                ccc->setSelectedKeyframeTime(at);
                if (!(*ccc)(CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime)) return false;
            }
        }
    } // RIGHT-CLICK --- Drag & Drop of keyframe OR pan axes ...
    else if (button == MouseButton::BUTTON_RIGHT) {
        if (down) {
            //Check all keyframes if they are hit
            this->dragDropActive = false;
            float offset = this->keyfMarkSize / 2.0f;
            float animAxisX, simAxisY, posX, posY;
            bool hit = false;
            for (unsigned int i = 0; i < keyframes->size(); i++) {
                animAxisX = this->animScaleOffset + (*keyframes)[i].GetAnimTime() * this->animLenTimeFrac;
                simAxisY = this->simScaleOffset + (*keyframes)[i].GetSimTime() * this->simTotalTime  * this->simLenTimeFrac;
                if ((animAxisX >= 0.0f) && (animAxisX <= this->animAxisLen)) {
                    posX = this->axisStartPos.x + animAxisX;
                    posY = this->axisStartPos.y + simAxisY;
                    if (((this->mouseX < (posX + offset)) && (this->mouseX > (posX - offset))) &&
                        ((this->mouseY < (posY + 2.0*offset)) && (this->mouseY > (posY)))) {
                        // If another keyframe is already hit, check which keyframe is closer to mouse position
                        if (hit) {
                            float deltaX = glm::abs(posX - this->mouseX);
                            animAxisX = this->animScaleOffset + ccc->getSelectedKeyframe().GetAnimTime() * this->animLenTimeFrac;
                            if ((animAxisX >= 0.0f) && (animAxisX <= this->animAxisLen)) {
                                posX = this->axisStartPos.x + animAxisX;
                                if (deltaX < glm::abs(posX - this->mouseX)) {
                                    this->dragDropKeyframe = (*keyframes)[i];
                                    ccc->setSelectedKeyframeTime((*keyframes)[i].GetAnimTime());
                                }
                            }
                        }
                        else {
                            this->dragDropKeyframe = (*keyframes)[i];
                            ccc->setSelectedKeyframeTime((*keyframes)[i].GetAnimTime());
                        }
                        hit = true;
                    }
                }
            }

            if (hit) {
                // Store hit keyframe locally
                this->dragDropActive = true;
                this->dragDropAxis = 0;
                if (!(*ccc)(CallKeyframeKeeper::CallForSetDragKeyframe)) return false;
            }
            this->lastMouseX = this->mouseX;
            this->lastMouseY = this->mouseY;
        }
        else {
            // Drop currently dragged keyframe
            if (this->dragDropActive) {
                float at = this->dragDropKeyframe.GetAnimTime();
                float st = this->dragDropKeyframe.GetSimTime();;
                if (this->dragDropAxis == 1) { // animation axis - X
                    at = this->dragDropKeyframe.GetAnimTime() + ((this->mouseX - this->lastMouseX) / this->animScaleFac) / this->animAxisLen * this->animTotalTime;
                    if (this->mouseX <= this->axisStartPos.x) {
                        at = 0.0f;
                    }
                    if (this->mouseX >= this->animAxisEndPos.x) {
                        at = this->animTotalTime;
                    }
                    st = this->dragDropKeyframe.GetSimTime();
                }
                else if (this->dragDropAxis == 2) { // simulation axis - Y
                    st = this->dragDropKeyframe.GetSimTime() + ((this->mouseY - this->lastMouseY) / this->simScaleFac) / this->simAxisLen;
                    if (this->mouseY < this->axisStartPos.y) {
                        st = 0.0f;
                    }
                    if (this->mouseY > this->simAxisEndPos.y) {
                        st = 1.0f;
                    }
                    at = this->dragDropKeyframe.GetAnimTime();
                }
                ccc->setDropTimes(at, st);
                if (!(*ccc)(CallKeyframeKeeper::CallForSetDropKeyframe)) return false;

                this->dragDropActive = false;
                this->dragDropAxis = 0;
            }
        }
    } // MIDDLE-CLICK --- Axis scaling
    else if (button == MouseButton::BUTTON_MIDDLE) {
        if (down) {
            // Just save current mouse position
            this->scaleAxis  = 0;
            this->lastMouseX = this->mouseX;
            this->lastMouseY = this->mouseY;

            this->animScalePos = glm::clamp(this->mouseX - this->axisStartPos.x, 0.0f, this->animAxisLen);
            this->simScalePos  = glm::clamp(this->mouseY - this->axisStartPos.y, 0.0f, this->simAxisLen);

            this->simScaleDelta = (this->simScalePos - this->simScaleOffset) / this->simScaleFac;
            this->animScaleDelta = (this->animScalePos - this->animScaleOffset) / this->animScaleFac;
        }
    }

    return true;
}


bool TimeLineRenderer::OnMouseMove(double x, double y) {

    CallKeyframeKeeper *ccc = this->keyframeKeeperSlot.CallAs<CallKeyframeKeeper>();
    if (ccc == nullptr) return false;
    // Updated data from cinematic camera call
    if (!(*ccc)(CallKeyframeKeeper::CallForGetUpdatedKeyframeData)) return false;

    bool down = (this->mouseAction == MouseButtonAction::PRESS);

    // Store current mouse position
    this->mouseX = (float)static_cast<int>(x);
    this->mouseY = (float)static_cast<int>(y);

    // LEFT-CLICK --- keyframe selection
    if (this->mouseButton == MouseButton::BUTTON_LEFT) {
        if (down) {
            // Get interpolated keyframe selection
            if ((this->mouseX >= this->axisStartPos.x) && (this->mouseX <= this->animAxisEndPos.x)) {
                // Set an interpolated keyframe as selected
                float at = (((-1.0f)*this->animScaleOffset + (this->mouseX - this->axisStartPos.x)) / this->animScaleFac) / this->animAxisLen * this->animTotalTime;
                ccc->setSelectedKeyframeTime(at);
                if (!(*ccc)(CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime)) return false;
            }
        }
    } // RIGHT-CLICK --- Drag & Drop of keyframe OR pan axes ...
    else if (this->mouseButton == MouseButton::BUTTON_RIGHT) {
        if (down) {
            // Update time of dragged keyframe. Only for locally stored dragged keyframe -> just for drawing
            if (this->dragDropActive) {
                if (this->dragDropAxis == 0) { // first time after activation of dragging a keyframe
                    if (glm::abs(this->mouseX - this->lastMouseX) > glm::abs(this->mouseY - this->lastMouseY)) {
                        this->dragDropAxis = 1;
                    }
                    else {
                        this->dragDropAxis = 2;
                    }
                }

                if (this->dragDropAxis == 1) { // animation axis - X
                    float at = this->dragDropKeyframe.GetAnimTime() + ((this->mouseX - this->lastMouseX) / this->animScaleFac) / this->animAxisLen * this->animTotalTime;
                    if (this->mouseX < this->axisStartPos.x) {
                        at = 0.0f;
                    }
                    if (this->mouseX > this->animAxisEndPos.x) {
                        at = this->animTotalTime;
                    }
                    this->dragDropKeyframe.SetAnimTime(at);
                }
                else if (this->dragDropAxis == 2) { // simulation axis - Y
                    float st = this->dragDropKeyframe.GetSimTime() + ((this->mouseY - this->lastMouseY) / this->simScaleFac) / this->simAxisLen;
                    if (this->mouseY < this->axisStartPos.y) {
                        st = 0.0f;
                    }
                    if (this->mouseY > this->simAxisEndPos.y) {
                        st = 1.0f;
                    }
                    this->dragDropKeyframe.SetSimTime(st);
                }
            }
            else {
                // Pan axes ...
                float panFac = 0.5f;
                this->animScaleOffset += (this->mouseX - this->lastMouseX) * panFac;
                this->simScaleOffset  += (this->mouseY - this->lastMouseY) * panFac;

                // Limit pan
                if (this->animScaleOffset >= 0.0f) {
                    this->animScaleOffset = 0.0f;
                }
                else if ((this->animScaleOffset + (this->animTotalTime * this->animLenTimeFrac)) < this->animAxisLen) {
                    this->animScaleOffset = this->animAxisLen - (this->animTotalTime * this->animLenTimeFrac);
                }
                if (this->simScaleOffset >= 0.0f) {
                    this->simScaleOffset = 0.0f;
                }
                else if ((this->simScaleOffset + (this->simTotalTime * this->simLenTimeFrac)) < this->simAxisLen) {
                    this->simScaleOffset = this->simAxisLen - (this->simTotalTime * this->simLenTimeFrac);
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

            if (this->scaleAxis == 0) { // first time after activation of dragging a keyframe
                if (glm::abs(diffX) > glm::abs(diffY)) {
                    this->scaleAxis = 1;
                }
                else {
                    this->scaleAxis = 2;
                }
            }

            if (this->scaleAxis == 1) { // animation axis - X

                this->animScaleFac += diffX * sensitivityX;
                //vislib::sys::Log::DefaultLog.WriteWarn("[animScaleFac] %f", this->animScaleFac);

                this->animScaleFac = (this->animScaleFac < 1.0f) ? (1.0f) : (this->animScaleFac);
                this->adaptAxis();
            }
            else if (this->scaleAxis == 2) { // simulation axis - Y

                this->simScaleFac += diffY * sensitivityY;
                //vislib::sys::Log::DefaultLog.WriteWarn("[simScaleFac] %f", this->simScaleFac);

                this->simScaleFac = (this->simScaleFac < 1.0f) ? (1.0f) : (this->simScaleFac);
                this->adaptAxis();
            }
            this->lastMouseX = this->mouseX;
            this->lastMouseY = this->mouseY;
        }
    }

    return true;
}
