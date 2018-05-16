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
#include "vislib/sys/KeyCode.h"
#include "vislib/math/ShallowMatrix.h"
#include "vislib/math/Matrix.h"

#include "TimeLineRenderer.h"
#include "CallCinematicCamera.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::utility;
using namespace megamol::cinematiccamera;

using namespace vislib;


/*
* cinematiccamera::TimeLineRenderer::TimeLineRenderer
*/
TimeLineRenderer::TimeLineRenderer(void) : view::Renderer2DModule(),
    theFont(megamol::core::utility::SDFFont::FontName::ROBOTO_SANS),
	keyframeKeeperSlot("getkeyframes", "Connects to the KeyframeKeeper"),
    rulerFontParam( "01_fontSize", "The font size."),
    moveRightFrame ("02_rightFrame", "Move to right animation time frame."),
    moveLeftFrame( "03_leftFrame", "Move to left animation time frame.")
	{

    this->keyframeKeeperSlot.SetCompatibleCall<CallCinematicCameraDescription>();
    this->MakeSlotAvailable(&this->keyframeKeeperSlot);

    // init variables
    this->fontSize          = 22.0f;

    this->markerTextures.Clear();

    this->dragDropKeyframe  = Keyframe();
    this->viewport          = vislib::math::Vector<float, 2>(1.0f, 1.0f);
    this->axisStartPos      = vislib::math::Vector<float, 2>(0.0f, 0.0f);
    this->lastMousePos      = vislib::math::Vector<float, 2>(0.0f, 0.0f);
    this->scaleAxis         = 0;
    this->dragDropActive    = false;
    this->dragDropAxis      = 0;
    this->rulerMarkSize     = 1.0f;
    this->keyfMarkSize      = 1.0f;
    this->fps               = 24;

    this->animAxisEndPos    = vislib::math::Vector<float, 2>(0.0f, 0.0f);
    this->animAxisLen       = 0.0f;
    this->animSegmSize      = 0.0f;
    this->animTotalTime     = 1.0f;
    this->animSegmValue     = 0.0f;
    this->animScaleFac      = 0.0f;
    this->animScaleOffset   = 0.0f;
    this->animLenTimeFrac   = 0.0f;
    this->animScalePos      = 0.0f;
    this->animScaleDelta    = 0.0f;
    this->animFormatStr     = "%.5f ";

    this->simAxisEndPos     = vislib::math::Vector<float, 2>(0.0f, 0.0f);
    this->simAxisLen        = 0.0f;
    this->simSegmSize       = 0.0f;
    this->simTotalTime      = 1.0f;
    this->simSegmValue      = 0.0f;
    this->simScaleFac       = 0.0f;
    this->simScaleOffset    = 0.0f;
    this->simLenTimeFrac    = 0.0f;
    this->simScalePos       = 0.0f;
    this->simScaleDelta     = 0.0f;
    this->simFormatStr      = "%.5f ";

    // init parameters
    this->rulerFontParam.SetParameter(new param::FloatParam(this->fontSize, 0.000001f));
    this->MakeSlotAvailable(&this->rulerFontParam);

    this->moveRightFrame.SetParameter(new param::ButtonParam(vislib::sys::KeyCode::KEY_RIGHT));
    this->MakeSlotAvailable(&this->moveRightFrame);

    this->moveLeftFrame.SetParameter(new param::ButtonParam(vislib::sys::KeyCode::KEY_LEFT));
    this->MakeSlotAvailable(&this->moveLeftFrame);
}


/*
* cinematiccamera::TimeLineRenderer::~TimeLineRenderer
*/
TimeLineRenderer::~TimeLineRenderer(void) {

	this->Release();
}


/*
* cinematiccamera::TimeLineRenderer::create
*/
bool TimeLineRenderer::create(void) {
	
    // Initialise font
    if (!this->theFont.Initialise(this->GetCoreInstance())) {
        vislib::sys::Log::DefaultLog.WriteError("[TIMELINE RENDERER] [Render] Couldn't initialize the font.");
        return false;
    }

    // Initialise texture
    if (!this->loadTexture("arrow.png")) {
        vislib::sys::Log::DefaultLog.WriteError("[TIMELINE RENDERER] [Render] Couldn't initialize the texture.");
        return false;
    }

    return true;
}


/*
* cinematiccamera::TimeLineRenderer::release
*/
void TimeLineRenderer::release(void) {

    // nothing to do here ...
}


/*
* cinematiccamera::TimeLineRenderer::GetExtents
*/
bool TimeLineRenderer::GetExtents(view::CallRender2D& call) {

	core::view::CallRender2D *cr = dynamic_cast<core::view::CallRender2D*>(&call);
	if (cr == nullptr) return false;

    cr->SetBoundingBox(cr->GetViewport());

    vislib::math::Vector<float, 2> currentViewport;
    currentViewport.SetX(static_cast<float>(cr->GetViewport().GetSize().GetWidth()));
    currentViewport.SetY(static_cast<float>(cr->GetViewport().GetSize().GetHeight()));

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

        this->axisStartPos   = vislib::math::Vector<float, 2>(strWidth + strHeight*1.5f, strHeight*2.5f);
        this->animAxisEndPos = vislib::math::Vector<float, 2>(currentViewport.X() - strWidth, strHeight * 2.5f);
        this->simAxisEndPos  = vislib::math::Vector<float, 2>(strWidth + strHeight * 1.5f, currentViewport.Y() - this->keyfMarkSize - CC_MENU_HEIGHT); 

        this->animAxisLen  = (this->animAxisEndPos - this->axisStartPos).Norm();
        this->simAxisLen   = (this->simAxisEndPos - this->axisStartPos).Norm();

        // Reset scaling factor
        this->animScaleFac = 1.0f; 
        this->simScaleFac  = 1.0f;

        this->axisAdaptation();

        this->viewport = currentViewport;
    }

    return true;
}


/*
 * cinematiccamera::TimeLineRenderer::axisAdaptation
 */
void TimeLineRenderer::axisAdaptation(void) {

    vislib::StringA tmpStr;
    float strWidth;

    // ANIMATION
    if (this->animTotalTime <= 0.0f) {
        vislib::sys::Log::DefaultLog.WriteError("[TIMELINE RENDERER] [axisAdaptation] Invalid total animation time: %f", this->animTotalTime);
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
        vislib::sys::Log::DefaultLog.WriteError("[TIMELINE RENDERER] [axisAdaptation] Invalid total simulation time: %f", this->simTotalTime);
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

    this->simLenTimeFrac = this->simAxisLen * this->simScaleFac;
    this->simScaleOffset = this->simScalePos - (this->simScaleDelta * this->simScaleFac);
    this->simScaleOffset = (this->simScaleOffset > 0.0f) ? (0.0f) : (this->simScaleOffset);

    // hard reset if scaling factor is one
    if (this->simScaleFac <= 1.0f) {
        this->simScaleOffset = 0.0f;
    }
}


/*
* cinematiccamera::TimeLineRenderer::Render
*/
bool TimeLineRenderer::Render(view::CallRender2D& call) {

    core::view::CallRender2D *cr = dynamic_cast<core::view::CallRender2D*>(&call);
    if (cr == nullptr) return false;

    // Update data in cinematic camera call
    CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
    if (!ccc) return false;
    // Updated data from cinematic camera call
    if (!(*ccc)(CallCinematicCamera::CallForGetUpdatedKeyframeData)) return false;

    vislib::Array<Keyframe> *keyframes = ccc->getKeyframes();
    if (keyframes == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn("[TIMELINE RENDERER] [Render] Pointer to keyframe array is nullptr.");
        return false;
    }

    // Get maximum animation time 
    if (this->animTotalTime != ccc->getTotalAnimTime()) {
        this->animTotalTime = ccc->getTotalAnimTime();
        this->axisAdaptation();
    }
    // Get max simulation time
    if (this->simTotalTime != ccc->getTotalSimTime()) {
        this->simTotalTime = ccc->getTotalSimTime();
        this->axisAdaptation();
    }
    // Get fps
    this->fps = ccc->getFps();

    // Update parameter
    if (this->rulerFontParam.IsDirty()) {
        this->rulerFontParam.ResetDirty();
        this->fontSize = this->rulerFontParam.Param<param::FloatParam>()->Value();
        // Recalc extends of time line which depends on font size
        this->GetExtents(call);
    }
    if (this->moveRightFrame.IsDirty()) {
        this->moveRightFrame.ResetDirty();
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
        if (!(*ccc)(CallCinematicCamera::CallForGetSelectedKeyframeAtTime)) return false;
        std::cout << "right: " << t << std::endl;
    }
    if (this->moveLeftFrame.IsDirty()) {
        this->moveLeftFrame.ResetDirty();
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
        if (!(*ccc)(CallCinematicCamera::CallForGetSelectedKeyframeAtTime)) return false;
        std::cout << "left: " << t << std::endl;
    }

    // Get the foreground color (inverse background color)
    float bgColor[4];
    float fgColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glGetFloatv(GL_COLOR_CLEAR_VALUE, bgColor);
    for (unsigned int i = 0; i < 3; i++) {
        fgColor[i] -= bgColor[i];
    }
    // COLORS
    float kColor[4]    = { 0.7f, 0.7f, 1.0f, 1.0f }; // Color for KEYFRAME 
    float dkmColor[4]  = { 0.5f, 0.5f, 1.0f, 1.0f }; // Color for DRAGGED KEYFRAME MARKER 
    float skColor[4]   = { 0.2f, 0.2f, 1.0f, 1.0f }; // Color for SELECTED KEYFRAME 
    float sColor[4]    = { 0.4f, 0.4f, 1.0f, 1.0f }; // Color for SPLINE
    float fColor[4]    = { 1.0f, 0.6f, 0.6f, 1.0f }; // Color for FRAME MARKER
    float white[4]     = { 1.0f, 1.0f, 1.0f, 1.0f };
    float menu[4]      = { 0.0f, 0.0f, 0.3f, 1.0f };
    //float yellow[4]    = { 1.0f, 1.0f, 0.0f, 1.0f };
    //float armColor[4]  = { 0.8f, 0.0f, 0.0f, 1.0f }; // Color for ANIMATION REPEAT MARKER
    
    // Adapt colors depending on  Lightness
    float L = (vislib::math::Max(bgColor[0], vislib::math::Max(bgColor[1], bgColor[2])) + vislib::math::Min(bgColor[0], vislib::math::Min(bgColor[1], bgColor[2]))) / 2.0f;
    if (L < 0.5f) {
        float tmp;
        // Swap keyframe colors
        for (unsigned int i = 0; i < 4; i++) {
            tmp        = kColor[i];
            kColor[i]  = skColor[i];
            skColor[i] = tmp;
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

    // Draw frame markers -----------------------------------------------------
    float frameFrac = this->animAxisLen / ((float)(this->fps) * (this->animTotalTime)) * this->animScaleFac;
    glLineWidth(1.0f);
    glColor4fv(fColor);
    glBegin(GL_LINES);
        for (float f = this->animScaleOffset; f <= this->animAxisLen; f = f + frameFrac) {
            if (f >= 0.0f) {
                glVertex2f(this->axisStartPos.X() + f, this->axisStartPos.Y());
                glVertex2f(this->axisStartPos.X() + f, this->axisStartPos.Y() + this->rulerMarkSize);
            }
        }
    glEnd();

    // Draw rulers ------------------------------------------------------------
    glLineWidth(2.5f);
    glColor4fv(fgColor);
    glBegin(GL_LINES);
        glVertex2f(this->axisStartPos.X()   - this->rulerMarkSize, this->axisStartPos.Y());
        glVertex2f(this->animAxisEndPos.X() + this->rulerMarkSize, this->animAxisEndPos.Y());
        // Draw animation ruler lines
        glVertex2f(this->axisStartPos.X(),                     this->axisStartPos.Y() + this->rulerMarkSize);
        glVertex2f(this->axisStartPos.X(),                     this->axisStartPos.Y() - this->rulerMarkSize);
        glVertex2f(this->axisStartPos.X() + this->animAxisLen, this->axisStartPos.Y() + this->rulerMarkSize);
        glVertex2f(this->axisStartPos.X() + this->animAxisLen, this->axisStartPos.Y() - this->rulerMarkSize);
        for (float f = this->animScaleOffset; f < this->animAxisLen; f = f + this->animSegmSize) {
            if (f >= 0.0f) {
                glVertex2f(this->axisStartPos.X() + f, this->axisStartPos.Y());
                glVertex2f(this->axisStartPos.X() + f, this->axisStartPos.Y() - this->rulerMarkSize);
            }
        }
        // Draw simulation ruler lines
        glVertex2f(this->axisStartPos.X(),  this->axisStartPos.Y()  - this->rulerMarkSize);
        glVertex2f(this->simAxisEndPos.X(), this->simAxisEndPos.Y() + this->rulerMarkSize);
        glVertex2f(this->simAxisEndPos.X() - this->rulerMarkSize, this->simAxisEndPos.Y());
        glVertex2f(this->simAxisEndPos.X() + this->rulerMarkSize, this->simAxisEndPos.Y());
        for (float f = this->simScaleOffset; f < this->simAxisLen; f = f + this->simSegmSize) {
            if (f >= 0.0f) {
                glVertex2f(this->axisStartPos.X() - this->rulerMarkSize, this->axisStartPos.Y() + f);
                glVertex2f(this->axisStartPos.X(), this->axisStartPos.Y() + f);
            }
        }
    glEnd();

    float x, y;
    Keyframe skf = ccc->getSelectedKeyframe();

    // Draw line strip between keyframes --------------------------------------
    if (keyframes->Count() > 0) {
        glColor4fv(sColor);
        glBegin(GL_LINE_STRIP);
            // First vertex
            x = this->animScaleOffset;
            y = this->simScaleOffset + (*keyframes).First().GetSimTime()  * this->simLenTimeFrac;
            glVertex2f(this->axisStartPos.X() + x, this->axisStartPos.Y() + y);
            for (unsigned int i = 0; i < keyframes->Count(); i++) {
                x = this->animScaleOffset + (*keyframes)[i].GetAnimTime() * this->animLenTimeFrac;
                y = this->simScaleOffset  + (*keyframes)[i].GetSimTime()  * this->simLenTimeFrac;
                glVertex2f(this->axisStartPos.X() + x, this->axisStartPos.Y() + y);
            }
            // Last vertex
            x = this->animScaleOffset + this->animTotalTime * this->animLenTimeFrac;
            y = this->simScaleOffset + (*keyframes).Last().GetSimTime()  * this->simLenTimeFrac;
            glVertex2f(this->axisStartPos.X() + x, this->axisStartPos.Y() + y);
        glEnd();
    }

    // Draw markers for existing keyframes in array ---------------------------
    for (unsigned int i = 0; i < keyframes->Count(); i++) {
        x = this->animScaleOffset + (*keyframes)[i].GetAnimTime() * this->animLenTimeFrac;
        y = this->simScaleOffset + (*keyframes)[i].GetSimTime()  * this->simLenTimeFrac;
        if (((x >= 0.0f) && (x <= this->animAxisLen)) && ((y >= 0.0f) && (y <= this->simAxisLen))) {
            if ((*keyframes)[i] == skf) {
                glColor4fv(skColor);
            }
            else {
                glColor4fv(kColor);
            }
            this->drawKeyframeMarker(this->axisStartPos.X() + x, this->axisStartPos.Y() + y);
        }
    }

    // Draw interpolated selected keyframe marker -----------------------------
    x = this->animScaleOffset + skf.GetAnimTime() * this->animLenTimeFrac;
    y = this->simScaleOffset + skf.GetSimTime()  * this->simLenTimeFrac;
    if (((x >= 0.0f) && (x <= this->animAxisLen)) && ((y >= 0.0f) && (y <= this->simAxisLen))) {
        float tmpMarkerSize = this->keyfMarkSize;
        this->keyfMarkSize = this->keyfMarkSize*0.75f;
        glColor4fv(skColor);
        this->drawKeyframeMarker(this->axisStartPos.X() + x, this->axisStartPos.Y() + y);
        this->keyfMarkSize = tmpMarkerSize;
        glLineWidth(1.0f);
        glBegin(GL_LINES);
            glVertex2f(this->axisStartPos.X() + x, this->axisStartPos.Y());
            glVertex2f(this->axisStartPos.X() + x, this->axisStartPos.Y() + y);
            glVertex2f(this->axisStartPos.X(), this->axisStartPos.Y() + y);
            glVertex2f(this->axisStartPos.X() + x, this->axisStartPos.Y() + y);
        glEnd();
    }

    // Draw dragged keyframe --------------------------------------------------
    if (this->dragDropActive) {
        x = this->animScaleOffset + this->dragDropKeyframe.GetAnimTime() * this->animLenTimeFrac;
        y = this->simScaleOffset + this->dragDropKeyframe.GetSimTime()  * this->simLenTimeFrac;
        if (((x >= 0.0f) && (x <= this->animAxisLen)) && ((y >= 0.0f) && (y <= this->simAxisLen))) {
            glColor4fv(dkmColor);
            this->drawKeyframeMarker(this->axisStartPos.X() + x, this->axisStartPos.Y() + y);
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
            this->theFont.DrawString(fgColor, this->axisStartPos.X() + f - strWidth / 2.0f, this->axisStartPos.Y() - this->rulerMarkSize ,
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
            this->theFont.DrawString(fgColor, this->axisStartPos.X() - this->rulerMarkSize - strWidth, this->axisStartPos.Y() + strHeight / 2.0f + f,
                strWidth, strHeight, this->fontSize, false, tmpStr, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
        }
        timeStep += this->simSegmValue;
    }

    // axis captions
    tmpStr = "animation time / frames ";
    strWidth = this->theFont.LineWidth(this->fontSize, tmpStr);
    this->theFont.DrawString(fgColor, this->axisStartPos.X() + this->animAxisLen / 2.0f - strWidth / 2.0f, this->axisStartPos.Y() - this->theFont.LineHeight(this->fontSize) - this->rulerMarkSize, 
        this->fontSize, false, tmpStr, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
    glRotatef(90.0f, 0.0f, 0.0f, 1.0f);
    tmpStr = "simulation time ";
    strWidth = this->theFont.LineWidth(this->fontSize, tmpStr);
    this->theFont.DrawString(fgColor, this->axisStartPos.Y() + this->simAxisLen / 2.0f - strWidth / 2.0f, (-1.0f)*this->axisStartPos.X() + tmpStrWidth + this->rulerMarkSize + 1.5f*strHeight, 
        this->fontSize, false, tmpStr, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
    glRotatef(-90.0f, 0.0f, 0.0f, 1.0f);

    // DRAW MENU --------------------------------------------------------------

    // selected keyframe info
    glColor4fv(fgColor);
    float aT = skf.GetAnimTime();
    float aF = skf.GetAnimTime() * (float)(this->fps);;
    float sT = skf.GetSimTime()*this->simTotalTime;
    if (this->dragDropActive) {
        aT = this->dragDropKeyframe.GetAnimTime();
        aF = this->dragDropKeyframe.GetAnimTime() * (float)(this->fps);
        sT = this->dragDropKeyframe.GetSimTime()*this->simTotalTime;
    }

    float vpH = static_cast<float>(cr->GetViewport().GetSize().GetHeight());
    float vpW = static_cast<float>(cr->GetViewport().GetSize().GetWidth());

    vislib::StringA leftLabel = " TIME LINE VIEW ";
    vislib::StringA midLabel = "";
    midLabel.Format("animation time: %.3f | animation frame: %.3f | simulation time: %.3f ", aT, aF, sT);
    vislib::StringA rightLabel = "";
    
    float lbFontSize = (CC_MENU_HEIGHT);
    float leftLabelWidth = this->theFont.LineWidth(lbFontSize, leftLabel);
    float midleftLabelWidth = this->theFont.LineWidth(lbFontSize, midLabel);
    float rightLabelWidth = this->theFont.LineWidth(lbFontSize, rightLabel);

    // Adapt font size if height of menu text is greater than menu height
    float vpWhalf = vpW / 2.0f;
    while (((leftLabelWidth + midleftLabelWidth / 2.0f) > vpWhalf) || ((rightLabelWidth + midleftLabelWidth / 2.0f) > vpWhalf)) {
        lbFontSize -= 0.5f;
        leftLabelWidth = this->theFont.LineWidth(lbFontSize, leftLabel);
        midleftLabelWidth = this->theFont.LineWidth(lbFontSize, midLabel);
        rightLabelWidth = this->theFont.LineWidth(lbFontSize, rightLabel);
    }

    // Draw menu background
    glColor4fv(menu);
    glBegin(GL_QUADS);
        glVertex2f(0.0f, vpH);
        glVertex2f(0.0f, vpH - (CC_MENU_HEIGHT));
        glVertex2f(vpW,  vpH - (CC_MENU_HEIGHT));
        glVertex2f(vpW,  vpH);
    glEnd();

    // Draw menu labels
    float labelPosY = vpH - (CC_MENU_HEIGHT) / 2.0f + lbFontSize / 2.0f;
    this->theFont.DrawString(white, 0.0f, labelPosY, lbFontSize, false, leftLabel, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
    this->theFont.DrawString(white, (vpW - midleftLabelWidth) / 2.0f, labelPosY, lbFontSize, false, midLabel, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);
    this->theFont.DrawString(white, (vpW - rightLabelWidth), labelPosY, lbFontSize, false, rightLabel, megamol::core::utility::AbstractFont::ALIGN_LEFT_TOP);

    // ------------------------------------------------------------------------

    // Reset opengl 
    glLineWidth(tmpLw);
    glDisable(GL_BLEND);

	return true;
}


/*
* cinematiccamera::TimeLineRenderer::drawKeyframeMarker
*/
void TimeLineRenderer::drawKeyframeMarker(float posX, float posY) {

    glEnable(GL_TEXTURE_2D);

    this->markerTextures[0]->Bind();
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBegin(GL_QUADS);
        glTexCoord2f(1.0f, 1.0f);
        glVertex2f(posX - (this->keyfMarkSize/ 2.0f), posY);

        glTexCoord2f(1.0f, 0.0f);
        glVertex2f(posX - (this->keyfMarkSize/ 2.0f), posY + (this->keyfMarkSize));

        glTexCoord2f(0.0f, 0.0f);
        glVertex2f(posX + (this->keyfMarkSize/ 2.0f), posY + (this->keyfMarkSize));

        glTexCoord2f(0.0f, 1.0f);
        glVertex2f(posX + (this->keyfMarkSize/ 2.0f), posY);
    glEnd();

    glDisable(GL_TEXTURE_2D);
}


/*
* cinematiccamera::TimeLineRenderer::MouseEvent
*/
bool TimeLineRenderer::MouseEvent(float x, float y, view::MouseFlags flags){

    CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
    if (ccc == nullptr) return false;
    // Updated data from cinematic camera call
    if (!(*ccc)(CallCinematicCamera::CallForGetUpdatedKeyframeData)) return false;

    //Get keyframes
    vislib::Array<Keyframe> *keyframes = ccc->getKeyframes();
    if (keyframes == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn("[TIMELINE RENDERER] [Mouse Event] Pointer to keyframe array is nullptr.");
        return false;
    }

// LEFT-CLICK --- keyframe selection
	if (flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN) { 
        // Do not snap to keyframe when mouse movement is continuous
        float offset = 0.0f;
        if (flags & view::MOUSEFLAG_BUTTON_LEFT_CHANGED || ((x == this->lastMousePos.X()) && (y == this->lastMousePos.Y()))) {
            offset = this->keyfMarkSize / 2.0f;
        }
        float animAxisX, posX;
		//Check all keyframes if they are hit
        bool hit = false;
		for (unsigned int i = 0; i < keyframes->Count(); i++){
            animAxisX = this->animScaleOffset + (*keyframes)[i].GetAnimTime() * this->animLenTimeFrac;
            if ((animAxisX >= 0.0f) && (animAxisX <= this->animAxisLen)) {
                posX = this->axisStartPos.X() + animAxisX;
                if ((x < (posX + offset)) && (x > (posX - offset))) {
                    // If another keyframe is already hit, check which keyframe is closer to mouse position
                    if (hit) { 
                        float deltaX = vislib::math::Abs(posX - x);
                        animAxisX = this->animScaleOffset + ccc->getSelectedKeyframe().GetAnimTime() * this->animLenTimeFrac;
                        if ((animAxisX >= 0.0f) && (animAxisX <= this->animAxisLen)) {
                            posX = this->axisStartPos.X() + animAxisX;
                            if (deltaX < vislib::math::Abs(posX - x)) {
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
            if (!(*ccc)(CallCinematicCamera::CallForGetSelectedKeyframeAtTime)) return false;
        }

        // Get interpolated keyframe selection
        if (!hit && ((x >= this->axisStartPos.X()) && (x <= this->animAxisEndPos.X()))) {
            // Set an interpolated keyframe as selected
            float at = (((-1.0f)*this->animScaleOffset + (x - this->axisStartPos.X())) / this->animScaleFac) / this->animAxisLen * this->animTotalTime;
            ccc->setSelectedKeyframeTime(at);
			if (!(*ccc)(CallCinematicCamera::CallForGetSelectedKeyframeAtTime)) return false;
        }

        // Store current mouse position for detecting continuous mouse movement
        this->lastMousePos.Set(x, y);
	} 
// RIGHT-CLICK --- Drag & Drop of keyframe OR pan axes ...
    else if ((flags & view::MOUSEFLAG_BUTTON_RIGHT_DOWN) && (flags & view::MOUSEFLAG_BUTTON_RIGHT_CHANGED)) {
        //Check all keyframes if they are hit
        this->dragDropActive = false;
        float animAxisX, posX;

        bool hit = false;
        for (unsigned int i = 0; i < keyframes->Count(); i++) {
            animAxisX = this->animScaleOffset + (*keyframes)[i].GetAnimTime() * this->animLenTimeFrac;
            if ((animAxisX >= 0.0f) && (animAxisX <= this->animAxisLen)) {
                posX = this->axisStartPos.X() + animAxisX;
                if ((x < (posX + (this->keyfMarkSize / 2.0f))) && (x >(posX - (this->keyfMarkSize / 2.0f)))) {
                    // If another keyframe is already hit, check which keyframe is closer to mouse position
                    if (hit) {
                        float deltaX = vislib::math::Abs(posX - x);
                        animAxisX = this->animScaleOffset + ccc->getSelectedKeyframe().GetAnimTime() * this->animLenTimeFrac;
                        if ((animAxisX >= 0.0f) && (animAxisX <= this->animAxisLen)) {
                            posX = this->axisStartPos.X() + animAxisX;
                            if (deltaX < vislib::math::Abs(posX - x)) {
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
            this->lastMousePos.Set(x, y);
            if (!(*ccc)(CallCinematicCamera::CallForSetDragKeyframe)) return false;
        } 

        this->lastMousePos.Set(x, y);
    }
    else if ((flags & view::MOUSEFLAG_BUTTON_RIGHT_DOWN) && !(flags & view::MOUSEFLAG_BUTTON_RIGHT_CHANGED)) {
        // Update time of dragged keyframe. Only for locally stored dragged keyframe -> just for drawing
        if (this->dragDropActive) {
            if (this->dragDropAxis == 0) { // first time after activation of dragging a keyframe
                if (vislib::math::Abs(x - this->lastMousePos.X()) > vislib::math::Abs(y - this->lastMousePos.Y())) {
                    this->dragDropAxis = 1;
                }
                else {
                    this->dragDropAxis = 2;
                }
            }
            else if (this->dragDropAxis == 1) { // animation axis - X
                float at = this->dragDropKeyframe.GetAnimTime() + ((x - this->lastMousePos.X()) / this->animScaleFac) / this->animAxisLen * this->animTotalTime;
                if (x < this->axisStartPos.X()) {
                    at = 0.0f;
                }
                if (x > this->animAxisEndPos.X()) {
                    at = this->animTotalTime;
                }
                this->dragDropKeyframe.SetAnimTime(at);
            }
            else if (this->dragDropAxis == 2) { // simulation axis - Y
                float st = this->dragDropKeyframe.GetSimTime() + ((y - this->lastMousePos.Y()) / this->simScaleFac) / this->simAxisLen;
                if (y < this->axisStartPos.Y()) {
                    st = 0.0f;
                }
                if (y > this->simAxisEndPos.Y()) {
                    st = 1.0f;
                }
                this->dragDropKeyframe.SetSimTime(st);
            }
        } 
        else {
            // Pan axes ...
            float panFac = 0.5f;
            this->animScaleOffset += (x - this->lastMousePos.X()) * panFac;
            this->simScaleOffset  += (y - this->lastMousePos.Y()) * panFac;
        }
        this->lastMousePos.Set(x, y);
    }
    else if (!(flags & view::MOUSEFLAG_BUTTON_RIGHT_DOWN) && (flags & view::MOUSEFLAG_BUTTON_RIGHT_CHANGED)) {
        // Drop currently dragged keyframe
        if (this->dragDropActive) {

            float at = 0.0f;
            float st = 0.0f;
            if (this->dragDropAxis == 1) { // animation axis - X
                at = this->dragDropKeyframe.GetAnimTime() + ((x - this->lastMousePos.X()) / this->animScaleFac) / this->animAxisLen * this->animTotalTime;
                if (x <= this->axisStartPos.X()) {
                    at = 0.0f;
                }
                if (x >= this->animAxisEndPos.X()) {
                    at = this->animTotalTime;
                }
                st = this->dragDropKeyframe.GetSimTime();
            }
            else if (this->dragDropAxis == 2) { // simulation axis - Y
                st = this->dragDropKeyframe.GetSimTime() + ((y - this->lastMousePos.Y()) / this->simScaleFac) / this->simAxisLen;
                if (y < this->axisStartPos.Y()) {
                    st = 0.0f;
                }
                if (y > this->simAxisEndPos.Y()) {
                    st = 1.0f;
                }
                at = this->dragDropKeyframe.GetAnimTime();
            }
            ccc->setDropTimes(at, st);
            if (!(*ccc)(CallCinematicCamera::CallForSetDropKeyframe)) return false;
            this->dragDropAxis = 0;
            this->dragDropActive = false;
        }
    }
// MIDDLE-CLICK --- Axis scaling
    else if ((flags & view::MOUSEFLAG_BUTTON_MIDDLE_DOWN) && (flags & view::MOUSEFLAG_BUTTON_MIDDLE_CHANGED)) {
        // Just save current mouse position
        this->scaleAxis = 0;
        this->lastMousePos.Set(x, y);

        this->animScalePos = vislib::math::Clamp(x - this->axisStartPos.X(), 0.0f, this->animAxisLen);
        this->simScalePos = vislib::math::Clamp(y - this->axisStartPos.Y(), 0.0f, this->simAxisLen);

        this->simScaleDelta = (this->simScalePos - this->simScaleOffset) / this->simScaleFac;
        this->animScaleDelta = (this->animScalePos - this->animScaleOffset) / this->animScaleFac;
    }
    else if ((flags & view::MOUSEFLAG_BUTTON_MIDDLE_DOWN) && !(flags & view::MOUSEFLAG_BUTTON_MIDDLE_CHANGED)) {

        float sensitivityX = 0.01f;
        float sensitivityY = 0.03f;
        float diffX = (x - this->lastMousePos.X());
        float diffY = (y - this->lastMousePos.Y());

        if (this->scaleAxis == 0) { // first time after activation of dragging a keyframe
            if (vislib::math::Abs(diffX) > vislib::math::Abs(diffY)) {
                this->scaleAxis = 1;
            }
            else {
                this->scaleAxis = 2;
            }
        }
        else if (this->scaleAxis == 1) { // animation axis - X

            this->animScaleFac += diffX * sensitivityX;
            //vislib::sys::Log::DefaultLog.WriteWarn("[animScaleFac] %f", this->animScaleFac);

            this->animScaleFac = (this->animScaleFac < 1.0f) ? (1.0f) : (this->animScaleFac);
            this->axisAdaptation();
        }
        else if (this->scaleAxis == 2) { // simulation axis - Y

            this->simScaleFac += diffY * sensitivityY;
            //vislib::sys::Log::DefaultLog.WriteWarn("[simScaleFac] %f", this->simScaleFac);

            this->simScaleFac = (this->simScaleFac < 1.0f) ? (1.0f) : (this->simScaleFac);
            this->axisAdaptation();
        }
        this->lastMousePos.Set(x, y);
    }

	// If true is returned, manipulator cannot move camera
	return true; // Consume all mouse events;
}


/*
* cinematiccamera::TimeLineRenderer::loadTexture
*/
bool TimeLineRenderer::loadTexture(vislib::StringA filename) {
    static vislib::graphics::BitmapImage img;
    static sg::graphics::PngBitmapCodec pbc;
    pbc.Image() = &img;
    ::glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    BYTE *buf = nullptr;
    SIZE_T size = 0;

    if ((size = megamol::core::utility::ResourceWrapper::LoadResource(this->GetCoreInstance()->Configuration(), filename, (void**)(&buf))) > 0) {
        if (pbc.Load(buf, size)) {
            img.Convert(vislib::graphics::BitmapImage::TemplateByteRGBA);
            for (unsigned int i = 0; i < img.Width() * img.Height(); i++) {
                BYTE r = img.PeekDataAs<BYTE>()[i * 4 + 0];
                BYTE g = img.PeekDataAs<BYTE>()[i * 4 + 1];
                BYTE b = img.PeekDataAs<BYTE>()[i * 4 + 2];
                if (r + g + b > 0) {
                    img.PeekDataAs<BYTE>()[i * 4 + 3] = 255;
                }
                else {
                    img.PeekDataAs<BYTE>()[i * 4 + 3] = 0;
                }
            }
            markerTextures.Add(vislib::SmartPtr<vislib::graphics::gl::OpenGLTexture2D>());
            markerTextures.Last() = new vislib::graphics::gl::OpenGLTexture2D();
            if (markerTextures.Last()->Create(img.Width(), img.Height(), false, img.PeekDataAs<BYTE>(), GL_RGBA) != GL_NO_ERROR) {
                vislib::sys::Log::DefaultLog.WriteError("[TIME LINE RENDERER] [Load Texture] Could not load \"%s\" texture.", filename.PeekBuffer());
                ARY_SAFE_DELETE(buf);
                return false;
            }
            markerTextures.Last()->SetFilter(GL_LINEAR, GL_LINEAR);
            ARY_SAFE_DELETE(buf);
            return true;
        }
        else {
            vislib::sys::Log::DefaultLog.WriteError("[TIME LINE RENDERER] [Load Texture] Could not read \"%s\" texture.", filename.PeekBuffer());
        }
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("[TIME LINE RENDERER] [Load Texture] Could not find \"%s\" texture.", filename.PeekBuffer());
    }
    return false;
}
