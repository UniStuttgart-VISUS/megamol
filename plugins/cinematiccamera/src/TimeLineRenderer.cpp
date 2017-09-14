/*
* TimeLineRenderer.cpp
*
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

//#define _USE_MATH_DEFINES

using namespace megamol::core;
using namespace megamol::cinematiccamera;


/*
* cinematiccamera::TimeLineRenderer::TimeLineRenderer
*/
TimeLineRenderer::TimeLineRenderer(void) : view::Renderer2DModule(),
	keyframeKeeperSlot("getkeyframes", "Connects to the KeyframeKeeper"),
    moveTimeLineParam(   "01 Moving time line", "Toggle if time should be moveable."),
    markerSizeParam(     "02 Marker size", "The size of the keyframe marker."),
    rulerModeParam(      "Ruler::01 Mode", "Switch between fixed ruler segmentation and adaptive ruler segemntation with fixed font size."),
    rulerFixedSegParam(  "Ruler::02 Fixed interval segmentation", "The fixed ruler segemntation interval."),
    rulerFixedFontParam( "Ruler::03 Fixed font size", "The fixed font size for adaptive ruler segmentation."),

#ifndef USE_SIMPLE_FONT
	theFont(vislib::graphics::gl::FontInfo_Verdana),
#endif // USE_SIMPLE_FONT
	dragDropKeyframe()
	{

    this->keyframeKeeperSlot.SetCompatibleCall<CallCinematicCameraDescription>();
    this->MakeSlotAvailable(&this->keyframeKeeperSlot);

    // init variables
    this->tlStartPos       = vislib::math::Vector<float, 2>(0.0f, 0.0f);
    this->tlEndPos         = vislib::math::Vector<float, 2>(0.0f, 1.0f);
    this->tlLength         = (this->tlEndPos - this->tlStartPos).Norm();
    this->devY             = 0.1f;
    this->devX             = 0.1f;
    this->totalTime        = 1.0f;
    this->lastMouseX       = 0.0f;
    this->lastMouseY       = 0.0f;
    this->aktiveDragDrop   = false;
    this->adaptFontSize    = 32.0f;
    this->adaptSegSize     = 10.0f;
    this->initScaleFac     = 1.0f;
    this->scaleFac         = 1.0f;
    this->moveTimeLine     = false;
    this->markerSize       = 30.0f;
    this->currentRulerMode = RULER_FIXED_FONT;
    this->fontSize         = 20.0f;
    this->segmentSize      = 10.0f;
    this->markerTextures.Clear();

    // init parameters
    this->moveTimeLineParam.SetParameter(new param::ButtonParam('t'));
    this->MakeSlotAvailable(&this->moveTimeLineParam);

    this->markerSizeParam.SetParameter(new param::FloatParam(this->markerSize, 1.0f));

    param::EnumParam *sbs = new param::EnumParam(this->currentRulerMode);
    sbs->SetTypePair(RULER_FIXED_FONT, "Fixed Font");
    sbs->SetTypePair(RULER_FIXED_SEGMENT, "Fixed Segmentation");
    this->rulerModeParam << sbs;

    this->rulerFixedSegParam.SetParameter(new param::FloatParam(this->segmentSize, 0.0f));

    this->rulerFixedFontParam.SetParameter(new param::FloatParam(this->fontSize, 0.0f));

    // Comment the following lines to hide parameters

    //this->MakeSlotAvailable(&this->markerSizeParam);
    this->MakeSlotAvailable(&this->rulerModeParam);
    this->MakeSlotAvailable(&this->rulerFixedSegParam);
    this->MakeSlotAvailable(&this->rulerFixedFontParam);
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
	
    this->LoadTexture("arrow.png");  // this->markerTextures[0]
	
	return true;
}


/*
* cinematiccamera::TimeLineRenderer::release
*/
void TimeLineRenderer::release(void) {

    // intentionally empty
}


/*
* cinematiccamera::TimeLineRenderer::GetExtents
*/
bool TimeLineRenderer::GetExtents(view::CallRender2D& call) {

	core::view::CallRender2D *cr = dynamic_cast<core::view::CallRender2D*>(&call);
	if (cr == NULL) return false;

    cr->SetBoundingBox(cr->GetViewport());

    // Set time line position in percentage of viewport
    this->devX         = cr->GetViewport().GetSize().GetWidth()  / 100.0f * 5.0f; // DO CHANGES HERE
    this->devY         = cr->GetViewport().GetSize().GetHeight() / 100.0f * 5.0f; // DO CHANGES HERE
    this->markerSize   = (this->devX > this->devY) ? (this->devX/2.0f) : (this->devY); // DO CHANGES HERE

    this->tlStartPos   = vislib::math::Vector<float, 2>(this->devX, cr->GetViewport().GetSize().GetHeight() / 2.0f);
    this->tlEndPos     = vislib::math::Vector<float, 2>(cr->GetViewport().GetSize().GetWidth() - this->devX, cr->GetViewport().GetSize().GetHeight() / 2.0f);

    float tmpLength    = this->tlLength;
    this->tlLength     = (this->tlEndPos - this->tlStartPos).Norm();

    if (tmpLength != this->tlLength) {

        this->initScaleFac = 1.0f;

        if (this->currentRulerMode == RULER_FIXED_FONT) {
            // Adapt segemnt size if necessary
            vislib::StringA tmpStr;
            tmpStr.Format("%.2f", this->totalTime);
            float strWidth     = this->theFont.LineWidth(this->fontSize, tmpStr);
            this->adaptSegSize = this->totalTime; // reset to max value
            float deltaSegSize = this->totalTime / 1000.0f; // one per mille steps
            float timeFrac     = this->tlLength / this->totalTime * this->adaptSegSize;
            while (timeFrac*0.7f > strWidth) {
                this->adaptSegSize = this->adaptSegSize - deltaSegSize;
                if (this->adaptSegSize < 0.0f) {
                    this->adaptSegSize = deltaSegSize;
                    break;
                }
                timeFrac = this->tlLength / this->totalTime * this->adaptSegSize;
            }
        }
        else { // this->currentRulerMode == RULER_FIXED_SEGMENT
            // Get suitable font size if length changed
            vislib::StringA tmpStr;
            tmpStr.Format("%.2f", this->totalTime);
            float timeFrac       = this->tlLength / this->totalTime * this->segmentSize;
            this->adaptFontSize  = 32.0f; //reset to max value
            float deltaFontSize  = 0.1f;
            float strWidth       = this->theFont.LineWidth(this->adaptFontSize, tmpStr);
            while (strWidth > timeFrac*0.7f) {
                this->adaptFontSize = this->adaptFontSize - deltaFontSize;
                if (this->adaptFontSize < 0.0f) {
                    this->adaptFontSize = deltaFontSize;
                    break;
                }
                strWidth = this->theFont.LineWidth(this->adaptFontSize, tmpStr);
            }
        }
    }

	return true;
}


/*
* cinematiccamera::TimeLineRenderer::Render
*/
bool TimeLineRenderer::Render(view::CallRender2D& call) {
	
    core::view::CallRender2D *cr = dynamic_cast<core::view::CallRender2D*>(&call);
    if (cr == NULL) return false;

    // Update data in cinematic camera call
    CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
    if (!ccc) return false;
    // Updated data from cinematic camera call
    if (!(*ccc)(CallCinematicCamera::CallForGetUpdatedKeyframeData)) return false;

    vislib::Array<Keyframe> *keyframes = ccc->getKeyframes();
    if (keyframes == NULL) {
        vislib::sys::Log::DefaultLog.WriteWarn("[TIMELINE RENDERER] [Render] Pointer to keyframe array is NULL.");
        return false;
    }

    Keyframe skf = ccc->getSelectedKeyframe();

    bool adaptFontSize = false;
    bool adaptSegSize  = false;

    // Get maximum time 
    if (this->totalTime != ccc->getTotalTime()) {
        this->totalTime = ccc->getTotalTime();
        adaptFontSize = true;
        adaptSegSize = true;
    }

    // Get scaling factor
    GLfloat modelViewMatrix_column[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    if (this->initScaleFac == 1.0f) {
        this->initScaleFac = modelViewMatrix_column[0];
    }
    if (this->scaleFac != (modelViewMatrix_column[0]/this->initScaleFac)) {
        this->scaleFac = modelViewMatrix_column[0] / this->initScaleFac;
        adaptFontSize = true;
        adaptSegSize = true;
    }

    // Update parameters
    if (this->moveTimeLineParam.IsDirty()) {
        this->moveTimeLine = !this->moveTimeLine;
        this->moveTimeLineParam.ResetDirty();
    }
    if (this->markerSizeParam.IsDirty()) {
        this->markerSizeParam.ResetDirty();
        this->markerSize = this->markerSizeParam.Param<param::FloatParam>()->Value();
    }
    if (this->rulerModeParam.IsDirty()) {
        this->rulerModeParam.ResetDirty();
        this->currentRulerMode = static_cast<rulerMode>(this->rulerModeParam.Param<param::EnumParam>()->Value());
        if (this->currentRulerMode == RULER_FIXED_FONT) {
            adaptSegSize = true;
        }
        else { // RULER_FIXED_SEGMENT
            adaptFontSize = true;
        }
    }
    if (this->rulerFixedFontParam.IsDirty()) {
        this->rulerFixedFontParam.ResetDirty();
        this->fontSize = this->rulerFixedFontParam.Param<param::FloatParam>()->Value();
        adaptSegSize = true;
    }
    if (this->rulerFixedSegParam.IsDirty()) {
        this->rulerFixedSegParam.ResetDirty();
        this->segmentSize = this->rulerFixedSegParam.Param<param::FloatParam>()->Value();
        if (this->segmentSize > this->totalTime) {
            this->segmentSize = this->totalTime;
            this->rulerFixedSegParam.Param<param::FloatParam>()->SetValue(this->segmentSize);
        }
        adaptFontSize = true;
    }

    // Initialise font
    if (!this->theFont.Initialise()) {
        vislib::sys::Log::DefaultLog.WriteWarn("[TIMELINE RENDERER] [Render] Couldn't initialize the font.");
        return false;
    }

    // Font or ruler segmentation adaptation
    vislib::StringA tmpStr;
    float strWidth;
    float currentFontSize; 
    float currentSegSize;
    float currentRulerFrac;
    if (this->currentRulerMode == RULER_FIXED_FONT) {
        // Adapt segemnt size if necessary
        if (adaptSegSize) {
            vislib::StringA tmpStr;
            tmpStr.Format("%.2f", this->totalTime);
            float strWidth     = this->theFont.LineWidth(this->fontSize / this->scaleFac, tmpStr);
            this->adaptSegSize = this->totalTime; // reset to max value
            float deltaSegSize = this->totalTime / 1000.0f; // one per mille steps
            float rulerFrac    = this->tlLength / this->totalTime * this->adaptSegSize / this->scaleFac;
            while (rulerFrac*0.7f > strWidth) {
                this->adaptSegSize = (this->adaptSegSize - deltaSegSize);
                if (this->adaptSegSize < 0.0f) {
                    this->adaptSegSize = deltaSegSize;
                    break;
                }
                rulerFrac = this->tlLength / this->totalTime * this->adaptSegSize / this->scaleFac;
            }
            // Rounding
            this->adaptSegSize /= this->scaleFac;
            if (this->adaptSegSize > 100.f) {
                float tmpFac = 1.0f;
                while (this->adaptSegSize > 1000.f) {
                    this->adaptSegSize /= 10.0f;
                    tmpFac *= 10.f;
                }
                this->adaptSegSize = this->adaptSegSize - (this->adaptSegSize - floorf(this->adaptSegSize));
                this->adaptSegSize *= tmpFac;
            }
            else {
                float tmpFac = 1.0f;
                while (this->adaptSegSize < 10.f) {
                    this->adaptSegSize *= 10.0f;
                    tmpFac *= 10.f;
                }
                this->adaptSegSize = this->adaptSegSize - (this->adaptSegSize - floorf(this->adaptSegSize));
                this->adaptSegSize /= tmpFac;
            }
            this->adaptSegSize *= this->scaleFac;
        }
        currentFontSize  = this->fontSize / this->scaleFac;
        currentSegSize   = this->adaptSegSize / this->scaleFac;
        currentRulerFrac = this->tlLength / this->totalTime * currentSegSize;
    }
    else { // RULER_FIXED_SEGMENT
        // Adapt font size if necessary
        if (adaptFontSize) {
            vislib::StringA tmpStr;
            tmpStr.Format("%.2f", this->totalTime);
            this->adaptFontSize = 32.0f; //reset to max value
            float deltaFontSize = 0.1f;
            float strWidth      = this->theFont.LineWidth(this->adaptFontSize, tmpStr);
            float rulerFrac     = this->tlLength / this->totalTime * this->segmentSize;
            while (strWidth > rulerFrac*0.7f) {
                this->adaptFontSize = this->adaptFontSize - deltaFontSize;
                if (this->adaptFontSize < 0.0f) {
                    this->adaptFontSize = deltaFontSize;
                    break;
                }
                strWidth = this->theFont.LineWidth(this->adaptFontSize, tmpStr);
            }
        }
        currentFontSize = this->adaptFontSize;
        currentSegSize = this->segmentSize;
        currentRulerFrac = this->tlLength / this->totalTime * this->segmentSize;
    }

    // Get the foreground color (inverse background color)
    float bgColor[4];
    float fgColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glGetFloatv(GL_COLOR_CLEAR_VALUE, bgColor);
    for (unsigned int i = 0; i < 4; i++) {
        fgColor[i] -= bgColor[i];
    }
    // COLORS
    float armColor[4]  = { 0.8f, 0.0f, 0.0f, 1.0f }; // Color for ANIMATION REPEAT MARKER
    float kColor[4]    = { 0.7f, 0.7f, 1.0f, 1.0f }; // Color for KEYFRAME 
    float dkmColor[4]  = { 0.5f, 0.5f, 1.0f, 1.0f }; // Color for SELECTED KEYFRAME 
    float skColor[4]   = { 0.1f, 0.1f, 1.0f, 1.0f }; // Color for SELECTED KEYFRAME 
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

    // Opengl setup
    GLfloat tmpLw;
    glGetFloatv(GL_LINE_WIDTH, &tmpLw);

    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

    float        frameFrac = this->tlLength / this->totalTime;
    float        maxAnimTime = ccc->getMaxAnimTime();
    unsigned int repeatAnimTime = static_cast<unsigned int>(floorf(this->totalTime / maxAnimTime));

    glDisable(GL_POLYGON_SMOOTH);
    // Draw markers for existing keyframes in array
    if (keyframes->Count() > 0) {
        for (unsigned int i = 0; i < keyframes->Count(); i++) {
            if ((*keyframes)[i] == skf) {
                glColor4fv(skColor);
            }
            else {
                glColor4fv(kColor);
            }
            this->DrawKeyframeMarker(this->tlStartPos.GetX() + (*keyframes)[i].getTime() * frameFrac, this->tlStartPos.GetY());
        }
    }
    // Draw dragged keyframe
    if (this->aktiveDragDrop) {
        glColor4fv(dkmColor); // Color for DRAGGED KEYFRAME MARKER
        this->DrawKeyframeMarker(this->tlStartPos.GetX() + this->dragDropKeyframe.getTime() * frameFrac, this->tlStartPos.GetY());
    }

    glEnable(GL_LINE_SMOOTH);
    // Draw marker when animation repeats
    glLineWidth(5.0f);
    glColor4fv(armColor);
    glBegin(GL_LINES);
    for (unsigned int i = 1; i <= repeatAnimTime; i++) {
        glVertex2f(this->tlStartPos.GetX() + frameFrac*maxAnimTime*(float)i, this->tlStartPos.GetY() + (this->devY / this->scaleFac));
        glVertex2f(this->tlStartPos.GetX() + frameFrac*maxAnimTime*(float)i, this->tlStartPos.GetY() - (this->devY / this->scaleFac));
    }
    glEnd();
    // Draw time line
    glLineWidth(2.5f);
    glColor4fv(fgColor);
    glBegin(GL_LINES);
        glVertex2fv(this->tlStartPos.PeekComponents());
        glVertex2fv(this->tlEndPos.PeekComponents());
        // Draw ruler lines
        glVertex2f(this->tlStartPos.GetX(),                  this->tlStartPos.GetY() + (this->devY / this->scaleFac));
        glVertex2f(this->tlStartPos.GetX(),                  this->tlStartPos.GetY() - (this->devY / this->scaleFac));
        glVertex2f(this->tlStartPos.GetX() + this->tlLength, this->tlStartPos.GetY() + (this->devY / this->scaleFac));
        glVertex2f(this->tlStartPos.GetX() + this->tlLength, this->tlStartPos.GetY() - (this->devY / this->scaleFac));
        for (float f = currentRulerFrac; f < this->tlLength; f = f + currentRulerFrac) {
            glVertex2f(this->tlStartPos.GetX() + f, this->tlStartPos.GetY());
            glVertex2f(this->tlStartPos.GetX() + f, this->tlStartPos.GetY() - (this->devY / this->scaleFac));
        }
    glEnd();
    // Draw interpolated selected keyframe marker
    if (!keyframes->Contains(skf)) {
        float x = this->tlStartPos.GetX() + skf.getTime() * frameFrac;
        glLineWidth(5.0f);
        glColor4fv(skColor);
        glBegin(GL_LINES);
            glVertex2f(x, this->tlStartPos.GetY() + (this->markerSize / this->scaleFac));
            glVertex2f(x, this->tlStartPos.GetY());
        glEnd();
    }

    glEnable(GL_POLYGON_SMOOTH);
    // Draw time captions
    glColor4fv(fgColor);
    tmpStr.Format("%.2f ", 0.0f);
    strWidth = this->theFont.LineWidth(currentFontSize, tmpStr);
    this->theFont.DrawString(this->tlStartPos.GetX() - strWidth, this->tlStartPos.GetY(), strWidth, 1.0f, currentFontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_BOTTOM);

    tmpStr.Format(" %.2f", this->totalTime);
    strWidth = this->theFont.LineWidth(currentFontSize, tmpStr);
    this->theFont.DrawString(this->tlEndPos.GetX() , this->tlStartPos.GetY(), strWidth, 1.0f, currentFontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_BOTTOM);

    float timeStep = currentSegSize;
    for (float f = currentRulerFrac; f < (this->tlLength - (currentRulerFrac /100.0f)); f = f + currentRulerFrac) {
        tmpStr.Format("%.2f ", timeStep);
        timeStep += currentSegSize;
        strWidth = this->theFont.LineWidth(currentFontSize, tmpStr);
        this->theFont.DrawString(this->tlStartPos.GetX() + f - strWidth/2.0f, this->tlStartPos.GetY() - 1.0f - (this->devY / this->scaleFac), strWidth, 1.0f, currentFontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
    }

    // Reset opengl 
    glLineWidth(tmpLw);
    glDisable(GL_BLEND);
    glDisable(GL_LINE_SMOOTH);

	return true;
}


/*
* cinematiccamera::TimeLineRenderer::DrawKeyframeMarker
*/
void TimeLineRenderer::DrawKeyframeMarker(float posX, float posY) {

    glEnable(GL_TEXTURE_2D);

    this->markerTextures[0]->Bind();
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBegin(GL_QUADS);
        glTexCoord2f(1.0f, 1.0f);
        glVertex2f(posX - (this->markerSize/ (2.0f*this->scaleFac)), posY);

        glTexCoord2f(1.0f, 0.0f);
        glVertex2f(posX - (this->markerSize/ (2.0f*this->scaleFac)), posY + (this->markerSize/this->scaleFac));

        glTexCoord2f(0.0f, 0.0f);
        glVertex2f(posX + (this->markerSize/ (2.0f*this->scaleFac)), posY + (this->markerSize/this->scaleFac));

        glTexCoord2f(0.0f, 1.0f);
        glVertex2f(posX + (this->markerSize/ (2.0f*this->scaleFac)), posY);
    glEnd();

    glDisable(GL_TEXTURE_2D);
}


/*
* cinematiccamera::TimeLineRenderer::MouseEvent
*/
bool TimeLineRenderer::MouseEvent(float x, float y, view::MouseFlags flags){

    bool consume = false;

    CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
    if (ccc == NULL) return false;
    // Updated data from cinematic camera call
    if (!(*ccc)(CallCinematicCamera::CallForGetUpdatedKeyframeData)) return false;

    //Get keyframes
    vislib::Array<Keyframe> *keyframes = ccc->getKeyframes();
    if (keyframes == NULL) {
        vislib::sys::Log::DefaultLog.WriteWarn("[TIMELINE RENDERER] [Mouse Event] Pointer to keyframe array is NULL.");
        return false;
    }
    float maxTime = ccc->getTotalTime();

	// on leftclick, check if a keyframe is hit and set the selected keyframe
	if (!this->moveTimeLine && (flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN)) { 
        consume = true; // Consume all left click events

        // Do not snap to keyframe when mouse movement is continuous
        float offset = 0.0f;
        if (flags & view::MOUSEFLAG_BUTTON_LEFT_CHANGED || ((x == this->lastMouseX) && (y == this->lastMouseY))) {
            offset = this->markerSize / (2.0f*this->scaleFac);
        }

		//Check all keyframes if they are hit
        bool hit = false;
		for (unsigned int i = 0; i < keyframes->Count(); i++){
			float posX = this->tlStartPos.GetX() + (*keyframes)[i].getTime() / maxTime * this->tlLength;
			if ((x < (posX + offset)) && (x > (posX - offset))) {
                // Set hit keyframe as selected
				ccc->setSelectedKeyframeTime((*keyframes)[i].getTime());
				if (!(*ccc)(CallCinematicCamera::CallForSetSelectedKeyframe)) return false;
                hit = true;
                break; // Exit loop on hit
			}
		}

        // Get interpolated keyframe selection
        if (!hit && ((x > this->tlStartPos.GetX()) && (x < this->tlEndPos.GetX()))) {
            // Set an interpolated keyframe as selected
            float st = ((x - this->tlStartPos.GetX()) / this->tlLength * maxTime);
            ccc->setSelectedKeyframeTime(st);
			if (!(*ccc)(CallCinematicCamera::CallForSetSelectedKeyframe)) return false;
        }

        // Store current mouse position for detecting continuous mouse movement
        this->lastMouseX = x;
        this->lastMouseY = y;
	} 
   

    // Drag & Drop of keyframe with right-click
    if ((flags & view::MOUSEFLAG_BUTTON_RIGHT_DOWN) && (flags & view::MOUSEFLAG_BUTTON_RIGHT_CHANGED)) {
        consume = true; // Consume all right click events

        //Check all keyframes if they are hit
        this->aktiveDragDrop = false;
        for (unsigned int i = 0; i < keyframes->Count(); i++) {
            float posX = this->tlStartPos.GetX() + (*keyframes)[i].getTime() / maxTime * this->tlLength;
            if ((x < (posX + (this->markerSize / (2.0f*this->scaleFac)))) && (x > (posX - (this->markerSize / (2.0f*this->scaleFac))))) {
                // Store hit keyframe locally
                this->dragDropKeyframe = (*keyframes)[i];
                this->aktiveDragDrop   = true;
                ccc->setSelectedKeyframeTime((*keyframes)[i].getTime());
                if (!(*ccc)(CallCinematicCamera::CallForDragKeyframe)) return false;
                break; // Exit loop on hit
            }
        }
    }
    else if ((flags & view::MOUSEFLAG_BUTTON_RIGHT_DOWN) && !(flags & view::MOUSEFLAG_BUTTON_RIGHT_CHANGED)) {
        consume = true; // Consume all right click events

        // Update time of dragged keyframe. Only for locally stored dragged keyframe -> just for drawing
        if (this->aktiveDragDrop) {
            float st = ((x - this->tlStartPos.GetX()) / this->tlLength * maxTime);
            if (x < this->tlStartPos.GetX()) {
                st = 0.0f;
            }
            if (x > this->tlEndPos.GetX()) {
                st = this->totalTime;
            }
            this->dragDropKeyframe.setTime(st);
        }
    }
    else if (!(flags & view::MOUSEFLAG_BUTTON_RIGHT_DOWN) && (flags & view::MOUSEFLAG_BUTTON_RIGHT_CHANGED)) {
        consume = true; // Consume all right click events

        // Drop currently dragged keyframe
        if (this->aktiveDragDrop) {
            float st = ((x - this->tlStartPos.GetX()) / this->tlLength * maxTime);
            if (x <= this->tlStartPos.GetX()) {
                st = 0.0f;
            }
            if (x >= this->tlEndPos.GetX()) {
                st = this->totalTime;
            }

            ccc->setDropTime(st);
            if (!(*ccc)(CallCinematicCamera::CallForDropKeyframe)) return false;
            this->aktiveDragDrop = false;
        }
    }

	// If true is returned, manipulator cannot move camera
	return consume;
}


/*
* cinematiccamera::TimeLineRenderer::LoadTexture
*/
bool TimeLineRenderer::LoadTexture(vislib::StringA filename) {
    static vislib::graphics::BitmapImage img;
    static sg::graphics::PngBitmapCodec pbc;
    pbc.Image() = &img;
    ::glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    void *buf = NULL;
    SIZE_T size = 0;

    if ((size = megamol::core::utility::ResourceWrapper::LoadResource(
        this->GetCoreInstance()->Configuration(), filename, &buf)) > 0) {
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