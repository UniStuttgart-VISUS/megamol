/*
* TimeLineRenderer.cpp
*
* Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
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

#include "vislib/String.h"
#include "vislib/graphics/gl/OpenGLTexture2D.h"
#include "vislib/sys/Log.h"
#include "vislib/graphics/gl/SimpleFont.h"
#include "vislib/graphics/BitmapImage.h"
#include "vislib/sys/KeyCode.h"

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
    resolutionParam(  "01 Time Resolution", "The resolution of time on the time line."),
    markerSizeParam(  "02 Marker Size", "The size of the keyframe marker."),
    moveTimeLineParam("03 Moving time line", "Toggle if time should be moveable."),
#ifndef USE_SIMPLE_FONT
	theFont(vislib::graphics::gl::FontInfo_Verdana),
#endif // USE_SIMPLE_FONT
	markerTextures(0), dragDropKeyframe()
	{

    this->keyframeKeeperSlot.SetCompatibleCall<CallCinematicCameraDescription>();
    this->MakeSlotAvailable(&this->keyframeKeeperSlot);

    // Init variables
    this->tlStartPos     = vislib::math::Vector<float, 2>(0.0f, 0.0f);
    this->tlEndPos       = vislib::math::Vector<float, 2>(0.0f, 1.0f);
    this->tlLength       = (this->tlEndPos - this->tlStartPos).Norm();
    this->devY           = 0.1f;
    this->devX           = 0.1f;
    this->fontSize       = 24.0f;
    this->maxTime        = 1.0f;
    this->lastMouseX     = 0.0f;
    this->lastMouseY     = 0.0f;
    this->moveTimeLine   = false;
    this->aktiveDragDrop = false;

    this->timeStep      = 10.0f;
    this->markerSize    = 30.0f;

	// Set up the resolution of the time line
	this->resolutionParam.SetParameter(new param::FloatParam(this->timeStep, 0.0f));
	this->MakeSlotAvailable(&this->resolutionParam);

    // Set up the size of the keyframe marker
    this->markerSizeParam.SetParameter(new param::FloatParam(this->markerSize, 1.0f));
    this->MakeSlotAvailable(&this->markerSizeParam);

    this->moveTimeLineParam.SetParameter(new param::ButtonParam('t'));
    this->MakeSlotAvailable(&this->moveTimeLineParam);

    // Adapt font size at startup
    this->resolutionParam.ForceSetDirty();
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
* cinematiccamera::TimeLineRenderer::GetExtents
*/
bool TimeLineRenderer::GetExtents(view::CallRender2D& call) {

	core::view::CallRender2D *cr = dynamic_cast<core::view::CallRender2D*>(&call);
	if (cr == NULL) return false;

    cr->SetBoundingBox(cr->GetViewport());

    // Set time line position in percentage of viewport
    this->devX        = cr->GetViewport().GetSize().GetWidth() / 100.0f  * 8.0f; // DO CHANGES HERE
    this->devY        = cr->GetViewport().GetSize().GetHeight() / 100.0f * 5.0f; // DO CHANGES HERE

    this->tlStartPos  = vislib::math::Vector<float, 2>(this->devX, cr->GetViewport().GetSize().GetHeight() / 2.0f);
    this->tlEndPos    = vislib::math::Vector<float, 2>(cr->GetViewport().GetSize().GetWidth() - this->devX, cr->GetViewport().GetSize().GetHeight() / 2.0f);
    this->tlLength    = (this->tlEndPos - this->tlStartPos).Norm();

	// Get suitable font size
	vislib::StringA tmpStr;
	float strWidth;
    float timeFrac = this->tlLength / this->maxTime * this->timeStep;
	tmpStr.Format("%.2f", this->maxTime);
	this->fontSize = 32.0f; //reset
	strWidth = this->theFont.LineWidth(this->fontSize, tmpStr);
	while (strWidth > timeFrac*0.8f) {
		this->fontSize = this->fontSize - 0.1f;
		strWidth = this->theFont.LineWidth(this->fontSize, tmpStr);
		if (this->fontSize < 0.0f) {
			this->fontSize = 0.1f;
			break;
		}
	}

	return true;
}


/*
* cinematiccamera::TimeLineRenderer::release
*/
void TimeLineRenderer::release(void) {

	// intentionally empty
}


/*
* cinematiccamera::TimeLineRenderer::Render
*/
bool TimeLineRenderer::Render(view::CallRender2D& call) {
	
    // Update data in cinematic camera call
    CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
    if (!ccc) return false;
    // Updated data from cinematic camera call
    if (!(*ccc)(CallCinematicCamera::CallForGetUpdatedKeyframeData)) return false;

    bool updateFontSize = false;

    // Get maximum time 
    if (this->maxTime != ccc->getTotalTime()) {
        this->maxTime = ccc->getTotalTime();
        updateFontSize = true;
    }

    // Update parameters
    if (this->markerSizeParam.IsDirty()) {
        this->markerSizeParam.ResetDirty();
        this->markerSize = this->markerSizeParam.Param<param::FloatParam>()->Value();
    }
    if (this->resolutionParam.IsDirty()) {
        this->resolutionParam.ResetDirty();
        this->timeStep = this->resolutionParam.Param<param::FloatParam>()->Value();
        if (this->timeStep > this->maxTime) {
            this->timeStep = this->maxTime;
            this->resolutionParam.Param<param::FloatParam>()->SetValue(this->maxTime);
        }
        updateFontSize = true;
    }
    if (this->moveTimeLineParam.IsDirty()) {
        this->moveTimeLine = !this->moveTimeLine;
        this->moveTimeLineParam.ResetDirty();
    }

    vislib::StringA tmpStr;
    float strWidth;
    float timeFrac = this->tlLength / this->maxTime * this->timeStep;

    // Adapt font size
    if (updateFontSize) {
        tmpStr.Format("%.2f", this->maxTime);
        this->fontSize = 32.0f; //reset
        strWidth = this->theFont.LineWidth(this->fontSize, tmpStr);
        while (strWidth > timeFrac*0.8f) {
            this->fontSize = this->fontSize - 0.1f;
            strWidth = this->theFont.LineWidth(this->fontSize, tmpStr);
            if (this->fontSize < 0.0f) {
                this->fontSize = 0.1f;
                break;
            }
        }
    }

	// Get the diagram color (inverse background color)
	float bgColor[4];
	float fgColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
	glGetFloatv(GL_COLOR_CLEAR_VALUE, bgColor);
	for (unsigned int i = 0; i < 4; i++) {
		fgColor[i] -= bgColor[i];
	}

    // Opengl setup
	glColor3fv(fgColor);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
    glLineWidth(2.0f);

    // Draw time line
	glBegin(GL_LINE_STRIP);
	    glVertex2fv(this->tlStartPos.PeekComponents());
        glVertex2fv(this->tlEndPos.PeekComponents());
	glEnd();

    // Draw ruler lines
    float offset;
    glBegin(GL_LINES);
    glVertex2f(this->tlStartPos.GetX(),                  this->tlStartPos.GetY() + this->devY);
    glVertex2f(this->tlStartPos.GetX(),                  this->tlStartPos.GetY() - this->devY);
    glVertex2f(this->tlStartPos.GetX() + this->tlLength, this->tlStartPos.GetY() + this->devY);
    glVertex2f(this->tlStartPos.GetX() + this->tlLength, this->tlStartPos.GetY() - this->devY);
    for (float f = timeFrac; f < this->tlLength; f = f + timeFrac) {
        glVertex2f(this->tlStartPos.GetX() + f, this->tlStartPos.GetY());
        glVertex2f(this->tlStartPos.GetX() + f, this->tlStartPos.GetY() - this->devY);
    }
    glEnd();

    // Draw time captions
    if (this->theFont.Initialise()) {
        tmpStr.Format("%.2f ", 0.0f);
        strWidth = this->theFont.LineWidth(this->fontSize, tmpStr);
        this->theFont.DrawString(this->tlStartPos.GetX() - strWidth, this->tlStartPos.GetY(), strWidth, 1.0f, this->fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_BOTTOM);

        tmpStr.Format(" %.2f", this->maxTime);
        strWidth = this->theFont.LineWidth(this->fontSize, tmpStr);
        this->theFont.DrawString(this->tlEndPos.GetX() , this->tlStartPos.GetY(), strWidth, 1.0f, this->fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_BOTTOM);

        float timeStep = this->timeStep;
        for (float f = timeFrac; f < this->tlLength; f = f + timeFrac) {
            tmpStr.Format("%.2f", timeStep);
            timeStep += this->timeStep;
            strWidth = this->theFont.LineWidth(this->fontSize, tmpStr);
            this->theFont.DrawString(this->tlStartPos.GetX() + f - strWidth/2.0f, this->tlStartPos.GetY() - 1.0f - this->devY, strWidth, 1.0f, this->fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
        }
    }

	// Draw keyframes
    vislib::Array<Keyframe> *keyframes = ccc->getKeyframes();
    if (keyframes == NULL) {
        vislib::sys::Log::DefaultLog.WriteWarn("[TIMELINE RENDERER] [Render] Pointer to keyframe array is NULL.");
        return false;
    }
    Keyframe s         = ccc->getSelectedKeyframe();
    float    frameFrac = this->tlLength / maxTime;
    if (keyframes->Count() > 0) {
        // draw the defined keyframes
        for (unsigned int i = 0; i < keyframes->Count(); i++) {
            ColorMode cMode = ((*keyframes)[i] == s) ? (ColorMode::SELECTED_COLOR) : (ColorMode::DEFAULT_COLOR);
            this->DrawKeyframeMarker(this->tlStartPos.GetX() + (*keyframes)[i].getTime() * frameFrac, this->tlStartPos.GetY(), cMode);
        }
    }

    // Draw interpolated selected keyframe
    if (!keyframes->Contains(s)) {
        float x = this->tlStartPos.GetX() + s.getTime() * frameFrac;
        glLineWidth(3.0f);
        glColor3f(0.1f, 0.1f, 1.0f);
        glBegin(GL_LINES);
            glVertex2f(x, this->tlStartPos.GetY() + this->markerSize);
            glVertex2f(x, this->tlStartPos.GetY());
        glEnd();
    }

    // Draw dragged keyframe
    if (this->aktiveDragDrop) {
        this->DrawKeyframeMarker(this->tlStartPos.GetX() + this->dragDropKeyframe.getTime() * frameFrac, this->tlStartPos.GetY(), ColorMode::DRAGDROP_COLOR);
    }

	return true;
}


/*
* cinematiccamera::TimeLineRenderer::DrawKeyframeMarker
*/
void TimeLineRenderer::DrawKeyframeMarker(float posX, float posY, ColorMode color) {

    switch (color) {
        case(ColorMode::DEFAULT_COLOR):  glColor3f(0.7f, 0.7f, 1.0f); break;
        case(ColorMode::SELECTED_COLOR): glColor3f(0.1f, 0.1f, 1.0f); break;
        case(ColorMode::DRAGDROP_COLOR): glColor3f(0.4f, 0.4f, 1.0f); break;
        default: glColor3f(0.5f, 0.5f, 0.5f); break;
    }

    // as texture
    glEnable(GL_TEXTURE_2D);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    this->markerTextures[0]->Bind();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
    glBegin(GL_QUADS);
        glTexCoord2f(1.0f, 1.0f);
        glVertex2f(posX - this->markerSize/2.0f, posY);

        glTexCoord2f(1.0f, 0.0f);
        glVertex2f(posX - this->markerSize/2.0f, posY + this->markerSize);

        glTexCoord2f(0.0f, 0.0f);
        glVertex2f(posX + this->markerSize/2.0f, posY + this->markerSize);

        glTexCoord2f(0.0f, 1.0f);
        glVertex2f(posX + this->markerSize/2.0f, posY);
    glEnd();

    glDisable(GL_BLEND);
    glDisable(GL_TEXTURE_2D);

    // as geometry
    /*
    glBegin(GL_TRIANGLES);
    glVertex2f(posX, posY);
    glVertex2f(posX - this->markerSize/2.0f, posY + this->markerSize);
    glVertex2f(posX + this->markerSize/2.0f, posY + this->markerSize);
    glEnd();
    */
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
            offset = this->markerSize / 2.0f;
        }

		//Check all keyframes if they are hit
        bool hit = false;
		for (unsigned int i = 0; i < keyframes->Count(); i++){
			float posX = this->tlStartPos.GetX() + (*keyframes)[i].getTime() / maxTime * this->tlLength;
			if ((x < (posX + offset)) && (x > (posX - offset))) {
                // Set keyframe as selected
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
            if ((x < (posX + this->markerSize / 2.0f)) && (x > (posX - this->markerSize / 2.0f))) {
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

        // Update time of dragged keyframe. Only for locally stred dragged keyframe -> just for drawing
        if (this->aktiveDragDrop) {
            float st = ((x - this->tlStartPos.GetX()) / this->tlLength * maxTime);
            if (x < this->tlStartPos.GetX()) {
                st = 0.0f;
            }
            if (x > this->tlEndPos.GetX()) {
                st = this->maxTime;
            }
            this->dragDropKeyframe.setTime(st);
        }
    }
    else if (!(flags & view::MOUSEFLAG_BUTTON_RIGHT_DOWN) && (flags & view::MOUSEFLAG_BUTTON_RIGHT_CHANGED)) {
        consume = true; // Consume all right click events

        if (this->aktiveDragDrop) {
            float st = ((x - this->tlStartPos.GetX()) / this->tlLength * maxTime);
            if (x <= this->tlStartPos.GetX()) {
                st = 0.0f;
            }
            if (x >= this->tlEndPos.GetX()) {
                st = this->maxTime;
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