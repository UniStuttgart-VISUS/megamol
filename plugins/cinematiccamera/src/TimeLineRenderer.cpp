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

#include "TimeLineRenderer.h"
#include "CallCinematicCamera.h"

#include <iostream>

//#define _USE_MATH_DEFINES

using namespace megamol::core;
using namespace megamol::cinematiccamera;


/*
* cinematiccamera::TimeLineRenderer::TimeLineRenderer
*/
TimeLineRenderer::TimeLineRenderer(void) : view::Renderer2DModule(),
	keyframeKeeperSlot("getkeyframes", "Connects to the KeyframeKeeper"),
    resolutionParam( "01 Time Resolution", "the resolution of time on the time line."),
    markerSizeParam( "02 Keyframe Marker Size", "the size of the keyframe marker."),
#ifndef USE_SIMPLE_FONT
	theFont(vislib::graphics::gl::FontInfo_Verdana),
#endif // USE_SIMPLE_FONT
	markerTextures(0)
	{

    this->keyframeKeeperSlot.SetCompatibleCall<CallCinematicCameraDescription>();
    this->MakeSlotAvailable(&this->keyframeKeeperSlot);

    // Init variables
    this->tlStartPos  = vislib::math::Vector<float, 2>(0.0f, 0.0f);
    this->tlEndPos    = vislib::math::Vector<float, 2>(0.0f, 1.0f);
    this->tlLength    = (this->tlEndPos - this->tlStartPos).Norm();
    this->devY        = 0.1f;
    this->devX        = 0.1f;
    this->tlRes       = 25;
    this->fontSize    = 24.0f;
    this->markerSize  = 30.0f;
    this->maxTime     = 1.0f;

	// Set up the resolution of the time line
	this->resolutionParam.SetParameter(new param::IntParam(this->tlRes, 1));
	this->MakeSlotAvailable(&this->resolutionParam);

    // Set up the size of the keyframe marker
    this->markerSizeParam.SetParameter(new param::FloatParam(this->markerSize, 1.0f));
    this->MakeSlotAvailable(&this->markerSizeParam);

    // Adjust font size right at the beginning
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

    call.SetBoundingBox(cr->GetViewport());

    // Set time line position in percentage of viewport
    this->devX        = cr->GetViewport().GetSize().GetWidth() / 100.0f  * 8.0f; // DO CHANGES HERE
    this->devY        = cr->GetViewport().GetSize().GetHeight() / 100.0f * 5.0f; // DO CHANGES HERE

    this->tlStartPos  = vislib::math::Vector<float, 2>(this->devX, cr->GetViewport().GetSize().GetHeight() / 2.0f);
    this->tlEndPos    = vislib::math::Vector<float, 2>(cr->GetViewport().GetSize().GetWidth() - this->devX, cr->GetViewport().GetSize().GetHeight() / 2.0f);
    this->tlLength    = (this->tlEndPos - this->tlStartPos).Norm();

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
    if (!(*ccc)(CallCinematicCamera::CallForUpdateKeyframeKeeper)) return false;

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
        this->tlRes = static_cast<int>(this->resolutionParam.Param<param::IntParam>()->Value());
        updateFontSize = true;
    }

    vislib::StringA tmpStr;
    float strWidth;
    float timeFrac = maxTime / (float)this->tlRes;
    float posFrac  = this->tlLength / (float)this->tlRes;

    // Get suitable font size
    if (updateFontSize) {
        tmpStr.Format("%.2f", maxTime);
        this->fontSize = 32.0f; //reset
        strWidth = theFont.LineWidth(this->fontSize, tmpStr);
        while (strWidth > posFrac*0.8f) {
            this->fontSize = this->fontSize - 0.1f;
            strWidth = theFont.LineWidth(this->fontSize, tmpStr);
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
    for (unsigned int i = 0; i <= this->tlRes; i++) {
        offset = 0.0f;
        if ((i == 0) || (i == this->tlRes)) {
            offset = this->devY;
        }
        glVertex2f(this->tlStartPos.GetX() + (float)i*posFrac, this->tlStartPos.GetY() + offset);
        glVertex2f(this->tlStartPos.GetX() + (float)i*posFrac, this->tlStartPos.GetY() - this->devY);
    }
    glEnd();

    // Draw captions
    if (theFont.Initialise()) {
        tmpStr = "Time Line:  ";
        theFont.DrawString(this->tlStartPos.GetX() - this->devX, this->tlStartPos.GetY(), this->devX, 2.0f, 20, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_RIGHT_MIDDLE);
        for (unsigned int i = 0; i <= this->tlRes; i++) {
            tmpStr.Format("%.2f ", (float)i * timeFrac);
            strWidth = theFont.LineWidth(this->fontSize, tmpStr);
            theFont.DrawString(this->tlStartPos.GetX() + (float)i*posFrac - strWidth/2.0f, this->tlStartPos.GetY() - 1.0f - this->devY, strWidth, 1.0f, this->fontSize, true, tmpStr, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
        }
    }

	// Draw keyframes
    vislib::Array<Keyframe> *keyframes = ccc->getKeyframes();
    Keyframe                *s = ccc->getSelectedKeyframe();
    if (keyframes->Count() > 0) {
        float frameFrac = this->tlLength / maxTime;
        // draw the defined keyframes
        for (unsigned int i = 0; i < keyframes->Count(); i++) {
            this->DrawKeyframeSymbol(this->tlStartPos.GetX() + (*keyframes)[i].getTime() * frameFrac, this->tlStartPos.GetY(), ((*keyframes)[i] == (*s)));
        }

        // Draw interpolated selected keyframe
        if (!keyframes->Contains(*s)) {
            float x = this->tlStartPos.GetX() + s->getTime() * frameFrac;
            glColor3f(1.0f, 0.0f, 0.0f);
            glBegin(GL_LINES);
                glVertex2f(x, this->tlStartPos.GetY() + this->devY);
                glVertex2f(x, this->tlStartPos.GetY() - this->devY);
            glEnd();
        }
    }

	return true;
}


/*
* cinematiccamera::TimeLineRenderer::DrawKeyframeSymbol
*/
void TimeLineRenderer::DrawKeyframeSymbol(float posX, float posY, bool selected) {

    if (selected)
        glColor3f(0.0f, 0.0f, 1.0f);
    else
        glColor3f(0.0f, 0.0f, 0.3f);

    // as geometry
    /*
    glBegin(GL_TRIANGLES);
        glVertex2f(posX, posY);
        glVertex2f(posX - this->markerSize/2.0f, posY + this->markerSize);
        glVertex2f(posX + this->markerSize/2.0f, posY + this->markerSize);
    glEnd();
    */

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
                vislib::sys::Log::DefaultLog.WriteError("TIME LINE RENDERER [LoadTexture] Could not load \"%s\" texture.", filename.PeekBuffer());
                ARY_SAFE_DELETE(buf);
                return false;
            }
            markerTextures.Last()->SetFilter(GL_LINEAR, GL_LINEAR);
            ARY_SAFE_DELETE(buf);
            return true;
        }
        else {
            vislib::sys::Log::DefaultLog.WriteError("TIME LINE RENDERER [LoadTexture] Could not read \"%s\" texture.", filename.PeekBuffer());
        }
    }
    else {
        vislib::sys::Log::DefaultLog.WriteError("TIME LINE RENDERER [LoadTexture] Could not find \"%s\" texture.", filename.PeekBuffer());
    }
    return false;
}


/*
* cinematiccamera::TimeLineRenderer::MouseEvent
*/
bool TimeLineRenderer::MouseEvent(float x, float y, view::MouseFlags flags){
	
    const bool error = true;

	// on leftclick, check if a keyframe is hit and set the selected keyframe: MOUSEFLAG_BUTTON_LEFT_DOWN
	if (flags == view::MOUSEFLAG_BUTTON_LEFT_CHANGED+1){  // WHY THIS FLAG ???

		// y-Position of mouse within keyframe symbol range?
		if ((y < this->markerSize + this->tlStartPos.GetY()) && (y > this->tlStartPos.GetY() - this->devY*1.0f)) {

            bool hit = false;

			CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
			if (ccc == NULL) return error;
            if (!(*ccc)(CallCinematicCamera::CallForUpdateKeyframeKeeper)) return error;

            //Get keyframes
			vislib::Array<Keyframe> *keyframes = ccc->getKeyframes();

			// Get maximum time of keyframes
			float maxTime = ccc->getTotalTime();

			//Check all keyframes if they are hit
			for (unsigned int i = 0; i < keyframes->Count(); i++){

				float posX = this->tlStartPos.GetX() + (*keyframes)[i].getTime() / maxTime * this->tlLength;

				if ((x < (posX + this->markerSize/2.0f)) && (x > (posX - this->markerSize/2.0f))) {
                    // Set keyframe as selected
					ccc->setSelectedKeyframeTime((*keyframes)[i].getTime());
                    ccc->setChangedSelectedKeyframeTime(true);
                    hit = true;
                    break;
				}
			}

            // Get interpolated keyframe selection
            if (!hit) {
                if ((x > this->tlStartPos.GetX()) && (x < this->tlEndPos.GetX())) {
                    // Set an interpolated keyframe as selected
                    float st = ((x - this->tlStartPos.GetX()) / this->tlLength * maxTime);
                    ccc->setSelectedKeyframeTime(st);
                    ccc->setChangedSelectedKeyframeTime(true);
                    hit = true;
                }
            }

			return hit;
		}
	}

	// If true is returned, manipulator cannot move camera
	return false;
}
