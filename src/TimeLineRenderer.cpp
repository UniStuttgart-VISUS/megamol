/*
* TimeLineRenderer.cpp
*
* Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
//#define _USE_MATH_DEFINES
#include "TimeLineRenderer.h"
#include "CallCinematicCamera.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/ButtonParam.h"
#include "vislib/graphics/gl/SimpleFont.h"
#include "vislib/graphics/BitmapImage.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "vislib/graphics/gl/OpenGLTexture2D.h"
#include "vislib/sys/Log.h"
#include "mmcore/misc/PngBitmapCodec.h"
#include "mmcore/view/Renderer2DModule.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/FloatParam.h"
//#include "Keyframe.h"
//#include "vislib/Array.h"
#include "vislib/String.h"

//#include <cmath>

using namespace megamol::core;
using namespace megamol::cinematiccamera;


/*
* cinematiccamera::TimeLineRenderer::TimeLineRenderer
*/
megamol::cinematiccamera::TimeLineRenderer::TimeLineRenderer(void) : view::Renderer2DModule(),
	getDataSlot("getkeyframes", "Connects to the KeyframeKeeper"),
	resolutionParam("resolution", "The plotting resolution of the diagram."),
#ifndef USE_SIMPLE_FONT
	theFont(vislib::graphics::gl::FontInfo_Verdana),
#endif // USE_SIMPLE_FONT
	markerTextures(0)
	{

	this->getDataSlot.SetCompatibleCall<cinematiccamera::CallCinematicCameraDescription>();
	this->MakeSlotAvailable(&this->getDataSlot);

	// set up the resolution param for the texture
	this->resolutionParam.SetParameter(new param::FloatParam(1024, 0, 8192));
	this->MakeSlotAvailable(&this->resolutionParam);
}


/*
* cinematiccamera::TimeLineRenderer::~TimeLineRenderer
*/
megamol::cinematiccamera::TimeLineRenderer::~TimeLineRenderer(void) {
	this->Release();
}


/*
* cinematiccamera::TimeLineRenderer::create
*/
bool megamol::cinematiccamera::TimeLineRenderer::create(void) {
	
	//this->LoadTexture("test.png");
	
	return true;
}


/*
* cinematiccamera::TimeLineRenderer::GetExtents
*/
bool megamol::cinematiccamera::TimeLineRenderer::GetExtents(view::CallRender2D& call) {

	//necessary?
	core::view::CallRender2D *cr = dynamic_cast<core::view::CallRender2D*>(&call);
	if (cr == NULL) return false;

	CallCinematicCamera *ccc = this->getDataSlot.CallAs<CallCinematicCamera>();
	if (ccc != NULL) {
		if (!(*ccc) (CallCinematicCamera::CallForGetKeyframes)){
			ccc = NULL;
		}
	}

	call.SetBoundingBox(cr->GetViewport());
	lineLength = cr->GetViewport().GetSize().GetWidth() - 10.0f;
	lineYPos = cr->GetViewport().GetSize().GetHeight() / 2.0f;
	return true;
}


/*
* cinematiccamera::TimeLineRenderer::release
*/
void megamol::cinematiccamera::TimeLineRenderer::release(void) {
	// intentionally empty
}


// sequenceRenderer protein-plugin anschauen!!!!


/*
* cinematiccamera::TimeLineRenderer::Render
*/
bool megamol::cinematiccamera::TimeLineRenderer::Render(view::CallRender2D& call) {
	updateParameters();
	view::CallRender2D *cr = dynamic_cast<view::CallRender2D*>(&call);
	if (cr == NULL) return false;
	

	vislib::Array<Keyframe> *keyframes;
	
	cinematiccamera::CallCinematicCamera *kfc = this->getDataSlot.CallAs<cinematiccamera::CallCinematicCamera>();
	if (kfc == NULL) return false;
	if (!(*kfc)(CallCinematicCamera::CallForGetKeyframes)) return false;
	
	keyframes = kfc->getKeyframes();
		 
	glDisable(GL_CULL_FACE);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	// get the diagram color (inverse background color)
	float bgColor[4];
	float fgColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
	glGetFloatv(GL_COLOR_CLEAR_VALUE, bgColor);
	for (unsigned int i = 0; i < 4; i++) {
		fgColor[i] -= bgColor[i];
	}

	vislib::StringA myString;

	glColor3fv(fgColor);
	glDisable(GL_CULL_FACE);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDisable(GL_VERTEX_ARRAY);
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);


	glBegin(GL_LINE_STRIP);
	glVertex2f(5.0f, cr->GetViewport().GetSize().GetHeight() / 2.0f - 5.0f);
	glVertex2f(5.0f, cr->GetViewport().GetSize().GetHeight() / 2.0f);
	glVertex2f(cr->GetViewport().GetSize().GetWidth() - 5.0f, cr->GetViewport().GetSize().GetHeight() / 2.0f);
	glVertex2f(cr->GetViewport().GetSize().GetWidth() - 5.0f, cr->GetViewport().GetSize().GetHeight() / 2.0f - 5.0f);
	glEnd();

	theFont.DrawString(5.0f, cr->GetViewport().GetSize().GetHeight() / 2.0f - 10.0f, theFont.LineWidth(16, "t = 0"), 1.0f, 16.0f, true, "t = 0", vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);
	
	// get maximum time of keyframes
	if (!(*kfc)(CallCinematicCamera::CallForGetTotalTime))return false;
	float max = kfc->getTotalTime();

	// create string
	myString.Format("t = %f", max);
	// cut off zeros at the end of float
	myString.Truncate(myString.Length()-5);
	theFont.DrawString(cr->GetViewport().GetSize().GetWidth() - 5.0f - theFont.LineWidth(16, myString), cr->GetViewport().GetSize().GetHeight() / 2 - 10.0f, theFont.LineWidth(16, myString), 1.0f, 16, true, myString, vislib::graphics::AbstractFont::ALIGN_LEFT_TOP);

	(*kfc)(CallCinematicCamera::CallForGetSelectedKeyframe);
	// draw the keyframe symbols (not if first keyframe is dummy keyframe)
	if ((*keyframes)[0].getID() != -2){
		for (int i = 0; i < keyframes->Count(); i++){
			DrawKeyframeSymbol((*keyframes)[i], cr->GetViewport().GetSize().GetWidth() - 10.0f, cr->GetViewport().GetSize().GetHeight() / 2.0f, kfc->getSelectedKeyframeIndex() == (float)i);
		}
	}
	// draw selected interoplated keyframe if appropriate
	if (kfc->getSelectedKeyframe().getID() == -1){
		kfc->setIndexToInterpolate(kfc->getSelectedKeyframeIndex());
		if (!(*kfc)(CallCinematicCamera::CallForInterpolatedKeyframe)){
			vislib::sys::Log::DefaultLog.WriteError("CallForInterpolatedKeyframe failed!");
			return false;
		}

		float posX = lineLength * kfc->getInterpolatedKeyframe().getTime() + 5.0f;
		float posY = cr->GetViewport().GetSize().GetHeight() / 2.0f;
		glColor3f(1.0, 1.0, 0.0);
		glBegin(GL_LINE_STRIP);
		glVertex2f(posX, 5.0f + lineYPos);
		glVertex2f(posX, -5.0f + lineYPos);
		glEnd();

	}

	return true;
}


bool megamol::cinematiccamera::TimeLineRenderer::LoadTexture(vislib::StringA filename) {
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
				vislib::sys::Log::DefaultLog.WriteError("could not load %s texture.", filename.PeekBuffer());
				ARY_SAFE_DELETE(buf);
				return false;
			}
			markerTextures.Last()->SetFilter(GL_LINEAR, GL_LINEAR);
			ARY_SAFE_DELETE(buf);
			return true;
		}
		else {
			vislib::sys::Log::DefaultLog.WriteError("could not read %s texture.", filename.PeekBuffer());
		}
	}
	else {
		vislib::sys::Log::DefaultLog.WriteError("could not find %s texture.", filename.PeekBuffer());
	}
	return false;
}

void TimeLineRenderer::DrawKeyframeSymbol(Keyframe k, float lineLength, float lineYPos, bool selected){
	float posX = lineLength * k.getTime() + 5.0f;
	if (selected) glColor3f(1.0, 1.0, 0.0); else glColor3f(1.0, 0.0, 0.0);
	glBegin(GL_LINE_STRIP);
	glVertex2f(posX, 2.0f + lineYPos);
	glVertex2f(posX, lineYPos);
	glVertex2f(posX + 2.0f, lineYPos);
	glVertex2f(posX + 2.0f, 3.0f + lineYPos);
	glVertex2f(posX + 4.0f, 5.0f + lineYPos);
	glVertex2f(posX - 4.0f, 5.0f + lineYPos);
	glVertex2f(posX - 2.0f, 3.0f + lineYPos);
	glVertex2f(posX - 2.0f, lineYPos);
	glVertex2f(posX, lineYPos);
	glEnd();
}


bool TimeLineRenderer::MouseEvent(float x, float y, megamol::core::view::MouseFlags flags){
	
	// on leftclick, check if a keyframe is hit and set the selected keyframe
	if (flags & view::MOUSEFLAG_BUTTON_LEFT_DOWN){

		float lineMiddle = resolutionParam.Param<param::FloatParam>()->Value() / 2;

		// y-Position of mouse within keyframe symbol range?
		if (y < 14.0f + lineYPos && y > 1.0f + lineYPos){
			//need to know keyframes
			CallCinematicCamera *kfc = this->getDataSlot.CallAs<CallCinematicCamera>();
			if (kfc == NULL) return false;
			if (!(*kfc)(CallCinematicCamera::CallForGetKeyframes)) return false;
			vislib::Array<Keyframe> *keyframes;
			keyframes = kfc->getKeyframes();

			// get maximum time of keyframes
			if (!(*kfc)(CallCinematicCamera::CallForGetTotalTime))return false;
			float max = kfc->getTotalTime();

			//check all keyframes if they are hit
			bool hit = false;
			for (int i = 0; i < keyframes->Count(); i++){
				float posX = lineLength * (*keyframes)[i].getTime();
				if (x < posX + 8.0f && x > posX - 8.0f){
					hit = true;
					kfc->setSelectedKeyframeIndex(static_cast<float>(i));
					if (!(*kfc)(CallCinematicCamera::CallForSelectKeyframe)) return false;					
				}
				// requested an interpolated keyframe
				if (x < posX - 8.0f && !hit){
					hit = true;
					float prevPosX = lineLength * (*keyframes)[i-1].getTime();

					kfc->setSelectedKeyframeIndex((float)i - (posX - x) / (posX - prevPosX));
					if (!(*kfc)(CallCinematicCamera::CallForSelectKeyframe)) {
						vislib::sys::Log::DefaultLog.WriteError("could not select interpolated keyframe.");
					}
				}
			}
			return hit;
		}
	}
	// if true is returned, manipulator cannot move camera
	return false;
}

bool TimeLineRenderer::updateParameters(){
	return true;
}