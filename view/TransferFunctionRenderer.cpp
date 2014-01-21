/*
 * TransferFunctionRenderer.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "TransferFunctionRenderer.h"
#include "CallGetTransferFunction.h"

using namespace megamol::core;


/*
 * view::Renderer3DModule::Renderer3DModule
 */

view::TransferFunctionRenderer::TransferFunctionRenderer(void) : Renderer2DModule(),
	getTFSlot("tf", "transfer function")
{
	getTFSlot.SetCompatibleCall<view::CallGetTransferFunctionDescription>();
	this->MakeSlotAvailable(&this->getTFSlot);

	ctFont = new vislib::graphics::gl::SimpleFont();
}


view::TransferFunctionRenderer::~TransferFunctionRenderer(void)
{
}

bool view::TransferFunctionRenderer::create(void)
{
	return ctFont->Initialise();
}

void view::TransferFunctionRenderer::release(void)
{
}

bool view::TransferFunctionRenderer::GetCapabilities(CallRender2D& call)
{
	return true;
}

bool view::TransferFunctionRenderer::GetExtents(CallRender2D& call)
{
	return true;
}

bool view::TransferFunctionRenderer::Render(CallRender2D& call)
{
	view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
	if ((cgtf != NULL) && ((*cgtf)())){
		
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glEnable(GL_TEXTURE_1D);
		glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
		
		::glColor4ub(255, 255, 255, 255);
        ::glBegin(GL_QUADS);
			::glTexCoord2f(1.0f, 1.0f); ::glVertex2f(-1.0f,-0.8f);
			::glTexCoord2f(0.0f, 1.0f); ::glVertex2f(-1.0f, 1.0f);
			::glTexCoord2f(0.0f, 0.0f); ::glVertex2f( 1.0f, 1.0f);
			::glTexCoord2f(1.0f, 0.0f); ::glVertex2f( 1.0f,-.8f);
        ::glEnd();
		glDisable(GL_TEXTURE_1D);

		vislib::StringA ctStr = cgtf->PeekCalleeSlot()->Parent()->Name();
		if( ctFont->IsInitialised() ) {
			ctFont->DrawString(  0.0, -1.0f, 0.2, true, ctStr.PeekBuffer(), vislib::graphics::AbstractFont::ALIGN_CENTER_BOTTOM);
		}
	}

	return true;
}