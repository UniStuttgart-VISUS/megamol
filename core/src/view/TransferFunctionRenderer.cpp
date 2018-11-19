/*
 * TransferFunctionRenderer.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/TransferFunctionRenderer.h"
#include "mmcore/view/CallGetTransferFunction.h"

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

bool view::TransferFunctionRenderer::GetExtents(CallRender2D& call)
{
	call.SetBoundingBox( 0.0f, 0.0f, 1.0f, 5.0f);
	return true;
}

bool view::TransferFunctionRenderer::Render(CallRender2D& call)
{
	view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
	if ((cgtf != NULL) && ((*cgtf)())){
		
		glEnable(GL_TEXTURE_1D);
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_LIGHTING);
		glDisable(GL_BLEND);
		glDisable(GL_CULL_FACE);
		
		glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
		
		::glColor4ub(255, 255, 255, 255);
        ::glBegin(GL_QUADS);
			::glTexCoord2f(1.0f, 1.0f); ::glVertex2f( 0.0f, 1.0f);
			::glTexCoord2f(0.0f, 1.0f); ::glVertex2f( 0.0f, 5.0f);
			::glTexCoord2f(0.0f, 0.0f); ::glVertex2f( 1.0f, 5.0f);
			::glTexCoord2f(1.0f, 0.0f); ::glVertex2f( 1.0f, 1.0f);
        ::glEnd();
		
		glBindTexture(GL_TEXTURE_1D, 0);
		glDisable(GL_TEXTURE_1D);

		vislib::StringA ctStr = cgtf->PeekCalleeSlot()->Parent()->Name();
		if( ctFont->IsInitialised() ) {
			ctFont->DrawString( 0.5f, 0.0f, 0.6f, true, ctStr.PeekBuffer(), vislib::graphics::AbstractFont::ALIGN_CENTER_BOTTOM);
		}
	}

	return true;
}