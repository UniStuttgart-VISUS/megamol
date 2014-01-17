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
}


view::TransferFunctionRenderer::~TransferFunctionRenderer(void)
{
}

bool view::TransferFunctionRenderer::create(void)
{
	return true;
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
			::glTexCoord2f(1.0f, 1.0f); ::glVertex2f(-1.0f,-1.0f);
			::glTexCoord2f(0.0f, 1.0f); ::glVertex2f(-1.0f, 1.0f);
			::glTexCoord2f(0.0f, 0.0f); ::glVertex2f( 1.0f, 1.0f);
			::glTexCoord2f(1.0f, 0.0f); ::glVertex2f( 1.0f,-1.0f);
        ::glEnd();
		glDisable(GL_TEXTURE_1D);
	}

	return true;
}