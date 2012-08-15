//
// SphereRendererMouse.cpp
//
// Copyright (C) 2012 by University of Stuttgart (VISUS).
// All rights reserved.
//


#include "stdafx.h"
#define _USE_MATH_DEFINES
#include <SphereRendererMouse.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include "math.h"
#include "CoreInstance.h"
#include "utility/ShaderSourceFactory.h"
#include "MolecularDataCall.h"
#include "view/AbstractCallRender3D.h"
#include "param/FloatParam.h"

using namespace megamol;

/*
 * protein::special::SphereRendererMouse::SphereRendererMouse
 */
protein::SphereRendererMouse::SphereRendererMouse() : Renderer3DModuleMouse(),
	    molDataCallerSlot("getData", "Connects the molecule rendering with molecule data storage."),
	    sphereRadSclParam("sphereRadScl", "Scale factor for the sphere radius."),
        mouseX(0.0f), mouseY(0.0f), startSelect(-1, -1), endSelect(-1, -1),
        drag(false),
        startSelectCurr(-1, -1), endSelectCurr(-1, -1), filter(false) {

	// Data caller slot
    this->molDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable( &this->molDataCallerSlot);

    // Param slot for sphere radius
    this->sphereRadSclParam << new core::param::FloatParam(1.0f, 0.0);
    this->MakeSlotAvailable(&this->sphereRadSclParam);
}


/*
 * protein::SphereRendererMouse::~SphereRendererMouse
 */
protein::SphereRendererMouse::~SphereRendererMouse() {
    this->Release();
}


/*
 * protein::SphereRendererMouse::create
 */
bool protein::SphereRendererMouse::create(void) {
	if(glh_init_extensions("GL_ARB_vertex_program") == 0) {
		return false;
	}
	if(!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
		return false;
	}

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);

	using namespace vislib::sys;
	using namespace vislib::graphics::gl;

	ShaderSource vertSrc;
	ShaderSource fragSrc;

	// Load sphere shader
	if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::sphereVertex", vertSrc)) {
		Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for sphere shader");
		return false;
	}
	if (!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("protein::std::sphereFragment", fragSrc)) {
		Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to load vertex shader source for sphere shader");
		return false;
	}
	try {
		if (!this->sphereShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
			throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
		}
	} catch(vislib::Exception e) {
		Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to create sphere shader: %s\n", e.GetMsgA());
		return false;
	}

	return true;
}


/*
 * protein::SphereRendererMouse::GetCapabilities
 */
bool protein::SphereRendererMouse::GetCapabilities(core::Call& call) {
    core::view::AbstractCallRender3D *cr3d = dynamic_cast<core::view::AbstractCallRender3D *>(&call);
    if (cr3d == NULL) return false;

    cr3d->SetCapabilities(core::view::AbstractCallRender3D::CAP_RENDER
        | core::view::AbstractCallRender3D::CAP_LIGHTING
        | core::view::AbstractCallRender3D::CAP_ANIMATION );

    return true;
}


/*
 * protein::SphereRendererMouse::GetExtents
 */
bool protein::SphereRendererMouse::GetExtents(core::Call& call) {

	core::view::AbstractCallRender3D *cr3d = dynamic_cast<core::view::AbstractCallRender3D *>(&call);
    if( cr3d == NULL ) return false;

    MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
    if( mol == NULL ) return false;
    if (!(*mol)(MolecularDataCall::CallForGetExtent)) return false;

    float scale;
    if( !vislib::math::IsEqual( mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }

    cr3d->AccessBoundingBoxes() = mol->AccessBoundingBoxes();
    cr3d->AccessBoundingBoxes().MakeScaledWorld(scale);
    cr3d->SetTimeFramesCount(mol->FrameCount());

    return true;
}


/*
 * protein::SphereRendererMouse::Render
 */
bool protein::SphereRendererMouse::Render(core::Call& call) {
	// cast the call to Render3D
	core::view::AbstractCallRender3D *cr3d =
			dynamic_cast<core::view::AbstractCallRender3D *>(&call);

	if(cr3d == NULL) return false;

	// Get camera information of render call
	this->cameraInfo = cr3d->GetCameraParameters();

	// Get calltime of render call
	float callTime = cr3d->Time();

	// get pointer to MolecularDataCall
	MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
	if( mol == NULL) return false;

	int cnt;

	// Set call time in data call
	mol->SetCalltime(callTime);

	// Set frame ID in data call and call data
	mol->SetFrameID(static_cast<int>( callTime));
	if (!(*mol)(MolecularDataCall::CallForGetData)) return false;

	// Check if atom count is zero
	if( mol->AtomCount() == 0 ) return true;

	// Init selection array if necessary
	if(this->atomSelect.Count() != mol->AtomCount()) {
		this->atomSelect.SetCount(mol->AtomCount());
	}

	// get positions of the first frame
	float *pos0 = new float[mol->AtomCount() * 3];
	memcpy(pos0, mol->AtomPositions(), mol->AtomCount()*3*sizeof(float));

	// set next frame ID and get positions of the second frame
	if((static_cast<int>(callTime)+1) < int(mol->FrameCount()))
		mol->SetFrameID(static_cast<int>(callTime)+1);
	else
		mol->SetFrameID(static_cast<int>( callTime));
	if (!(*mol)(MolecularDataCall::CallForGetData)) {
		delete[] pos0;
		return false;
	}
	float *pos1 = new float[mol->AtomCount()*3];
	memcpy(pos1, mol->AtomPositions(), mol->AtomCount()*3*sizeof(float));

	// interpolate atom positions between frames
	float *posInter = new float[mol->AtomCount()*4];
	float inter = callTime - static_cast<float>(static_cast<int>(callTime));
	float threshold = vislib::math::Min(mol->AccessBoundingBoxes().ObjectSpaceBBox().Width(),
			vislib::math::Min( mol->AccessBoundingBoxes().ObjectSpaceBBox().Height(),
					mol->AccessBoundingBoxes().ObjectSpaceBBox().Depth())) * 0.75f;
#pragma omp parallel for
	for(cnt = 0; cnt < int(mol->AtomCount()); ++cnt) {
		if(std::sqrt( std::pow( pos0[3*cnt+0] - pos1[3*cnt+0], 2) +
				std::pow( pos0[3*cnt+1] - pos1[3*cnt+1], 2) +
				std::pow( pos0[3*cnt+2] - pos1[3*cnt+2], 2) ) < threshold ) {
			posInter[4*cnt+0] = (1.0f - inter) * pos0[3*cnt+0] + inter * pos1[3*cnt+0];
			posInter[4*cnt+1] = (1.0f - inter) * pos0[3*cnt+1] + inter * pos1[3*cnt+1];
			posInter[4*cnt+2] = (1.0f - inter) * pos0[3*cnt+2] + inter * pos1[3*cnt+2];
		} else if(inter < 0.5f) {
			posInter[4*cnt+0] = pos0[3*cnt+0];
			posInter[4*cnt+1] = pos0[3*cnt+1];
			posInter[4*cnt+2] = pos0[3*cnt+2];
		} else {
			posInter[4*cnt+0] = pos1[3*cnt+0];
			posInter[4*cnt+1] = pos1[3*cnt+1];
			posInter[4*cnt+2] = pos1[3*cnt+2];
		}
		posInter[4*cnt+3] = this->sphereRadSclParam.Param<core::param::FloatParam>()->Value();
	}

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	// compute scale factor and scale world
	float scale;
	if(!vislib::math::IsEqual( mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
		scale = 2.0f / mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
	} else {
		scale = 1.0f;
	}
	glScalef(scale, scale, scale);

	float viewportStuff[4] = {
			this->cameraInfo->TileRect().Left(),
			this->cameraInfo->TileRect().Bottom(),
			this->cameraInfo->TileRect().Width(),
			this->cameraInfo->TileRect().Height()};
	if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
	if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
	viewportStuff[2] = 2.0f / viewportStuff[2];
	viewportStuff[3] = 2.0f / viewportStuff[3];

	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);

	if(this->filter) {
		printf("Filter RECT (%i %i) (%i %i)\n",
    			this->startSelectCurr.X(), this->startSelectCurr.Y(),
    			this->endSelectCurr.X(), this->endSelectCurr.Y());

		// Calculate screen space positions of the atoms
		GLdouble modelMatrix[16];
		glGetDoublev(GL_MODELVIEW_MATRIX, modelMatrix);
		GLdouble projMatrix[16];
		glGetDoublev(GL_PROJECTION_MATRIX, projMatrix);
		int viewport[4];
		glGetIntegerv(GL_VIEWPORT, viewport);
#pragma omp parallel for
		for(int at = 0; at < static_cast<int>(mol->AtomCount()); at++) {
			this->atomSelect[at] = false;
		}

		// Determine which atoms are selected
#pragma omp parallel for
		for(int at = 0; at < static_cast<int>(mol->AtomCount()); at++) {
			this->atomSelect[at] = false;
		}
		this->filter = false;
	}
	else {
#pragma omp parallel for
		for(int at = 0; at < static_cast<int>(mol->AtomCount()); at++) {
			this->atomSelect[at] = false;
		}
	}

	// Set color
	this->atomColor.SetCount(mol->AtomCount()*3);
#pragma omp parallel for
	for(int at = 0; at < static_cast<int>(mol->AtomCount()); at++) {
		if(this->atomSelect[at] == false) { // Atom is not selected
			this->atomColor[3*at+0] = 0.0f;
			this->atomColor[3*at+1] = 1.0f;
			this->atomColor[3*at+2] = 0.0f;
		}
		else { // Atom is selected
			this->atomColor[3*at+0] = 0.0f;
			this->atomColor[3*at+1] = 0.0f;
			this->atomColor[3*at+2] = 1.0f;
		}
	}
	//glColor3f(0.0f, 1.0f, 0.0f);

	// Enable sphere shader
	this->sphereShader.Enable();
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	// set shader variables
	glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
	glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
	glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
	glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());

	// Draw points
	glVertexPointer(4, GL_FLOAT, 0, posInter);
	glColorPointer(3, GL_FLOAT, 0, this->atomColor.PeekElements());
	glDrawArrays(GL_POINTS, 0, mol->AtomCount());

	// disable sphere shader
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	this->sphereShader.Disable();

	delete[] pos0;
	delete[] pos1;
	delete[] posInter;

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	// Render rectangle

    float curVP[4];
    glGetFloatv(GL_VIEWPORT, curVP);

    // Draw rectangle if there is a valid end point
    // TODO Find better criterion?
    if(this->endSelect.X() != -1) {

    	glDisable(GL_DEPTH_TEST);
    	glDisable(GL_LIGHTING);
    	glDisable(GL_CULL_FACE);
    	glEnable(GL_BLEND);

    	glMatrixMode(GL_PROJECTION);
    	glPushMatrix();
    	glLoadIdentity();

    	// This sets up the OpenGL window so that (0,0) corresponds to the top left corner
    	//printf("Current viewport (%f %f %f %f)\n", curVP[0], curVP[1], curVP[2], curVP[3]); // DEBUG
    	glOrtho(curVP[0], curVP[2], curVP[3], curVP[1], -1.0, 1.0);

    	glMatrixMode(GL_MODELVIEW);
    	glPushMatrix();
    	glLoadIdentity();

    	// Draw transparent quad
    	glPolygonMode(GL_FRONT_AND_BACK,  GL_FILL);
    	glColor4f(1.0f, 1.0f, 1.0f, 0.2f);
    	glBegin(GL_QUADS);
    		glVertex2i(this->startSelect.X(), this->startSelect.Y());
    		glVertex2i(this->endSelect.X(), this->startSelect.Y());
    		glVertex2i(this->endSelect.X(),   this->endSelect.Y());
    		glVertex2i(this->startSelect.X(),   this->endSelect.Y());
    	glEnd();

    	// Draw outline
    	glPolygonMode(GL_FRONT_AND_BACK,  GL_LINE);
    	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    	glBegin(GL_QUADS);
    		glVertex2i(this->startSelect.X(), this->startSelect.Y());
    		glVertex2i(this->endSelect.X(), this->startSelect.Y());
    		glVertex2i(this->endSelect.X(),   this->endSelect.Y());
    		glVertex2i(this->startSelect.X(),   this->endSelect.Y());
    	glEnd();

    	glPopMatrix();

    	glMatrixMode(GL_PROJECTION);
    	glPopMatrix();

    	glEnable(GL_CULL_FACE);
    	glDisable(GL_BLEND);
    }


	return true;

}


/*
 * protein::SphereRendererMouse::release
 */
void protein::SphereRendererMouse::release(void) {
    // intentionally empty
}


/*
 * protein::SphereRendererMouse::MouseEvent
 */
bool protein::SphereRendererMouse::MouseEvent(int x, int y, core::view::MouseFlags flags) {

    this->mouseX = x;
    this->mouseY = y;

    printf("POS (%i %i)\n", this->mouseX, this->mouseY);

    if ((flags & core::view::MOUSEFLAG_BUTTON_LEFT_DOWN) != 0) {
    	if(!this->drag) {
    		this->startSelect.Set(this->mouseX, this->mouseY);
    		this->startSelectCurr.Set(-1, -1);
    		this->endSelectCurr.Set(-1, -1);
    		this->drag = true;
    	}
    	else {
    		this->endSelect.Set(this->mouseX, this->mouseY);
    	}
    }
    else {
    	if(this->endSelect.X() != -1) {
    		this->startSelectCurr.Set(this->startSelect.X(), this->startSelect.Y());
    		this->endSelectCurr.Set(this->endSelect.X(), this->endSelect.Y());
    		this->filter = true;
    	}
		this->startSelect.Set(-1, -1);
		this->endSelect.Set(-1, -1);
    	this->drag = false;
    }
    return true;
}
