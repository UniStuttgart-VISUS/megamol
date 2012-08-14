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
#include "math.h"
#include "CoreInstance.h"
#include "utility/ShaderSourceFactory.h"
#include "MolecularDataCall.h"
#include "view/AbstractCallRender3D.h"

using namespace megamol;

/*
 * protein::special::SphereRendererMouse::SphereRendererMouse
 */
protein::SphereRendererMouse::SphereRendererMouse() : Renderer3DModuleMouse(),
	    molDataCallerSlot("getData", "Connects the molecule rendering with molecule data storage."),
        mouseX(0.0f), mouseY(0.0f) {

    this->molDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable( &this->molDataCallerSlot);
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
	if (glh_init_extensions("GL_ARB_vertex_program") == 0) {
		return false;
	}
	if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
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
	core::view::AbstractCallRender3D *cr3d = dynamic_cast<core::view::AbstractCallRender3D *>(&call);
	if( cr3d == NULL ) return false;

	// get camera information
	this->cameraInfo = cr3d->GetCameraParameters();

	float callTime = cr3d->Time();

	// get pointer to MolecularDataCall
	MolecularDataCall *mol = this->molDataCallerSlot.CallAs<MolecularDataCall>();
	if( mol == NULL) return false;

	int cnt;

	// set call time
	mol->SetCalltime(callTime);
	// set frame ID and call data
	mol->SetFrameID(static_cast<int>( callTime));

	if (!(*mol)(MolecularDataCall::CallForGetData)) return false;
	// check if atom count is zero
	if( mol->AtomCount() == 0 ) return true;
	// get positions of the first frame
	float *pos0 = new float[mol->AtomCount() * 3];
	memcpy( pos0, mol->AtomPositions(), mol->AtomCount() * 3 * sizeof( float));
	// set next frame ID and get positions of the second frame
	if((static_cast<int>( callTime) + 1) < int( mol->FrameCount()))
		mol->SetFrameID(static_cast<int>( callTime) + 1);
	else
		mol->SetFrameID(static_cast<int>( callTime));
	if (!(*mol)(MolecularDataCall::CallForGetData)) {
		delete[] pos0;
		return false;
	}
	float *pos1 = new float[mol->AtomCount() * 3];
	memcpy( pos1, mol->AtomPositions(), mol->AtomCount() * 3 * sizeof( float));

	// interpolate atom positions between frames
	float *posInter = new float[mol->AtomCount()*4];
	float inter = callTime - static_cast<float>(static_cast<int>( callTime));
	float threshold = vislib::math::Min( mol->AccessBoundingBoxes().ObjectSpaceBBox().Width(),
			vislib::math::Min( mol->AccessBoundingBoxes().ObjectSpaceBBox().Height(),
					mol->AccessBoundingBoxes().ObjectSpaceBBox().Depth())) * 0.75f;
#pragma omp parallel for
	for(cnt = 0; cnt < int(mol->AtomCount()); ++cnt ) {
		if( std::sqrt( std::pow( pos0[3*cnt+0] - pos1[3*cnt+0], 2) +
				std::pow( pos0[3*cnt+1] - pos1[3*cnt+1], 2) +
				std::pow( pos0[3*cnt+2] - pos1[3*cnt+2], 2) ) < threshold ) {
			posInter[4*cnt+0] = (1.0f - inter) * pos0[3*cnt+0] + inter * pos1[3*cnt+0];
			posInter[4*cnt+1] = (1.0f - inter) * pos0[3*cnt+1] + inter * pos1[3*cnt+1];
			posInter[4*cnt+2] = (1.0f - inter) * pos0[3*cnt+2] + inter * pos1[3*cnt+2];
		} else if( inter < 0.5f ) {
			posInter[4*cnt+0] = pos0[3*cnt+0];
			posInter[4*cnt+1] = pos0[3*cnt+1];
			posInter[4*cnt+2] = pos0[3*cnt+2];
		} else {
			posInter[4*cnt+0] = pos1[3*cnt+0];
			posInter[4*cnt+1] = pos1[3*cnt+1];
			posInter[4*cnt+2] = pos1[3*cnt+2];
		}

		posInter[4*cnt+3] = 1.0f;
	}

	glPushMatrix();
	// compute scale factor and scale world
	float scale;
	if( !vislib::math::IsEqual( mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
		scale = 2.0f / mol->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
	} else {
		scale = 1.0f;
	}
	glScalef( scale, scale, scale);

	float viewportStuff[4] = {
			this->cameraInfo->TileRect().Left(),
			this->cameraInfo->TileRect().Bottom(),
			this->cameraInfo->TileRect().Width(),
			this->cameraInfo->TileRect().Height()};
	if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
	if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
	viewportStuff[2] = 2.0f / viewportStuff[2];
	viewportStuff[3] = 2.0f / viewportStuff[3];

	glDisable( GL_BLEND);
	glEnable( GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_ARB);

	// enable sphere shader
	this->sphereShader.Enable();
	glEnableClientState(GL_VERTEX_ARRAY);
	//glEnableClientState(GL_COLOR_ARRAY);
	// set shader variables
	glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
	glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"), 1, cameraInfo->Front().PeekComponents());
	glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"), 1, cameraInfo->Right().PeekComponents());
	glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"), 1, cameraInfo->Up().PeekComponents());
	// draw points
	// set vertex and color pointers and draw them
	glVertexPointer( 4, GL_FLOAT, 0, posInter);
	//glColorPointer( 3, GL_UNSIGNED_BYTE, 0, sphere->SphereColors() );
	//glColorPointer( 3, GL_FLOAT, 0, this->colors.PeekElements() );
	glDrawArrays( GL_POINTS, 0, mol->AtomCount());
	// disable sphere shader
	//glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	this->sphereShader.Disable();

	delete[] pos0;
	delete[] pos1;
	delete[] posInter;

	glDisable(GL_DEPTH_TEST);

	glPopMatrix();

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

    printf("Pos (%i %i)\n", this->mouseX, this->mouseY);

    if ((flags & core::view::MOUSEFLAG_BUTTON_LEFT_DOWN) != 0) {
    	printf("Left Button DOWN\n");
    }

    /*if ((flags & core::view::MOUSEFLAG_BUTTON_RIGHT_DOWN) != 0) {
    	printf("Right Button DOWN\n");
    }

    if ((flags & core::view::MOUSEFLAG_BUTTON_MIDDLE_DOWN) != 0) {
    	printf("Middle Button DOWN\n");
    }*/

    return true;
}
