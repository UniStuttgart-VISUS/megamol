/*
* AuroraBorealisRenderer.cpp
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"

#include "aurora_borealis_renderer.h"

#include <fstream>

#include "mmcore\moldyn\VolumeDataCall.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/view/AnimDataModule.h"


using namespace megamol;


ab::AuroraBorealisRenderer::AuroraBorealisRenderer(void) : Renderer3DModule(),
	getDataSlot("getdata", "Connects to the data source from the simulation") // only for testing purposes
	
{
	this->getDataSlot.SetCompatibleCall<core::moldyn::VolumeDataCallDescription>();
	this->MakeSlotAvailable(&this->getDataSlot);
}

ab::AuroraBorealisRenderer::~AuroraBorealisRenderer(void) {
	this->Release();
}

bool ab::AuroraBorealisRenderer::create() {
	mVis.initialize();

	return true;
}

void ab::AuroraBorealisRenderer::release() {
	// Renderer3DModule::release();
}

bool ab::AuroraBorealisRenderer::GetCapabilities(core::Call& call) {
	/*core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
	if (cr == NULL) return false;

	cr->SetCapabilities(
		core::view::CallRender3D::CAP_ANIMATION
	);*/

	return true;
}

bool ab::AuroraBorealisRenderer::GetExtents(core::Call& call) {
	return false;
}

bool ab::AuroraBorealisRenderer::Render(core::Call& call) {
	// connect camera to plugin to get camera movement
	core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
	if (cr == NULL) return false;


	mVis._window_width = cr->GetViewport().Width();
	mVis._window_height = cr->GetViewport().Height();
	mVis.zNear = cr->GetCameraParameters()->NearClip();
	mVis.zFar = cr->GetCameraParameters()->FarClip();
	mVis.aspectRatio = cr->GetViewport().AspectRatio();
	vec3_assign(mVis.eye, cr->GetCameraParameters()->EyePosition().PeekCoordinates());
	vec3_assign(mVis.lookat, cr->GetCameraParameters()->EyeDirection().PeekComponents());
	vec3_assign(mVis.up, cr->GetCameraParameters()->EyeUpVector().PeekComponents());

	

	mVis.VisRender();

	return true;
}