/**
 * Screenshotview.cpp
 */

#include "stdafx.h"
#include "Screenshotview.h"
#include "CallCinematicCamera.h"
#include "CameraParamsOverride.h"

#include "mmcore/view/CallRender3D.h"

using namespace megamol;
using namespace cinematiccamera;

Screenshotview::Screenshotview(void) : View3D(),
keyframeKeeperSlot("keyframeKeeper", "Connects to the Keyframe Keeper."),
paramsOverride(nullptr){

	this->keyframeKeeperSlot.SetCompatibleCall<CallCinematicCameraDescription>();
	this->MakeSlotAvailable(&this->keyframeKeeperSlot);
}


megamol::cinematiccamera::Screenshotview::~Screenshotview() {}

void megamol::cinematiccamera::Screenshotview::Render(const mmcRenderViewContext& context) {
	
	{
		auto curParams = this->cam.Parameters();	// DO NOT USE MEMBER VARIABLE!
		if (curParams != this->paramsOverride) {
			if (this->paramsOverride == nullptr) {
				this->paramsOverride = new CameraParamsOverride(curParams);
			}
			this->cam.SetParameters(this->paramsOverride);
		}
		if (this->camParams != this->paramsOverride) {
			this->camParams = this->paramsOverride;
		}
	}

	auto cr3d = this->rendererSlot.CallAs<core::view::CallRender3D>();

	auto kfc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();

	kfc->setTimeofKeyframeToGet(context.Time);
	if ((*kfc)(CallCinematicCamera::CallForGetKeyframeAtTime)){
//		vislib::SmartPtr<vislib::graphics::CameraParameters> hackCamParams(
//			new CameraParamsOverride(camParams));
//		kfc->getInterpolatedKeyframe().putCamParameters(hackCamParams);
//		cr3d->SetCameraParameters(hackCamParams);
//		this->cam.SetParameters(hackCamParams);

		kfc->getInterpolatedKeyframe().putCamParameters(this->paramsOverride);
		
		Base::Render(context);
	}
}