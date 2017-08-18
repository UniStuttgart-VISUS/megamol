/**
* PreviewView.cpp
*/

#include "stdafx.h"
#include "PreviewView.h"
#include "CallCinematicCamera.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/sys/Log.h"
#include "vislib/Trace.h"
#include "vislib/sys/Log.h"
#include "vislib/StringSerialiser.h"

using namespace megamol;
using namespace core;
using namespace cinematiccamera;

PreviewView::PreviewView(void) : View3D(),
keyframeKeeperSlot("keyframeKeeper", "Connects to the Keyframe Keeper."),
paramsOverride(nullptr) {

	this->keyframeKeeperSlot.SetCompatibleCall<CallCinematicCameraDescription>();
	this->MakeSlotAvailable(&this->keyframeKeeperSlot);
	showBBox.Param<param::BoolParam>()->SetValue(false);
}


megamol::cinematiccamera::PreviewView::~PreviewView() {}

void megamol::cinematiccamera::PreviewView::Render(const mmcRenderViewContext& context) {
	
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

	auto kfc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();

	if ((*kfc)(CallCinematicCamera::CallForGetSelectedKeyframe)){

		auto cr3d = this->rendererSlot.CallAs<core::view::CallRender3D>();
	
		if (kfc->getSelectedKeyframe().getID() != -2){
			/*
			camParams->SetFocalDistance(1);
			paramsOverride->SetFocalDistance(1);
			camParams->SetClip(0.1, 10);
			paramsOverride->SetClip(0.1, 10);
			*/
			if (kfc->getSelectedKeyframe().getID() == -1)
				kfc->getInterpolatedKeyframe().putCamParameters(this->paramsOverride);
			else
				kfc->getSelectedKeyframe().putCamParameters(this->paramsOverride);
		
	//		cr3d->SetTime(kfc->getSelectedKeyframe().getTime()*kfc->getTotalTime());
			
			// Focal Distance is ~4.7 instead of 1
			// (Half-) Aperture Angle is smaller by factor 2
			// ViewSize is larger than in Renderer (~ factor 6)
			// FarClip is off! should be 10
			// NearClip is off! should be 0.1


/*			vislib::StringSerialiserA ser;

			paramsOverride->Serialise(ser);
			vislib::sys::Log::DefaultLog.WriteInfo("Preview:");
			vislib::sys::Log::DefaultLog.WriteInfo(ser.GetString().PeekBuffer());
			
			*/
			
			Base::Render(context);

			vislib::sys::Log::DefaultLog.WriteInfo("Preview near, far: %f, %f", paramsOverride->NearClip(), paramsOverride->FarClip());
			vislib::sys::Log::DefaultLog.WriteInfo("Preview Right: %f, %f, %f", paramsOverride->Right().GetX(), paramsOverride->Right().GetY(), paramsOverride->Right().GetZ());
		}
	}
}