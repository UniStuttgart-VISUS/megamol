

#include "stdafx.h"
#include "PositionScene.h"
#include "mmcore/param/EnumParam.h"
#include "vislib/sys/Log.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "CallCinematicCamera.h"

using namespace megamol::core;
using namespace megamol::cinematiccamera;
/*
* PositionScene::PositionScene
*/
PositionScene::PositionScene(void) : view::View3D(),
getDataSlot("getdata", "Connects to the data source"),
getCinematicCameraSlot("keyframes", "Connects to the KeyframeKeeper"){

	this->MakeSlotAvailable(&this->getDataSlot);

	this->getCinematicCameraSlot.SetCompatibleCall<cinematiccamera::CallCinematicCameraDescription>();
	this->MakeSlotAvailable(&this->getCinematicCameraSlot);
}

/*
* PositionScene::~PositionScene
*/
PositionScene::~PositionScene(void) {
	this->Release();
}

/*
* PositionScene::ResetView
*/
void PositionScene::ResetView(void) {
	view::View3D::ResetView(); // parent function call

}


/*
* PositionScene::create
*/
bool PositionScene::create(void) {

	view::View3D::create();

	cinematiccamera::CallCinematicCamera *keyframeCall = this->getCinematicCameraSlot.CallAs<cinematiccamera::CallCinematicCamera>();
	if (keyframeCall != NULL){

		//NACHFRAGEN: Ist das so korrekt? (ref.: injectCamera)
		this->camParams->CopyFrom(keyframeCall->getSelectedKeyframe().getCamera().Parameters());
	}
	else {
		return false;
	}
	
	return true;
}

/*
* PositionScene::release
*/
void PositionScene::release(void) {
	view::View3D::release();

}

void PositionScene::injectCamera(vislib::graphics::Camera injectionCam){
	this->camParams->CopyFrom(injectionCam.Parameters());
}

bool PositionScene::injectCamera(){
	cinematiccamera::CallCinematicCamera *keyframeCall = this->getCinematicCameraSlot.CallAs<cinematiccamera::CallCinematicCamera>();
	if (keyframeCall != NULL){
		this->camParams->CopyFrom(keyframeCall->getSelectedKeyframe().getCamera().Parameters());
		return true;
	}
	else {
		return false;
	}
}