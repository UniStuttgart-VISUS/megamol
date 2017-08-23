/**
* CinematicView.cpp
*/

#include "stdafx.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/view/CallRender3D.h"

#include "CinematicView.h"
#include "CallCinematicCamera.h"



using namespace megamol;
using namespace megamol::core;
using namespace cinematiccamera;


/*
* CinematicView::CinematicView
*/
CinematicView::CinematicView(void) : View3D(),
	keyframeKeeperSlot("keyframeKeeper", "Connects to the Keyframe Keeper."),
	selectedSkyboxSideParam("cinematic::01 Skybox Side", "Skybox side rendering"),
    selectedKeyframe()
    {

    this->keyframeKeeperSlot.SetCompatibleCall<CallCinematicCameraDescription>();
    this->MakeSlotAvailable(&this->keyframeKeeperSlot);

	param::EnumParam *sbs = new param::EnumParam(SKYBOX_NONE);
	sbs->SetTypePair(SKYBOX_NONE,   "None");
	sbs->SetTypePair(SKYBOX_FRONT,	"Front");
	sbs->SetTypePair(SKYBOX_BACK,	"Back");
	sbs->SetTypePair(SKYBOX_LEFT,	"Left");
	sbs->SetTypePair(SKYBOX_RIGHT,	"Right");
	sbs->SetTypePair(SKYBOX_UP,		"Up");
	sbs->SetTypePair(SKYBOX_DOWN,	"Down");
	this->selectedSkyboxSideParam << sbs;
	this->MakeSlotAvailable(&this->selectedSkyboxSideParam);

    this->currentTime  = 0.0f;
}


/*
* CinematicView::~CinematicView
*/
CinematicView::~CinematicView() {

}


/*
* CinematicView::Render
*/
void CinematicView::Render(const mmcRenderViewContext& context) {

	view::CallRender3D *cr3d = this->rendererSlot.CallAs<core::view::CallRender3D>();
	if (!(*cr3d)(1)) return; // get extents

	CallCinematicCamera *ccc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
    if (!ccc) return;
    if (!(*ccc)(CallCinematicCamera::CallForUpdateKeyframeKeeperData)) return;

    // ...
    float viewTime = static_cast<float>(context.Time);
    if (this->currentTime != viewTime) {

        // Select the keyframe based on the current animation time.
        ccc->setSelectedKeyframeTime(viewTime);
        // Update data in cinematic camera call
        if (!(*ccc)(CallCinematicCamera::CallForSetSelectedKeyframe)) return;
        this->currentTime = viewTime;
    }

    if (ccc->getKeyframes()->Count() > 0) {
        // Set camera parameters of selected keyframe for this view
        Keyframe s = ccc->getSelectedKeyframe();
        if (!(this->selectedKeyframe.getCamera() == s.getCamera())) {
            vislib::SmartPtr<vislib::graphics::CameraParameters> p = s.getCamParameters();
            this->cam.Parameters()->SetView(p->Position(), p->LookAt(), p->Up());
            this->cam.Parameters()->SetApertureAngle(p->ApertureAngle());
            this->selectedKeyframe = s;
        }
    }

    // Propagate camera parameter to keyframe keeper
    ccc->setCameraParameter(this->cam.Parameters());
    if (!(*ccc)(CallCinematicCamera::CallForSetCameraForKeyframe)) return;

    // Set time of keyframe to animation time
    // this->timeCtrl.SetTimeExtend(static_cast<int>(s.getTime()), false); // -> TimeControl

    // Adjust cam to selected skybox side
	vislib::SmartPtr<vislib::graphics::CameraParameters> cp = this->cam.Parameters();
	vislib::math::Point<float, 3> camPos = cp->Position();
	vislib::math::Vector<float, 3> camRight = cp->Right();
	vislib::math::Vector<float, 3> camUp = cp->Up();
	vislib::math::Vector<float, 3> camFront = cp->Front();
	float tmpDist = cp->FocalDistance();
	SkyboxSides side = static_cast<SkyboxSides>(this->selectedSkyboxSideParam.Param<param::EnumParam>()->Value());
	if (side != SKYBOX_NONE) {
		// set aperture angle to 90 deg
		cp->SetApertureAngle(90.0f);
		if (side == SKYBOX_BACK) {
			cp->SetView(camPos, camPos - camFront * tmpDist, camUp);
		}
		else if (side == SKYBOX_RIGHT) {
			cp->SetView(camPos, camPos + camRight * tmpDist, camUp);
		}
		else if (side == SKYBOX_LEFT) {
			cp->SetView(camPos, camPos - camRight * tmpDist, camUp);
		}
		else if (side == SKYBOX_UP) {
			cp->SetView(camPos, camPos + camUp * tmpDist, -camFront);
		}
		else if (side == SKYBOX_DOWN) {
			cp->SetView(camPos, camPos - camUp * tmpDist, camFront);
		}

	}

    // Call Renderer
    Base::Render(context);

}