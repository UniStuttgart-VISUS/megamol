/**
* CinematicView.cpp
*/

#include "stdafx.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/view/CallRender3D.h"

#include "CinematicView.h"
#include "CallCinematicCamera.h"

#include <iostream>


using namespace megamol;
using namespace megamol::core;
using namespace cinematiccamera;


/*
* CinematicView::CinematicView
*/
CinematicView::CinematicView(void) : View3D(),
	keyframeKeeperSlot("keyframeKeeper", "Connects to the Keyframe Keeper."),
    viewModeParam(          "cinematic::01 VIEW MODE", "render only selected keyframe or render keyframe at animation time"),
	selectedSkyboxSideParam("cinematic::02 Skybox Side", "Skybox side rendering")
    {

    this->currentViewMode = VIEWMODE_SELECTION;

    param::EnumParam *rm = new param::EnumParam(int(this->currentViewMode));
    rm->SetTypePair(VIEWMODE_SELECTION, "Selected Keyframe");
    rm->SetTypePair(VIEWMODE_ANIMATION, "Animation");
    this->viewModeParam << rm;
    this->MakeSlotAvailable(&this->viewModeParam);

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
    // Update data in cinematic camera call
    if (!(*ccc)(CallCinematicCamera::CallForUpdateKeyframeKeeper)) return;

    // Update parameter
    if (this->viewModeParam.IsDirty()) {
        this->currentViewMode =static_cast<ViewMode>(this->viewModeParam.Param<param::EnumParam>()->Value());
        this->viewModeParam.ResetDirty();
    }

    // Preview the currently selected key frame.
    if (this->currentViewMode == VIEWMODE_SELECTION) {

        // set camera parameters of selected keyframe for this view
        if (ccc->getKeyframes()->Count() > 0) {
            Keyframe *s = ccc->getSelectedKeyframe();
            vislib::SmartPtr<vislib::graphics::CameraParameters> p = s->getCamParameters();
            this->cam.Parameters()->SetView(p->Position(), p->LookAt(), p->Up());
            this->cam.Parameters()->SetApertureAngle(p->ApertureAngle());
        }

        Base::Render(context);
           
	} 
    else { // this->currentViewMode == VIEWMODE_ANIMATION

        if (ccc->getKeyframes()->Count() > 0) {

            // Select the keyframe based on the current animation time.
            ccc->setSelectedKeyframeTime(static_cast<float>(context.Time));
            ccc->setChangedSelectedKeyframeTime(true);
            // Update data in cinematic camera call
            if (!(*ccc)(CallCinematicCamera::CallForUpdateKeyframeKeeper)) return;
    
            // Set camera parameters of this view based on selected keyframe 
            Keyframe *k = ccc->getSelectedKeyframe();
            vislib::SmartPtr<vislib::graphics::CameraParameters> p = k->getCamParameters();
            this->cam.Parameters()->SetView(p->Position(), p->LookAt(), p->Up());
            this->cam.Parameters()->SetApertureAngle(p->ApertureAngle());
        }

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

        // ...
		Base::Render(context);

		// reset cam (in case some skybox side was rendered)
		//if (side != SKYBOX_NONE) {
		//	cp->SetView(camPos, camPos + camFront * tmpDist, camUp);
		//}
	}
}