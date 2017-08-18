/**
* CinematicView.cpp
*/

#include "stdafx.h"
#include "CinematicView.h"
#include "CallCinematicCamera.h"
#include "CameraParamsOverride.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/view/CallRender3D.h"

using namespace megamol;
using namespace megamol::core;
using namespace cinematiccamera;

CinematicView::CinematicView(void) : View3D(),
		keyframeKeeperSlot("keyframeKeeper", "Connects to the Keyframe Keeper."),
		selectedKeyframeParam("cinematicCam::selectedKeyframeParam", "render selected Keyframe if true. render keyframe at animationtime if false"),
		selectedSkyboxSideParam("cinematicCam::skyboxSide", "Skybox side rendering"),
		autoLoadKeyframes("keyframeAutoLoad","shall the keyframes be loaded automatically from the file provided in the keyframe-keeper?"),
		autoSetTotalTime("totalTimeAutoSet", "shall the total animation time be determined automatically?"),
        paramEdit("cinematicCam::edit", "Do not overwrite the camera such that the selected key frame can be edited."),
		firstframe(true) {


	this->keyframeKeeperSlot.SetCompatibleCall<CallCinematicCameraDescription>();
	this->MakeSlotAvailable(&this->keyframeKeeperSlot);

	this->selectedKeyframeParam << new core::param::BoolParam(true);
	this->MakeSlotAvailable(&this->selectedKeyframeParam);

	this->autoLoadKeyframes.SetParameter(new core::param::BoolParam(false));
	this->MakeSlotAvailable(&this->autoLoadKeyframes);

	this->autoSetTotalTime.SetParameter(new core::param::BoolParam(true));
	this->MakeSlotAvailable(&this->autoSetTotalTime);

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

    this->paramEdit << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->paramEdit);
}


megamol::cinematiccamera::CinematicView::~CinematicView() {}

void megamol::cinematiccamera::CinematicView::Render(const mmcRenderViewContext& context) {
	auto cr3d = this->rendererSlot.CallAs<core::view::CallRender3D>();
	(*cr3d)(1); // get extents

	auto kfc = this->keyframeKeeperSlot.CallAs<CallCinematicCamera>();
	if (!kfc) {
		vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "KeyframeKeeper slot not connected!");
	}

	

	if (firstframe) {
		if (autoLoadKeyframes.Param<param::BoolParam>()->Value()) {
			(*kfc)(CallCinematicCamera::CallForLoadKeyframe);
		}
		if (autoSetTotalTime.Param<param::BoolParam>()->Value()) {
			kfc->setTotalTime(static_cast<float>(cr3d->TimeFramesCount()));
			(*kfc)(CallCinematicCamera::CallForSetTotalTime);
		}
		firstframe = false;
	}


    bool isSelectedKeyFrame = this->selectedKeyframeParam.Param<core::param::BoolParam>()->Value();
    if (isSelectedKeyFrame) {
        // Preview the currently selected key frame.

        if ((*kfc)(CallCinematicCamera::CallForGetSelectedKeyframe)) {
            auto id = kfc->getSelectedKeyframe().getID();
            bool isEdit = this->paramEdit.Param<core::param::BoolParam>()->Value();

            // Note: It is important that we do that before interpolation.
            if (this->paramEdit.IsDirty()) {
                if (!isEdit && isSelectedKeyFrame && (id >= 0)) {
                    vislib::sys::Log::DefaultLog.WriteInfo("Updating key frame %d ...", id);
                    // Note: It is important to create a deep copy here
                    kfc->setCameraForNewKeyframe(vislib::SmartPtr<vislib::graphics::CameraParameters>(
                        new vislib::graphics::CameraParamsStore(*this->cam.Parameters())));
                    if ((*kfc)(CallCinematicCamera::CallForKeyFrameUpdate)) {
                        vislib::sys::Log::DefaultLog.WriteInfo("Key frame %d was updated.", id);
                    }
                }
                this->paramEdit.ResetDirty();
            }

            switch (id) {
                case -2:
                    // There is no key frame yet.
                    break;

                case -1:
                    // An interpolated position was selected.
                    kfc->getInterpolatedKeyframe().putCamParameters(this->cam.Parameters());
                    break;

                default:
                    // A key frame was selected.
                    if (!isEdit) {
                        kfc->getSelectedKeyframe().putCamParameters(this->cam.Parameters());
                    }
                    break;
            }

            Base::Render(context);

        } else {
            vislib::sys::Log::DefaultLog.WriteError("CallForGetSelectedKeyframe failed!");
        }

	} else {
        // Select the key frame based on the current animation time.

		kfc->setTimeofKeyframeToGet(static_cast<float>(context.Time));
		if ((*kfc)(CallCinematicCamera::CallForGetKeyframeAtTime)){
			kfc->getInterpolatedKeyframe().putCamParameters(this->cam.Parameters());

		}

		vislib::SmartPtr<vislib::graphics::CameraParameters> cp = this->cam.Parameters();
		vislib::math::Point<float, 3> camPos = cp->Position();
		vislib::math::Vector<float, 3> camRight = cp->Right();
		vislib::math::Vector<float, 3> camUp = cp->Up();
		vislib::math::Vector<float, 3> camFront = cp->Front();
		float tmpDist = cp->FocalDistance();
		// adjust cam to selected skybox side
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

		Base::Render(context);

		// reset cam (in case some skybox side was rendered)
		//if (side != SKYBOX_NONE) {
		//	cp->SetView(camPos, camPos + camFront * tmpDist, camUp);
		//}

	}

}