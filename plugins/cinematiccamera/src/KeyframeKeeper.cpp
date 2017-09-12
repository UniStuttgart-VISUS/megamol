/*
* KeyframeKeeper.cpp
*
*/


#include "stdafx.h"

#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/AbstractSlot.h"
#include "mmcore/utility/xml/XmlParser.h"
#include "mmcore/utility/xml/XmlReader.h"

#include "vislib/math/Point.h"
#include "vislib/math/Vector.h"
#include "vislib/StringSerialiser.h"
#include "vislib/assert.h"

#include <fstream>

#include "KeyframeKeeper.h"
#include "CallCinematicCamera.h"


using namespace megamol;
using namespace megamol::cinematiccamera;
using namespace vislib;
using namespace vislib::math;
using namespace megamol::core;


/*
* KeyframeKeeper::KeyframeKeeper
*/
KeyframeKeeper::KeyframeKeeper(void) : core::Module(),
    cinematicCallSlot("scene3D", "holds keyframe data"),
    addKeyframeParam(              "01 Add new keyframe", "Adds new keyframe at the currently selected time."),
    replaceKeyframeParam(          "02 Replace selected keyframe", "Replaces selected keyframe at the currently selected time."),
    deleteSelectedKeyframeParam(   "03 Delete selected keyframe", "Deletes the currently selected keyframe."),
    setTotalTimeParam(             "04 Total time", "The total timespan of the animation."),
    setKeyframesToSameSpeed(       "05 Set same speed", "Move keyframes to get same speed between all keyframes."),

    editCurrentTimeParam(          "Edit Selection::01 Time", "Edit time of the selected keyframe."),
    editCurrentPosParam(           "Edit Selection::02 Position", "Edit  position vector of the selected keyframe."),
    editCurrentLookAtParam(        "Edit Selection::03 LookAt", "Edit LookAt vector of the selected keyframe."),
    resetLookAtParam(              "Edit Selection::04 Reset LookAt", "Reset the LookAt vector of the selected keyframe."),
    editCurrentUpParam(            "Edit Selection::05 UP", "Edit Up vector of the selected keyframe."),
    editCurrentApertureParam(      "Edit Selection::06 Aperture", "Edit apperture angle of the selected keyframe."),

    fileNameParam(                 "Storage::01 Filename", "The name of the file to load or save keyframes."),
    saveKeyframesParam(            "Storage::02 Save keyframes", "Save keyframes to file."),
    loadKeyframesParam(            "Storage::03 (Auto) Load keyframes", "Load keyframes from file when filename changes."),
    selectedKeyframe(), dragDropKeyframe(), cameraParam(NULL)
    {

	// setting up callback
	this->cinematicCallSlot.SetCallback(CallCinematicCamera::ClassName(),
		CallCinematicCamera::FunctionName(CallCinematicCamera::CallForGetUpdatedKeyframeData), &KeyframeKeeper::CallForGetUpdatedKeyframeData);

	this->cinematicCallSlot.SetCallback(CallCinematicCamera::ClassName(),
		CallCinematicCamera::FunctionName(CallCinematicCamera::CallForSetAnimationData), &KeyframeKeeper::CallForSetAnimationData);

	this->cinematicCallSlot.SetCallback(CallCinematicCamera::ClassName(),
		CallCinematicCamera::FunctionName(CallCinematicCamera::CallForInterpolatedCamPos), &KeyframeKeeper::CallForInterpolatedCamPos);

	this->cinematicCallSlot.SetCallback(CallCinematicCamera::ClassName(),
		CallCinematicCamera::FunctionName(CallCinematicCamera::CallForSetSelectedKeyframe), &KeyframeKeeper::CallForSetSelectedKeyframe);

	this->cinematicCallSlot.SetCallback(CallCinematicCamera::ClassName(),
		CallCinematicCamera::FunctionName(CallCinematicCamera::CallForSetCameraForKeyframe), &KeyframeKeeper::CallForSetCameraForKeyframe);

    this->cinematicCallSlot.SetCallback(CallCinematicCamera::ClassName(),
        CallCinematicCamera::FunctionName(CallCinematicCamera::CallForDragKeyframe), &KeyframeKeeper::CallForDragKeyframe);

    this->cinematicCallSlot.SetCallback(CallCinematicCamera::ClassName(),
        CallCinematicCamera::FunctionName(CallCinematicCamera::CallForDropKeyframe), &KeyframeKeeper::CallForDropKeyframe);

	this->MakeSlotAvailable(&this->cinematicCallSlot);


    // init variables
    this->keyframes.Clear();
    this->keyframes.AssertCapacity(1000);

    this->interpolCamPos.Clear();
    this->interpolCamPos.AssertCapacity(1000);

    this->boundingBox.SetNull();

    this->totalTime            = 1.0f;
    this->filename             = "";
    this->interpolSteps        = 10;
    this->sameSpeed            = false;
    this->totVelocity          = 1.0f;
    this->maxAnimTime          = 1.0f;
    this->bboxCenter           = vislib::math::Point<float, 3>(0.0f, 0.0f, 0.0f); 

    // init parameters
    this->addKeyframeParam.SetParameter(new param::ButtonParam('a'));
    this->MakeSlotAvailable(&this->addKeyframeParam);

    this->replaceKeyframeParam.SetParameter(new param::ButtonParam('r'));
    this->MakeSlotAvailable(&this->replaceKeyframeParam);

    this->deleteSelectedKeyframeParam.SetParameter(new param::ButtonParam('d'));
    this->MakeSlotAvailable(&this->deleteSelectedKeyframeParam);

    this->editCurrentTimeParam.SetParameter(new param::FloatParam(this->selectedKeyframe.getTime(), 0.0f));
    this->MakeSlotAvailable(&this->editCurrentTimeParam);

    this->setKeyframesToSameSpeed.SetParameter(new param::BoolParam(this->sameSpeed));
    this->MakeSlotAvailable(&this->setKeyframesToSameSpeed);

    this->editCurrentPosParam.SetParameter(new param::Vector3fParam(this->selectedKeyframe.getCamPosition()));
    this->MakeSlotAvailable(&this->editCurrentPosParam);

    this->editCurrentLookAtParam.SetParameter(new param::Vector3fParam(this->selectedKeyframe.getCamLookAt()));
    this->MakeSlotAvailable(&this->editCurrentLookAtParam);

    this->resetLookAtParam.SetParameter(new param::ButtonParam('l'));
    this->MakeSlotAvailable(&this->resetLookAtParam);
    
    this->editCurrentUpParam.SetParameter(new param::Vector3fParam(this->selectedKeyframe.getCamUp()));
    this->MakeSlotAvailable(&this->editCurrentUpParam);

    this->editCurrentApertureParam.SetParameter(new param::FloatParam(this->selectedKeyframe.getCamApertureAngle(), 0.0f, 180.0f));
    this->MakeSlotAvailable(&this->editCurrentApertureParam);

    this->setTotalTimeParam.SetParameter(new param::FloatParam(this->totalTime, 0.0f));
    this->MakeSlotAvailable(&this->setTotalTimeParam);

    this->fileNameParam.SetParameter(new param::FilePathParam(this->filename));
    this->MakeSlotAvailable(&this->fileNameParam);

	this->saveKeyframesParam.SetParameter(new param::ButtonParam('s'));
	this->MakeSlotAvailable(&this->saveKeyframesParam);

	this->loadKeyframesParam.SetParameter(new param::BoolParam(true));
	this->MakeSlotAvailable(&this->loadKeyframesParam);
}


/*
* KeyframeKeeper::~KeyframeKeeper
*/
KeyframeKeeper::~KeyframeKeeper(void) {

	this->Release();
}


/*
* KeyframeKeeper::create(void)
*/
bool KeyframeKeeper::create(void) {

	return true;
}


/*
* KeyframeKeeper::release(void)
*/
void KeyframeKeeper::release(void) {

	// intentionally empty
}


/*
* KeyframeKeeper::CallForSetAnimationData
*/
bool KeyframeKeeper::CallForSetAnimationData(core::Call& c) {

	CallCinematicCamera *ccc = dynamic_cast<CallCinematicCamera*>(&c);
	if (ccc == NULL) return false;

    // Get max anim time
    this->maxAnimTime = ccc->getMaxAnimTime();
    if (this->totalTime < this->maxAnimTime) {
        this->totalTime = this->maxAnimTime;
        // Put new value of changed total time to parameter
        this->setTotalTimeParam.Param<param::FloatParam>()->SetValue(this->totalTime, false);
    }

    // Get bounding box center
    this->bboxCenter = ccc->getBboxCenter();

	return true;
}


/*
* KeyframeKeeper::CallForRequestInterpolatedKeyframe
*/
bool KeyframeKeeper::CallForInterpolatedCamPos(core::Call& c) {

	CallCinematicCamera *ccc = dynamic_cast<CallCinematicCamera*>(&c);
	if (ccc == NULL) return false;

    this->interpolSteps = ccc->getInterpolationSteps();
    this->refreshInterpolCamPos(this->interpolSteps);

	return true;
}


/*
* KeyframeKeeper::CallForSetSelectedKeyframe
*/
bool KeyframeKeeper::CallForSetSelectedKeyframe(core::Call& c) {

	CallCinematicCamera *ccc = dynamic_cast<CallCinematicCamera*>(&c);
	if (ccc == NULL) return false;

    // Update selected keyframe
    Keyframe s = ccc->getSelectedKeyframe();

    if (this->sameSpeed) {

////////
// TODO: Set truly! same speed between to consecutive keyframes
////////

        this->selectedKeyframe = this->interpolateKeyframe(s.getTime());
    }
    else {
        this->selectedKeyframe = this->interpolateKeyframe(s.getTime());
    }

    // Apply changes to existing keyframe
    int selIndex = static_cast<int>(this->keyframes.IndexOf(this->selectedKeyframe));
    if (selIndex >= 0) {
        if (this->replaceKeyframe(this->selectedKeyframe)) {
            this->refreshInterpolCamPos(this->interpolSteps);
            if (this->sameSpeed) {
                this->setSameSpeed();
            }
        }
    }
    this->updateEditParameters(this->selectedKeyframe);

	return true;
}


/*
* KeyframeKeeper::CallForSetCameraForKeyframe
*/
bool KeyframeKeeper::CallForSetCameraForKeyframe(core::Call& c) {

	CallCinematicCamera *ccc = dynamic_cast<CallCinematicCamera*>(&c);
	if (ccc == NULL) return false;

    this->cameraParam = ccc->getCameraParameter();

	return true;
}


/*
* KeyframeKeeper::CallForGrabDragDropKeyframe
*/
bool KeyframeKeeper::CallForDragKeyframe(core::Call& c) {

    CallCinematicCamera *ccc = dynamic_cast<CallCinematicCamera*>(&c);
    if (ccc == NULL) return false;

    // Checking if selected keyframe exists in keyframe array is done by caller

    // Update selected keyframe
    Keyframe s = ccc->getSelectedKeyframe();

    this->selectedKeyframe = this->interpolateKeyframe(s.getTime());
    this->updateEditParameters(this->selectedKeyframe);

    // Save currently selected keyframe as drag and drop keyframe
    this->dragDropKeyframe = this->selectedKeyframe;

    return true;
}


/*
* KeyframeKeeper::CallForReleaseDragDropKeyframe
*/
bool KeyframeKeeper::CallForDropKeyframe(core::Call& c) {

    CallCinematicCamera *ccc = dynamic_cast<CallCinematicCamera*>(&c);
    if (ccc == NULL) return false;

    // Delete selected keyframe from keyframe array
    this->deleteKeyframe(this->selectedKeyframe);

    // Insert dragged keyframe at new position
    this->dragDropKeyframe.setTime(ccc->getDropTime());
    if (!this->addKeyframe(this->dragDropKeyframe)) {
        this->replaceKeyframe(this->dragDropKeyframe);
    }
    // Set new slected keyframe
    this->selectedKeyframe = this->dragDropKeyframe;
    this->updateEditParameters(this->selectedKeyframe);
    this->refreshInterpolCamPos(this->interpolSteps);
    if (this->sameSpeed) {
        this->setSameSpeed();
    }

    return true;
}


/*
* KeyframeKeeper::CallForGetUpdatedKeyframeData
*/
bool KeyframeKeeper::CallForGetUpdatedKeyframeData(core::Call& c) {

	CallCinematicCamera *ccc = dynamic_cast<CallCinematicCamera*>(&c);
	if (ccc == NULL) return false;


    // UPDATE PARAMETERS

    // addKeyframeParam -------------------------------------------------------
    if (this->addKeyframeParam.IsDirty()) {
        this->addKeyframeParam.ResetDirty();

        // Get current camera for selected keyframe
        // Creating a hard copy of camera parameters is necessary
        vislib::graphics::Camera c;
        if (!this->cameraParam.IsNull()) {
            c.Parameters()->SetPosition(this->cameraParam->Position());
            c.Parameters()->SetLookAt(this->cameraParam->LookAt());
            c.Parameters()->SetUp(this->cameraParam->Up());
            c.Parameters()->SetApertureAngle(this->cameraParam->ApertureAngle());
        }
        this->selectedKeyframe.setCamera(c);

        // Add keyframe to array
        if(!this->addKeyframe(this->selectedKeyframe)) {
            // Choose new time
            if (this->keyframes.Count() > 0) {
                float t = this->keyframes.Last().getTime() + 0.1f*this->totalTime;
                t = (t < 0.0f) ? (0.0f) : (t);
                t = (t > this->totalTime) ? (this->totalTime) : (t);

                this->selectedKeyframe.setTime(t);
                this->addKeyframe(this->selectedKeyframe);
            }
        }
        this->refreshInterpolCamPos(this->interpolSteps);
        if (this->sameSpeed) {
            this->setSameSpeed();
        }
        this->updateEditParameters(this->selectedKeyframe);
    }

    // replaceKeyframeParam -------------------------------------------------------
    if (this->replaceKeyframeParam.IsDirty()) {
        this->replaceKeyframeParam.ResetDirty();

        // Get current camera for selected keyframe
        // Creating a hard copy of camera parameters is necessary
        vislib::graphics::Camera c;
        if (!this->cameraParam.IsNull()) {
            c.Parameters()->SetPosition(this->cameraParam->Position());
            c.Parameters()->SetLookAt(this->cameraParam->LookAt());
            c.Parameters()->SetUp(this->cameraParam->Up());
            c.Parameters()->SetApertureAngle(this->cameraParam->ApertureAngle());
        }
        this->selectedKeyframe.setCamera(c);

        // Replace existing keyframe
        if (this->replaceKeyframe(this->selectedKeyframe)) {
            // Refresh interoplated camera positions
            this->refreshInterpolCamPos(this->interpolSteps);
            if (this->sameSpeed) {
                this->setSameSpeed();
            }
        }
        this->updateEditParameters(this->selectedKeyframe);
    }

    // deleteSelectedKeyframeParam --------------------------------------------
    if (this->deleteSelectedKeyframeParam.IsDirty()) {
        this->deleteSelectedKeyframeParam.ResetDirty();

        this->deleteKeyframe(this->selectedKeyframe);
        this->selectedKeyframe = this->interpolateKeyframe(this->selectedKeyframe.getTime());

        // Refresh interoplated camera positions
        this->refreshInterpolCamPos(this->interpolSteps);
        if (this->sameSpeed) {
            this->setSameSpeed();
        }
    }

    // setTotalTimeParam ------------------------------------------------------
    if (this->setTotalTimeParam.IsDirty()) {
        this->setTotalTimeParam.ResetDirty();

        float tt = this->setTotalTimeParam.Param<param::FloatParam>()->Value();
        if (!this->keyframes.IsEmpty()) {
            if (tt < this->keyframes.Last().getTime()) {
                tt = this->keyframes.Last().getTime();
                this->setTotalTimeParam.Param<param::FloatParam>()->SetValue(tt, false);
                vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Set Total Time ] Total time is smaller than time of last keyframe. Delete Keyframe(s) to reduce total time to desired value.");
            }
        }
        this->totalTime = tt;
    }

    // setKeyframesToSameSpeed ------------------------------------------------
    if (this->setKeyframesToSameSpeed.IsDirty()) {
        this->setKeyframesToSameSpeed.ResetDirty();
        this->sameSpeed = this->setKeyframesToSameSpeed.Param<param::BoolParam>()->Value();
        if (this->sameSpeed) {
            this->setSameSpeed();
        }
    }

    // editCurrentTimeParam ---------------------------------------------------
    if (this->editCurrentTimeParam.IsDirty()) {
        this->editCurrentTimeParam.ResetDirty();

        // Clamp time value to allowed min max
        float t = this->editCurrentTimeParam.Param<param::FloatParam>()->Value();
        t = (t < 0.0f) ? (0.0f) : (t);
        t = (t > this->totalTime) ? (this->totalTime) : (t);

        // Get index of existing keyframe
        int selIndex = static_cast<int>(this->keyframes.IndexOf(this->selectedKeyframe));
        if (selIndex >= 0) { // If existing keyframe is selected, delete keyframe an add at the right position
            this->deleteKeyframe(this->selectedKeyframe);
            this->selectedKeyframe.setTime(t);
            if (!this->replaceKeyframe(this->selectedKeyframe)) {
                this->addKeyframe(this->selectedKeyframe); 
            }
        }
        else { // Else just change time of interpolated selected keyframe
            this->selectedKeyframe = this->interpolateKeyframe(t);
        }

        // Write back clamped total time to parameter
        this->editCurrentTimeParam.Param<param::FloatParam>()->SetValue(t, false);

    }

    // editCurrentPosParam ----------------------------------------------------
    if (this->editCurrentPosParam.IsDirty()) {
        this->editCurrentPosParam.ResetDirty();

        vislib::math::Vector<float, 3> posV = this->editCurrentPosParam.Param<param::Vector3fParam>()->Value();
        vislib::math::Point<float, 3>  pos = vislib::math::Point<float, 3>(posV.GetX(), posV.GetY(), posV.GetZ());

        // Get index of existing keyframe
        int selIndex = static_cast<int>(this->keyframes.IndexOf(this->selectedKeyframe));
        if (selIndex >= 0) {
            this->selectedKeyframe.setCameraPosition(pos);
            this->keyframes[selIndex] = this->selectedKeyframe;
            // Refresh interoplated camera positions
            this->refreshInterpolCamPos(this->interpolSteps);
            if (this->sameSpeed) {
                this->setSameSpeed();
            }
        }
        else {
            vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Edit Current Pos] No existing keyframe selected.");
        }
    }

    // editCurrentLookAtParam -------------------------------------------------
    if (this->editCurrentLookAtParam.IsDirty()) {
        this->editCurrentLookAtParam.ResetDirty();

        vislib::math::Vector<float, 3> lookatV = this->editCurrentLookAtParam.Param<param::Vector3fParam>()->Value();
        vislib::math::Point<float, 3>  lookat = vislib::math::Point<float, 3>(lookatV.GetX(), lookatV.GetY(), lookatV.GetZ());

        // Get index of existing keyframe
        int selIndex = static_cast<int>(this->keyframes.IndexOf(this->selectedKeyframe));
        if (selIndex >= 0) {
            this->selectedKeyframe.setCameraLookAt(lookat);
            this->keyframes[selIndex] = this->selectedKeyframe;
        }
        else {
            vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Edit Current LookAt] No existing keyframe selected.");
        }
    }

    // resetLookAtParam -------------------------------------------------
    if (this->resetLookAtParam.IsDirty()) {
        this->resetLookAtParam.ResetDirty();

        this->editCurrentLookAtParam.Param<param::Vector3fParam>()->SetValue(this->bboxCenter);
        // Get index of existing keyframe
        int selIndex = static_cast<int>(this->keyframes.IndexOf(this->selectedKeyframe));
        if (selIndex >= 0) {
            this->selectedKeyframe.setCameraLookAt(this->bboxCenter);
            this->keyframes[selIndex] = this->selectedKeyframe;
        }
        else {
            vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Edit Current LookAt] No existing keyframe selected.");
        }
    }
    
    // editCurrentUpParam -----------------------------------------------------
    if (this->editCurrentUpParam.IsDirty()) {
        this->editCurrentUpParam.ResetDirty();

        vislib::math::Vector<float, 3> up = this->editCurrentUpParam.Param<param::Vector3fParam>()->Value();

        // Get index of existing keyframe
        int selIndex = static_cast<int>(this->keyframes.IndexOf(this->selectedKeyframe));
        if (selIndex >= 0) {
            this->selectedKeyframe.setCameraUp(up);
            this->keyframes[selIndex] = this->selectedKeyframe;
        }
        else {
            vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Edit Current Up] No existing keyframe selected.");
        }
    }

    // editCurrentUpParam -----------------------------------------------------
    if (this->editCurrentApertureParam.IsDirty()) {
        this->editCurrentApertureParam.ResetDirty();

        float aperture = this->editCurrentApertureParam.Param<param::FloatParam>()->Value();

        // Get index of existing keyframe
        int selIndex = static_cast<int>(this->keyframes.IndexOf(this->selectedKeyframe));
        if (selIndex >= 0) {
            this->selectedKeyframe.setCameraApertureAngele(aperture);
            this->keyframes[selIndex] = this->selectedKeyframe;
        }
        else {
            vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Edit Current Aperture] No existing keyframe selected.");
        }
    }

    // fileNameParam ----------------------------------------------------------
    if (this->fileNameParam.IsDirty()) {
        this->fileNameParam.ResetDirty();

        this->filename = this->fileNameParam.Param<param::FilePathParam>()->Value();
        // Auto loading keyframe file when new filename is given
        this->loadKeyframesParam.ForceSetDirty();
    }

    // saveKeyframesParam -----------------------------------------------------
    if (this->saveKeyframesParam.IsDirty()) {
        this->saveKeyframesParam.ResetDirty();

        this->saveKeyframes();
    }

    // loadKeyframesParam -----------------------------------------------------
    if (this->loadKeyframesParam.IsDirty()) {
        this->loadKeyframesParam.ResetDirty();

        if (this->loadKeyframesParam.Param<param::BoolParam>()->Value()) {
            this->loadKeyframes();
        }
    }

    // PROPAGATE CURRENT DATA TO CALL -----------------------------------------
    ccc->setKeyframes(&this->keyframes);
    ccc->setBoundingBox(&this->boundingBox);
    ccc->setSelectedKeyframe(this->selectedKeyframe);
    ccc->setTotalTime(this->totalTime);
    ccc->setInterpolatedCamPos(&this->interpolCamPos);
    ccc->setMaxAnimTime(this->maxAnimTime);

    return true;
}

/*
* KeyframeKeeper::setSameSpeed
*/
void KeyframeKeeper::setSameSpeed() {

    if (this->keyframes.Count() > 2) {

        // Store index of selected keyframe to restore seleection after changing time of keyframes
        int selIndex = static_cast<int>(this->keyframes.IndexOf(this->selectedKeyframe));

        // Get total values
        float totTime = this->keyframes.Last().getTime() - this->keyframes.First().getTime();
        float totDist = 0.0f;

        for (unsigned int i = 0; i < this->interpolCamPos.Count() - 1; i++) {
            totDist += (this->interpolCamPos[i + 1] - this->interpolCamPos[i]).Length();
        }
        this->totVelocity = totDist / totTime; // unit doesn't matter ... it is only relative

        // Get values between two consecutive keyframes and shift remoter keyframe if necessary
        float kfTime = 0.0f;
        float kfDist = 0.0f;
        for (unsigned int i = 0; i < this->interpolCamPos.Count() - 2; i++) {
            if ((i > 0) && (i % this->interpolSteps == 0)) {  // skip checking for first keyframe (last keyframe is skipped by prior loop)
                kfTime = kfDist / this->totVelocity;

                unsigned int index = static_cast<unsigned int>(floorf(((float)i / (float)this->interpolSteps)));
                this->keyframes[index].setTime(this->keyframes[index - 1].getTime() + kfTime);

                kfDist = 0.0f;
            }
            // Add distance up to existing keyframe
            kfDist += (this->interpolCamPos[i + 1] - this->interpolCamPos[i]).Length();
        }

        // Restore previous selected keyframe
        if (selIndex >= 0) {
            this->selectedKeyframe = this->keyframes[selIndex];
        }
    }
}


/*
* KeyframeKeeper::refreshInterpolCamPos
*/
void KeyframeKeeper::refreshInterpolCamPos(unsigned int s) {

    this->interpolCamPos.Clear();
    this->interpolCamPos.AssertCapacity(1000);

    float startTime;
    float deltaTimeStep;
    Keyframe k;
    vislib::math::Vector<float, 3> manipulator;
    if (this->keyframes.Count() > 1) {
        for (unsigned int i = 0; i < this->keyframes.Count() - 1; i++) {
            startTime = this->keyframes[i].getTime();
            deltaTimeStep = (this->keyframes[i + 1].getTime() - startTime) / (float)s;

            for (unsigned int j = 0; j < s; j++) {
                k = this->interpolateKeyframe(startTime + deltaTimeStep*(float)j);
                this->interpolCamPos.Add(k.getCamPosition());
                // Extend camera position for bounding box to cover manipulator axis
                manipulator = vislib::math::Vector<float, 3>(k.getCamPosition().GetX(), k.getCamPosition().GetY(), k.getCamPosition().GetZ());
                manipulator.ScaleToLength(1.0f);
                this->boundingBox.GrowToPoint(static_cast<vislib::math::Point<float, 3>>(k.getCamPosition() + manipulator));
            }
        }
        // Add last existing camera position
        this->interpolCamPos.Add(this->keyframes.Last().getCamPosition());
    }
}


/*
* KeyframeKeeper::deleteKeyframe
*/
bool KeyframeKeeper::deleteKeyframe(Keyframe kf) {

    // Get index of keyframe to delete
    int selIndex = static_cast<int>(this->keyframes.IndexOf(kf));

    // Choose new selected keyframe
    if (selIndex >= 0) {
        // Remove keyframe from keyframe array
        this->keyframes.RemoveAt(selIndex);

        // Reset bounding box
        this->boundingBox.SetNull();
        for (unsigned int i = 0; i < this->keyframes.Count(); i++) {
            this->boundingBox.GrowToPoint(this->keyframes[i].getCamPosition());
        }
    }
    else {
        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Delete Keyframe] No existing keyframe selected.");
        return false;
    }
    return true;
}


/*
* KeyframeKeeper::replaceKeyframe
*/
bool KeyframeKeeper::replaceKeyframe(Keyframe kf) {

    float time = kf.getTime();

    // Check if keyframe already exists and override it
    for (unsigned int i = 0; i < this->keyframes.Count(); i++) {
        if (this->keyframes[i].getTime() == time) {
            // Replace keyframe
            this->keyframes[i] = kf;
            // Updating Bounding Box
            this->boundingBox.GrowToPoint(kf.getCamPosition());
            //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Replace Keyframe] Replacing existing keyframe.");
            return true;
        }
    }
    vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Replace Keyframe] Found no keyframe to replace.");
    return false;
}


/*
* KeyframeKeeper::addKeyframe
*/
bool KeyframeKeeper::addKeyframe(Keyframe kf) {

    float time = kf.getTime();

    // Check if keyframe already exists and override it
    for (unsigned int i = 0; i < this->keyframes.Count(); i++) {
        if (this->keyframes[i].getTime() == time) {
            vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Add Keyframe] Keyframe already exists.");
            return false;
        }
    }

    // Sort new keyframe to keyframe array
    if (this->keyframes.IsEmpty() || (this->keyframes.Last().getTime() <= time)) {
        this->keyframes.Add(kf);
    }
    else if (this->keyframes.First().getTime() >= time) {
        this->keyframes.Prepend(kf);
    }
    else { // Insert keyframe in-between existing keyframes
        unsigned int insertIdx = 0;
        for (unsigned int i = 0; i < this->keyframes.Count(); i++) {
            if (time < this->keyframes[i].getTime()) {
                insertIdx = i;
                break;
            }
        }
        this->keyframes.Insert(insertIdx, kf);
    }
    this->boundingBox.GrowToPoint(kf.getCamPosition());

    return true;
}


/*
* KeyframeKeeper::interpolateKeyframe
*/
Keyframe KeyframeKeeper::interpolateKeyframe(float time) {

    float t = time;

    if (t < 0.0f) {
        t = 0.0f;
    }
    if (t > this->totalTime) {
        t = this->totalTime;
    }

    // Check if there is an existing keyframe at requested time
    for (int i = 0; i < this->keyframes.Count(); i++) {
        if (t == this->keyframes[i].getTime()) {
            return this->keyframes[i];
        }
    }

    if (this->keyframes.IsEmpty()) {
        // vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Interpolate Keyframe] Empty keyframe array.");
        Keyframe k = Keyframe(vislib::graphics::Camera(), t);
        if (!this->cameraParam.IsNull()) {
            k.setCameraParameters(this->cameraParam);
        }
        return k;
    }
    else if (t <= this->keyframes.First().getTime()) {
        return Keyframe(this->keyframes.First().getCamera(), t);
    }
    else if (t >= this->keyframes.Last().getTime()) {
        return Keyframe(this->keyframes.Last().getCamera(), t);
    }
    else { // if ((t > this->keyframes.First().getTime()) && (t < this->keyframes.Last().getTime())) {

        // new default keyframe
        Keyframe k = Keyframe(vislib::graphics::Camera(), t);

        // determine indices for interpolation 
        int i0 = 0;
        int i1 = 0;
        int i2 = 0;
        int i3 = 0;
        int kfIdxCnt = (int)keyframes.Count()-1;
        float iT = 0.0f;
        for (int i = 0; i < kfIdxCnt; i++) {
            float tMin = this->keyframes[i].getTime();
            float tMax = this->keyframes[i + 1].getTime();
            if ((tMin< t) && (t < tMax)) {
                iT = (t - tMin) / (tMax - tMin); // Map current time to [0,1] between two keyframes
                i1 = i;
                i2 = i + 1;
                break;
            }
        }
        i0 = (i1 > 0) ? (i1 - 1) : (0);
        i3 = (i2 < kfIdxCnt) ? (i2 + 1) : (kfIdxCnt);

        //interpolate position
        vislib::math::Vector<float, 3> p0(keyframes[i0].getCamPosition());
        vislib::math::Vector<float, 3> p1(keyframes[i1].getCamPosition());
        vislib::math::Vector<float, 3> p2(keyframes[i2].getCamPosition());
        vislib::math::Vector<float, 3> p3(keyframes[i3].getCamPosition());

        vislib::math::Vector<float, 3> pk = (((p1 * 2.0f) +
            (p2 - p0) * iT +
            (p0 * 2 - p1 * 5 + p2 * 4 - p3) * iT * iT +
            (-p0 + p1 * 3 - p2 * 3 + p3) * iT * iT * iT) * 0.5);
        k.setCameraPosition(Point<float, 3>(pk.GetX(), pk.GetY(), pk.GetZ()));

        //interpolate lookAt
        vislib::math::Vector<float, 3> l0(keyframes[i0].getCamLookAt());
        vislib::math::Vector<float, 3> l1(keyframes[i1].getCamLookAt());
        vislib::math::Vector<float, 3> l2(keyframes[i2].getCamLookAt());
        vislib::math::Vector<float, 3> l3(keyframes[i3].getCamLookAt());

        vislib::math::Vector<float, 3> lk = (((l1 * 2) +
            (l2 - l0) * iT +
            (l0 * 2 - l1 * 5 + l2 * 4 - l3) * iT * iT +
            (-l0 + l1 * 3 - l2 * 3 + l3) * iT * iT * iT) * 0.5);
        k.setCameraLookAt(Point<float, 3>(lk.GetX(), lk.GetY(), lk.GetZ()));

        //interpolate up
        vislib::math::Vector<float, 3> u0 = p0 + keyframes[i0].getCamUp();
        vislib::math::Vector<float, 3> u1 = p1 + keyframes[i1].getCamUp();
        vislib::math::Vector<float, 3> u2 = p2 + keyframes[i2].getCamUp();
        vislib::math::Vector<float, 3> u3 = p3 + keyframes[i3].getCamUp();

        vislib::math::Vector<float, 3> uk = (((u1 * 2) +
            (u2 - u0) * iT +
            (u0 * 2 - u1 * 5 + u2 * 4 - u3) * iT * iT +
            (-u0 + u1 * 3 - u2 * 3 + u3) * iT * iT * iT) * 0.5);
        k.setCameraUp(uk - pk);

        //interpolate aperture angle
        float a0 = keyframes[i0].getCamApertureAngle();
        float a1 = keyframes[i1].getCamApertureAngle();
        float a2 = keyframes[i2].getCamApertureAngle();
        float a3 = keyframes[i3].getCamApertureAngle();

        a0 = (((a1 * 2) +
            (a2 - a0) * iT +
            (a0 * 2 - a1 * 5 + a2 * 4 - a3) * iT * iT +
            (-a0 + a1 * 3 - a2 * 3 + a3) * iT * iT * iT) * 0.5f);

        k.setCameraApertureAngele(a0);

        return k;
    }
}


/*
* KeyframeKeeper::saveKeyframes
*/
void KeyframeKeeper::saveKeyframes() {

    if (this->filename.IsEmpty()) {
        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Save Keyframes] No filename given.");
    } 
    else {

        std::ofstream outfile;
        outfile.open(this->filename, std::ios::binary);
        vislib::StringSerialiserA ser;
        outfile << "totalTime=" << this->totalTime << "\n\n";
        for (unsigned int i = 0; i < this->keyframes.Count(); i++) {
            outfile << "time=" << keyframes[i].getTime() << "\n";
            this->keyframes[i].getCamParameters()->Serialise(ser);
            outfile << ser.GetString().PeekBuffer() << "\n";
        }
        outfile.close();

        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Save Keyframes] Successfully stored keyframes to file.");
    }
}


/*
* KeyframeKeeper::loadKeyframes
*/
void KeyframeKeeper::loadKeyframes() {


    if (this->filename.IsEmpty()) {
        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Load Keyframes] No filename given.");
    }
    else {
        
        // Reset keyframe array and bounding box
        this->keyframes.Clear();
        this->keyframes.AssertCapacity(1000);
        this->boundingBox.SetNull();

        std::ifstream infile;
        infile.open(this->filename);
        if (!infile.is_open()) {
            vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Load Keyframes] Failed to open keyframe file.");
            return;
        }

        vislib::StringSerialiserA ser;
        std::string               line;
        vislib::StringA           cameraStr = "";;
        float                     time      = 0.0f;

        // get total time
        std::getline(infile, line); 
        this->totalTime = std::stof(line.erase(0, 10)); // "totalTime="
        this->setTotalTimeParam.Param<param::FloatParam>()->SetValue(this->totalTime, false);
        // Consume empty line
        std::getline(infile, line); 

        // One frame consists of an initial "time"-line followed by the serialized camera parameters and an final empty line
        while (std::getline(infile, line)) {
            if (line.substr(0, 5) == "time=") { // read time for new frame
                time = static_cast<float>(std::stof(line.erase(0, 5)));
            }
            else if (line.empty()) { // new empty line indicates current frame is complete
                ser.SetInputString(cameraStr);
                Keyframe k;
                k.setTime(time);
                k.getCamParameters()->Deserialise(ser);
                this->keyframes.Add(k);
                cameraStr.Clear();
                ser.ClearData();
            }
            else {
                cameraStr.Append(line.c_str());
                cameraStr.Append("\n");
            }
        }
        infile.close();

        // Set selected keyframe to first in keyframe array
        if (!this->keyframes.IsEmpty()) {
            this->selectedKeyframe = this->keyframes.First();

            // Put new values of changes selected keyframe to parameters (and update parameters to apply new selected keyframe to all calls)
            vislib::math::Vector<float, 3> lookatV = this->selectedKeyframe.getCamLookAt();
            vislib::math::Point<float, 3>  lookat = vislib::math::Point<float, 3>(lookatV.GetX(), lookatV.GetY(), lookatV.GetZ());
            vislib::math::Vector<float, 3> posV = this->selectedKeyframe.getCamPosition();
            vislib::math::Point<float, 3>  pos = vislib::math::Point<float, 3>(posV.GetX(), posV.GetY(), posV.GetZ());
            this->editCurrentTimeParam.Param<param::FloatParam>()->SetValue(this->selectedKeyframe.getTime(), true);
            this->editCurrentPosParam.Param<param::Vector3fParam>()->SetValue(pos, true);
            this->editCurrentLookAtParam.Param<param::Vector3fParam>()->SetValue(lookat, true);
            this->editCurrentUpParam.Param<param::Vector3fParam>()->SetValue(this->selectedKeyframe.getCamUp(), true);
            this->editCurrentApertureParam.Param<param::FloatParam>()->SetValue(this->selectedKeyframe.getCamApertureAngle(), true);

            this->refreshInterpolCamPos(this->interpolSteps);
            if (this->sameSpeed) {
                this->setSameSpeed();
            }
        }
        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Load Keyframes] Successfully loaded keyframes from file.");
    }
}


/*
* KeyframeKeeper::updateEditParameters
*/
void KeyframeKeeper::updateEditParameters(Keyframe k) {

    // Put new values of changes selected keyframe to parameters
    vislib::math::Vector<float, 3> lookatV = k.getCamLookAt();
    vislib::math::Point<float, 3>  lookat = vislib::math::Point<float, 3>(lookatV.GetX(), lookatV.GetY(), lookatV.GetZ());
    vislib::math::Vector<float, 3> posV = k.getCamPosition();
    vislib::math::Point<float, 3>  pos = vislib::math::Point<float, 3>(posV.GetX(), posV.GetY(), posV.GetZ());
    this->editCurrentTimeParam.Param<param::FloatParam>()->SetValue(k.getTime(), false);
    this->editCurrentPosParam.Param<param::Vector3fParam>()->SetValue(pos, false);
    this->editCurrentLookAtParam.Param<param::Vector3fParam>()->SetValue(lookat, false);
    this->editCurrentUpParam.Param<param::Vector3fParam>()->SetValue(k.getCamUp(), false);
    this->editCurrentApertureParam.Param<param::FloatParam>()->SetValue(k.getCamApertureAngle(), false);

}