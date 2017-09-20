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
#include <ctime>

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
    changeKeyframeParam(           "02 Change selected keyframe", "Changes selected keyframe at the currently selected time."),
    deleteSelectedKeyframeParam(   "03 Delete selected keyframe", "Deletes the currently selected keyframe."),
    setTotalAnimTimeParam(         "04 Total time", "The total timespan of the animation."),
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
    selectedKeyframe(), dragDropKeyframe()
    {

	// setting up callback
	this->cinematicCallSlot.SetCallback(CallCinematicCamera::ClassName(),
		CallCinematicCamera::FunctionName(CallCinematicCamera::CallForGetUpdatedKeyframeData), &KeyframeKeeper::CallForGetUpdatedKeyframeData);

	this->cinematicCallSlot.SetCallback(CallCinematicCamera::ClassName(),
		CallCinematicCamera::FunctionName(CallCinematicCamera::CallForSetSimulationData), &KeyframeKeeper::CallForSetSimulationData);

	this->cinematicCallSlot.SetCallback(CallCinematicCamera::ClassName(),
		CallCinematicCamera::FunctionName(CallCinematicCamera::CallForGetInterpolCamPositions), &KeyframeKeeper::CallForGetInterpolCamPositions);

	this->cinematicCallSlot.SetCallback(CallCinematicCamera::ClassName(),
		CallCinematicCamera::FunctionName(CallCinematicCamera::CallForSetSelectedKeyframe), &KeyframeKeeper::CallForSetSelectedKeyframe);

    this->cinematicCallSlot.SetCallback(CallCinematicCamera::ClassName(),
        CallCinematicCamera::FunctionName(CallCinematicCamera::CallForGetSelectedKeyframeAtTime), &KeyframeKeeper::CallForGetSelectedKeyframeAtTime);

	this->cinematicCallSlot.SetCallback(CallCinematicCamera::ClassName(),
		CallCinematicCamera::FunctionName(CallCinematicCamera::CallForSetCameraForKeyframe), &KeyframeKeeper::CallForSetCameraForKeyframe);

    this->cinematicCallSlot.SetCallback(CallCinematicCamera::ClassName(),
        CallCinematicCamera::FunctionName(CallCinematicCamera::CallForSetDragKeyframe), &KeyframeKeeper::CallForSetDragKeyframe);

    this->cinematicCallSlot.SetCallback(CallCinematicCamera::ClassName(),
        CallCinematicCamera::FunctionName(CallCinematicCamera::CallForSetDropKeyframe), &KeyframeKeeper::CallForSetDropKeyframe);

	this->MakeSlotAvailable(&this->cinematicCallSlot);

    // init variables
    this->keyframes.Clear();
    this->interpolCamPos.Clear();
    this->boundingBox.SetNull();
    this->totalAnimTime        = 1.0f;
    this->interpolSteps        = 10;
    this->totalSimTime         = 1.0f;
    this->bboxCenter           = vislib::math::Point<float, 3>(0.0f, 0.0f, 0.0f); 
    this->filename             = "keyframe_storage.kf";
    this->camViewUp            = vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f);
    this->camViewPosition      = vislib::math::Point<float, 3>(1.0f, 0.0f, 0.0f);
    this->camViewLookat        = vislib::math::Point<float, 3>(0.0f, 0.0f, 0.0f);
    this->camViewApertureangle = 30.0f;


    // init parameters
    this->addKeyframeParam.SetParameter(new param::ButtonParam('a'));
    this->MakeSlotAvailable(&this->addKeyframeParam);

    this->changeKeyframeParam.SetParameter(new param::ButtonParam('c'));
    this->MakeSlotAvailable(&this->changeKeyframeParam);

    this->deleteSelectedKeyframeParam.SetParameter(new param::ButtonParam('d'));
    this->MakeSlotAvailable(&this->deleteSelectedKeyframeParam);

    this->editCurrentTimeParam.SetParameter(new param::FloatParam(this->selectedKeyframe.getAnimTime(), 0.0f));
    this->MakeSlotAvailable(&this->editCurrentTimeParam);

    this->setKeyframesToSameSpeed.SetParameter(new param::ButtonParam('v'));
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

    this->setTotalAnimTimeParam.SetParameter(new param::FloatParam(this->totalAnimTime, 0.0f));
    this->MakeSlotAvailable(&this->setTotalAnimTimeParam);

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
bool KeyframeKeeper::CallForSetSimulationData(core::Call& c) {

	CallCinematicCamera *ccc = dynamic_cast<CallCinematicCamera*>(&c);
	if (ccc == NULL) return false;

    // Get max simulation time
    this->totalSimTime = ccc->getTotalSimTime();
    if (this->totalAnimTime < this->totalSimTime) {
        this->totalAnimTime = this->totalSimTime;
        // Put new value of changed total animation time to parameter
        this->setTotalAnimTimeParam.Param<param::FloatParam>()->SetValue(this->totalAnimTime, false);
    }

    // Get bounding box center
    this->bboxCenter = ccc->getBboxCenter();

	return true;
}


/*
* KeyframeKeeper::CallForSetInterpolCamPositions
*/
bool KeyframeKeeper::CallForGetInterpolCamPositions(core::Call& c) {

	CallCinematicCamera *ccc = dynamic_cast<CallCinematicCamera*>(&c);
	if (ccc == NULL) return false;

    this->interpolSteps = ccc->getInterpolationSteps();
    this->refreshInterpolCamPos(this->interpolSteps);
    ccc->setInterpolCamPositions(&this->interpolCamPos);

	return true;
}


/*
* KeyframeKeeper::CallForSetSelectedKeyframe
*/
bool KeyframeKeeper::CallForSetSelectedKeyframe(core::Call& c) {

    CallCinematicCamera *ccc = dynamic_cast<CallCinematicCamera*>(&c);
    if (ccc == NULL) return false;

    // Apply changes of camera parameters only to existing keyframe
    if (this->changeKeyframe(ccc->getSelectedKeyframe())) {
        // Update selected keyframe
        this->selectedKeyframe = ccc->getSelectedKeyframe();
        this->updateEditParameters(this->selectedKeyframe, false);
    }
    else {
        this->selectedKeyframe = this->interpolateKeyframe(ccc->getSelectedKeyframe().getAnimTime());
        this->updateEditParameters(this->selectedKeyframe, false);
        ccc->setSelectedKeyframe(this->selectedKeyframe);
        vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [CallForSetSelectedKeyframe] Selected keyframe doesn't exist. Changes are omitted.");
    }

    return true;
}


/*
* KeyframeKeeper::CallForGetSelectedKeyframeAtTime
*/
bool KeyframeKeeper::CallForGetSelectedKeyframeAtTime(core::Call& c) {

	CallCinematicCamera *ccc = dynamic_cast<CallCinematicCamera*>(&c);
	if (ccc == NULL) return false;

    // Update selected keyframe
    this->selectedKeyframe = this->interpolateKeyframe(ccc->getSelectedKeyframe().getAnimTime());
    this->updateEditParameters(this->selectedKeyframe, false);
    ccc->setSelectedKeyframe(this->selectedKeyframe);

	return true;
}


/*
* KeyframeKeeper::CallForSetCameraForKeyframe
*/
bool KeyframeKeeper::CallForSetCameraForKeyframe(core::Call& c) {

	CallCinematicCamera *ccc = dynamic_cast<CallCinematicCamera*>(&c);
	if (ccc == NULL) return false;

    this->camViewUp            = ccc->getCameraParameters()->Up();
    this->camViewPosition      = ccc->getCameraParameters()->Position();
    this->camViewLookat        = ccc->getCameraParameters()->LookAt();
    this->camViewApertureangle = ccc->getCameraParameters()->ApertureAngle();

	return true;
}


/*
* KeyframeKeeper::CallForSetDragKeyframe
*/
bool KeyframeKeeper::CallForSetDragKeyframe(core::Call& c) {

    CallCinematicCamera *ccc = dynamic_cast<CallCinematicCamera*>(&c);
    if (ccc == NULL) return false;

    // Checking if selected keyframe exists in keyframe array is done by caller

    // Update selected keyframe
    Keyframe skf = ccc->getSelectedKeyframe();

    this->selectedKeyframe = this->interpolateKeyframe(skf.getAnimTime());
    this->updateEditParameters(this->selectedKeyframe, false);

    // Save currently selected keyframe as drag and drop keyframe
    this->dragDropKeyframe = this->selectedKeyframe;

    return true;
}


/*
* KeyframeKeeper::CallForSetDropKeyframe
*/
bool KeyframeKeeper::CallForSetDropKeyframe(core::Call& c) {

    CallCinematicCamera *ccc = dynamic_cast<CallCinematicCamera*>(&c);
    if (ccc == NULL) return false;

    // Delete selected keyframe from keyframe array
    this->deleteKeyframe(this->selectedKeyframe);

    // Insert dragged keyframe at new position
    this->dragDropKeyframe.setAnimTime(ccc->getDropTime());
    if (!this->addKeyframe(this->dragDropKeyframe)) {
        this->changeKeyframe(this->dragDropKeyframe);
    }
    // Set new slected keyframe
    this->selectedKeyframe = this->dragDropKeyframe;
    this->updateEditParameters(this->selectedKeyframe, false);

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
        this->selectedKeyframe.setCameraUp(this->camViewUp);
        this->selectedKeyframe.setCameraPosition(this->camViewPosition);
        this->selectedKeyframe.setCameraLookAt(this->camViewLookat);
        this->selectedKeyframe.setCameraApertureAngele(this->camViewApertureangle);

        // Add keyframe to array
        if(!this->addKeyframe(this->selectedKeyframe)) {
            // Choose new time
            if (this->keyframes.Count() > 0) {
                float t = this->keyframes.Last().getAnimTime() + 0.1f*this->totalAnimTime;
                t = (t < 0.0f) ? (0.0f) : (t);
                t = (t > this->totalAnimTime) ? (this->totalAnimTime) : (t);

                this->selectedKeyframe.setAnimTime(t);
                this->addKeyframe(this->selectedKeyframe);
            }
        }
        this->updateEditParameters(this->selectedKeyframe, false);
    }

    // changeKeyframeParam -------------------------------------------------------
    if (this->changeKeyframeParam.IsDirty()) {
        this->changeKeyframeParam.ResetDirty();

        // Get current camera for selected keyframe
        Keyframe tmpKf;
        tmpKf.setCameraUp(this->camViewUp);
        tmpKf.setCameraPosition(this->camViewPosition);
        tmpKf.setCameraLookAt(this->camViewLookat);
        tmpKf.setCameraApertureAngele(this->camViewApertureangle);
        tmpKf.setAnimTime(this->selectedKeyframe.getAnimTime());

        // change existing keyframe
        if (this->changeKeyframe(tmpKf)) {
            this->selectedKeyframe = tmpKf;
            this->updateEditParameters(this->selectedKeyframe, false);
        }
    }

    // deleteSelectedKeyframeParam --------------------------------------------
    if (this->deleteSelectedKeyframeParam.IsDirty()) {
        this->deleteSelectedKeyframeParam.ResetDirty();

        this->deleteKeyframe(this->selectedKeyframe);
        // Update to new selected keyframe
        this->selectedKeyframe = this->interpolateKeyframe(this->selectedKeyframe.getAnimTime());
        this->updateEditParameters(this->selectedKeyframe, false);
    }

    // setTotalAnimTimeParam ------------------------------------------------------
    if (this->setTotalAnimTimeParam.IsDirty()) {
        this->setTotalAnimTimeParam.ResetDirty();

        float tt = this->setTotalAnimTimeParam.Param<param::FloatParam>()->Value();
        if (!this->keyframes.IsEmpty()) {
            if (tt < this->keyframes.Last().getAnimTime()) {
                tt = this->keyframes.Last().getAnimTime();
                this->setTotalAnimTimeParam.Param<param::FloatParam>()->SetValue(tt, false);
                vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Set Total Time ] Total time is smaller than time of last keyframe. Delete Keyframe(s) to reduce total time to desired value.");
            }
        }
        this->totalAnimTime = tt;
    }

    // setKeyframesToSameSpeed ------------------------------------------------
    if (this->setKeyframesToSameSpeed.IsDirty()) {
        this->setKeyframesToSameSpeed.ResetDirty();

        this->setSameSpeed();
    }

    // editCurrentTimeParam ---------------------------------------------------
    if (this->editCurrentTimeParam.IsDirty()) {
        this->editCurrentTimeParam.ResetDirty();

        // Clamp time value to allowed min max
        float t = this->editCurrentTimeParam.Param<param::FloatParam>()->Value();
        t = (t < 0.0f) ? (0.0f) : (t);
        t = (t > this->totalAnimTime) ? (this->totalAnimTime) : (t);

        // Get index of existing keyframe
        int selIndex = static_cast<int>(this->keyframes.IndexOf(this->selectedKeyframe));
        if (selIndex >= 0) { // If existing keyframe is selected, delete keyframe an add at the right position
            this->deleteKeyframe(this->selectedKeyframe);
            this->selectedKeyframe.setAnimTime(t);
            if (!this->changeKeyframe(this->selectedKeyframe)) {
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
        vislib::math::Point<float, 3>  pos = vislib::math::Point<float, 3>(posV.X(), posV.Y(), posV.Z());

        // Get index of existing keyframe
        int selIndex = static_cast<int>(this->keyframes.IndexOf(this->selectedKeyframe));
        if (selIndex >= 0) {
            this->selectedKeyframe.setCameraPosition(pos);
            this->keyframes[selIndex] = this->selectedKeyframe;
            // Refresh interoplated camera positions
            this->refreshInterpolCamPos(this->interpolSteps);
        }
        else {
            vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Edit Current Pos] No existing keyframe selected.");
        }
    }

    // editCurrentLookAtParam -------------------------------------------------
    if (this->editCurrentLookAtParam.IsDirty()) {
        this->editCurrentLookAtParam.ResetDirty();

        vislib::math::Vector<float, 3> lookatV = this->editCurrentLookAtParam.Param<param::Vector3fParam>()->Value();
        vislib::math::Point<float, 3>  lookat = vislib::math::Point<float, 3>(lookatV.X(), lookatV.Y(), lookatV.Z());

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

        this->filename = static_cast<vislib::StringA>(this->fileNameParam.Param<param::FilePathParam>()->Value());
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
    ccc->setTotalAnimTime(this->totalAnimTime);
    ccc->setInterpolCamPositions(&this->interpolCamPos);
    ccc->setTotalSimTime(this->totalSimTime);

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
        float totTime = this->keyframes.Last().getAnimTime() - this->keyframes.First().getAnimTime();
        float totDist = 0.0f;

        for (unsigned int i = 0; i < this->interpolCamPos.Count() - 1; i++) {
            totDist += (this->interpolCamPos[i + 1] - this->interpolCamPos[i]).Length();
        }
        float totalVelocity = totDist / totTime; // unit doesn't matter ... it is only relative

        // Get values between two consecutive keyframes and shift remoter keyframe if necessary
        float kfTime = 0.0f;
        float kfDist = 0.0f;
        for (unsigned int i = 0; i < this->interpolCamPos.Count() - 2; i++) {
            if ((i > 0) && (i % this->interpolSteps == 0)) {  // skip checking for first keyframe (last keyframe is skipped by prior loop)
                kfTime = kfDist / totalVelocity;

                unsigned int index = static_cast<unsigned int>(floorf(((float)i / (float)this->interpolSteps)));
                this->keyframes[index].setAnimTime(this->keyframes[index - 1].getAnimTime() + kfTime);

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
    Keyframe kf;
    if (this->keyframes.Count() > 1) {
        for (unsigned int i = 0; i < this->keyframes.Count() - 1; i++) {
            startTime = this->keyframes[i].getAnimTime();
            deltaTimeStep = (this->keyframes[i + 1].getAnimTime() - startTime) / (float)s;

            for (unsigned int j = 0; j < s; j++) {
                kf = this->interpolateKeyframe(startTime + deltaTimeStep*(float)j);
                this->interpolCamPos.Add(kf.getCamPosition());
                this->boundingBox.GrowToPoint(this->interpolCamPos.Last());
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

        // No changes for bounding box necesary

        // Refresh interoplated camera positions
        this->refreshInterpolCamPos(this->interpolSteps);
    }
    else {
        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Delete Keyframe] No existing keyframe selected.");
        return false;
    }
    return true;
}


/*
* KeyframeKeeper::changeKeyframe
*/
bool KeyframeKeeper::changeKeyframe(Keyframe kf) {

    float time = kf.getAnimTime();

    // Check if keyframe already exists and override it
    for (unsigned int i = 0; i < this->keyframes.Count(); i++) {
        if (this->keyframes[i].getAnimTime() == time) {
            // change keyframe
            this->keyframes[i] = kf;
            // Updating Bounding Box
            // Extend camera position for bounding box to cover manipulator axis
            vislib::math::Vector<float, 3> manipulator = vislib::math::Vector<float, 3>(kf.getCamLookAt().X(), kf.getCamLookAt().Y(), kf.getCamLookAt().Z());
            manipulator = kf.getCamPosition() - manipulator;
            manipulator.ScaleToLength(1.5f);
            this->boundingBox.GrowToPoint(static_cast<vislib::math::Point<float, 3> >(kf.getCamPosition() + manipulator));
            // Refresh interoplated camera positions
            this->refreshInterpolCamPos(this->interpolSteps);
            //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [change Keyframe] Replacing existing keyframe.");
            return true;
        }
    }
    vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [change Keyframe] Found no keyframe to change.");
    return false;
}


/*
* KeyframeKeeper::addKeyframe
*/
bool KeyframeKeeper::addKeyframe(Keyframe kf) {

    float time = kf.getAnimTime();

    // Check if keyframe already exists and override it
    for (unsigned int i = 0; i < this->keyframes.Count(); i++) {
        if (this->keyframes[i].getAnimTime() == time) {
            //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Add Keyframe] Keyframe already exists.");
            return false;
        }
    }

    // Sort new keyframe to keyframe array
    if (this->keyframes.IsEmpty() || (this->keyframes.Last().getAnimTime() <= time)) {
        this->keyframes.Add(kf);
    }
    else if (this->keyframes.First().getAnimTime() >= time) {
        this->keyframes.Prepend(kf);
    }
    else { // Insert keyframe in-between existing keyframes
        unsigned int insertIdx = 0;
        for (unsigned int i = 0; i < this->keyframes.Count(); i++) {
            if (time < this->keyframes[i].getAnimTime()) {
                insertIdx = i;
                break;
            }
        }
        this->keyframes.Insert(insertIdx, kf);
    }
    // Update bounding box
    // Extend camera position for bounding box to cover manipulator axis
    vislib::math::Vector<float, 3> manipulator = vislib::math::Vector<float, 3>(kf.getCamLookAt().X(), kf.getCamLookAt().Y(), kf.getCamLookAt().Z());
    manipulator = kf.getCamPosition() - manipulator;
    manipulator.ScaleToLength(1.5f);
    this->boundingBox.GrowToPoint(static_cast<vislib::math::Point<float, 3> >(kf.getCamPosition() + manipulator));
    // Refresh interoplated camera positions
    this->refreshInterpolCamPos(this->interpolSteps);

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
    if (t > this->totalAnimTime) {
        t = this->totalAnimTime;
    }

    // Check if there is an existing keyframe at requested time
    for (int i = 0; i < this->keyframes.Count(); i++) {
        if (t == this->keyframes[i].getAnimTime()) {
            return this->keyframes[i];
        }
    }

    if (this->keyframes.IsEmpty()) {
        // vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Interpolate Keyframe] Empty keyframe array.");
        Keyframe kf = Keyframe(t);
        kf.setCameraUp(this->camViewUp);
        kf.setCameraPosition(this->camViewPosition);
        kf.setCameraLookAt(this->camViewLookat);
        kf.setCameraApertureAngele(this->camViewApertureangle);
        return kf;
    }
    else if (t <= this->keyframes.First().getAnimTime()) {
        Keyframe kf = this->keyframes.First();
        kf.setAnimTime(t);
        return kf;
    }
    else if (t >= this->keyframes.Last().getAnimTime()) {
        Keyframe kf = this->keyframes.Last();
        kf.setAnimTime(t);
        return kf;
    }
    else { // if ((t > this->keyframes.First().getAnimTime()) && (t < this->keyframes.Last().getAnimTime())) {

        // new default keyframe
        Keyframe kf = Keyframe(t);

        // determine indices for interpolation 
        int i0 = 0;
        int i1 = 0;
        int i2 = 0;
        int i3 = 0;
        int kfIdxCnt = (int)keyframes.Count()-1;
        float iT = 0.0f;
        for (int i = 0; i < kfIdxCnt; i++) {
            float tMin = this->keyframes[i].getAnimTime();
            float tMax = this->keyframes[i + 1].getAnimTime();
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
        kf.setCameraPosition(Point<float, 3>(pk.X(), pk.Y(), pk.Z()));

        //interpolate lookAt
        vislib::math::Vector<float, 3> l0(keyframes[i0].getCamLookAt());
        vislib::math::Vector<float, 3> l1(keyframes[i1].getCamLookAt());
        vislib::math::Vector<float, 3> l2(keyframes[i2].getCamLookAt());
        vislib::math::Vector<float, 3> l3(keyframes[i3].getCamLookAt());

        vislib::math::Vector<float, 3> lk = (((l1 * 2) +
            (l2 - l0) * iT +
            (l0 * 2 - l1 * 5 + l2 * 4 - l3) * iT * iT +
            (-l0 + l1 * 3 - l2 * 3 + l3) * iT * iT * iT) * 0.5);
        kf.setCameraLookAt(Point<float, 3>(lk.X(), lk.Y(), lk.Z()));

        //interpolate up
        vislib::math::Vector<float, 3> u0 = p0 + keyframes[i0].getCamUp();
        vislib::math::Vector<float, 3> u1 = p1 + keyframes[i1].getCamUp();
        vislib::math::Vector<float, 3> u2 = p2 + keyframes[i2].getCamUp();
        vislib::math::Vector<float, 3> u3 = p3 + keyframes[i3].getCamUp();

        vislib::math::Vector<float, 3> uk = (((u1 * 2) +
            (u2 - u0) * iT +
            (u0 * 2 - u1 * 5 + u2 * 4 - u3) * iT * iT +
            (-u0 + u1 * 3 - u2 * 3 + u3) * iT * iT * iT) * 0.5);
        kf.setCameraUp(uk - pk);

        //interpolate aperture angle
        float a0 = keyframes[i0].getCamApertureAngle();
        float a1 = keyframes[i1].getCamApertureAngle();
        float a2 = keyframes[i2].getCamApertureAngle();
        float a3 = keyframes[i3].getCamApertureAngle();

        a0 = (((a1 * 2) +
            (a2 - a0) * iT +
            (a0 * 2 - a1 * 5 + a2 * 4 - a3) * iT * iT +
            (-a0 + a1 * 3 - a2 * 3 + a3) * iT * iT * iT) * 0.5f);

        kf.setCameraApertureAngele(a0);

        return kf;
    }
}


/*
* KeyframeKeeper::saveKeyframes
*/
void KeyframeKeeper::saveKeyframes() {

    if (this->filename.IsEmpty()) {
        vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [Save Keyframes] No filename given. Using default filename.");

        time_t t = std::time(0);  // get time now
        struct tm * now = std::localtime(&t);
        this->filename.Format("keyframe_storage_%i%i%i-%i%i%i.kf", (now->tm_year + 1900), (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
        this->fileNameParam.Param<param::FilePathParam>()->SetValue(this->filename, false);
    } 

    std::ofstream outfile;
    outfile.open(this->filename.PeekBuffer(), std::ios::binary);
    vislib::StringSerialiserA ser;
    outfile << "totalAnimTime=" << this->totalAnimTime << "\n\n";
    for (unsigned int i = 0; i < this->keyframes.Count(); i++) {
        this->keyframes[i].serialise(ser);
        outfile << ser.GetString().PeekBuffer() << "\n";
    }
    outfile.close();

    vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Save Keyframes] Successfully stored keyframes to file: %s", this->filename.PeekBuffer());

}


/*
* KeyframeKeeper::loadKeyframes
*/
void KeyframeKeeper::loadKeyframes() {

    if (this->filename.IsEmpty()) {
        vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [Load Keyframes] No filename given.");
    }
    else {
        
        std::ifstream infile;
        infile.open(this->filename.PeekBuffer());
        if (!infile.is_open()) {
            vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [Load Keyframes] Failed to open keyframe file.");
            return;
        }

        // Reset keyframe array and bounding box
        this->keyframes.Clear();
        this->keyframes.AssertCapacity(1000);
        this->boundingBox.SetNull();

        vislib::StringSerialiserA ser;
        std::string               line;
        vislib::StringA           cameraStr = "";;
        float                     time      = 0.0f;

        // get total time
        std::getline(infile, line); 
        this->totalAnimTime = std::stof(line.erase(0, 14)); // "totalAnimTime="
        this->setTotalAnimTimeParam.Param<param::FloatParam>()->SetValue(this->totalAnimTime, false);
        // Consume empty line
        std::getline(infile, line); 

        // One frame consists of an initial "time"-line followed by the serialized camera parameters and an final empty line
        while (std::getline(infile, line)) {
            if ((line.empty()) && !(cameraStr.IsEmpty())) { // new empty line indicates current frame is complete
                ser.SetInputString(cameraStr);
                Keyframe kf;
                kf.deserialise(ser);
                this->keyframes.Add(kf);
                // Extend camera position for bounding box to cover manipulator axis
                vislib::math::Vector<float, 3> manipulator = vislib::math::Vector<float, 3>(kf.getCamLookAt().X(), kf.getCamLookAt().Y(), kf.getCamLookAt().Z());
                manipulator = kf.getCamPosition() - manipulator;
                manipulator.ScaleToLength(1.5f);
                this->boundingBox.GrowToPoint(static_cast<vislib::math::Point<float, 3> >(kf.getCamPosition() + manipulator));
                cameraStr.Clear();
                ser.ClearData();
            }
            else {
                cameraStr.Append(line.c_str());
                cameraStr.Append("\n");
            }
        }
        infile.close();

        if (!this->keyframes.IsEmpty()) {
            // Set selected keyframe to first in keyframe array
            this->selectedKeyframe = this->keyframes.First();
            // Put new values of changes selected keyframe to parameters (and update parameters to apply new selected keyframe to all calls)
            this->updateEditParameters(this->selectedKeyframe, true);
            // Refresh interoplated camera positions
            this->refreshInterpolCamPos(this->interpolSteps);
        }
        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Load Keyframes] Successfully loaded keyframes from file: %s", this->filename.PeekBuffer());
    }
}


/*
* KeyframeKeeper::updateEditParameters
*/
void KeyframeKeeper::updateEditParameters(Keyframe kf, bool setDirty) {

    // Put new values of changes selected keyframe to parameters
    vislib::math::Vector<float, 3> lookatV = kf.getCamLookAt();
    vislib::math::Point<float, 3>  lookat = vislib::math::Point<float, 3>(lookatV.X(), lookatV.Y(), lookatV.Z());
    vislib::math::Vector<float, 3> posV = kf.getCamPosition();
    vislib::math::Point<float, 3>  pos = vislib::math::Point<float, 3>(posV.X(), posV.Y(), posV.Z());
    this->editCurrentTimeParam.Param<param::FloatParam>()->SetValue(kf.getAnimTime(), setDirty);
    this->editCurrentPosParam.Param<param::Vector3fParam>()->SetValue(pos, setDirty);
    this->editCurrentLookAtParam.Param<param::Vector3fParam>()->SetValue(lookat, setDirty);
    this->editCurrentUpParam.Param<param::Vector3fParam>()->SetValue(kf.getCamUp(), setDirty);
    this->editCurrentApertureParam.Param<param::FloatParam>()->SetValue(kf.getCamApertureAngle(), setDirty);

}