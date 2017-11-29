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
    addKeyframeParam(              "01_addKeyframe", "Adds new keyframe at the currently selected time."),
    setTotalAnimTimeParam(         "02_maxAnimTime", "The total timespan of the animation."),
    setKeyframesToSameSpeed(       "03_setSameSpeed", "Move keyframes to get same speed between all keyframes."),
    snapAnimFramesParam(           "04_snapAnimFrames", "Snap animation time of all keyframes to fixed animation frames."),
    snapSimFramesParam(            "05_snapSimFrames", "Snap simulation time of all keyframes to integer simulation frames."),
    simTangentParam(               "06_straightenSimTangent", "Straighten tangent of simulation time between currently selectd keyframe and the following selected keyframe."),
    addFixedAnimTimeParam(         "07_addFixedPerCentAnimTime", "Adds fixed per cent of animation time to currently selected keyframe when new keyframe is added."),
    addFixedSimTimeParam(          "08_addFixedPerCentSimTime", "Adds fixed per cent of simulation time to currently selected keyframe when new keyframe is added."),

    deleteSelectedKeyframeParam(   "editSelected::01_deleteKeyframe", "Deletes the currently selected keyframe."),
    changeKeyframeParam(           "editSelected::02_applyView", "Apply current view to selected keyframe."),
    editCurrentAnimTimeParam(      "editSelected::03_animTime", "Edit animation time of the selected keyframe."),
    editCurrentSimTimeParam(       "editSelected::04_simTime", "Edit simulation time of the selected keyframe."),
    editCurrentPosParam(           "editSelected::05_position", "Edit  position vector of the selected keyframe."),
    editCurrentLookAtParam(        "editSelected::06_lookat", "Edit LookAt vector of the selected keyframe."),
    resetLookAtParam(              "editSelected::07_resetLookat", "Reset the LookAt vector of the selected keyframe."),
    editCurrentUpParam(            "editSelected::08_up", "Edit Up vector of the selected keyframe."),
    editCurrentApertureParam(      "editSelected::09_apertureAngle", "Edit apperture angle of the selected keyframe."),

    fileNameParam(                 "storage::01_filename", "The name of the file to load or save keyframes."),
    saveKeyframesParam(            "storage::02_save", "Save keyframes to file."),
    loadKeyframesParam(            "storage::03_autoLoad", "Load keyframes from file when filename changes."),
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
    this->fps                  = 24;
    this->totalSimTime         = 1.0f;
    this->bboxCenter           = vislib::math::Point<float, 3>(0.0f, 0.0f, 0.0f); 
    this->filename             = "keyframes.kf";
    this->camViewUp            = vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f);
    this->camViewPosition      = vislib::math::Point<float, 3>(1.0f, 0.0f, 0.0f);
    this->camViewLookat        = vislib::math::Point<float, 3>(0.0f, 0.0f, 0.0f);
    this->camViewApertureangle = 30.0f;
    this->simTangentStatus     = false;

    // init parameters
    this->addKeyframeParam.SetParameter(new param::ButtonParam('a'));
    this->MakeSlotAvailable(&this->addKeyframeParam);

    this->changeKeyframeParam.SetParameter(new param::ButtonParam('c'));
    this->MakeSlotAvailable(&this->changeKeyframeParam);

    this->deleteSelectedKeyframeParam.SetParameter(new param::ButtonParam('d'));
    this->MakeSlotAvailable(&this->deleteSelectedKeyframeParam);

    this->editCurrentAnimTimeParam.SetParameter(new param::FloatParam(this->selectedKeyframe.getAnimTime(), 0.0f));
    this->MakeSlotAvailable(&this->editCurrentAnimTimeParam);

    this->editCurrentSimTimeParam.SetParameter(new param::FloatParam(this->selectedKeyframe.getSimTime()*this->totalSimTime, 0.0f));
    this->MakeSlotAvailable(&this->editCurrentSimTimeParam);

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

    this->setTotalAnimTimeParam.SetParameter(new param::FloatParam(this->totalAnimTime, 1.0f));
    this->MakeSlotAvailable(&this->setTotalAnimTimeParam);

    this->fileNameParam.SetParameter(new param::FilePathParam(this->filename));
    this->MakeSlotAvailable(&this->fileNameParam);

	this->saveKeyframesParam.SetParameter(new param::ButtonParam('s'));
	this->MakeSlotAvailable(&this->saveKeyframesParam);

	this->loadKeyframesParam.SetParameter(new param::BoolParam(true));
	this->MakeSlotAvailable(&this->loadKeyframesParam);
    this->loadKeyframesParam.ForceSetDirty(); 

    this->snapAnimFramesParam.SetParameter(new param::ButtonParam('f'));
    this->MakeSlotAvailable(&this->snapAnimFramesParam);

    this->snapSimFramesParam.SetParameter(new param::ButtonParam('g'));
    this->MakeSlotAvailable(&this->snapSimFramesParam);

    this->simTangentParam.SetParameter(new param::ButtonParam('t'));
    this->MakeSlotAvailable(&this->simTangentParam);

    this->addFixedAnimTimeParam.SetParameter(new param::FloatParam(0.1f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->addFixedAnimTimeParam);

    this->addFixedSimTimeParam.SetParameter(new param::FloatParam(0.0f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->addFixedSimTimeParam);
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
* KeyframeKeeper::CallForSetSimulationData
*/
bool KeyframeKeeper::CallForSetSimulationData(core::Call& c) {

	CallCinematicCamera *ccc = dynamic_cast<CallCinematicCamera*>(&c);
	if (ccc == NULL) return false;

    // Get bounding box center
    this->bboxCenter = ccc->getBboxCenter();

    // Get total simulation time
    if (ccc->getTotalSimTime() != this->totalSimTime) {
        this->totalSimTime = ccc->getTotalSimTime();
        this->editCurrentSimTimeParam.Param<param::FloatParam>()->SetValue(this->selectedKeyframe.getSimTime() * this->totalSimTime, false);
    }

    // Get Frames per Second
    this->fps = ccc->getFps();

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
        //vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [CallForSetSelectedKeyframe] Selected keyframe doesn't exist. Changes are omitted.");
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

    // Straighten tangent between simTangentKf and currently selected keyframe by shifting all inbetween keyframes simulation time
    if (this->simTangentStatus) {
        if (this->keyframes.Contains(this->selectedKeyframe)) {
            if (this->keyframes.Contains(this->simTangentKf)) {
                // Calculate liner equation between the two selected keyframes 
                // f(x) = mx + b
                vislib::math::Point<float, 2> p1 = vislib::math::Point<float, 2>(this->selectedKeyframe.getAnimTime(), this->selectedKeyframe.getSimTime());
                vislib::math::Point<float, 2> p2 = vislib::math::Point<float, 2>(this->simTangentKf.getAnimTime(), this->simTangentKf.getSimTime());
                float m = (p1.Y() - p2.Y()) / (p1.X() - p2.X());
                float b = m * (-p1.X()) + p1.Y();
                // Get indices
                int iKf1 = this->keyframes.IndexOf(this->selectedKeyframe);
                int iKf2 = this->keyframes.IndexOf(this->simTangentKf);
                if (iKf1 > iKf2) {
                    int tmp = iKf1;
                    iKf1 = iKf2;
                    iKf2 = tmp;
                }
                // Consider only keyframes lying between the two selected ones
                float newSimTime;
                for (unsigned int i = iKf1 + 1; i < iKf2; i++) {
                    newSimTime = m * (this->keyframes[i].getAnimTime()) + b;
                    this->keyframes[i].setSimTime(newSimTime);
                }
            }
            else {
                vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [straighten tangent] First selectred keyframe doesn't exist any more.");
            }
            this->simTangentStatus = false;
        }
        else {
            vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [straighten tangent] Select existing keyframe to finish straightening the tangent.");
        }
    }

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
    this->dragDropKeyframe.setAnimTime(ccc->getDropAnimTime());
    this->dragDropKeyframe.setSimTime(ccc->getDropSimTime());
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
        if (!this->addKeyframe(this->selectedKeyframe)) {
            // Choose new time if the last keyframe was tried to be replaced
            if (this->keyframes.Last().getAnimTime() < this->totalAnimTime) {

                float t = this->selectedKeyframe.getAnimTime();
                t += (this->totalAnimTime * this->addFixedAnimTimeParam.Param<param::FloatParam>()->Value());
                t = (t > this->totalAnimTime) ? (this->totalAnimTime) : (t);
                this->selectedKeyframe.setAnimTime(t);

                float s = this->selectedKeyframe.getSimTime();
                s += this->addFixedSimTimeParam.Param<param::FloatParam>()->Value();
                s = (s > 1.0f) ? (1.0f) : (s);
                this->selectedKeyframe.setSimTime(s);

                this->selectedKeyframe.setAnimTime(t);
                this->addKeyframe(this->selectedKeyframe);
            }
            else {
                vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [Add Keyframe] Unable to add new keyframe. Keyframe already exists.");
            }
        }

        this->updateEditParameters(this->selectedKeyframe, false);
    }

    // changeKeyframeParam -------------------------------------------------------
    if (this->changeKeyframeParam.IsDirty()) {
        this->changeKeyframeParam.ResetDirty();

        // Get current camera for selected keyframe
        Keyframe tmpKf = this->selectedKeyframe;
        tmpKf.setCameraUp(this->camViewUp);
        tmpKf.setCameraPosition(this->camViewPosition);
        tmpKf.setCameraLookAt(this->camViewLookat);
        tmpKf.setCameraApertureAngele(this->camViewApertureangle);

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
                vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Set Total Time] Total time is smaller than time of last keyframe. Delete Keyframe(s) to reduce total time to desired value.");
            }
        }
        this->totalAnimTime = tt;
    }

    // setKeyframesToSameSpeed ------------------------------------------------
    if (this->setKeyframesToSameSpeed.IsDirty()) {
        this->setKeyframesToSameSpeed.ResetDirty();

        this->setSameSpeed();
    }

    // editCurrentAnimTimeParam ---------------------------------------------------
    if (this->editCurrentAnimTimeParam.IsDirty()) {
        this->editCurrentAnimTimeParam.ResetDirty();

        // Clamp time value to allowed min max
        float t = this->editCurrentAnimTimeParam.Param<param::FloatParam>()->Value(); 
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
        this->editCurrentAnimTimeParam.Param<param::FloatParam>()->SetValue(t, false);
    }

    // editCurrentSimTimeParam ------------------------------------------------
    if (this->editCurrentSimTimeParam.IsDirty()) {
        this->editCurrentSimTimeParam.ResetDirty();

        // Clamp time value to allowed min max
        float t = this->editCurrentSimTimeParam.Param<param::FloatParam>()->Value();
        t = vislib::math::Clamp(t, 0.0f, this->totalSimTime);

        // Get index of existing keyframe
        int selIndex = static_cast<int>(this->keyframes.IndexOf(this->selectedKeyframe));
        if (selIndex >= 0) { // If existing keyframe is selected, delete keyframe an add at the right position
            this->selectedKeyframe.setSimTime(t / this->totalSimTime);
            if (!this->changeKeyframe(this->selectedKeyframe)) {
                this->addKeyframe(this->selectedKeyframe);
            }
        }
        // Write back clamped total time to parameter
        this->editCurrentSimTimeParam.Param<param::FloatParam>()->SetValue(t, false);
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
            //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Edit Current Pos] No existing keyframe selected.");
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
            //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Edit Current LookAt] No existing keyframe selected.");
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
            //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Edit Current LookAt] No existing keyframe selected.");
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
            //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Edit Current Up] No existing keyframe selected.");
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
            //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Edit Current Aperture] No existing keyframe selected.");
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

    // snapAnimFramesParam -----------------------------------------------------
    if (this->snapAnimFramesParam.IsDirty()) {
        this->snapAnimFramesParam.ResetDirty();

        for (unsigned int i = 0; i < this->keyframes.Count(); i++) {
            this->snapKeyframe2AnimFrame(&this->keyframes[i]);
        }
        this->snapKeyframe2AnimFrame(&this->selectedKeyframe);
    }

    // snapSimFramesParam -----------------------------------------------------
    if (this->snapSimFramesParam.IsDirty()) {
        this->snapSimFramesParam.ResetDirty();

        for (unsigned int i = 0; i < this->keyframes.Count(); i++) {
            this->snapKeyframe2SimFrame(&this->keyframes[i]);
        }
        this->snapKeyframe2SimFrame(&this->selectedKeyframe);      
    }

    // simTangentParam -----------------------------------------------------
    if (this->simTangentParam.IsDirty()) {
        this->simTangentParam.ResetDirty();

        this->simTangentStatus = true;

        // Straighten tangent only between existing keyframes
        if (this->keyframes.Contains(this->selectedKeyframe)) {
            this->simTangentKf = this->selectedKeyframe;
        }
        else {
            vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [straighten tangent] Select existing keyframe before trying to straighten the tangent.");
            this->simTangentStatus = false;
        }
    }


    // PROPAGATE CURRENT DATA TO CALL -----------------------------------------
    ccc->setKeyframes(&this->keyframes);
    ccc->setBoundingBox(&this->boundingBox);
    ccc->setSelectedKeyframe(this->selectedKeyframe);
    ccc->setTotalAnimTime(this->totalAnimTime);
    ccc->setInterpolCamPositions(&this->interpolCamPos);
    ccc->setTotalSimTime(this->totalSimTime);
    ccc->setFps(this->fps);

    return true;
}


/*
* KeyframeKeeper::snapKeyframe2AnimFrame
*/
void KeyframeKeeper::snapKeyframe2AnimFrame(Keyframe *kf) {

    float snapAnimTime = kf->getAnimTime();

    if (this->fps == 0) {
        vislib::sys::Log::DefaultLog.WriteError("[KEYFRAME KEEPER] [snapKeyframe2AnimFrame] FPS is ZERO.");
        return;
    }

    float fpsFrac = 1.0f / (float)(this->fps);
    // Round to 5th position after the comma
    snapAnimTime = floorf(snapAnimTime / fpsFrac + 0.5f) * fpsFrac;

    kf->setAnimTime(snapAnimTime);
}


/*
* KeyframeKeeper::snapKeyframe2SimFrame
*/
void KeyframeKeeper::snapKeyframe2SimFrame(Keyframe *kf) {

    float snapSimTime = kf->getSimTime();
    snapSimTime = floorf(snapSimTime*this->totalSimTime + 0.5f) / this->totalSimTime;
    kf->setSimTime(snapSimTime);
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

        if (totTime == 0.0f) {
            vislib::sys::Log::DefaultLog.WriteError("[KEYFRAME KEEPER] [setSameSpeed] totTime is ZERO.");
            return;
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

    if (s == 0) {
        vislib::sys::Log::DefaultLog.WriteError("[KEYFRAME KEEPER] [refreshInterpolCamPos] Interpolation step count is ZERO.");
        return;
    }

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

        // No changes for bounding box necessary

        // Refresh interoplated camera positions
        this->refreshInterpolCamPos(this->interpolSteps);

        // Adjusting selected keyframe
        if (selIndex > 0) {
            this->selectedKeyframe = this->keyframes[selIndex - 1];
        }
        else if (selIndex < this->keyframes.Count()) {
            this->selectedKeyframe = this->keyframes[selIndex];
        }
    }
    else {
        //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Delete Keyframe] No existing keyframe selected.");
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
            this->boundingBox.GrowToPoint(kf.getCamPosition());
            // Refresh interoplated camera positions
            this->refreshInterpolCamPos(this->interpolSteps);
            //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [change Keyframe] Replacing existing keyframe.");
            return true;
        }
    }
    //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [change Keyframe] Found no keyframe to change.");
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
    if (this->keyframes.IsEmpty() || (this->keyframes.Last().getAnimTime() < time)) {
        this->keyframes.Add(kf);
    }
    else if (time < this->keyframes.First().getAnimTime()) {
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
        Keyframe kf = Keyframe();
        kf.setAnimTime(t);
        kf.setSimTime(0.0f);
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
        Keyframe kf = Keyframe();
        kf.setAnimTime(t);

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
            if ((tMin < t) && (t < tMax)) {
                iT = (t - tMin) / (tMax - tMin); // Map current time to [0,1] between two keyframes
                i1 = i;
                i2 = i + 1;
                break;
            }
        }
        i0 = (i1 > 0) ? (i1 - 1) : (0);
        i3 = (i2 < kfIdxCnt) ? (i2 + 1) : (kfIdxCnt);

        // Interpolate simulation time linear between i1 and i2
        float simT1 = this->keyframes[i1].getSimTime();
        float simT2 = this->keyframes[i2].getSimTime();
        float simT = simT1 + (simT2 - simT1)*iT;

        kf.setSimTime(simT);

        // Skip interpolation of camera parameters if p1 == p2 
        // => Prevent loops if time of keyframes is different, but postion is the same
        if (keyframes[i1].getCamPosition() == keyframes[i2].getCamPosition()) {
            kf.setCameraPosition(keyframes[i1].getCamPosition());
            kf.setCameraLookAt(keyframes[i1].getCamLookAt());
            kf.setCameraUp(keyframes[i1].getCamUp());
            kf.setCameraApertureAngele(keyframes[i1].getCamApertureAngle());
            return kf;
        }

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
        this->filename.Format("keyframes_%i%i%i-%i%i%i.kf", (now->tm_year + 1900), (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
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
        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Load Keyframes] No filename given.");
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
    this->editCurrentAnimTimeParam.Param<param::FloatParam>()->SetValue(kf.getAnimTime(), setDirty);
    this->editCurrentSimTimeParam.Param<param::FloatParam>()->SetValue(kf.getSimTime() * this->totalSimTime, setDirty);
    this->editCurrentPosParam.Param<param::Vector3fParam>()->SetValue(pos, setDirty);
    this->editCurrentLookAtParam.Param<param::Vector3fParam>()->SetValue(lookat, setDirty);
    this->editCurrentUpParam.Param<param::Vector3fParam>()->SetValue(kf.getCamUp(), setDirty);
    this->editCurrentApertureParam.Param<param::FloatParam>()->SetValue(kf.getCamApertureAngle(), setDirty);

}