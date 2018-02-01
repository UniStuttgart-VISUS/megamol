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

#include <iostream>
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
    applyKeyframeParam(            "01_applyKeyframe", "Apply current settings to selected/new keyframe."),
    undoChangesParam(              "02_undoChanges", "Undo changes."),
    redoChangesParam(              "03_redoChanges", "Redo changes."),
    deleteSelectedKeyframeParam(   "04_deleteKeyframe", "Deletes the currently selected keyframe."),
    setTotalAnimTimeParam(         "05_maxAnimTime", "The total timespan of the animation."),
    snapAnimFramesParam(           "06_snapAnimFrames", "Snap animation time of all keyframes to fixed animation frames."),
    snapSimFramesParam(            "07_snapSimFrames", "Snap simulation time of all keyframes to integer simulation frames."),
    simTangentParam(               "08_linearizeSimTime", "Linearize simulation time between two keyframes between currently selected keyframe and subsequently selected keyframe."),
    interpolTangentParam(          "09_interpolTangent", "Length of keyframe tangets affecting curvature of interpolation spline."),
    //UNUSED setKeyframesToSameSpeed("10_setSameSpeed", "Move keyframes to get same speed between all keyframes."),
    editCurrentAnimTimeParam(      "editSelected::01_animTime", "Edit animation time of the selected keyframe."),
    editCurrentSimTimeParam(       "editSelected::02_simTime", "Edit simulation time of the selected keyframe."),
    editCurrentPosParam(           "editSelected::03_position", "Edit  position vector of the selected keyframe."),
    editCurrentLookAtParam(        "editSelected::04_lookat", "Edit LookAt vector of the selected keyframe."),
    resetLookAtParam(              "editSelected::05_resetLookat", "Reset the LookAt vector of the selected keyframe."),
    editCurrentUpParam(            "editSelected::06_up", "Edit Up vector of the selected keyframe."),
    editCurrentApertureParam(      "editSelected::07_apertureAngle", "Edit apperture angle of the selected keyframe."),
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

    this->cinematicCallSlot.SetCallback(CallCinematicCamera::ClassName(),
        CallCinematicCamera::FunctionName(CallCinematicCamera::CallForSetCtrlPoints), &KeyframeKeeper::CallForSetCtrlPoints);

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
    this->firstCtrllPos = vislib::math::Point<float, 3>(0.0f, 0.0f, 0.0f);
    this->lastCtrllPos  = vislib::math::Point<float, 3>(0.0f, 0.0f, 0.0f);
    this->camViewApertureangle = 30.0f;
    this->simTangentStatus     = false;
    this->undoQueueIndex       = 0;
    this->tl                   = 0.5f;
    this->undoQueue.Clear();

    // init parameters
    this->applyKeyframeParam.SetParameter(new param::ButtonParam('a'));
    this->MakeSlotAvailable(&this->applyKeyframeParam);

    this->undoChangesParam.SetParameter(new param::ButtonParam(vislib::sys::KeyCode::KEY_MOD_CTRL + 'y')); // = z in german keyboard layout
    this->MakeSlotAvailable(&this->undoChangesParam);

    this->redoChangesParam.SetParameter(new param::ButtonParam(vislib::sys::KeyCode::KEY_MOD_CTRL + 'z')); // = y in german keyboard layout
    this->MakeSlotAvailable(&this->redoChangesParam);

    this->deleteSelectedKeyframeParam.SetParameter(new param::ButtonParam('d'));
    this->MakeSlotAvailable(&this->deleteSelectedKeyframeParam);

    this->editCurrentAnimTimeParam.SetParameter(new param::FloatParam(this->selectedKeyframe.getAnimTime(), 0.0f));
    this->MakeSlotAvailable(&this->editCurrentAnimTimeParam);

    this->editCurrentSimTimeParam.SetParameter(new param::FloatParam(this->selectedKeyframe.getSimTime()*this->totalSimTime, 0.0f));
    this->MakeSlotAvailable(&this->editCurrentSimTimeParam);

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

    this->interpolTangentParam.SetParameter(new param::FloatParam(this->tl)); // , -10.0f, 10.0f));
    this->MakeSlotAvailable(&this->interpolTangentParam);

    //UNUSED this->setKeyframesToSameSpeed.SetParameter(new param::ButtonParam('v'));
    //UNUSED this->MakeSlotAvailable(&this->setKeyframesToSameSpeed);
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

    bool appliedChanges = false;

    // Apply changes of camera parameters only to existing keyframe
    float selAnimTime = ccc->getSelectedKeyframe().getAnimTime();
    for (unsigned int i = 0; i < this->keyframes.Count(); i++) {
        if (this->keyframes[i].getAnimTime() == selAnimTime) {
            this->replaceKeyframe(this->keyframes[i], ccc->getSelectedKeyframe(), true);
            appliedChanges = true;
        }
    }

    if (!appliedChanges) {
        this->selectedKeyframe = this->interpolateKeyframe(ccc->getSelectedKeyframe().getAnimTime());
        this->updateEditParameters(this->selectedKeyframe);
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

    Keyframe prevSelKf = this->selectedKeyframe;

    // Update selected keyframe
    this->selectedKeyframe = this->interpolateKeyframe(ccc->getSelectedKeyframe().getAnimTime());
    this->updateEditParameters(this->selectedKeyframe);
    ccc->setSelectedKeyframe(this->selectedKeyframe);

    // Linearize tangent
    this->linearizeSimTangent(prevSelKf);

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
    this->updateEditParameters(this->selectedKeyframe);

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

    // Insert dragged keyframe at new position
    float t = ccc->getDropAnimTime();
    t = (t < 0.0f) ? (0.0f) : (t);
    t = (t > this->totalAnimTime) ? (this->totalAnimTime) : (t);
    this->dragDropKeyframe.setAnimTime(t);
    this->dragDropKeyframe.setSimTime(ccc->getDropSimTime());

    this->replaceKeyframe(this->selectedKeyframe, this->dragDropKeyframe, true);

    return true;
}


/*
* KeyframeKeeper::CallForSetCtrlPoints
*/
bool KeyframeKeeper::CallForSetCtrlPoints(core::Call& c) {

    CallCinematicCamera *ccc = dynamic_cast<CallCinematicCamera*>(&c);
    if (ccc == NULL) return false;

    this->firstCtrllPos = ccc->getFirstControlPointPosition();
    this->lastCtrllPos = ccc->getLastControlPointPosition();

    // Refresh interoplated camera positions
    this->refreshInterpolCamPos(this->interpolSteps);

    return true;
}


/*
* KeyframeKeeper::CallForGetUpdatedKeyframeData
*/
bool KeyframeKeeper::CallForGetUpdatedKeyframeData(core::Call& c) {

    CallCinematicCamera *ccc = dynamic_cast<CallCinematicCamera*>(&c);
    if (ccc == NULL) return false;


    // UPDATE PARAMETERS

    // interpolTangentParam ---------------------------------------------------
    if (this->interpolTangentParam.IsDirty()) {
        this->interpolTangentParam.ResetDirty();

        this->tl = this->interpolTangentParam.Param<param::FloatParam>()->Value();
        this->refreshInterpolCamPos(this->interpolSteps);
    }

    // applyKeyframeParam -----------------------------------------------------
    if (this->applyKeyframeParam.IsDirty()) {
        this->applyKeyframeParam.ResetDirty();

        // Get current camera for selected keyframe
        Keyframe tmpKf = this->selectedKeyframe;
        tmpKf.setCameraUp(this->camViewUp);
        tmpKf.setCameraPosition(this->camViewPosition);
        tmpKf.setCameraLookAt(this->camViewLookat);
        tmpKf.setCameraApertureAngele(this->camViewApertureangle);

        // Try adding keyframe to array
        if (!this->addKeyframe(tmpKf, true)) {
            if (!this->replaceKeyframe(this->selectedKeyframe, tmpKf, true)) {
                vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [Add Keyframe] Unable to apply settings to new/selested keyframe.");
            }
        }
    }

    // deleteSelectedKeyframeParam --------------------------------------------
    if (this->deleteSelectedKeyframeParam.IsDirty()) {
        this->deleteSelectedKeyframeParam.ResetDirty();

        this->deleteKeyframe(this->selectedKeyframe, true);
    }

    // undoChangesParam -------------------------------------------------------
    if (this->undoChangesParam.IsDirty()) {
        this->undoChangesParam.ResetDirty();

        this->undo();
    }

    // redoChangesParam -------------------------------------------------------
    if (this->redoChangesParam.IsDirty()) {
        this->redoChangesParam.ResetDirty();

        this->redo();
    }

    // setTotalAnimTimeParam --------------------------------------------------
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

    /* UNUSED
    // setKeyframesToSameSpeed ------------------------------------------------
    if (this->setKeyframesToSameSpeed.IsDirty()) {
        this->setKeyframesToSameSpeed.ResetDirty();

        this->setSameSpeed();
    }
    */

    // editCurrentAnimTimeParam -----------------------------------------------
    if (this->editCurrentAnimTimeParam.IsDirty()) {
        this->editCurrentAnimTimeParam.ResetDirty();

        // Clamp time value to allowed min max
        float t = this->editCurrentAnimTimeParam.Param<param::FloatParam>()->Value(); 
        t = (t < 0.0f) ? (0.0f) : (t);
        t = (t > this->totalAnimTime) ? (this->totalAnimTime) : (t);

        // Get index of existing keyframe
        int selIndex = static_cast<int>(this->keyframes.IndexOf(this->selectedKeyframe));
        if (selIndex >= 0) { // If existing keyframe is selected, delete keyframe an add at the right position
            Keyframe tmpKf = this->selectedKeyframe;
            this->selectedKeyframe.setAnimTime(t);
            this->replaceKeyframe(tmpKf, this->selectedKeyframe, true);
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
        float s = this->editCurrentSimTimeParam.Param<param::FloatParam>()->Value();
        s = vislib::math::Clamp(s, 0.0f, this->totalSimTime);

        // Get index of existing keyframe
        int selIndex = static_cast<int>(this->keyframes.IndexOf(this->selectedKeyframe));
        if (selIndex >= 0) { // If existing keyframe is selected, delete keyframe an add at the right position
            Keyframe tmpKf = this->selectedKeyframe;
            this->selectedKeyframe.setSimTime(s / this->totalSimTime);
            this->replaceKeyframe(tmpKf, this->selectedKeyframe, true);
        }
        // Write back clamped total time to parameter
        this->editCurrentSimTimeParam.Param<param::FloatParam>()->SetValue(s, false);
    }

    // editCurrentPosParam ----------------------------------------------------
    if (this->editCurrentPosParam.IsDirty()) {
        this->editCurrentPosParam.ResetDirty();

        vislib::math::Vector<float, 3> posV = this->editCurrentPosParam.Param<param::Vector3fParam>()->Value();
        vislib::math::Point<float, 3>  pos = vislib::math::Point<float, 3>(posV.X(), posV.Y(), posV.Z());

        // Get index of existing keyframe
        int selIndex = static_cast<int>(this->keyframes.IndexOf(this->selectedKeyframe));
        if (selIndex >= 0) {
            Keyframe tmpKf = this->selectedKeyframe;
            this->selectedKeyframe.setCameraPosition(pos);
            this->replaceKeyframe(tmpKf, this->selectedKeyframe, true);
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
            Keyframe tmpKf = this->selectedKeyframe;
            this->selectedKeyframe.setCameraLookAt(lookat);
            this->replaceKeyframe(tmpKf, this->selectedKeyframe, true);
        }
        else {
            //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Edit Current LookAt] No existing keyframe selected.");
        }
    }

    // resetLookAtParam -------------------------------------------------------
    if (this->resetLookAtParam.IsDirty()) {
        this->resetLookAtParam.ResetDirty();

        this->editCurrentLookAtParam.Param<param::Vector3fParam>()->SetValue(this->bboxCenter);
        // Get index of existing keyframe
        int selIndex = static_cast<int>(this->keyframes.IndexOf(this->selectedKeyframe));
        if (selIndex >= 0) {
            Keyframe tmpKf = this->selectedKeyframe;
            this->selectedKeyframe.setCameraLookAt(this->bboxCenter);
            this->replaceKeyframe(tmpKf, this->selectedKeyframe, true);
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
            Keyframe tmpKf = this->selectedKeyframe;
            this->selectedKeyframe.setCameraUp(up);
            this->replaceKeyframe(tmpKf, this->selectedKeyframe, true);
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
            Keyframe tmpKf = this->selectedKeyframe;
            this->selectedKeyframe.setCameraApertureAngele(aperture);
            this->replaceKeyframe(tmpKf, this->selectedKeyframe, true);
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

    // simTangentParam --------------------------------------------------------
    if (this->simTangentParam.IsDirty()) {
        this->simTangentParam.ResetDirty();

        this->simTangentStatus = true;
    }


    // PROPAGATE CURRENT DATA TO CALL -----------------------------------------
    ccc->setKeyframes(&this->keyframes);
    ccc->setBoundingBox(&this->boundingBox);
    ccc->setSelectedKeyframe(this->selectedKeyframe);
    ccc->setTotalAnimTime(this->totalAnimTime);
    ccc->setInterpolCamPositions(&this->interpolCamPos);
    ccc->setTotalSimTime(this->totalSimTime);
    ccc->setFps(this->fps);
    ccc->setControlPointPosition(this->firstCtrllPos, this->lastCtrllPos);

    return true;
}



/*
* KeyframeKeeper::undo
*/
bool KeyframeKeeper::addNewUndoAction(KeyframeKeeper::UndoActionEnum act, Keyframe kf, Keyframe prevkf) {

    bool retVal = false;

    // Remove all already undone actions in list
    if (!this->undoQueue.IsEmpty() && (this->undoQueueIndex >= -1)) {
        if (this->undoQueueIndex < (int)(this->undoQueue.Count() - 1)) {
            this->undoQueue.Erase((SIZE_T)(this->undoQueueIndex + 1), ((this->undoQueue.Count() + 1) - (SIZE_T)(this->undoQueueIndex + 1)));
        }
    }

    this->undoQueue.Add(UndoAction(act, kf, prevkf));
    this->undoQueueIndex = (int)(this->undoQueue.Count()) - 1;
    retVal = true;

    if (!retVal) {
        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [addNewUndoAction] Failed to add new undo action.");
    }

    return retVal;
}


/*
* KeyframeKeeper::undo
*/
bool KeyframeKeeper::undo() {

    bool retVal  = false;

    if (!this->undoQueue.IsEmpty() && (this->undoQueueIndex >= 0)) {

        UndoAction currentUndo = this->undoQueue[this->undoQueueIndex];

        switch (currentUndo.action) {
            case (KeyframeKeeper::UndoActionEnum::UNDO_NONE): 
                break;
            case (KeyframeKeeper::UndoActionEnum::UNDO_ADD): 
                // Revert adding a keyframe (= delete)
                if (this->deleteKeyframe(currentUndo.keyframe, false)) {
                    this->undoQueueIndex--;
                    retVal = true;
                }
                break;
            case (KeyframeKeeper::UndoActionEnum::UNDO_DELETE):
                // Revert deleting a keyframe (= add)
                if (this->addKeyframe(currentUndo.keyframe, false)) {
                    this->undoQueueIndex--;
                    retVal = true; 
                }
                break;
            case (KeyframeKeeper::UndoActionEnum::UNDO_MODIFY): 
                // Revert changes made to a keyframe.
                if (this->replaceKeyframe(currentUndo.keyframe, currentUndo.prevKeyframe, false)) {
                    this->undoQueueIndex--;
                    retVal = true;
                }
                break;
            default: break;
        }
    }    

    if (!retVal) {
        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [undo] Failed to undo changes.");
    }

    return retVal;
}


/*
* KeyframeKeeper::redo
*/
bool KeyframeKeeper::redo() {

    bool retVal = false;


    if (!this->undoQueue.IsEmpty() && (this->undoQueueIndex < (int)(this->undoQueue.Count() - 1))) {

        this->undoQueueIndex++;
        if (this->undoQueueIndex < 0) {
            this->undoQueueIndex = 0;
        }
       
        UndoAction currentUndo = this->undoQueue[this->undoQueueIndex];

        switch (currentUndo.action) {
        case (KeyframeKeeper::UndoActionEnum::UNDO_NONE):
            break;
        case (KeyframeKeeper::UndoActionEnum::UNDO_ADD):
            // Redo adding a keyframe (= delete)
            if (this->addKeyframe(currentUndo.keyframe, false)) {
                retVal = true;
            }
            break;
        case (KeyframeKeeper::UndoActionEnum::UNDO_DELETE):
            // Redo deleting a keyframe (= add)
            if (this->deleteKeyframe(currentUndo.keyframe, false)) {
                retVal = true;
            }
            break;
        case (KeyframeKeeper::UndoActionEnum::UNDO_MODIFY):
            // Redo changes made to a keyframe.
            if (this->replaceKeyframe(currentUndo.prevKeyframe, currentUndo.keyframe, false)) {
                retVal = true;
            }
            break;
        default: break;
        }
    }

    if (!retVal) {
        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [redo] Failed to redo changes.");
    }

    return retVal;
}


/*
* KeyframeKeeper::linearizeSimTangent
*/
void KeyframeKeeper::linearizeSimTangent(Keyframe stkf) {

    // Linearize tangent between simTangentKf and currently selected keyframe by shifting all inbetween keyframes simulation time
    if (this->simTangentStatus) {

        // Linearize tangent only between existing keyframes
        if (!(this->keyframes.Contains(stkf))) {
            vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [linearize tangent] Select existing keyframe before trying to linearize the tangent.");
            this->simTangentStatus = false;
        }
        else if (this->keyframes.Contains(this->selectedKeyframe)) {

            // Calculate liner equation between the two selected keyframes 
            // f(x) = mx + b
            vislib::math::Point<float, 2> p1 = vislib::math::Point<float, 2>(this->selectedKeyframe.getAnimTime(), this->selectedKeyframe.getSimTime());
            vislib::math::Point<float, 2> p2 = vislib::math::Point<float, 2>(stkf.getAnimTime(), stkf.getSimTime());
            float m = (p1.Y() - p2.Y()) / (p1.X() - p2.X());
            float b = m * (-p1.X()) + p1.Y();
            // Get indices
            int iKf1 = static_cast<int>(this->keyframes.IndexOf(this->selectedKeyframe));
            int iKf2 = static_cast<int>(this->keyframes.IndexOf(stkf));
            if (iKf1 > iKf2) {
                int tmp = iKf1;
                iKf1 = iKf2;
                iKf2 = tmp;
            }
            // Consider only keyframes lying between the two selected ones
            float newSimTime;
            for (int i = iKf1 + 1; i < iKf2; i++) {
                newSimTime = m * (this->keyframes[i].getAnimTime()) + b;

                // MODIFY - UNDO //
                // Store old keyframe
                Keyframe tmpKf = this->keyframes[i];
                // Apply changes to keyframe
                this->keyframes[i].setSimTime(newSimTime);
                // Add modification to undo queue
                this->addNewUndoAction(KeyframeKeeper::UndoActionEnum::UNDO_MODIFY, this->keyframes[i], tmpKf);
            }
    
            this->simTangentStatus = false;
        }
        else {
            vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [linearize tangent] Select existing keyframe to finish linearizing the tangent.");
        }
    }
}


/*
* KeyframeKeeper::snapKeyframe2AnimFrame
*/
void KeyframeKeeper::snapKeyframe2AnimFrame(Keyframe *kf) {

    if (this->fps == 0) {
        vislib::sys::Log::DefaultLog.WriteError("[KEYFRAME KEEPER] [snapKeyframe2AnimFrame] FPS is ZERO.");
        return;
    }

    float fpsFrac = 1.0f / (float)(this->fps);
    float t = std::round(kf->getAnimTime() / fpsFrac) * fpsFrac;
    if (std::abs(t - std::round(t)) < (fpsFrac / 2.0)) {
        t = std::round(t);
    }
    t = (t < 0.0f) ? (0.0f) : (t);
    t = (t > this->totalAnimTime) ? (this->totalAnimTime) : (t);

    // MODIFY - UNDO //
    // Store old keyframe
    Keyframe tmpKf = *kf;
    // Apply changes to keyframe
    kf->setAnimTime(t);
    // Add modification to undo queue
    this->addNewUndoAction(KeyframeKeeper::UndoActionEnum::UNDO_MODIFY, *kf, tmpKf);
}


/*
* KeyframeKeeper::snapKeyframe2SimFrame
*/
void KeyframeKeeper::snapKeyframe2SimFrame(Keyframe *kf) {

    float s = std::round(kf->getSimTime() * this->totalSimTime) / this->totalSimTime;

    // MODIFY - UNDO //
    // Store old keyframe
    Keyframe tmpKf = *kf;
    // Apply changes to keyframe
    kf->setSimTime(s);
    // Add modification to undo queue
    this->addNewUndoAction(KeyframeKeeper::UndoActionEnum::UNDO_MODIFY, *kf, tmpKf);
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
                float t = this->keyframes[index - 1].getAnimTime() + kfTime;
                t = (t < 0.0f) ? (0.0f) : (t);
                t = (t > this->totalAnimTime) ? (this->totalAnimTime) : (t);

                // MODIFY - UNDO //
                // Store old keyframe
                Keyframe tmpKf = this->keyframes[index];
                // Apply changes to keyframe
                this->keyframes[index].setAnimTime(t);
                // Add modification to undo queue
                this->addNewUndoAction(KeyframeKeeper::UndoActionEnum::UNDO_MODIFY, this->keyframes[index], tmpKf);

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
* KeyframeKeeper::replaceKeyframe
*/
bool KeyframeKeeper::replaceKeyframe(Keyframe oldkf, Keyframe newkf, bool undo) {

    // Both are equal ... nothing to do
    if (oldkf == newkf) {
        return true;
    }

    // Check if old keyframe exists
    int selIndex = static_cast<int>(this->keyframes.IndexOf(oldkf));
    if (selIndex >= 0) {
        // Delete old keyframe
        this->deleteKeyframe(oldkf, false);
        // Try to add new keyframe
        if (!this->addKeyframe(newkf, false)) {
            // There is alredy a keyframe on the new position ... overwrite existing keyframe.
            float newAnimTime = newkf.getAnimTime();
            for (unsigned int i = 0; i < this->keyframes.Count(); i++) {
                if (this->keyframes[i].getAnimTime() == newAnimTime) {
                    this->deleteKeyframe(this->keyframes[i], true);
                    break;
                }
            }
            this->addKeyframe(newkf, false);
        }
        if (undo) {
            // Add modification to undo queue
            this->addNewUndoAction(KeyframeKeeper::UndoActionEnum::UNDO_MODIFY, newkf, oldkf);
        }
    }
    else {
        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [replace Keyframe] Could not find keyframe which should be replaced.");
        return false;
    }

    return true;
}


/*
* KeyframeKeeper::deleteKeyframe
*/
bool KeyframeKeeper::deleteKeyframe(Keyframe kf, bool undo) {

    // Get index of keyframe to delete
    int selIndex = static_cast<int>(this->keyframes.IndexOf(kf));

    // Choose new selected keyframe
    if (selIndex >= 0) {

        // DELETE - UNDO //
        // Remove keyframe from keyframe array
        this->keyframes.RemoveAt(selIndex);
        if (undo) {
            // Add modification to undo queue
            this->addNewUndoAction(KeyframeKeeper::UndoActionEnum::UNDO_DELETE, kf, kf);

            // Adjust first/last control point position - ONLY if it is a "real" delete and no replace
            vislib::math::Vector<float, 3> tmpV;
            if (this->keyframes.Count() > 1) {
                if (selIndex == 0) {
                    tmpV = (this->keyframes[0].getCamPosition() - this->keyframes[1].getCamPosition());
                    tmpV.Normalise();
                    this->firstCtrllPos = this->keyframes[0].getCamPosition() + tmpV;
                }
                if (selIndex == this->keyframes.Count()) { // Element is already removed so the index is now: (this->keyframes.Count() - 1) + 1
                    tmpV = (this->keyframes.Last().getCamPosition() - this->keyframes[(int)this->keyframes.Count() - 2].getCamPosition());
                    tmpV.Normalise();
                    this->lastCtrllPos = this->keyframes.Last().getCamPosition() + tmpV;
                }
            }
        }

        // Reset bounding box
        this->boundingBox.SetNull();

        // Refresh interoplated camera positions
        this->refreshInterpolCamPos(this->interpolSteps);

        // Adjusting selected keyframe
        if (selIndex > 0) {
            this->selectedKeyframe = this->keyframes[selIndex - 1];
        }
        else if (selIndex < this->keyframes.Count()) {
            this->selectedKeyframe = this->keyframes[selIndex];
        }
        this->updateEditParameters(this->selectedKeyframe);
    }
    else {
        //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Delete Keyframe] No existing keyframe selected.");
        return false;
    }
    return true;
}


/*
* KeyframeKeeper::addKeyframe
*/
bool KeyframeKeeper::addKeyframe(Keyframe kf, bool undo) {

    float time = kf.getAnimTime();

    // Check if keyframe already exists
    for (unsigned int i = 0; i < this->keyframes.Count(); i++) {
        if (this->keyframes[i].getAnimTime() == time) {
            //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Add Keyframe] Keyframe already exists.");
            return false;
        }
    }

    // Sort new keyframe to keyframe array
    if (this->keyframes.IsEmpty() || (this->keyframes.Last().getAnimTime() < time)) {
        this->keyframes.Add(kf);
        // Adjust first/last control point position - ONLY if it is a "real" add and no replace
        if (undo && this->keyframes.Count() > 1) {
            vislib::math::Vector<float, 3> tmpV = (this->keyframes.Last().getCamPosition() - this->keyframes[(int)this->keyframes.Count() - 2].getCamPosition());
            tmpV.Normalise();
            this->lastCtrllPos = this->keyframes.Last().getCamPosition() + tmpV;
        }
    }
    else if (time < this->keyframes.First().getAnimTime()) {
        this->keyframes.Prepend(kf);
        // Adjust first/last control point position - ONLY if it is a "real" add and no replace
        if (undo && this->keyframes.Count() > 1) {
            vislib::math::Vector<float, 3> tmpV = (this->keyframes[0].getCamPosition() - this->keyframes[1].getCamPosition());
            tmpV.Normalise();
            this->firstCtrllPos = this->keyframes[0].getCamPosition() + tmpV;
        }
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

    // ADD - UNDO //
    if (undo) {
        // Add modification to undo queue
        this->addNewUndoAction(KeyframeKeeper::UndoActionEnum::UNDO_ADD, kf, kf);
    }

    // Update bounding box
    // Extend camera position for bounding box to cover manipulator axis
    vislib::math::Vector<float, 3> manipulator = vislib::math::Vector<float, 3>(kf.getCamLookAt().X(), kf.getCamLookAt().Y(), kf.getCamLookAt().Z());
    manipulator = kf.getCamPosition() - manipulator;
    manipulator.ScaleToLength(1.5f);
    this->boundingBox.GrowToPoint(static_cast<vislib::math::Point<float, 3> >(kf.getCamPosition() + manipulator));
    // Refresh interoplated camera positions
    this->refreshInterpolCamPos(this->interpolSteps);

    // Set new slected keyframe
    this->selectedKeyframe = kf;
    this->updateEditParameters(this->selectedKeyframe);

    return true;
}


/*
* KeyframeKeeper::interpolateKeyframe
*/
Keyframe KeyframeKeeper::interpolateKeyframe(float time) {

    float t = time;
    t = (t < 0.0f) ? (0.0f) : (t);
    t = (t > this->totalAnimTime) ? (this->totalAnimTime) : (t);

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

        // ! Skip interpolation of camera parameters if they are equal for ?1 and ?2.
        // => Prevent interpolation loops if time of keyframes is different, but cam params are the same.

        //interpolate position
        vislib::math::Vector<float, 3> p0(keyframes[i0].getCamPosition());
        vislib::math::Vector<float, 3> p1(keyframes[i1].getCamPosition());
        vislib::math::Vector<float, 3> p2(keyframes[i2].getCamPosition());
        vislib::math::Vector<float, 3> p3(keyframes[i3].getCamPosition());

        // Use additional control point positions to manipulate interpolation curve for first and last keyframe
        if (p0 == p1) {
            p0 = this->firstCtrllPos;
        }
        if (p2 == p3) {
            p3 = this->lastCtrllPos;
        }


        vislib::math::Vector<float, 3> pk = this->interpolation(iT, p0, p1, p2, p3);
        if (p1 == p2) {
            kf.setCameraPosition(keyframes[i1].getCamPosition());
        }
        else {
            kf.setCameraPosition(Point<float, 3>(pk.X(), pk.Y(), pk.Z()));
        }

        vislib::math::Vector<float, 3> l0(keyframes[i0].getCamLookAt());
        vislib::math::Vector<float, 3> l1(keyframes[i1].getCamLookAt());
        vislib::math::Vector<float, 3> l2(keyframes[i2].getCamLookAt());
        vislib::math::Vector<float, 3> l3(keyframes[i3].getCamLookAt());
        if (l1 == l2) {
            kf.setCameraLookAt(keyframes[i1].getCamLookAt());
        }
        else {
            //interpolate lookAt
            vislib::math::Vector<float, 3> lk = this->interpolation(iT, l0, l1, l2, l3);
            kf.setCameraLookAt(Point<float, 3>(lk.X(), lk.Y(), lk.Z()));
        }

        vislib::math::Vector<float, 3> u0 = p0 + keyframes[i0].getCamUp();
        vislib::math::Vector<float, 3> u1 = p1 + keyframes[i1].getCamUp();
        vislib::math::Vector<float, 3> u2 = p2 + keyframes[i2].getCamUp();
        vislib::math::Vector<float, 3> u3 = p3 + keyframes[i3].getCamUp();
        if (u1 == u2) {
            kf.setCameraUp(keyframes[i1].getCamUp());
        }
        else {
            //interpolate up
            vislib::math::Vector<float, 3> uk = this->interpolation(iT, u0, u1, u2, u3);
            kf.setCameraUp(uk - pk);
        }

        //interpolate aperture angle
        float a0 = keyframes[i0].getCamApertureAngle();
        float a1 = keyframes[i1].getCamApertureAngle();
        float a2 = keyframes[i2].getCamApertureAngle();
        float a3 = keyframes[i3].getCamApertureAngle();
        if (a1 == a2) {
            kf.setCameraApertureAngele(keyframes[i1].getCamApertureAngle());
        }
        else {
            float ak = this->interpolation(iT, a0, a1, a2, a3);
            kf.setCameraApertureAngele(ak);
        }

        return kf;
    }
}


/*
* KeyframeKeeper::interpolation
*/
// Catmull-Rom
float KeyframeKeeper::interpolation(float u, float f0, float f1, float f2, float f3) {

    /* 
    // Original version 
    float f = ((f1 * 2.0f) +
               (f2 - f0) * u +
              ((f0 * 2.0f) - (f1 * 5.0f) + (f2 * 4.0f) - f3) * u * u +
              (-f0 + (f1 * 3.0f) - (f2 * 3.0f) + f3) * u * u * u) * 0.5f;
    */

    // Considering global tangent length 
    // SOURCE: https://www.cs.cmu.edu/~462/projects/assn2/assn2/catmullRom.pdf
    float f = (f1) +
              (-(this->tl * f0) + (this->tl* f2)) * u + 
              ((2.0f*this->tl * f0) + ((this->tl - 3.0f) * f1) + ((3.0f - 2.0f*this->tl) * f2) - (this->tl* f3)) * u * u + 
              (-(this->tl * f0) + ((2.0f - this->tl) * f1) + ((this->tl - 2.0f) * f2) + (this->tl* f3)) * u * u * u;


    return f;
}

vislib::math::Vector<float, 3> KeyframeKeeper::interpolation(float u, vislib::math::Vector<float, 3> v0, vislib::math::Vector<float, 3> v1, vislib::math::Vector<float, 3> v2, vislib::math::Vector<float, 3> v3) {

    vislib::math::Vector<float, 3> v;
    v.SetX(this->interpolation(u, v0.X(), v1.X(), v2.X(), v3.X()));
    v.SetY(this->interpolation(u, v0.Y(), v1.Y(), v2.Y(), v3.Y()));
    v.SetZ(this->interpolation(u, v0.Z(), v1.Z(), v2.Z(), v3.Z()));

    return v;
}


/*
* KeyframeKeeper::saveKeyframes
*/
void KeyframeKeeper::saveKeyframes() {

    if (this->filename.IsEmpty()) {
        vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [Save Keyframes] No filename given. Using default filename.");

        time_t t = std::time(0);  // get time now
        struct tm *now = nullptr;
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
        struct tm nowdata;
        now = &nowdata;
        localtime_s(now, &t);
#else /* defined(_WIN32) && (_MSC_VER >= 1400) */
        now = localtime(&t);
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
        this->filename.Format("keyframes_%i%02i%02i-%02i%02i%02i.kf", (now->tm_year + 1900), (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
        this->fileNameParam.Param<param::FilePathParam>()->SetValue(this->filename, false);
    } 

    std::ofstream outfile;
    outfile.open(this->filename.PeekBuffer(), std::ios::binary);
    vislib::StringSerialiserA ser;
    outfile << "totalAnimTime=" << this->totalAnimTime << "\n";
    outfile << "tangentLength=" << this->tl << "\n";
    outfile << "firstCtrllPosX=" << this->firstCtrllPos.X() << "\n";
    outfile << "firstCtrllPosY=" << this->firstCtrllPos.Y() << "\n";
    outfile << "firstCtrllPosZ=" << this->firstCtrllPos.Z() << "\n";
    outfile << "lastCtrllPosX=" << this->lastCtrllPos.X() << "\n";
    outfile << "lastCtrllPosY=" << this->lastCtrllPos.Y() << "\n";
    outfile << "lastCtrllPosZ=" << this->lastCtrllPos.Z() << "\n\n";
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

        // Make compatible with previous version
        if (line.find("Length",0) < line.length()) {
            // get tangentLength
            //std::getline(infile, line);
            this->tl = std::stof(line.erase(0, 14)); // "tangentLength="
            // get firstCtrllPos
            std::getline(infile, line);
            this->firstCtrllPos.SetX(std::stof(line.erase(0, 15))); // "firstCtrllPosX="
            std::getline(infile, line);
            this->firstCtrllPos.SetY(std::stof(line.erase(0, 15))); // "firstCtrllPosY="
            std::getline(infile, line);
            this->firstCtrllPos.SetZ(std::stof(line.erase(0, 15))); // "firstCtrllPosZ="
            // get lastCtrllPos
            std::getline(infile, line);
            this->lastCtrllPos.SetX(std::stof(line.erase(0, 14))); // "lastCtrllPosX="
            std::getline(infile, line);
            this->lastCtrllPos.SetY(std::stof(line.erase(0, 14))); // "lastCtrllPosY="
            std::getline(infile, line);
            this->lastCtrllPos.SetZ(std::stof(line.erase(0, 14))); // "lastCtrllPosZ="
            // Consume empty line
            std::getline(infile, line);
        }
        else {
            vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [Load Keyframes] Loading keyframes stored in OLD format - Save keyframes to current file to convert to new format.");
        }


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
            this->selectedKeyframe = this->interpolateKeyframe(0.0f);
            this->updateEditParameters(this->selectedKeyframe);
            // Refresh interoplated camera positions
            this->refreshInterpolCamPos(this->interpolSteps);
        }
        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Load Keyframes] Successfully loaded keyframes from file: %s", this->filename.PeekBuffer());
    }
}


/*
* KeyframeKeeper::updateEditParameters
*/
void KeyframeKeeper::updateEditParameters(Keyframe kf) {

    // Put new values of changes selected keyframe to parameters
    vislib::math::Vector<float, 3> lookatV = kf.getCamLookAt();
    vislib::math::Point<float, 3>  lookat = vislib::math::Point<float, 3>(lookatV.X(), lookatV.Y(), lookatV.Z());
    vislib::math::Vector<float, 3> posV = kf.getCamPosition();
    vislib::math::Point<float, 3>  pos = vislib::math::Point<float, 3>(posV.X(), posV.Y(), posV.Z());
    this->editCurrentAnimTimeParam.Param<param::FloatParam>()->SetValue(kf.getAnimTime(), false);
    this->editCurrentSimTimeParam.Param<param::FloatParam>()->SetValue(kf.getSimTime() * this->totalSimTime, false);
    this->editCurrentPosParam.Param<param::Vector3fParam>()->SetValue(pos, false);
    this->editCurrentLookAtParam.Param<param::Vector3fParam>()->SetValue(lookat, false);
    this->editCurrentUpParam.Param<param::Vector3fParam>()->SetValue(kf.getCamUp(), false);
    this->editCurrentApertureParam.Param<param::FloatParam>()->SetValue(kf.getCamApertureAngle(), false);
}