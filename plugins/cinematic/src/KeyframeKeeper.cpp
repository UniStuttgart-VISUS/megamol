/*
* KeyframeKeeper.cpp
*
* Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
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
#include "CallKeyframeKeeper.h"

using namespace megamol;
using namespace megamol::cinematic;
using namespace vislib;
using namespace vislib::math;
using namespace megamol::core;

KeyframeKeeper::KeyframeKeeper(void) : core::Module(),
    cinematicCallSlot("scene3D", "holds keyframe data"),
    applyKeyframeParam(            "applyKeyframe", "Apply current settings to selected/new keyframe."),
    undoChangesParam(              "undoChanges", "Undo changes."),
    redoChangesParam(              "redoChanges", "Redo changes."),
    deleteSelectedKeyframeParam(   "deleteKeyframe", "Deletes the currently selected keyframe."),
    setTotalAnimTimeParam(         "maxAnimTime", "The total timespan of the animation."),
    snapAnimFramesParam(           "snapAnimFrames", "Snap animation time of all keyframes to fixed animation frames."),
    snapSimFramesParam(            "snapSimFrames", "Snap simulation time of all keyframes to integer simulation frames."),
    simTangentParam(               "linearizeSimTime", "Linearize simulation time between two keyframes between currently selected keyframe and subsequently selected keyframe."),
    interpolTangentParam(          "interpolTangent", "Length of keyframe tangets affecting curvature of interpolation spline."),
    setKeyframesToSameSpeed(       "setSameSpeed", "Move keyframes to get same speed between all keyframes."),
    editCurrentAnimTimeParam(      "editSelected::animTime", "Edit animation time of the selected keyframe."),
    editCurrentSimTimeParam(       "editSelected::simTime", "Edit simulation time of the selected keyframe."),
    editCurrentPosParam(           "editSelected::positionVector", "Edit  position vector of the selected keyframe."),
    editCurrentLookAtParam(        "editSelected::lookatVector", "Edit LookAt vector of the selected keyframe."),
    resetLookAtParam(              "editSelected::resetLookat", "Reset the LookAt vector of the selected keyframe."),
    editCurrentUpParam(            "editSelected::upVector", "Edit Up vector of the selected keyframe."),
    editCurrentApertureParam(      "editSelected::apertureAngle", "Edit apperture angle of the selected keyframe."),
    fileNameParam(                 "storage::filename", "The name of the file to load or save keyframes."),
    saveKeyframesParam(            "storage::save", "Save keyframes to file."),
    loadKeyframesParam(            "storage::load", "Load keyframes from file."),

    interpolCamPos(),
    keyframes(),
    boundingBox(),
    selectedKeyframe(), 
    dragDropKeyframe(),
    startCtrllPos(),
    endCtrllPos(),
    totalAnimTime(1.0f),
    totalSimTime(1.0f),
    interpolSteps(10),
    modelBboxCenter(),
    fps(24),
    camViewUp(0.0f, 1.0f, 0.0f),
    camViewPosition(1.0f, 0.0f, 0.0f),
    camViewLookat(),
    camViewApertureangle(30.0f),
    filename("keyframes.kf"),
    simTangentStatus(false),
    tl(0.5f),
    undoQueue(),
    undoQueueIndex(0)
{

    // setting up callback
    this->cinematicCallSlot.SetCallback(CallKeyframeKeeper::ClassName(),
        CallKeyframeKeeper::FunctionName(CallKeyframeKeeper::CallForGetUpdatedKeyframeData), &KeyframeKeeper::CallForGetUpdatedKeyframeData);

    this->cinematicCallSlot.SetCallback(CallKeyframeKeeper::ClassName(),
        CallKeyframeKeeper::FunctionName(CallKeyframeKeeper::CallForSetSimulationData), &KeyframeKeeper::CallForSetSimulationData);

    this->cinematicCallSlot.SetCallback(CallKeyframeKeeper::ClassName(),
        CallKeyframeKeeper::FunctionName(CallKeyframeKeeper::CallForGetInterpolCamPositions), &KeyframeKeeper::CallForGetInterpolCamPositions);

    this->cinematicCallSlot.SetCallback(CallKeyframeKeeper::ClassName(),
        CallKeyframeKeeper::FunctionName(CallKeyframeKeeper::CallForSetSelectedKeyframe), &KeyframeKeeper::CallForSetSelectedKeyframe);

    this->cinematicCallSlot.SetCallback(CallKeyframeKeeper::ClassName(),
        CallKeyframeKeeper::FunctionName(CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime), &KeyframeKeeper::CallForGetSelectedKeyframeAtTime);

    this->cinematicCallSlot.SetCallback(CallKeyframeKeeper::ClassName(),
        CallKeyframeKeeper::FunctionName(CallKeyframeKeeper::CallForSetCameraForKeyframe), &KeyframeKeeper::CallForSetCameraForKeyframe);

    this->cinematicCallSlot.SetCallback(CallKeyframeKeeper::ClassName(),
        CallKeyframeKeeper::FunctionName(CallKeyframeKeeper::CallForSetDragKeyframe), &KeyframeKeeper::CallForSetDragKeyframe);

    this->cinematicCallSlot.SetCallback(CallKeyframeKeeper::ClassName(),
        CallKeyframeKeeper::FunctionName(CallKeyframeKeeper::CallForSetDropKeyframe), &KeyframeKeeper::CallForSetDropKeyframe);

    this->cinematicCallSlot.SetCallback(CallKeyframeKeeper::ClassName(),
        CallKeyframeKeeper::FunctionName(CallKeyframeKeeper::CallForSetCtrlPoints), &KeyframeKeeper::CallForSetCtrlPoints);

    this->MakeSlotAvailable(&this->cinematicCallSlot);

    // init parameters
    this->applyKeyframeParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_A));
    this->MakeSlotAvailable(&this->applyKeyframeParam);

    this->undoChangesParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_Y, core::view::Modifier::CTRL)); // = z in german keyboard layout
    this->MakeSlotAvailable(&this->undoChangesParam);

    this->redoChangesParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_Z, core::view::Modifier::CTRL)); // = y in german keyboard layout
    this->MakeSlotAvailable(&this->redoChangesParam);

    this->deleteSelectedKeyframeParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_D));
    this->MakeSlotAvailable(&this->deleteSelectedKeyframeParam);

    this->setTotalAnimTimeParam.SetParameter(new param::FloatParam(this->totalAnimTime, 0.000001f));
    this->MakeSlotAvailable(&this->setTotalAnimTimeParam);

    this->snapAnimFramesParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_F));
    this->MakeSlotAvailable(&this->snapAnimFramesParam);

    this->snapSimFramesParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_G));
    this->MakeSlotAvailable(&this->snapSimFramesParam);

    this->simTangentParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_T));
    this->MakeSlotAvailable(&this->simTangentParam);

    this->interpolTangentParam.SetParameter(new param::FloatParam(this->tl)); // , -10.0f, 10.0f));
    this->MakeSlotAvailable(&this->interpolTangentParam);

    //this->setKeyframesToSameSpeed.SetParameter(new param::ButtonParam(core::view::Key::KEY_V));
    //this->MakeSlotAvailable(&this->setKeyframesToSameSpeed);

    this->editCurrentAnimTimeParam.SetParameter(new param::FloatParam(this->selectedKeyframe.GetAnimTime(), 0.0f));
    this->MakeSlotAvailable(&this->editCurrentAnimTimeParam);

    this->editCurrentSimTimeParam.SetParameter(new param::FloatParam(this->selectedKeyframe.GetSimTime()*this->totalSimTime, 0.0f));
    this->MakeSlotAvailable(&this->editCurrentSimTimeParam);

    this->editCurrentPosParam.SetParameter(new param::Vector3fParam(this->selectedKeyframe.GetCamPosition() - this->modelBboxCenter.operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>()));
    this->MakeSlotAvailable(&this->editCurrentPosParam);

    this->editCurrentLookAtParam.SetParameter(new param::Vector3fParam(this->selectedKeyframe.GetCamLookAt()));
    this->MakeSlotAvailable(&this->editCurrentLookAtParam);

    this->resetLookAtParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_L));
    this->MakeSlotAvailable(&this->resetLookAtParam);
    
    this->editCurrentUpParam.SetParameter(new param::Vector3fParam(this->selectedKeyframe.GetCamUp()));
    this->MakeSlotAvailable(&this->editCurrentUpParam);

    this->editCurrentApertureParam.SetParameter(new param::FloatParam(this->selectedKeyframe.GetCamApertureAngle(), 0.0f, 180.0f));
    this->MakeSlotAvailable(&this->editCurrentApertureParam);

    this->fileNameParam.SetParameter(new param::FilePathParam(this->filename));
    this->MakeSlotAvailable(&this->fileNameParam);

    this->saveKeyframesParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_S, core::view::Modifier::CTRL));
    this->MakeSlotAvailable(&this->saveKeyframesParam);

    this->loadKeyframesParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_L, core::view::Modifier::CTRL));
    this->MakeSlotAvailable(&this->loadKeyframesParam);
    this->loadKeyframesParam.ForceSetDirty(); // Try to load keyframe file at program start

}


KeyframeKeeper::~KeyframeKeeper(void) {

    this->Release();
}


bool KeyframeKeeper::create(void) {

    return true;
}


void KeyframeKeeper::release(void) {

    // nothing to do here ...
}


bool KeyframeKeeper::CallForSetSimulationData(core::Call& c) {

    CallKeyframeKeeper *ccc = dynamic_cast<CallKeyframeKeeper*>(&c);
    if (ccc == nullptr) return false;

    // Get bounding box center
    this->modelBboxCenter = ccc->getBboxCenter();

    // Get total simulation time
    if (ccc->getTotalSimTime() != this->totalSimTime) {
        this->totalSimTime = ccc->getTotalSimTime();
        this->editCurrentSimTimeParam.Param<param::FloatParam>()->SetValue(this->selectedKeyframe.GetSimTime() * this->totalSimTime, false);
    }

    // Get Frames per Second
    this->fps = ccc->getFps();

    return true;
}


bool KeyframeKeeper::CallForGetInterpolCamPositions(core::Call& c) {

    CallKeyframeKeeper *ccc = dynamic_cast<CallKeyframeKeeper*>(&c);
    if (ccc == nullptr) return false;

    this->interpolSteps = ccc->getInterpolationSteps();
    this->refreshInterpolCamPos(this->interpolSteps);
    ccc->setInterpolCamPositions(&this->interpolCamPos);

    return true;
}


bool KeyframeKeeper::CallForSetSelectedKeyframe(core::Call& c) {

    CallKeyframeKeeper *ccc = dynamic_cast<CallKeyframeKeeper*>(&c);
    if (ccc == nullptr) return false;

    bool appliedChanges = false;

    // Apply changes of camera parameters only to existing keyframe
    float selAnimTime = ccc->getSelectedKeyframe().GetAnimTime();
    for (unsigned int i = 0; i < this->keyframes.Count(); i++) {
        if (this->keyframes[i].GetAnimTime() == selAnimTime) {
            this->replaceKeyframe(this->keyframes[i], ccc->getSelectedKeyframe(), true);
            appliedChanges = true;
        }
    }

    if (!appliedChanges) {
        this->selectedKeyframe = this->interpolateKeyframe(ccc->getSelectedKeyframe().GetAnimTime());
        this->updateEditParameters(this->selectedKeyframe);
        ccc->setSelectedKeyframe(this->selectedKeyframe);
        //vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [CallForSetSelectedKeyframe] Selected keyframe doesn't exist. Changes are omitted.");
    }

    return true;
}


bool KeyframeKeeper::CallForGetSelectedKeyframeAtTime(core::Call& c) {

    CallKeyframeKeeper *ccc = dynamic_cast<CallKeyframeKeeper*>(&c);
    if (ccc == nullptr) return false;

    Keyframe prevSelKf = this->selectedKeyframe;

    // Update selected keyframe
    this->selectedKeyframe = this->interpolateKeyframe(ccc->getSelectedKeyframe().GetAnimTime());
    this->updateEditParameters(this->selectedKeyframe);
    ccc->setSelectedKeyframe(this->selectedKeyframe);

    // Linearize tangent
    this->linearizeSimTangent(prevSelKf);

    return true;
}


bool KeyframeKeeper::CallForSetCameraForKeyframe(core::Call& c) {

    CallKeyframeKeeper *ccc = dynamic_cast<CallKeyframeKeeper*>(&c);
    if (ccc == nullptr) return false;

    this->camViewUp            = ccc->getCameraParameters()->Up();
    this->camViewPosition      = ccc->getCameraParameters()->Position();
    this->camViewLookat        = ccc->getCameraParameters()->LookAt();
    this->camViewApertureangle = ccc->getCameraParameters()->ApertureAngle();

    return true;
}


bool KeyframeKeeper::CallForSetDragKeyframe(core::Call& c) {

    CallKeyframeKeeper *ccc = dynamic_cast<CallKeyframeKeeper*>(&c);
    if (ccc == nullptr) return false;

    // Checking if selected keyframe exists in keyframe array is done by caller

    // Update selected keyframe
    Keyframe skf = ccc->getSelectedKeyframe();

    this->selectedKeyframe = this->interpolateKeyframe(skf.GetAnimTime());
    this->updateEditParameters(this->selectedKeyframe);

    // Save currently selected keyframe as drag and drop keyframe
    this->dragDropKeyframe = this->selectedKeyframe;

    return true;
}


bool KeyframeKeeper::CallForSetDropKeyframe(core::Call& c) {

    CallKeyframeKeeper *ccc = dynamic_cast<CallKeyframeKeeper*>(&c);
    if (ccc == nullptr) return false;

    // Insert dragged keyframe at new position
    float t = ccc->getDropAnimTime();
    t = (t < 0.0f) ? (0.0f) : (t);
    t = (t > this->totalAnimTime) ? (this->totalAnimTime) : (t);
    this->dragDropKeyframe.SetAnimTime(t);
    this->dragDropKeyframe.SetSimTime(ccc->getDropSimTime());

    this->replaceKeyframe(this->selectedKeyframe, this->dragDropKeyframe, true);

    return true;
}


bool KeyframeKeeper::CallForSetCtrlPoints(core::Call& c) {

    CallKeyframeKeeper *ccc = dynamic_cast<CallKeyframeKeeper*>(&c);
    if (ccc == nullptr) return false;

    auto prev_StartCP = this->startCtrllPos;
    auto prev_EndCP   = this->endCtrllPos;

    this->startCtrllPos = ccc->getStartControlPointPosition();
    this->endCtrllPos   = ccc->getEndControlPointPosition();

    // ADD UNDO //
    if ((prev_StartCP != this->startCtrllPos) || (prev_EndCP != this->endCtrllPos)) {
        this->addCpUndoAction(KeyframeKeeper::UndoActionEnum::UNDO_CP_MODIFY, this->startCtrllPos, this->endCtrllPos, prev_StartCP, prev_EndCP);
        vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [CallForSetCtrlPoints] ADDED undo for CTRL POINT ......");
    }

    // Refresh interoplated camera positions
    this->refreshInterpolCamPos(this->interpolSteps);


    ccc->setControlPointPosition(this->startCtrllPos, this->endCtrllPos);

    return true;
}


bool KeyframeKeeper::CallForGetUpdatedKeyframeData(core::Call& c) {

    CallKeyframeKeeper *ccc = dynamic_cast<CallKeyframeKeeper*>(&c);
    if (ccc == nullptr) return false;


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
        tmpKf.SetCameraUp(this->camViewUp);
        tmpKf.SetCameraPosition(this->camViewPosition);
        tmpKf.SetCameraLookAt(this->camViewLookat);
        tmpKf.SetCameraApertureAngele(this->camViewApertureangle);

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

        this->undoAction();
    }

    // redoChangesParam -------------------------------------------------------
    if (this->redoChangesParam.IsDirty()) {
        this->redoChangesParam.ResetDirty();

        this->redoAction();
    }

    // setTotalAnimTimeParam --------------------------------------------------
    if (this->setTotalAnimTimeParam.IsDirty()) {
        this->setTotalAnimTimeParam.ResetDirty();

        float tt = this->setTotalAnimTimeParam.Param<param::FloatParam>()->Value();
        if (!this->keyframes.IsEmpty()) {
            if (tt < this->keyframes.Last().GetAnimTime()) {
                tt = this->keyframes.Last().GetAnimTime();
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
            this->selectedKeyframe.SetAnimTime(t);
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
            this->selectedKeyframe.SetSimTime(s / this->totalSimTime);
            this->replaceKeyframe(tmpKf, this->selectedKeyframe, true);
        }
        // Write back clamped total time to parameter
        this->editCurrentSimTimeParam.Param<param::FloatParam>()->SetValue(s, false);
    }

    // editCurrentPosParam ----------------------------------------------------
    if (this->editCurrentPosParam.IsDirty()) {
        this->editCurrentPosParam.ResetDirty();

        v3f posV = this->editCurrentPosParam.Param<param::Vector3fParam>()->Value() + this->modelBboxCenter.operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>();
        p3f  pos = p3f(posV.X(), posV.Y(), posV.Z());

        // Get index of existing keyframe
        int selIndex = static_cast<int>(this->keyframes.IndexOf(this->selectedKeyframe));
        if (selIndex >= 0) {
            Keyframe tmpKf = this->selectedKeyframe;
            this->selectedKeyframe.SetCameraPosition(pos);
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

        v3f lookatV = this->editCurrentLookAtParam.Param<param::Vector3fParam>()->Value();
        p3f  lookat = p3f(lookatV.X(), lookatV.Y(), lookatV.Z());

        // Get index of existing keyframe
        int selIndex = static_cast<int>(this->keyframes.IndexOf(this->selectedKeyframe));
        if (selIndex >= 0) {
            Keyframe tmpKf = this->selectedKeyframe;
            this->selectedKeyframe.SetCameraLookAt(lookat);
            this->replaceKeyframe(tmpKf, this->selectedKeyframe, true);
        }
        else {
            //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Edit Current LookAt] No existing keyframe selected.");
        }
    }

    // resetLookAtParam -------------------------------------------------------
    if (this->resetLookAtParam.IsDirty()) {
        this->resetLookAtParam.ResetDirty();

        this->editCurrentLookAtParam.Param<param::Vector3fParam>()->SetValue(this->modelBboxCenter);
        // Get index of existing keyframe
        int selIndex = static_cast<int>(this->keyframes.IndexOf(this->selectedKeyframe));
        if (selIndex >= 0) {
            Keyframe tmpKf = this->selectedKeyframe;
            this->selectedKeyframe.SetCameraLookAt(this->modelBboxCenter);
            this->replaceKeyframe(tmpKf, this->selectedKeyframe, true);
        }
        else {
            //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Edit Current LookAt] No existing keyframe selected.");
        }
    }
    
    // editCurrentUpParam -----------------------------------------------------
    if (this->editCurrentUpParam.IsDirty()) {
        this->editCurrentUpParam.ResetDirty();

        v3f up = this->editCurrentUpParam.Param<param::Vector3fParam>()->Value();

        // Get index of existing keyframe
        int selIndex = static_cast<int>(this->keyframes.IndexOf(this->selectedKeyframe));
        if (selIndex >= 0) {
            Keyframe tmpKf = this->selectedKeyframe;
            this->selectedKeyframe.SetCameraUp(up);
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
            this->selectedKeyframe.SetCameraApertureAngele(aperture);
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

        this->loadKeyframes();
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
    ccc->setControlPointPosition(this->startCtrllPos, this->endCtrllPos);

    return true;
}


bool KeyframeKeeper::addKfUndoAction(KeyframeKeeper::UndoActionEnum act, Keyframe kf, Keyframe prev_kf) {

    return (this->addUndoAction(act, kf, prev_kf, v3f(), v3f(), v3f(), v3f()));
}



bool KeyframeKeeper::addCpUndoAction(KeyframeKeeper::UndoActionEnum act, v3f startcp, v3f endcp, v3f prev_startcp, v3f prev_endcp) {

    return (this->addUndoAction(act, Keyframe(), Keyframe(), startcp, endcp, prev_startcp, prev_endcp));
}


bool KeyframeKeeper::addUndoAction(KeyframeKeeper::UndoActionEnum act, Keyframe kf, Keyframe prev_kf, v3f startcp, v3f endcp, v3f prev_startcp, v3f prev_endcp) {

    bool retVal = false;

    // Remove all already undone actions in list
    if (!this->undoQueue.IsEmpty() && (this->undoQueueIndex >= -1)) {
        if (this->undoQueueIndex < (int)(this->undoQueue.Count() - 1)) {
            this->undoQueue.Erase((SIZE_T)(this->undoQueueIndex + 1), ((this->undoQueue.Count() + 1) - (SIZE_T)(this->undoQueueIndex + 1)));
        }
    }

    this->undoQueue.Add(UndoAction(act, kf, prev_kf, startcp, endcp, prev_startcp, prev_endcp));
    this->undoQueueIndex = (int)(this->undoQueue.Count()) - 1;
    retVal = true;

    if (!retVal) {
        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [addKfUndoAction] Failed to add new undo action.");
    }

    return retVal;
}


bool KeyframeKeeper::undoAction() {

    bool retVal  = false;

    if (!this->undoQueue.IsEmpty() && (this->undoQueueIndex >= 0)) {

        UndoAction currentUndo = this->undoQueue[this->undoQueueIndex];

        switch (currentUndo.action) {
            case (KeyframeKeeper::UndoActionEnum::UNDO_NONE): 
                break;
            case (KeyframeKeeper::UndoActionEnum::UNDO_KF_ADD): 
                // Revert adding a keyframe (= delete)
                if (this->deleteKeyframe(currentUndo.keyframe, false)) {
                    this->undoQueueIndex--;
                    retVal = true;
                }
                break;
            case (KeyframeKeeper::UndoActionEnum::UNDO_KF_DELETE):
                // Revert deleting a keyframe (= add)
                if (this->addKeyframe(currentUndo.keyframe, false)) {
                    this->undoQueueIndex--;
                    retVal = true; 
                }
                break;
            case (KeyframeKeeper::UndoActionEnum::UNDO_KF_MODIFY): 
                // Revert changes made to a keyframe.
                if (this->replaceKeyframe(currentUndo.keyframe, currentUndo.prev_keyframe, false)) {
                    this->undoQueueIndex--;
                    retVal = true;
                }
                break;
            case (KeyframeKeeper::UndoActionEnum::UNDO_CP_MODIFY):
                // Revert changes made to the control points.
                this->startCtrllPos = currentUndo.prev_startcp;
                this->endCtrllPos   = currentUndo.prev_endcp;
                this->undoQueueIndex--;
                retVal = true;
                break;
            default: break;
        }
    }    
    //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] Undo queue index: %d - Undo queue size: %d", this->undoQueueIndex, this->undoQueue.Count());

    if (!retVal) {
        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [undoAction] Failed to undo changes.");
    }

    return retVal;
}


bool KeyframeKeeper::redoAction() {

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
        case (KeyframeKeeper::UndoActionEnum::UNDO_KF_ADD):
            // Redo adding a keyframe (= delete)
            if (this->addKeyframe(currentUndo.keyframe, false)) {
                retVal = true;
            }
            break;
        case (KeyframeKeeper::UndoActionEnum::UNDO_KF_DELETE):
            // Redo deleting a keyframe (= add)
            if (this->deleteKeyframe(currentUndo.keyframe, false)) {
                retVal = true;
            }
            break;
        case (KeyframeKeeper::UndoActionEnum::UNDO_KF_MODIFY):
            // Redo changes made to a keyframe.
            if (this->replaceKeyframe(currentUndo.prev_keyframe, currentUndo.keyframe, false)) {
                retVal = true;
            }
            break;
        case (KeyframeKeeper::UndoActionEnum::UNDO_CP_MODIFY):
            // Revert changes made to the control points.
            this->startCtrllPos = currentUndo.startcp;
            this->endCtrllPos   = currentUndo.endcp;
            retVal = true;
            break;
        default: break;
        }
    }

    //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] Undo queue index: %d - Undo queue size: %d", this->undoQueueIndex, this->undoQueue.Count());

    if (!retVal) {
        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [redoAction] Failed to redo changes.");
    }

    return retVal;
}


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
            vislib::math::Point<float, 2> p1 = vislib::math::Point<float, 2>(this->selectedKeyframe.GetAnimTime(), this->selectedKeyframe.GetSimTime());
            vislib::math::Point<float, 2> p2 = vislib::math::Point<float, 2>(stkf.GetAnimTime(), stkf.GetSimTime());
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
                newSimTime = m * (this->keyframes[i].GetAnimTime()) + b;

                // ADD UNDO //
                // Store old keyframe
                Keyframe tmpKf = this->keyframes[i];
                // Apply changes to keyframe
                this->keyframes[i].SetSimTime(newSimTime);
                // Add modification to undo queue
                this->addKfUndoAction(KeyframeKeeper::UndoActionEnum::UNDO_KF_MODIFY, this->keyframes[i], tmpKf);
            }
    
            this->simTangentStatus = false;
        }
        else {
            vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [linearize tangent] Select existing keyframe to finish linearizing the tangent.");
        }
    }
}


void KeyframeKeeper::snapKeyframe2AnimFrame(Keyframe *kf) {

    if (this->fps == 0) {
        vislib::sys::Log::DefaultLog.WriteError("[KEYFRAME KEEPER] [snapKeyframe2AnimFrame] FPS is ZERO.");
        return;
    }

    float fpsFrac = 1.0f / (float)(this->fps);
    float t = std::round(kf->GetAnimTime() / fpsFrac) * fpsFrac;
    if (std::abs(t - std::round(t)) < (fpsFrac / 2.0)) {
        t = std::round(t);
    }
    t = (t < 0.0f) ? (0.0f) : (t);
    t = (t > this->totalAnimTime) ? (this->totalAnimTime) : (t);

    // ADD UNDO //
    // Store old keyframe
    Keyframe tmpKf = *kf;
    // Apply changes to keyframe
    kf->SetAnimTime(t);
    // Add modification to undo queue
    this->addKfUndoAction(KeyframeKeeper::UndoActionEnum::UNDO_KF_MODIFY, *kf, tmpKf);
}


void KeyframeKeeper::snapKeyframe2SimFrame(Keyframe *kf) {

    float s = std::round(kf->GetSimTime() * this->totalSimTime) / this->totalSimTime;

    // ADD UNDO //
    // Store old keyframe
    Keyframe tmpKf = *kf;
    // Apply changes to keyframe
    kf->SetSimTime(s);
    // Add modification to undo queue
    this->addKfUndoAction(KeyframeKeeper::UndoActionEnum::UNDO_KF_MODIFY, *kf, tmpKf);
}


void KeyframeKeeper::setSameSpeed() {

    if (this->keyframes.Count() > 2) {

        // Store index of selected keyframe to restore seleection after changing time of keyframes
        int selIndex = static_cast<int>(this->keyframes.IndexOf(this->selectedKeyframe));

        // Get total values
        float totTime = this->keyframes.Last().GetAnimTime() - this->keyframes.First().GetAnimTime();
        if (totTime == 0.0f) {
            vislib::sys::Log::DefaultLog.WriteError("[KEYFRAME KEEPER] [setSameSpeed] totTime is ZERO.");
            return;
        }

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
                float t = this->keyframes[index - 1].GetAnimTime() + kfTime;
                t = (t < 0.0f) ? (0.0f) : (t);
                t = (t > this->totalAnimTime) ? (this->totalAnimTime) : (t);

                // ADD UNDO //
                // Store old keyframe
                Keyframe tmpKf = this->keyframes[index];
                // Apply changes to keyframe
                this->keyframes[index].SetAnimTime(t);
                // Add modification to undo queue
                this->addKfUndoAction(KeyframeKeeper::UndoActionEnum::UNDO_KF_MODIFY, this->keyframes[index], tmpKf);

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
            startTime = this->keyframes[i].GetAnimTime();
            deltaTimeStep = (this->keyframes[i + 1].GetAnimTime() - startTime) / (float)s;

            for (unsigned int j = 0; j < s; j++) {
                kf = this->interpolateKeyframe(startTime + deltaTimeStep*(float)j);
                this->interpolCamPos.Add(kf.GetCamPosition());
                this->boundingBox.GrowToPoint(this->interpolCamPos.Last());
            }
        }
        // Add last existing camera position
        this->interpolCamPos.Add(this->keyframes.Last().GetCamPosition());
    }
}


bool KeyframeKeeper::replaceKeyframe(Keyframe oldkf, Keyframe newkf, bool undo) {

    if (!this->keyframes.IsEmpty()) {

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
                float newAnimTime = newkf.GetAnimTime();
                for (unsigned int i = 0; i < this->keyframes.Count(); i++) {
                    if (this->keyframes[i].GetAnimTime() == newAnimTime) {
                        this->deleteKeyframe(this->keyframes[i], true);
                        break;
                    }
                }
                this->addKeyframe(newkf, false);
            }
            if (undo) {
                // ADD UNDO //
                this->addKfUndoAction(KeyframeKeeper::UndoActionEnum::UNDO_KF_MODIFY, newkf, oldkf);
            }
        }
        else {
            vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [replace Keyframe] Could not find keyframe which should be replaced.");
            return false;
        }
    }

    return true;
}


bool KeyframeKeeper::deleteKeyframe(Keyframe kf, bool undo) {

    if (!this->keyframes.IsEmpty()) {

        // Get index of keyframe to delete
        unsigned int selIndex = static_cast<unsigned int>(this->keyframes.IndexOf(kf));

        // Choose new selected keyframe
        if (selIndex >= 0) {

            // DELETE - UNDO //
            // Remove keyframe from keyframe array
            this->keyframes.RemoveAt(selIndex);
            if (undo) {
                // ADD UNDO //
                this->addKfUndoAction(KeyframeKeeper::UndoActionEnum::UNDO_KF_DELETE, kf, kf);

                // Adjust first/last control point position - ONLY if it is a "real" delete and no replace
                v3f tmpV;
                if (this->keyframes.Count() > 1) {
                    if (selIndex == 0) {
                        tmpV = (this->keyframes[0].GetCamPosition() - this->keyframes[1].GetCamPosition());
                        tmpV.Normalise();
                        this->startCtrllPos = this->keyframes[0].GetCamPosition() + tmpV;
                    }
                    if (selIndex == this->keyframes.Count()) { // Element is already removed so the index is now: (this->keyframes.Count() - 1) + 1
                        tmpV = (this->keyframes.Last().GetCamPosition() - this->keyframes[(int)this->keyframes.Count() - 2].GetCamPosition());
                        tmpV.Normalise();
                        this->endCtrllPos = this->keyframes.Last().GetCamPosition() + tmpV;
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

    }
    return true;
}


bool KeyframeKeeper::addKeyframe(Keyframe kf, bool undo) {

    float time = kf.GetAnimTime();

    // Check if keyframe already exists
    for (unsigned int i = 0; i < this->keyframes.Count(); i++) {
        if (this->keyframes[i].GetAnimTime() == time) {
            //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Add Keyframe] Keyframe already exists.");
            return false;
        }
    }

    // Sort new keyframe to keyframe array
    if (this->keyframes.IsEmpty() || (this->keyframes.Last().GetAnimTime() < time)) {
        this->keyframes.Add(kf);
        // Adjust first/last control point position - ONLY if it is a "real" add and no replace
        if (undo && this->keyframes.Count() > 1) {
            v3f tmpV = (this->keyframes.Last().GetCamPosition() - this->keyframes[(int)this->keyframes.Count() - 2].GetCamPosition());
            tmpV.Normalise();
            this->endCtrllPos = this->keyframes.Last().GetCamPosition() + tmpV;
        }
    }
    else if (time < this->keyframes.First().GetAnimTime()) {
        this->keyframes.Prepend(kf);
        // Adjust first/last control point position - ONLY if it is a "real" add and no replace
        if (undo && this->keyframes.Count() > 1) {
            v3f tmpV = (this->keyframes[0].GetCamPosition() - this->keyframes[1].GetCamPosition());
            tmpV.Normalise();
            this->startCtrllPos = this->keyframes[0].GetCamPosition() + tmpV;
        }
    }
    else { // Insert keyframe in-between existing keyframes
        unsigned int insertIdx = 0;
        for (unsigned int i = 0; i < this->keyframes.Count(); i++) {
            if (time < this->keyframes[i].GetAnimTime()) {
                insertIdx = i;
                break;
            }
        }
        this->keyframes.Insert(insertIdx, kf);
    }

    // ADD - UNDO //
    if (undo) {
        // ADD UNDO //
        this->addKfUndoAction(KeyframeKeeper::UndoActionEnum::UNDO_KF_ADD, kf, kf);
    }

    // Update bounding box
    // Extend camera position for bounding box to cover manipulator axis
    v3f manipulator = v3f(kf.GetCamLookAt().X(), kf.GetCamLookAt().Y(), kf.GetCamLookAt().Z());
    manipulator = kf.GetCamPosition() - manipulator;
    manipulator.ScaleToLength(1.5f);
    this->boundingBox.GrowToPoint(static_cast<p3f >(kf.GetCamPosition() + manipulator));
    // Refresh interoplated camera positions
    this->refreshInterpolCamPos(this->interpolSteps);

    // Set new slected keyframe
    this->selectedKeyframe = kf;
    this->updateEditParameters(this->selectedKeyframe);

    return true;
}


Keyframe KeyframeKeeper::interpolateKeyframe(float time) {

    float t = time;
    t = (t < 0.0f) ? (0.0f) : (t);
    t = (t > this->totalAnimTime) ? (this->totalAnimTime) : (t);

    // Check if there is an existing keyframe at requested time
    for (SIZE_T i = 0; i < this->keyframes.Count(); i++) {
        if (t == this->keyframes[i].GetAnimTime()) {
            return this->keyframes[i];
        }
    }

    if (this->keyframes.IsEmpty()) {
        // vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Interpolate Keyframe] Empty keyframe array.");
        Keyframe kf = Keyframe();
        kf.SetAnimTime(t);
        kf.SetSimTime(0.0f);
        kf.SetCameraUp(this->camViewUp);
        kf.SetCameraPosition(this->camViewPosition);
        kf.SetCameraLookAt(this->camViewLookat);
        kf.SetCameraApertureAngele(this->camViewApertureangle);
        return kf;
    }
    else if (t < this->keyframes.First().GetAnimTime()) {
        /**/
        Keyframe kf = this->keyframes.First();
        kf.SetAnimTime(t);
        /**/
        /*
        Keyframe kf = Keyframe();
        kf.SetAnimTime(t);
        kf.SetSimTime(this->keyframes.First().GetSimTime());
        kf.SetCameraUp(this->camViewUp);
        kf.SetCameraPosition(this->camViewPosition);
        kf.SetCameraLookAt(this->camViewLookat);
        kf.SetCameraApertureAngele(this->camViewApertureangle);
        */
        return kf;

    }
    else if (t > this->keyframes.Last().GetAnimTime()) {
        /*
        Keyframe kf = this->keyframes.Last();
        kf.SetAnimTime(t);
        */
        Keyframe kf = Keyframe();
        kf.SetAnimTime(t);
        kf.SetSimTime(this->keyframes.Last().GetSimTime());
        kf.SetCameraUp(this->camViewUp);
        kf.SetCameraPosition(this->camViewPosition);
        kf.SetCameraLookAt(this->camViewLookat);
        kf.SetCameraApertureAngele(this->camViewApertureangle);
        return kf;
    }
    else { // if ((t > this->keyframes.First().GetAnimTime()) && (t < this->keyframes.Last().GetAnimTime())) {

        // new default keyframe
        Keyframe kf = Keyframe();
        kf.SetAnimTime(t);

        // determine indices for interpolation 
        int i0 = 0;
        int i1 = 0;
        int i2 = 0;
        int i3 = 0;
        int kfIdxCnt = (int)keyframes.Count()-1;
        float iT = 0.0f;
        for (int i = 0; i < kfIdxCnt; i++) {
            float tMin = this->keyframes[i].GetAnimTime();
            float tMax = this->keyframes[i + 1].GetAnimTime();
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
        float simT1 = this->keyframes[i1].GetSimTime();
        float simT2 = this->keyframes[i2].GetSimTime();
        float simT = simT1 + (simT2 - simT1)*iT;

        kf.SetSimTime(simT);

        // ! Skip interpolation of camera parameters if they are equal for ?1 and ?2.
        // => Prevent interpolation loops if time of keyframes is different, but cam params are the same.

        //interpolate position
        v3f p0(keyframes[i0].GetCamPosition());
        v3f p1(keyframes[i1].GetCamPosition());
        v3f p2(keyframes[i2].GetCamPosition());
        v3f p3(keyframes[i3].GetCamPosition());

        // Use additional control point positions to manipulate interpolation curve for first and last keyframe
        if (p0 == p1) {
            p0 = this->startCtrllPos;
        }
        if (p2 == p3) {
            p3 = this->endCtrllPos;
        }


        v3f pk = this->interpolate_v3f(iT, p0, p1, p2, p3);
        if (p1 == p2) {
            kf.SetCameraPosition(keyframes[i1].GetCamPosition());
        }
        else {
            kf.SetCameraPosition(Point<float, 3>(pk.X(), pk.Y(), pk.Z()));
        }

        v3f l0(keyframes[i0].GetCamLookAt());
        v3f l1(keyframes[i1].GetCamLookAt());
        v3f l2(keyframes[i2].GetCamLookAt());
        v3f l3(keyframes[i3].GetCamLookAt());
        if (l1 == l2) {
            kf.SetCameraLookAt(keyframes[i1].GetCamLookAt());
        }
        else {
            //interpolate lookAt
            v3f lk = this->interpolate_v3f(iT, l0, l1, l2, l3);
            kf.SetCameraLookAt(Point<float, 3>(lk.X(), lk.Y(), lk.Z()));
        }

        v3f u0 = p0 + keyframes[i0].GetCamUp();
        v3f u1 = p1 + keyframes[i1].GetCamUp();
        v3f u2 = p2 + keyframes[i2].GetCamUp();
        v3f u3 = p3 + keyframes[i3].GetCamUp();
        if (u1 == u2) {
            kf.SetCameraUp(keyframes[i1].GetCamUp());
        }
        else {
            //interpolate up
            v3f uk = this->interpolate_v3f(iT, u0, u1, u2, u3);
            kf.SetCameraUp(uk - pk);
        }

        //interpolate aperture angle
        float a0 = keyframes[i0].GetCamApertureAngle();
        float a1 = keyframes[i1].GetCamApertureAngle();
        float a2 = keyframes[i2].GetCamApertureAngle();
        float a3 = keyframes[i3].GetCamApertureAngle();
        if (a1 == a2) {
            kf.SetCameraApertureAngele(keyframes[i1].GetCamApertureAngle());
        }
        else {
            float ak = this->interpolate_f(iT, a0, a1, a2, a3);
            kf.SetCameraApertureAngele(ak);
        }

        return kf;
    }
}


float KeyframeKeeper::interpolate_f(float u, float f0, float f1, float f2, float f3) {

    // Catmull-Rom
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


vislib::math::Vector<float, 3> KeyframeKeeper::interpolate_v3f(float u, v3f v0, v3f v1, v3f v2, v3f v3) {

    v3f v;
    v.SetX(this->interpolate_f(u, v0.X(), v1.X(), v2.X(), v3.X()));
    v.SetY(this->interpolate_f(u, v0.Y(), v1.Y(), v2.Y(), v3.Y()));
    v.SetZ(this->interpolate_f(u, v0.Z(), v1.Z(), v2.Z(), v3.Z()));

    return v;
}


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
    outfile << "startCtrllPosX=" << this->startCtrllPos.X() << "\n";
    outfile << "startCtrllPosY=" << this->startCtrllPos.Y() << "\n";
    outfile << "startCtrllPosZ=" << this->startCtrllPos.Z() << "\n";
    outfile << "endCtrllPosX=" << this->endCtrllPos.X() << "\n";
    outfile << "endCtrllPosY=" << this->endCtrllPos.Y() << "\n";
    outfile << "endCtrllPosZ=" << this->endCtrllPos.Z() << "\n\n";
    for (unsigned int i = 0; i < this->keyframes.Count(); i++) {
        this->keyframes[i].Serialise(ser);
        outfile << ser.GetString().PeekBuffer() << "\n";
    }
    outfile.close();

    vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [Save Keyframes] Successfully stored keyframes to file: %s", this->filename.PeekBuffer());

}


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
            // get startCtrllPos
            std::getline(infile, line);
            this->startCtrllPos.SetX(std::stof(line.erase(0, 15))); // "startCtrllPosX="
            std::getline(infile, line);
            this->startCtrllPos.SetY(std::stof(line.erase(0, 15))); // "startCtrllPosY="
            std::getline(infile, line);
            this->startCtrllPos.SetZ(std::stof(line.erase(0, 15))); // "startCtrllPosZ="
            // get endCtrllPos
            std::getline(infile, line);
            this->endCtrllPos.SetX(std::stof(line.erase(0, 14))); // "endCtrllPosX="
            std::getline(infile, line);
            this->endCtrllPos.SetY(std::stof(line.erase(0, 14))); // "endCtrllPosY="
            std::getline(infile, line);
            this->endCtrllPos.SetZ(std::stof(line.erase(0, 14))); // "endCtrllPosZ="
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
                kf.Deserialise(ser);
                this->keyframes.Add(kf);
                // Extend camera position for bounding box to cover manipulator axis
                v3f manipulator = v3f(kf.GetCamLookAt().X(), kf.GetCamLookAt().Y(), kf.GetCamLookAt().Z());
                manipulator = kf.GetCamPosition() - manipulator;
                manipulator.ScaleToLength(1.5f);
                this->boundingBox.GrowToPoint(static_cast<p3f >(kf.GetCamPosition() + manipulator));
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


void KeyframeKeeper::updateEditParameters(Keyframe kf) {

    // Put new values of changes selected keyframe to parameters
    v3f lookatV = kf.GetCamLookAt();
    p3f  lookat = p3f(lookatV.X(), lookatV.Y(), lookatV.Z());
    v3f posV = kf.GetCamPosition();
    p3f  pos = p3f(posV.X(), posV.Y(), posV.Z());
    this->editCurrentAnimTimeParam.Param<param::FloatParam>()->SetValue(kf.GetAnimTime(), false);
    this->editCurrentSimTimeParam.Param<param::FloatParam>()->SetValue(kf.GetSimTime() * this->totalSimTime, false);
    this->editCurrentPosParam.Param<param::Vector3fParam>()->SetValue(pos - this->modelBboxCenter.operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>(), false);
    this->editCurrentLookAtParam.Param<param::Vector3fParam>()->SetValue(lookat, false);
    this->editCurrentUpParam.Param<param::Vector3fParam>()->SetValue(kf.GetCamUp(), false);
    this->editCurrentApertureParam.Param<param::FloatParam>()->SetValue(kf.GetCamApertureAngle(), false);
}