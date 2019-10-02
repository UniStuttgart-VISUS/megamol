/*
* KeyframeKeeper.cpp
*
* Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "KeyframeKeeper.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::cinematic;

using namespace vislib;
using namespace vislib::math;


KeyframeKeeper::KeyframeKeeper(void) : core::Module()
    , cinematicCallSlot("scene3D", "holds keyframe data")
    , applyKeyframeParam("applyKeyframe", "Apply current settings to selected/new keyframe.")
    , undoChangesParam("undoChanges", "Undo changes.")
    , redoChangesParam("redoChanges", "Redo changes.")
    , deleteSelectedKeyframeParam("deleteKeyframe", "Deletes the currently selected keyframe.")
    , setTotalAnimTimeParam("maxAnimTime", "The total timespan of the animation.")
    , snapAnimFramesParam("snapAnimFrames", "Snap animation time of all keyframes to fixed animation frames.")
    , snapSimFramesParam("snapSimFrames", "Snap simulation time of all keyframes to integer simulation frames.")
    , simTangentParam("linearizeSimTime", "Linearize simulation time between two keyframes between currently selected keyframe and subsequently selected keyframe.")
    , interpolTangentParam("interpolTangent", "Length of keyframe tangets affecting curvature of interpolation spline.")
    , setKeyframesToSameSpeed("setSameSpeed", "Move keyframes to get same speed between all keyframes.")
    , editCurrentAnimTimeParam("editSelected::animTime", "Edit animation time of the selected keyframe.")
    , editCurrentSimTimeParam("editSelected::simTime", "Edit simulation time of the selected keyframe.")
    , editCurrentPosParam("editSelected::positionVector", "Edit  position vector of the selected keyframe.")
    , editCurrentLookAtParam("editSelected::lookatVector", "Edit LookAt vector of the selected keyframe.")
    , resetLookAtParam("editSelected::resetLookat", "Reset the LookAt vector of the selected keyframe.")
    , editCurrentUpParam("editSelected::upVector", "Edit Up vector of the selected keyframe.")
    , editCurrentApertureParam("editSelected::apertureAngle", "Edit apperture angle of the selected keyframe.")
    , fileNameParam("storage::filename", "The name of the file to load or save keyframes.")
    , saveKeyframesParam("storage::save", "Save keyframes to file.")
    , loadKeyframesParam("storage::load", "Load keyframes from file.")
    , cameraState(nullptr)
    , interpolCamPos(nullptr)
    , keyframes(nullptr)
    , boundingBox(nullptr)
    , selectedKeyframe()
    , dragDropKeyframe()
    , startCtrllPos()
    , endCtrllPos()
    , totalAnimTime(1.0f)
    , totalSimTime(1.0f)
    , interpolSteps(10)
    , modelBboxCenter()
    , fps(24)
    , filename("keyframes.kf")
    , simTangentStatus(false)
    , splineTangentLength(0.5f)
    , undoQueue()
    , undoQueueIndex(0) {

    // init callbacks
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
    this->applyKeyframeParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_A, core::view::Modifier::CTRL));
    this->MakeSlotAvailable(&this->applyKeyframeParam);

    this->undoChangesParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_Y, core::view::Modifier::CTRL)); // = z in german keyboard layout
    this->MakeSlotAvailable(&this->undoChangesParam);

    this->redoChangesParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_Z, core::view::Modifier::CTRL)); // = y in german keyboard layout
    this->MakeSlotAvailable(&this->redoChangesParam);

    this->deleteSelectedKeyframeParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_D, core::view::Modifier::CTRL));
    this->MakeSlotAvailable(&this->deleteSelectedKeyframeParam);

    this->setTotalAnimTimeParam.SetParameter(new param::FloatParam(this->totalAnimTime, 0.000001f));
    this->MakeSlotAvailable(&this->setTotalAnimTimeParam);

    this->snapAnimFramesParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_F, core::view::Modifier::CTRL));
    this->MakeSlotAvailable(&this->snapAnimFramesParam);

    this->snapSimFramesParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_G, core::view::Modifier::CTRL));
    this->MakeSlotAvailable(&this->snapSimFramesParam);

    this->simTangentParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_T, core::view::Modifier::CTRL));
    this->MakeSlotAvailable(&this->simTangentParam);

    this->interpolTangentParam.SetParameter(new param::FloatParam(this->splineTangentLength)); // , -10.0f, 10.0f));
    this->MakeSlotAvailable(&this->interpolTangentParam);

    //this->setKeyframesToSameSpeed.SetParameter(new param::ButtonParam(core::view::Key::KEY_V, core::view::Modifier::CTRL));
    //this->MakeSlotAvailable(&this->setKeyframesToSameSpeed);

    this->editCurrentAnimTimeParam.SetParameter(new param::FloatParam(this->selectedKeyframe.GetAnimTime(), 0.0f));
    this->MakeSlotAvailable(&this->editCurrentAnimTimeParam);

    this->editCurrentSimTimeParam.SetParameter(new param::FloatParam(this->selectedKeyframe.GetSimTime()*this->totalSimTime, 0.0f));
    this->MakeSlotAvailable(&this->editCurrentSimTimeParam);

    this->editCurrentPosParam.SetParameter(new param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, -1.0f)));
    this->MakeSlotAvailable(&this->editCurrentPosParam);

    this->editCurrentLookAtParam.SetParameter(new param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f)));
    this->MakeSlotAvailable(&this->editCurrentLookAtParam);

    this->resetLookAtParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_U, core::view::Modifier::CTRL));
    this->MakeSlotAvailable(&this->resetLookAtParam);
    
    this->editCurrentUpParam.SetParameter(new param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f)));
    this->MakeSlotAvailable(&this->editCurrentUpParam);

    this->editCurrentApertureParam.SetParameter(new param::FloatParam(60.0f, 0.0f, 180.0f));
    this->MakeSlotAvailable(&this->editCurrentApertureParam);

    this->fileNameParam.SetParameter(new param::FilePathParam(this->filename));
    this->MakeSlotAvailable(&this->fileNameParam);

    this->saveKeyframesParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_S, core::view::Modifier::CTRL));
    this->MakeSlotAvailable(&this->saveKeyframesParam);

    this->loadKeyframesParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_L, core::view::Modifier::CTRL));
    this->MakeSlotAvailable(&this->loadKeyframesParam);
    this->loadKeyframesParam.ForceSetDirty(); // Try to load keyframe file at program start

    // init parameter pointers
    this->cameraState = std::make_shared<Keyframe::cam_state_type>();
	this->interpolCamPos = std::make_shared<std::vector<glm::vec3>>();
	this->keyframes = std::make_shared<std::vector<Keyframe>>();
	this->boundingBox = std::make_shared<vislib::math::Cuboid<float>>();
}


KeyframeKeeper::~KeyframeKeeper(void) {

    this->Release();
}


bool KeyframeKeeper::create(void) {

    return true;
}


void KeyframeKeeper::release(void) {

}


bool KeyframeKeeper::CallForSetSimulationData(core::Call& c) {

    auto ccc = dynamic_cast<CallKeyframeKeeper*>(&c);
    if (ccc == nullptr) return false;

    this->modelBboxCenter = ccc->GetBboxCenter();
    if (ccc->GetTotalSimTime() != this->totalSimTime) {
        this->totalSimTime = ccc->GetTotalSimTime();
        this->editCurrentSimTimeParam.Param<param::FloatParam>()->SetValue(this->selectedKeyframe.GetSimTime() * this->totalSimTime, false);
    }
    this->fps = ccc->GetFps();

    return true;
}


bool KeyframeKeeper::CallForGetInterpolCamPositions(core::Call& c) {

    auto ccc = dynamic_cast<CallKeyframeKeeper*>(&c);
    if (ccc == nullptr) return false;

    this->interpolSteps = ccc->GetInterpolationSteps();
    this->refreshInterpolCamPos(this->interpolSteps);
    ccc->SetInterpolCamPositions(this->interpolCamPos);

    return true;
}


bool KeyframeKeeper::CallForSetSelectedKeyframe(core::Call& c) {

    auto ccc = dynamic_cast<CallKeyframeKeeper*>(&c);
    if (ccc == nullptr) return false;

    // Apply changes of camera parameters only to existing keyframe
    bool appliedChanges = false;
    float selAnimTime = ccc->GetSelectedKeyframe().GetAnimTime();
    for (unsigned int i = 0; i < this->keyframes->size(); i++) {
        if (this->keyframes->operator[](i).GetAnimTime() == selAnimTime) {
            this->replaceKeyframe(this->keyframes->operator[](i), ccc->GetSelectedKeyframe(), true);
            appliedChanges = true;
        }
    }
    if (!appliedChanges) {
        this->selectedKeyframe = this->interpolateKeyframe(ccc->GetSelectedKeyframe().GetAnimTime());
        this->updateEditParameters(this->selectedKeyframe);
        ccc->SetSelectedKeyframe(this->selectedKeyframe);
        //vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [CallForSetSelectedKeyframe] Selected keyframe doesn't exist. Changes are omitted.");
    }

    return true;
}


bool KeyframeKeeper::CallForGetSelectedKeyframeAtTime(core::Call& c) {

    auto ccc = dynamic_cast<CallKeyframeKeeper*>(&c);
    if (ccc == nullptr) return false;

    // Update selected keyframe
    Keyframe prevSelKf = this->selectedKeyframe;
    this->selectedKeyframe = this->interpolateKeyframe(ccc->GetSelectedKeyframe().GetAnimTime());
    this->updateEditParameters(this->selectedKeyframe);
    ccc->SetSelectedKeyframe(this->selectedKeyframe);
    // Linearize tangent
    this->linearizeSimTangent(prevSelKf);

    return true;
}


bool KeyframeKeeper::CallForSetCameraForKeyframe(core::Call& c) {

    auto ccc = dynamic_cast<CallKeyframeKeeper*>(&c);
    if (ccc == nullptr) return false;

    (*this->cameraState) = (*ccc->GetCameraState());

    return true;
}


bool KeyframeKeeper::CallForSetDragKeyframe(core::Call& c) {

    auto ccc = dynamic_cast<CallKeyframeKeeper*>(&c);
    if (ccc == nullptr) return false;

    /// (Checking if selected keyframe exists in keyframe array is done by caller)
    Keyframe skf = ccc->GetSelectedKeyframe();
    this->selectedKeyframe = this->interpolateKeyframe(skf.GetAnimTime());
    this->updateEditParameters(this->selectedKeyframe);
    this->dragDropKeyframe = this->selectedKeyframe;

    return true;
}


bool KeyframeKeeper::CallForSetDropKeyframe(core::Call& c) {

    auto ccc = dynamic_cast<CallKeyframeKeeper*>(&c);
    if (ccc == nullptr) return false;

    float t = ccc->GetDropAnimTime();
    t = (t < 0.0f) ? (0.0f) : (t);
    t = (t > this->totalAnimTime) ? (this->totalAnimTime) : (t);
    this->dragDropKeyframe.SetAnimTime(t);
    this->dragDropKeyframe.SetSimTime(ccc->GetDropSimTime());
    this->replaceKeyframe(this->selectedKeyframe, this->dragDropKeyframe, true);

    return true;
}


bool KeyframeKeeper::CallForSetCtrlPoints(core::Call& c) {

    auto ccc = dynamic_cast<CallKeyframeKeeper*>(&c);
    if (ccc == nullptr) return false;

    auto prev_StartCP = this->startCtrllPos;
    auto prev_EndCP = this->endCtrllPos;
    this->startCtrllPos = ccc->GetStartControlPointPosition();
    this->endCtrllPos = ccc->GetEndControlPointPosition();
    if ((prev_StartCP != this->startCtrllPos) || (prev_EndCP != this->endCtrllPos)) {
        this->addControlPointUndoAction(KeyframeKeeper::Undo::Action::UNDO_CONTROLPOINT_MODIFY, this->startCtrllPos, this->endCtrllPos, prev_StartCP, prev_EndCP);
        vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [CallForSetCtrlPoints] ADDED undo for CTRL POINT ......");
    }
    this->refreshInterpolCamPos(this->interpolSteps);
    ccc->SetControlPointPosition(this->startCtrllPos, this->endCtrllPos);

    return true;
}


bool KeyframeKeeper::CallForGetUpdatedKeyframeData(core::Call& c) {

    auto ccc = dynamic_cast<CallKeyframeKeeper*>(&c);
    if (ccc == nullptr) return false;

    // UPDATE PARAMETERS

    // interpolTangentParam ---------------------------------------------------
    if (this->interpolTangentParam.IsDirty()) {
        this->interpolTangentParam.ResetDirty();

        this->splineTangentLength = this->interpolTangentParam.Param<param::FloatParam>()->Value();
        this->refreshInterpolCamPos(this->interpolSteps);
    }
    // applyKeyframeParam -----------------------------------------------------
    if (this->applyKeyframeParam.IsDirty()) {
        this->applyKeyframeParam.ResetDirty();

        // Get current camera for selected keyframe
        Keyframe tmpKf = this->selectedKeyframe;
        tmpKf.SetCameraState((*this->cameraState));

        // Try adding keyframe to array
        if (!this->addKeyframe(tmpKf, true)) {
            if (!this->replaceKeyframe(this->selectedKeyframe, tmpKf, true)) {
                vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [CallForGetUpdatedKeyframeData] Unable to apply settings to new/selested keyframe.");
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
        if (!this->keyframes->empty()) {
            if (tt < this->keyframes->back().GetAnimTime()) {
                tt = this->keyframes->back().GetAnimTime();
                this->setTotalAnimTimeParam.Param<param::FloatParam>()->SetValue(tt, false);
                vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [CallForGetUpdatedKeyframeData] Total time is smaller than time of last keyframe. Delete Keyframe(s) to reduce total time to desired value.");
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
        int selIndex = this->getKeyframeIndex(this->keyframes, this->selectedKeyframe);
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
        int selIndex = this->getKeyframeIndex(this->keyframes, this->selectedKeyframe);
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

        //glm::vec3 posV = V2G(this->editCurrentPosParam.Param<param::Vector3fParam>()->Value()) + this->modelBboxCenter;
        //glm::vec3 pos = glm::vec3(posV.x, posV.y, posV.z);

        //// Get index of existing keyframe
        //int selIndex = this->getKeyframeIndex(this->keyframes, this->selectedKeyframe);
        //if (selIndex >= 0) {
        //    Keyframe tmpKf = this->selectedKeyframe;
        //    this->selectedKeyframe.SetCameraPosition(pos);
        //    this->replaceKeyframe(tmpKf, this->selectedKeyframe, true);
        //    // Refresh interoplated camera positions
        //    this->refreshInterpolCamPos(this->interpolSteps);
        //}
        //else {
        //    //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [CallForGetUpdatedKeyframeData] No existing keyframe selected.");
        //}
    }
    // editCurrentLookAtParam -------------------------------------------------
    if (this->editCurrentLookAtParam.IsDirty()) {
        this->editCurrentLookAtParam.ResetDirty();

        //glm::vec3 lookatV = V2G(this->editCurrentLookAtParam.Param<param::Vector3fParam>()->Value());
        //glm::vec3 lookat = glm::vec3(lookatV.x, lookatV.y, lookatV.z);

        //// Get index of existing keyframe
        //int selIndex = this->getKeyframeIndex(this->keyframes, this->selectedKeyframe);
        //if (selIndex >= 0) {
        //    Keyframe tmpKf = this->selectedKeyframe;
        //    this->selectedKeyframe.SetCameraLookAt(lookat);
        //    this->replaceKeyframe(tmpKf, this->selectedKeyframe, true);
        //}
        //else {
        //    //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [CallForGetUpdatedKeyframeData] No existing keyframe selected.");
        //}
    }
    // resetLookAtParam -------------------------------------------------------
    if (this->resetLookAtParam.IsDirty()) {
        this->resetLookAtParam.ResetDirty();

        //this->editCurrentLookAtParam.Param<param::Vector3fParam>()->SetValue(G2V(this->modelBboxCenter));
        //// Get index of existing keyframe
        //int selIndex = this->getKeyframeIndex(this->keyframes, this->selectedKeyframe);
        //if (selIndex >= 0) {
        //    Keyframe tmpKf = this->selectedKeyframe;
        //    this->selectedKeyframe.SetCameraLookAt(this->modelBboxCenter);
        //    this->replaceKeyframe(tmpKf, this->selectedKeyframe, true);
        //}
        //else {
        //    //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [CallForGetUpdatedKeyframeData] No existing keyframe selected.");
        //}
    }
    // editCurrentUpParam -----------------------------------------------------
    if (this->editCurrentUpParam.IsDirty()) {
        this->editCurrentUpParam.ResetDirty();

        //glm::vec3 up = V2G(this->editCurrentUpParam.Param<param::Vector3fParam>()->Value());

        //// Get index of existing keyframe
        //int selIndex = this->getKeyframeIndex(this->keyframes, this->selectedKeyframe);
        //if (selIndex >= 0) {
        //    Keyframe tmpKf = this->selectedKeyframe;
        //    this->selectedKeyframe.SetCameraUp(up);
        //    this->replaceKeyframe(tmpKf, this->selectedKeyframe, true);
        //}
        //else {
        //    //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [CallForGetUpdatedKeyframeData] No existing keyframe selected.");
        //}
    }
    // editCurrentUpParam -----------------------------------------------------
    if (this->editCurrentApertureParam.IsDirty()) {
        this->editCurrentApertureParam.ResetDirty();

        //float aperture = this->editCurrentApertureParam.Param<param::FloatParam>()->Value();

        //// Get index of existing keyframe
        //int selIndex = this->getKeyframeIndex(this->keyframes, this->selectedKeyframe);
        //if (selIndex >= 0) {
        //    Keyframe tmpKf = this->selectedKeyframe;
        //    this->selectedKeyframe.SetCameraApertureAngele(aperture);
        //    this->replaceKeyframe(tmpKf, this->selectedKeyframe, true);
        //}
        //else {
        //    //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [CallForGetUpdatedKeyframeData] No existing keyframe selected.");
        //}
    }
    // fileNameParam ----------------------------------------------------------
    if (this->fileNameParam.IsDirty()) {
        this->fileNameParam.ResetDirty();

        this->filename = vislib::StringA(this->fileNameParam.Param<param::FilePathParam>()->Value().PeekBuffer());
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

        for (unsigned int i = 0; i < this->keyframes->size(); i++) {
            this->snapKeyframe2AnimFrame(&this->keyframes->operator[](i));
        }
        this->snapKeyframe2AnimFrame(&this->selectedKeyframe);
    }
    // snapSimFramesParam -----------------------------------------------------
    if (this->snapSimFramesParam.IsDirty()) {
        this->snapSimFramesParam.ResetDirty();

        for (unsigned int i = 0; i < this->keyframes->size(); i++) {
            this->snapKeyframe2SimFrame(&this->keyframes->operator[](i));
        }
        this->snapKeyframe2SimFrame(&this->selectedKeyframe);      
    }
    // simTangentParam --------------------------------------------------------
    if (this->simTangentParam.IsDirty()) {
        this->simTangentParam.ResetDirty();

        this->simTangentStatus = true;
    }

    // PROPAGATE UPDATED DATA TO CALL -----------------------------------------
    ccc->SetCameraState(this->cameraState);
    ccc->SetKeyframes(this->keyframes);
    ccc->SetBoundingBox(this->boundingBox);
    ccc->SetSelectedKeyframe(this->selectedKeyframe);
    ccc->SetTotalAnimTime(this->totalAnimTime);
    ccc->SetInterpolCamPositions(this->interpolCamPos);
    ccc->SetTotalSimTime(this->totalSimTime);
    ccc->SetFps(this->fps);
    ccc->SetControlPointPosition(this->startCtrllPos, this->endCtrllPos);

    return true;
}


bool KeyframeKeeper::addKeyframeUndoAction(KeyframeKeeper::Undo::Action act, Keyframe kf, Keyframe prev_kf) {

    return (this->addUndoAction(act, kf, prev_kf, glm::vec3(), glm::vec3(), glm::vec3(), glm::vec3()));
}


bool KeyframeKeeper::addControlPointUndoAction(KeyframeKeeper::Undo::Action act, glm::vec3 first_controlpoint, glm::vec3 last_controlpoint, glm::vec3 previous_first_controlpoint, glm::vec3 previous_last_controlpoint) {

    return (this->addUndoAction(act, Keyframe(), Keyframe(), first_controlpoint, last_controlpoint, previous_first_controlpoint, previous_last_controlpoint));
}


bool KeyframeKeeper::addUndoAction(KeyframeKeeper::Undo::Action act, Keyframe kf, Keyframe prev_kf, glm::vec3 first_controlpoint, glm::vec3 last_controlpoint, glm::vec3 previous_first_controlpoint, glm::vec3 previous_last_controlpoint) {

    bool retVal = false;

    // Remove all already undone actions in list
    if (!this->undoQueue.empty() && (this->undoQueueIndex >= -1)) {
        if (this->undoQueueIndex < (int)(this->undoQueue.size() - 1)) {
            this->undoQueue.erase(this->undoQueue.begin() + (this->undoQueueIndex + 1), this->undoQueue.begin() + ((this->undoQueue.size() + 1) - (SIZE_T)(this->undoQueueIndex + 1)));
        }
    }

    this->undoQueue.emplace_back(Undo(act, kf, prev_kf, first_controlpoint, last_controlpoint, previous_first_controlpoint, previous_last_controlpoint));
    this->undoQueueIndex = (int)(this->undoQueue.size()) - 1;
    retVal = true;

    if (!retVal) {
        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [addUndoAction] Failed to add new undo action.");
    }

    return retVal;
}


bool KeyframeKeeper::undoAction() {

    bool retVal  = false;

    if (!this->undoQueue.empty() && (this->undoQueueIndex >= 0)) {

        Undo currentUndo = this->undoQueue[this->undoQueueIndex];

        switch (currentUndo.action) {
            case (KeyframeKeeper::Undo::Action::UNDO_NONE): 
                break;
            case (KeyframeKeeper::Undo::Action::UNDO_KEYFRAME_ADD): 
                // Revert adding a keyframe (= delete)
                if (this->deleteKeyframe(currentUndo.keyframe, false)) {
                    this->undoQueueIndex--;
                    retVal = true;
                }
                break;
            case (KeyframeKeeper::Undo::Action::UNDO_KEYFRAME_DELETE):
                // Revert deleting a keyframe (= add)
                if (this->addKeyframe(currentUndo.keyframe, false)) {
                    this->undoQueueIndex--;
                    retVal = true; 
                }
                break;
            case (KeyframeKeeper::Undo::Action::UNDO_KEYFRAME_MODIFY): 
                // Revert changes made to a keyframe.
                if (this->replaceKeyframe(currentUndo.keyframe, currentUndo.previous_keyframe, false)) {
                    this->undoQueueIndex--;
                    retVal = true;
                }
                break;
            case (KeyframeKeeper::Undo::Action::UNDO_CONTROLPOINT_MODIFY):
                // Revert changes made to the control points.
                this->startCtrllPos = currentUndo.previous_first_controlpoint;
                this->endCtrllPos   = currentUndo.previous_last_controlpoint;
                this->undoQueueIndex--;
                retVal = true;
                break;
            default: break;
        }
    }    
    //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [undoAction] Undo queue index: %d - Undo queue size: %d", this->undoQueueIndex, this->undoQueue.size());

    if (!retVal) {
        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [undoAction] Failed to undo changes.");
    }

    return retVal;
}


bool KeyframeKeeper::redoAction() {

    bool retVal = false;

    if (!this->undoQueue.empty() && (this->undoQueueIndex < (int)(this->undoQueue.size() - 1))) {

        this->undoQueueIndex++;
        if (this->undoQueueIndex < 0) {
            this->undoQueueIndex = 0;
        }
        Undo currentUndo = this->undoQueue[this->undoQueueIndex];
        switch (currentUndo.action) {
        case (KeyframeKeeper::Undo::Action::UNDO_NONE):
            break;
        case (KeyframeKeeper::Undo::Action::UNDO_KEYFRAME_ADD):
            // Redo adding a keyframe (= delete)
            if (this->addKeyframe(currentUndo.keyframe, false)) {
                retVal = true;
            }
            break;
        case (KeyframeKeeper::Undo::Action::UNDO_KEYFRAME_DELETE):
            // Redo deleting a keyframe (= add)
            if (this->deleteKeyframe(currentUndo.keyframe, false)) {
                retVal = true;
            }
            break;
        case (KeyframeKeeper::Undo::Action::UNDO_KEYFRAME_MODIFY):
            // Redo changes made to a keyframe.
            if (this->replaceKeyframe(currentUndo.previous_keyframe, currentUndo.keyframe, false)) {
                retVal = true;
            }
            break;
        case (KeyframeKeeper::Undo::Action::UNDO_CONTROLPOINT_MODIFY):
            // Revert changes made to the control points.
            this->startCtrllPos = currentUndo.first_controlpoint;
            this->endCtrllPos   = currentUndo.last_controlpoint;
            retVal = true;
            break;
        default: break;
        }
    }
    //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] Undo queue index: %d - Undo queue size: %d", this->undoQueueIndex, this->undoQueue.size());

    if (!retVal) {
        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [redoAction] Failed to redo changes.");
    }

    return retVal;
}


void KeyframeKeeper::linearizeSimTangent(Keyframe stkf) {

    // Linearize tangent between simTangentKf and currently selected keyframe by shifting all inbetween keyframes simulation time
    if (this->simTangentStatus) {
        // Linearize tangent only between existing keyframes
        if (this->getKeyframeIndex(this->keyframes, stkf) >= 0) {
            vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [linearizeSimTangent] Select existing keyframe before trying to linearize the tangent.");
            this->simTangentStatus = false;
        }
        else if (this->getKeyframeIndex(this->keyframes, this->selectedKeyframe) >= 0) {

            // Calculate liner equation between the two selected keyframes 
            // f(x) = mx + b
            glm::vec2 p1 = glm::vec2(this->selectedKeyframe.GetAnimTime(), this->selectedKeyframe.GetSimTime());
            glm::vec2 p2 = glm::vec2(stkf.GetAnimTime(), stkf.GetSimTime());
            float m = (p1.y - p2.y) / (p1.x - p2.x);
            float b = m * (-p1.x) + p1.y;
            // Get indices
            int iKf1 = this->getKeyframeIndex(this->keyframes, this->selectedKeyframe);
            int iKf2 = this->getKeyframeIndex(this->keyframes, stkf);
            if (iKf1 > iKf2) {
                int tmp = iKf1;
                iKf1 = iKf2;
                iKf2 = tmp;
            }
            // Consider only keyframes lying between the two selected ones
            float newSimTime;
            for (int i = iKf1 + 1; i < iKf2; i++) {
                newSimTime = m * (this->keyframes->operator[](i).GetAnimTime()) + b;

                // ADD UNDO
                // Store old keyframe
                Keyframe tmpKf = this->keyframes->operator[](i);
                // Apply changes to keyframe
                this->keyframes->operator[](i).SetSimTime(newSimTime);
                // Add modification to undo queue
                this->addKeyframeUndoAction(KeyframeKeeper::Undo::Action::UNDO_KEYFRAME_MODIFY, this->keyframes->operator[](i), tmpKf);
            }
    
            this->simTangentStatus = false;
        }
        else {
            vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [linearizeSimTangent] Select existing keyframe to finish linearizing the tangent.");
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

    // ADD UNDO
    Keyframe tmpKf = *kf;
    kf->SetAnimTime(t);
    this->addKeyframeUndoAction(KeyframeKeeper::Undo::Action::UNDO_KEYFRAME_MODIFY, *kf, tmpKf);
}


void KeyframeKeeper::snapKeyframe2SimFrame(Keyframe *kf) {

    float s = std::round(kf->GetSimTime() * this->totalSimTime) / this->totalSimTime;

    // ADD UNDO
    Keyframe tmpKf = *kf;
    kf->SetSimTime(s);
    this->addKeyframeUndoAction(KeyframeKeeper::Undo::Action::UNDO_KEYFRAME_MODIFY, *kf, tmpKf);
}


void KeyframeKeeper::setSameSpeed() {

    if (this->keyframes->size() > 2) {
        // Store index of selected keyframe to restore seleection after changing time of keyframes
        int selIndex = this->getKeyframeIndex(this->keyframes, this->selectedKeyframe);

        // Get total values
        float totTime = this->keyframes->back().GetAnimTime() - this->keyframes->front().GetAnimTime();
        if (totTime == 0.0f) {
            vislib::sys::Log::DefaultLog.WriteError("[KEYFRAME KEEPER] [setSameSpeed] totTime is ZERO.");
            return;
        }
        float totDist = 0.0f;
        for (unsigned int i = 0; i < this->interpolCamPos->size() - 1; i++) {
            totDist += glm::length(this->interpolCamPos->operator[](i + 1) - this->interpolCamPos->operator[](i));
        }
        float totalVelocity = totDist / totTime; // unit doesn't matter ... it is only relative

        // Get values between two consecutive keyframes and shift remoter keyframe if necessary
        float kfTime = 0.0f;
        float kfDist = 0.0f;
        for (unsigned int i = 0; i < this->interpolCamPos->size() - 2; i++) {
            if ((i > 0) && (i % this->interpolSteps == 0)) {  // skip checking for first keyframe (last keyframe is skipped by prior loop)
                kfTime = kfDist / totalVelocity;

                unsigned int index = static_cast<unsigned int>(floorf(((float)i / (float)this->interpolSteps)));
                float t = this->keyframes->operator[](index - 1).GetAnimTime() + kfTime;
                t = (t < 0.0f) ? (0.0f) : (t);
                t = (t > this->totalAnimTime) ? (this->totalAnimTime) : (t);
                kfDist = 0.0f;

                // ADD UNDO
                Keyframe tmpKf = this->keyframes->operator[](index);
                this->keyframes->operator[](index).SetAnimTime(t);
                this->addKeyframeUndoAction(KeyframeKeeper::Undo::Action::UNDO_KEYFRAME_MODIFY, this->keyframes->operator[](index), tmpKf);
            }
            // Add distance up to existing keyframe
            kfDist += glm::length(this->interpolCamPos->operator[](i + 1) - this->interpolCamPos->operator[](i));
        }

        // Restore previous selected keyframe
        if (selIndex >= 0) {
            this->selectedKeyframe = this->keyframes->operator[](selIndex);
        }
    }
}


void KeyframeKeeper::refreshInterpolCamPos(unsigned int s) {

    this->interpolCamPos->clear();
    this->interpolCamPos->reserve(1000);
    if (s == 0) {
        vislib::sys::Log::DefaultLog.WriteError("[KEYFRAME KEEPER] [refreshInterpolCamPos] Interpolation step count is ZERO.");
        return;
    }

    float startTime;
    float deltaTimeStep;
    Keyframe kf;
    if (this->keyframes->size() > 1) {
        for (unsigned int i = 0; i < this->keyframes->size() - 1; i++) {
            startTime = this->keyframes->operator[](i).GetAnimTime();
            deltaTimeStep = (this->keyframes->operator[](i + 1).GetAnimTime() - startTime) / (float)s;

            for (unsigned int j = 0; j < s; j++) {
                kf = this->interpolateKeyframe(startTime + deltaTimeStep*(float)j);
                auto p = kf.GetCameraState().position;
                this->interpolCamPos->emplace_back(glm::vec3(p[0], p[1], p[2]));
				glm::vec3 grow = this->interpolCamPos->back();
                this->boundingBox->GrowToPoint(grow.x, grow.y, grow.z);
            }
        }
        // Add last existing camera position
        auto p = this->keyframes->back().GetCameraState().position;
        this->interpolCamPos->emplace_back(glm::vec3(p[0], p[1], p[2]));
    }
}


bool KeyframeKeeper::replaceKeyframe(Keyframe oldkf, Keyframe newkf, bool undo) {

    if (!this->keyframes->empty()) {
        // Both are equal ... nothing to do
        if (oldkf == newkf) {
            return true;
        }
        // Check if old keyframe exists
        int selIndex = this->getKeyframeIndex(this->keyframes, oldkf);
        if (selIndex >= 0) {
            // Delete old keyframe
            this->deleteKeyframe(oldkf, false);
            // Try to add new keyframe
            if (!this->addKeyframe(newkf, false)) {
                // There is alredy a keyframe on the new position ... overwrite existing keyframe.
                float newAnimTime = newkf.GetAnimTime();
                for (unsigned int i = 0; i < this->keyframes->size(); i++) {
                    if (this->keyframes->operator[](i).GetAnimTime() == newAnimTime) {
                        this->deleteKeyframe(this->keyframes->operator[](i), true);
                        break;
                    }
                }
                this->addKeyframe(newkf, false);
            }
            if (undo) {
                // ADD UNDO
                this->addKeyframeUndoAction(KeyframeKeeper::Undo::Action::UNDO_KEYFRAME_MODIFY, newkf, oldkf);
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

    if (!this->keyframes->empty()) {
        // Get index of keyframe to delete
        unsigned int selIndex = this->getKeyframeIndex(this->keyframes, kf);
        // Choose new selected keyframe
        if (selIndex >= 0) {
            // DELETE UNDO
            this->keyframes->erase(this->keyframes->begin() + selIndex);
            if (undo) {
                // ADD UNDO
                this->addKeyframeUndoAction(KeyframeKeeper::Undo::Action::UNDO_KEYFRAME_DELETE, kf, kf);
                // Adjust first/last control point position - ONLY if it is a "real" delete and no replace
                if (this->keyframes->size() > 1) {
                    if (selIndex == 0) {
                        auto p0 = this->keyframes->operator[](0).GetCameraState().position;
                        auto p1 = this->keyframes->operator[](1).GetCameraState().position;
                        glm::vec3 tmpV = glm::normalize(glm::vec3(p0[0], p0[1], p0[2]) - glm::vec3(p1[0], p1[1], p1[2]));
                        this->startCtrllPos = glm::vec3(p0[0], p0[1], p0[2]) + tmpV;
                    }
                    if (selIndex == this->keyframes->size()) { // Element is already removed so the index is now: (this->keyframes->size() - 1) + 1
                        auto p0 = this->keyframes->back().GetCameraState().position;
                        auto p1 = this->keyframes->operator[]((int)this->keyframes->size() - 2).GetCameraState().position;
                        glm::vec3 tmpV = glm::normalize(glm::vec3(p0[0], p0[1], p0[2]) - glm::vec3(p1[0], p1[1], p1[2]));
                        this->endCtrllPos = glm::vec3(p0[0], p0[1], p0[2]) + tmpV;
                    }
                }
            }
            this->boundingBox->SetNull();
            this->refreshInterpolCamPos(this->interpolSteps);
            if (selIndex > 0) {
                this->selectedKeyframe = this->keyframes->operator[](selIndex - 1);
            }
            else if (selIndex < this->keyframes->size()) {
                this->selectedKeyframe = this->keyframes->operator[](selIndex);
            }
            this->updateEditParameters(this->selectedKeyframe);
        }
        else {
            //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [deleteKeyframe] No existing keyframe selected.");
            return false;
        }
    }

    return true;
}


bool KeyframeKeeper::addKeyframe(Keyframe kf, bool undo) {

    float time = kf.GetAnimTime();

    // Check if keyframe already exists
    for (unsigned int i = 0; i < this->keyframes->size(); i++) {
        if (this->keyframes->operator[](i).GetAnimTime() == time) {
            //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [addKeyframe] Keyframe already exists.");
            return false;
        }
    }

    // Sort new keyframe to keyframe array
    if (this->keyframes->empty() || (this->keyframes->back().GetAnimTime() < time)) {
        this->keyframes->emplace_back(kf);
        // Adjust first/last control point position - ONLY if it is a "real" add and no replace
        if (undo && this->keyframes->size() > 1) {
            auto p0 = this->keyframes->back().GetCameraState().position;
            auto p1 = this->keyframes->operator[]((int)this->keyframes->size() - 2).GetCameraState().position;
            glm::vec3 tmpV = glm::normalize(glm::vec3(p0[0], p0[1], p0[2]) - glm::vec3(p1[0], p1[1], p1[2]));
            this->endCtrllPos = glm::vec3(p0[0], p0[1], p0[2]) + tmpV;
        }
    }
    else if (time < this->keyframes->front().GetAnimTime()) {
        this->keyframes->insert(this->keyframes->begin(), kf);
        // Adjust first/last control point position - ONLY if it is a "real" add and no replace
        if (undo && this->keyframes->size() > 1) {
            auto p0 = this->keyframes->back().GetCameraState().position;
            auto p1 = this->keyframes->operator[]((int)this->keyframes->size() - 2).GetCameraState().position;
            glm::vec3 tmpV = glm::normalize(glm::vec3(p0[0], p0[1], p0[2]) - glm::vec3(p1[0], p1[1], p1[2]));
            this->endCtrllPos = glm::vec3(p0[0], p0[1], p0[2]) + tmpV;
        }
    }
    else { // Insert keyframe in-between existing keyframes
        unsigned int insertIdx = 0;
        for (unsigned int i = 0; i < this->keyframes->size(); i++) {
            if (time < this->keyframes->operator[](i).GetAnimTime()) {
                insertIdx = i;
                break;
            }
        }
        this->keyframes->insert(this->keyframes->begin() + insertIdx, kf);
    }

    // ADD UNDO 
    if (undo) {
        this->addKeyframeUndoAction(KeyframeKeeper::Undo::Action::UNDO_KEYFRAME_ADD, kf, kf);
    }

    // Extend camera position for bounding box to cover manipulator axis
    megamol::core::view::Camera_2 cam(kf.GetCameraState());
    cam_type::snapshot_type snapshot;
    cam.take_snapshot(snapshot, thecam::snapshot_content::all);
    glm::vec4 view = snapshot.view_vector;
    glm::vec4 pos = snapshot.position;
	glm::vec4 grow = pos + glm::normalize(pos - view) * 1.5f;
    this->boundingBox->GrowToPoint(grow.x, grow.y, grow.z);
    // Refresh interoplated camera positions
    this->refreshInterpolCamPos(this->interpolSteps);

    this->selectedKeyframe = kf;
    this->updateEditParameters(this->selectedKeyframe);

    return true;
}


Keyframe KeyframeKeeper::interpolateKeyframe(float time) {

    float t = time;
    t = (t < 0.0f) ? (0.0f) : (t);
    t = (t > this->totalAnimTime) ? (this->totalAnimTime) : (t);

    // Check if there is an existing keyframe at requested time
    for (SIZE_T i = 0; i < this->keyframes->size(); i++) {
        if (t == this->keyframes->operator[](i).GetAnimTime()) {
            return this->keyframes->operator[](i);
        }
    }

    if (this->keyframes->empty()) {
        // vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [interpolateKeyframe] Empty keyframe array.");
        Keyframe kf = Keyframe();
        kf.SetAnimTime(t);
        kf.SetSimTime(0.0f);
        kf.SetCameraState((*this->cameraState));
        return kf;
    }
    else if (t < this->keyframes->front().GetAnimTime()) {
        /**/
        Keyframe kf = this->keyframes->front();
        kf.SetAnimTime(t);
        /**/
        /*
        Keyframe kf = Keyframe();
        kf.SetAnimTime(t);
        kf.SetSimTime(this->keyframes->front().GetSimTime());
        kf.SetCameraUp(this->camViewUp);
        kf.SetCameraPosition(this->camViewPosition);
        kf.SetCameraLookAt(this->camViewLookat);
        kf.SetCameraApertureAngele(this->camViewApertureangle);
        */
        return kf;

    }
    else if (t > this->keyframes->back().GetAnimTime()) {
        /*
        Keyframe kf = this->keyframes->back();
        kf.SetAnimTime(t);
        */
        Keyframe kf = Keyframe();
        kf.SetAnimTime(t);
        kf.SetSimTime(this->keyframes->back().GetSimTime());
        kf.SetCameraState((*this->cameraState));
        return kf;
    }
    else { // if ((t > this->keyframes->front().GetAnimTime()) && (t < this->keyframes->back().GetAnimTime())) {

        // New default keyframe
        Keyframe kf = Keyframe();
        megamol::core::view::Camera_2 cam_kf = kf.GetCameraState();

        // Nothing to do for animation time
        kf.SetAnimTime(t);

        // Determine indices for interpolation 
        int i0 = 0;
        int i1 = 0;
        int i2 = 0;
        int i3 = 0;
        int kfIdxCnt = (int)this->keyframes->size() - 1;
        float iT = 0.0f;
        for (int i = 0; i < kfIdxCnt; i++) {
            float tMin = this->keyframes->operator[](i).GetAnimTime();
            float tMax = this->keyframes->operator[](i + 1).GetAnimTime();
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
        float simT1 = this->keyframes->operator[](i1).GetSimTime();
        float simT2 = this->keyframes->operator[](i2).GetSimTime();
        float simT = simT1 + (simT2 - simT1)*iT;
        kf.SetSimTime(simT);

        // ! Skip interpolation of camera parameters if they are equal for ?1 and ?2.
        // => Prevent interpolation loops if time of keyframes is different, but cam params are the same.

        cam_type::snapshot_type s0;
        megamol::core::view::Camera_2 c0(this->keyframes->operator[](i0).GetCameraState());
        c0.take_snapshot(s0, thecam::snapshot_content::all);

        cam_type::snapshot_type s1;
        megamol::core::view::Camera_2 c1 = this->keyframes->operator[](i1).GetCameraState();
        c1.take_snapshot(s1, thecam::snapshot_content::all);

        cam_type::snapshot_type s2;
        megamol::core::view::Camera_2 c2 = this->keyframes->operator[](i2).GetCameraState();
        c2.take_snapshot(s2, thecam::snapshot_content::all);

        cam_type::snapshot_type s3;
        megamol::core::view::Camera_2 c3 = this->keyframes->operator[](i3).GetCameraState();
        c3.take_snapshot(s3, thecam::snapshot_content::all);

        //interpolate position ------------------------------------------------
        glm::vec4 ps0 = s0.position;
        glm::vec4 ps1 = s1.position;
        glm::vec4 ps2 = s2.position;
        glm::vec4 ps3 = s3.position;
        glm::vec3 p0 = glm::vec3(ps0.x, ps0.y, ps0.z);
        glm::vec3 p1 = glm::vec3(ps1.x, ps1.y, ps1.z);
        glm::vec3 p2 = glm::vec3(ps2.x, ps2.y, ps2.z);
        glm::vec3 p3 = glm::vec3(ps3.x, ps3.y, ps3.z);
        /// Use additional control point positions to manipulate interpolation curve for first and last keyframe
        if (p0 == p1) {
            p0 = this->startCtrllPos;
        }
        if (p2 == p3) {
            p3 = this->endCtrllPos;
        }
        if (p1 == p2) {
            cam_kf.position(s1.position);
        }
        else {
            glm::vec3 pk = this->interpolate_vec3(iT, p0, p1, p2, p3);
            cam_kf.position(glm::vec4(pk.x, pk.y, pk.z, 1.0f));
        }

        //interpolate look at -------------------------------------------------
        glm::vec4 ls0 = s0.view_vector;
        glm::vec4 ls1 = s1.view_vector;
        glm::vec4 ls2 = s2.view_vector;
        glm::vec4 ls3 = s3.view_vector;
        glm::vec3 l0 = glm::vec3(ls0.x, ls0.y, ls0.z);
        glm::vec3 l1 = glm::vec3(ls1.x, ls1.y, ls1.z);
        glm::vec3 l2 = glm::vec3(ls2.x, ls2.y, ls2.z);
        glm::vec3 l3 = glm::vec3(ls3.x, ls3.y, ls3.z);
        if (l1 == l2) {
            cam_kf.look_at(s1.view_vector);
        }
        else {
            glm::vec3 lk = this->interpolate_vec3(iT, l0, l1, l2, l3);
            cam_kf.look_at(glm::vec4(lk.x, lk.y, lk.z, 1.0f));
        }

        //interpolate up ------------------------------------------------------
        glm::vec4 us0 = s0.up_vector;
        glm::vec4 us1 = s1.up_vector;
        glm::vec4 us2 = s2.up_vector;
        glm::vec4 us3 = s3.up_vector;
        glm::vec3 u0 = glm::vec3(us0.x, us0.y, us0.z);
        glm::vec3 u1 = glm::vec3(us1.x, us1.y, us1.z);
        glm::vec3 u2 = glm::vec3(us2.x, us2.y, us2.z);
        glm::vec3 u3 = glm::vec3(us3.x, us3.y, us3.z);
        //if (u1 == u2) {
        //    kf.SetCameraUp(this->keyframes->operator[](i1).GetCamUp());
        //}
        //else {
        //    glm::vec3 uk = this->interpolate_vec3(iT, u0, u1, u2, u3);
        //    kf.SetCameraUp(uk - pk);
        //}

        //interpolate aperture angle ------------------------------------------
        float a0 = c0.aperture_angle();
        float a1 = c1.aperture_angle();
        float a2 = c2.aperture_angle();
        float a3 = c3.aperture_angle();
        if (a1 == a2) {
            cam_kf.aperture_angle(a1);
        }
        else {
            float ak = this->interpolate_f(iT, a0, a1, a2, a3);
            cam_kf.aperture_angle(ak);
        }

        // Finally set new interpoated camera of keyframe
        cam_type::minimal_state_type camera_state;
        cam_kf.get_minimal_state(camera_state);
        kf.SetCameraState(camera_state);

        return kf;
    }
}


float KeyframeKeeper::interpolate_f(float u, float f0, float f1, float f2, float f3) {

    /// Catmull-Rom
    // Original version:
    /* float f = ((f1 * 2.0f) +
               (f2 - f0) * u +
              ((f0 * 2.0f) - (f1 * 5.0f) + (f2 * 4.0f) - f3) * u * u +
              (-f0 + (f1 * 3.0f) - (f2 * 3.0f) + f3) * u * u * u) * 0.5f;*/
    // Considering global tangent length:
    // SOURCE: https://www.cs.cmu.edu/~462/projects/assn2/assn2/catmullRom.pdf
    float tl = this->splineTangentLength;
    float f = (f1) +
              (-(tl * f0) + (tl* f2)) * u + 
              ((2.0f*tl * f0) + ((tl - 3.0f) * f1) + ((3.0f - 2.0f*tl) * f2) - (tl* f3)) * u * u + 
              (-(tl * f0) + ((2.0f - tl) * f1) + ((tl - 2.0f) * f2) + (tl* f3)) * u * u * u;

    return f;
}


glm::vec3 KeyframeKeeper::interpolate_vec3(float u, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3) {

    glm::vec3 v;
    v.x = this->interpolate_f(u, v0.x, v1.x, v2.x, v3.x);
    v.y = this->interpolate_f(u, v0.y, v1.y, v2.y, v3.y);
    v.z = this->interpolate_f(u, v0.z, v1.z, v2.z, v3.z);

    return v;
}


void KeyframeKeeper::saveKeyframes() {

    ///TODO move to json format (keep for compatibility?)

//    if (this->filename.IsEmpty()) {
//        vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [saveKeyframes] No filename given. Using default filename.");
//        time_t t = std::time(0);  // get time now
//        struct tm *now = nullptr;
//#if (defined(_MSC_VER) && (_MSC_VER > 1000))
//        struct tm nowdata;
//        now = &nowdata;
//        localtime_s(now, &t);
//#else /* defined(_WIN32) && (_MSC_VER >= 1400) */
//        now = localtime(&t);
//#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
//        this->filename.Format("keyframes_%i%02i%02i-%02i%02i%02i.kf", (now->tm_year + 1900), (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
//        this->fileNameParam.Param<param::FilePathParam>()->SetValue(this->filename, false);
//    } 
//
//    std::ofstream outfile;
//    outfile.open(this->filename.PeekBuffer(), std::ios::binary);
//    vislib::StringSerialiserA ser;
//    outfile << "totalAnimTime=" << this->totalAnimTime << "\n";
//    outfile << "tangentLength=" << this->splineTangentLength << "\n";
//    outfile << "startCtrllPosX=" << this->startCtrllPos.x << "\n";
//    outfile << "startCtrllPosY=" << this->startCtrllPos.y << "\n";
//    outfile << "startCtrllPosZ=" << this->startCtrllPos.z << "\n";
//    outfile << "endCtrllPosX=" << this->endCtrllPos.x << "\n";
//    outfile << "endCtrllPosY=" << this->endCtrllPos.y << "\n";
//    outfile << "endCtrllPosZ=" << this->endCtrllPos.z << "\n\n";
//    for (unsigned int i = 0; i < this->keyframes->size(); i++) {
//        this->keyframes->operator[](i).Serialise(ser);
//        outfile << ser.GetString().PeekBuffer() << "\n";
//    }
//    outfile.close();
//
//    vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] Successfully stored keyframes to file: %s", this->filename.PeekBuffer());

}


void KeyframeKeeper::loadKeyframes() {

    ///TODO move to json format (keep for compatibility?)

    //if (this->filename.IsEmpty()) {
    //    vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] No filename given.");
    //}
    //else {
    //    std::ifstream infile;
    //    infile.open(this->filename.PeekBuffer());
    //    if (!infile.is_open()) {
    //        vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] Failed to open keyframe file.");
    //        return;
    //    }

    //    // Reset keyframe array and bounding box
    //    this->keyframes->clear();
    //    this->boundingBox->SetNull();

    //    vislib::StringSerialiserA ser;
    //    std::string line;
    //    vislib::StringA cameraStr = "";;

    //    // get total time
    //    std::getline(infile, line); 
    //    this->totalAnimTime = std::stof(line.erase(0, 14)); // "totalAnimTime="
    //    this->setTotalAnimTimeParam.Param<param::FloatParam>()->SetValue(this->totalAnimTime, false);

    //    // Consume empty line
    //    std::getline(infile, line);

    //    // Make compatible with previous version
    //    if (line.find("Length",0) < line.length()) {
    //        // get tangentLength
    //        //std::getline(infile, line);
    //        this->splineTangentLength = std::stof(line.erase(0, 14)); // "tangentLength="
    //        // get startCtrllPos
    //        std::getline(infile, line);
    //        this->startCtrllPos.x = std::stof(line.erase(0, 15)); // "startCtrllPosX="
    //        std::getline(infile, line);
    //        this->startCtrllPos.y = std::stof(line.erase(0, 15)); // "startCtrllPosY="
    //        std::getline(infile, line);
    //        this->startCtrllPos.z = std::stof(line.erase(0, 15)); // "startCtrllPosZ="
    //        // get endCtrllPos
    //        std::getline(infile, line);
    //        this->endCtrllPos.x = std::stof(line.erase(0, 13)); // "endCtrllPosX="
    //        std::getline(infile, line);
    //        this->endCtrllPos.y = std::stof(line.erase(0, 13)); // "endCtrllPosY="
    //        std::getline(infile, line);
    //        this->endCtrllPos.z = std::stof(line.erase(0, 13)); // "endCtrllPosZ="
    //        // Consume empty line
    //        std::getline(infile, line);
    //    }
    //    else {
    //        vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [Load Keyframes] Loading keyframes stored in OLD format - Save keyframes to current file to convert to new format.");
    //    }


    //    // One frame consists of an initial "time"-line followed by the serialized camera parameters and an final empty line
    //    while (std::getline(infile, line)) {
    //        if ((line.empty()) && !(cameraStr.IsEmpty())) { // new empty line indicates current frame is complete
    //            ser.SetInputString(cameraStr);
    //            Keyframe kf;
    //            kf.Deserialise(ser);
    //            this->keyframes->emplace_back(kf);
    //            // Extend camera position for bounding box to cover manipulator axis
    //            glm::vec3 manipulator = glm::vec3(kf.GetCamLookAt().x, kf.GetCamLookAt().y, kf.GetCamLookAt().z);
    //            manipulator = kf.GetCamPosition() - manipulator;
    //            manipulator = glm::normalize(manipulator) * 1.5f;
				//glm::vec3 grow = kf.GetCamPosition() + manipulator;
    //            this->boundingBox->GrowToPoint(grow.x, grow.y, grow.z);
    //            cameraStr.Clear();
    //            ser.ClearData();
    //        }
    //        else {
    //            cameraStr.Append(line.c_str());
    //            cameraStr.Append("\n");
    //        }
    //    }
    //    infile.close();

    //    if (!this->keyframes->empty()) {
    //        // Set selected keyframe to first in keyframe array
    //        this->selectedKeyframe = this->interpolateKeyframe(0.0f);
    //        this->updateEditParameters(this->selectedKeyframe);
    //        // Refresh interoplated camera positions
    //        this->refreshInterpolCamPos(this->interpolSteps);
    //    }
    //    vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] Successfully loaded keyframes from file: %s", this->filename.PeekBuffer());
    //}
}


void KeyframeKeeper::updateEditParameters(Keyframe kf) {

    //// Set new parameter values of changed selected keyframe
    //glm::vec3 lookatV = kf.GetCamera().l;
    //glm::vec3  lookat = glm::vec3(lookatV.x, lookatV.y, lookatV.z);
    //glm::vec3 posV = kf.GetCamPosition();
    //glm::vec3  pos = glm::vec3(posV.x, posV.y, posV.z);
    //this->editCurrentAnimTimeParam.Param<param::FloatParam>()->SetValue(kf.GetAnimTime(), false);
    //this->editCurrentSimTimeParam.Param<param::FloatParam>()->SetValue(kf.GetSimTime() * this->totalSimTime, false);
    //this->editCurrentPosParam.Param<param::Vector3fParam>()->SetValue(G2V(pos - this->modelBboxCenter), false);
    //this->editCurrentLookAtParam.Param<param::Vector3fParam>()->SetValue(G2V(lookat), false);
    //this->editCurrentUpParam.Param<param::Vector3fParam>()->SetValue(G2V(kf.GetCamUp()), false);
    //this->editCurrentApertureParam.Param<param::FloatParam>()->SetValue(kf.GetCamApertureAngle(), false);
}


int KeyframeKeeper::getKeyframeIndex(std::shared_ptr<std::vector<Keyframe>> keyframes, Keyframe keyframe) {

    int count = keyframes->size();
    for (int i = 0; i < count; ++i) {
        if (keyframes->operator[](i) == keyframe) {
            return i;
        }
    }
    return -1;
}
