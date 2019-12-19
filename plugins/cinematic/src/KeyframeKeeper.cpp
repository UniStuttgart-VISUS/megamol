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
    , keyframeCallSlot("keyframeData", "holds keyframe data")
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
    , resetViewParam("editSelected::resetLookAt", "Reset the 'look at' vector of the selected keyframe to the center of the model boundng box.")
    , editCurrentViewParam("editSelected::lookAtVector", "Edit 'look at' vector of the selected keyframe.")
    , editCurrentUpParam("editSelected::upVector", "Edit up vector direction relative to 'look at' vector of the selected keyframe.")
    , editCurrentApertureParam("editSelected::apertureAngle", "Edit apperture angle of the selected keyframe.")
    , fileNameParam("storage::filename", "The name of the file to load or save keyframes.")
    , saveKeyframesParam("storage::save", "Save keyframes to file.")
    , loadKeyframesParam("storage::load", "Load keyframes from file.")
    , cameraState()
    , interpolCamPos()
    , keyframes()
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
    this->keyframeCallSlot.SetCallback(CallKeyframeKeeper::ClassName(),
        CallKeyframeKeeper::FunctionName(CallKeyframeKeeper::CallForGetUpdatedKeyframeData), &KeyframeKeeper::CallForGetUpdatedKeyframeData);

    this->keyframeCallSlot.SetCallback(CallKeyframeKeeper::ClassName(),
        CallKeyframeKeeper::FunctionName(CallKeyframeKeeper::CallForSetSimulationData), &KeyframeKeeper::CallForSetSimulationData);

    this->keyframeCallSlot.SetCallback(CallKeyframeKeeper::ClassName(),
        CallKeyframeKeeper::FunctionName(CallKeyframeKeeper::CallForGetInterpolCamPositions), &KeyframeKeeper::CallForGetInterpolCamPositions);

    this->keyframeCallSlot.SetCallback(CallKeyframeKeeper::ClassName(),
        CallKeyframeKeeper::FunctionName(CallKeyframeKeeper::CallForSetSelectedKeyframe), &KeyframeKeeper::CallForSetSelectedKeyframe);

    this->keyframeCallSlot.SetCallback(CallKeyframeKeeper::ClassName(),
        CallKeyframeKeeper::FunctionName(CallKeyframeKeeper::CallForGetSelectedKeyframeAtTime), &KeyframeKeeper::CallForGetSelectedKeyframeAtTime);

    this->keyframeCallSlot.SetCallback(CallKeyframeKeeper::ClassName(),
        CallKeyframeKeeper::FunctionName(CallKeyframeKeeper::CallForSetCameraForKeyframe), &KeyframeKeeper::CallForSetCameraForKeyframe);

    this->keyframeCallSlot.SetCallback(CallKeyframeKeeper::ClassName(),
        CallKeyframeKeeper::FunctionName(CallKeyframeKeeper::CallForSetDragKeyframe), &KeyframeKeeper::CallForSetDragKeyframe);

    this->keyframeCallSlot.SetCallback(CallKeyframeKeeper::ClassName(),
        CallKeyframeKeeper::FunctionName(CallKeyframeKeeper::CallForSetDropKeyframe), &KeyframeKeeper::CallForSetDropKeyframe);

    this->keyframeCallSlot.SetCallback(CallKeyframeKeeper::ClassName(),
        CallKeyframeKeeper::FunctionName(CallKeyframeKeeper::CallForSetCtrlPoints), &KeyframeKeeper::CallForSetCtrlPoints);

    this->MakeSlotAvailable(&this->keyframeCallSlot);

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

    this->resetViewParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_U, core::view::Modifier::CTRL));
    this->MakeSlotAvailable(&this->resetViewParam);
    
    this->editCurrentViewParam.SetParameter(new param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f)));
    this->MakeSlotAvailable(&this->editCurrentViewParam);

    this->editCurrentUpParam.SetParameter(new param::Vector3fParam(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f)));
    this->MakeSlotAvailable(&this->editCurrentUpParam);

    this->editCurrentApertureParam.SetParameter(new param::FloatParam(60.0f, 0.0f, 180.0f));
    this->MakeSlotAvailable(&this->editCurrentApertureParam);

    this->fileNameParam.SetParameter(new param::FilePathParam(vislib::StringA(this->filename.c_str())));
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
    ccc->SetInterpolCamPositions(std::make_shared<std::vector<glm::vec3>>(this->interpolCamPos));

    return true;
}


bool KeyframeKeeper::CallForSetSelectedKeyframe(core::Call& c) {

    auto ccc = dynamic_cast<CallKeyframeKeeper*>(&c);
    if (ccc == nullptr) return false;

    bool appliedChanges = false; 
    float selAnimTime = ccc->GetSelectedKeyframe().GetAnimTime();
    for (unsigned int i = 0; i < this->keyframes.size(); i++) {
        if (this->keyframes[i].GetAnimTime() == selAnimTime) {
            this->replaceKeyframe(this->keyframes[i], ccc->GetSelectedKeyframe(), true);
            appliedChanges = true;
        }
    }
    if (!appliedChanges) {
        this->selectedKeyframe = this->interpolateKeyframe(ccc->GetSelectedKeyframe().GetAnimTime());
        ccc->SetSelectedKeyframe(this->selectedKeyframe);
        //vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [CallForSetSelectedKeyframe] Selected keyframe doesn't exist. Changes are omitted.");
    }
    this->updateEditParameters(this->selectedKeyframe);

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
    this->linearizeSimTangent(prevSelKf);

    return true;
}


bool KeyframeKeeper::CallForSetCameraForKeyframe(core::Call& c) {

    auto ccc = dynamic_cast<CallKeyframeKeeper*>(&c);
    if (ccc == nullptr) return false;

    this->cameraState = (*ccc->GetCameraState());

    return true;
}


bool KeyframeKeeper::CallForSetDragKeyframe(core::Call& c) {

    auto ccc = dynamic_cast<CallKeyframeKeeper*>(&c);
    if (ccc == nullptr) return false;

    // Checking if selected keyframe exists in keyframe array is done by caller.
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
        //vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [CallForSetCtrlPoints] ADDED undo for CTRL POINT ......");
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
        Keyframe tmp_kf = this->selectedKeyframe;
        tmp_kf.SetCameraState(this->cameraState);

        // Try adding keyframe to array
        if (!this->addKeyframe(tmp_kf, true)) {
            if (!this->replaceKeyframe(this->selectedKeyframe, tmp_kf, true)) {
                vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [CallForGetUpdatedKeyframeData] Unable to apply settings to new/selected keyframe.");
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
        if (!this->keyframes.empty()) {
            if (tt < this->keyframes.back().GetAnimTime()) {
                tt = this->keyframes.back().GetAnimTime();
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
        if (this->getKeyframeIndex(this->keyframes, this->selectedKeyframe) >= 0) { 
            // If existing keyframe is selected, delete keyframe an add at the right position
            Keyframe tmp_kf = this->selectedKeyframe;
            this->selectedKeyframe.SetAnimTime(t);
            this->replaceKeyframe(tmp_kf, this->selectedKeyframe, true);
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

        if (this->getKeyframeIndex(this->keyframes, this->selectedKeyframe) >= 0) { 
            // If existing keyframe is selected, delete keyframe an add at the right position
            Keyframe tmp_kf = this->selectedKeyframe;
            this->selectedKeyframe.SetSimTime(s / this->totalSimTime);
            this->replaceKeyframe(tmp_kf, this->selectedKeyframe, true);
        }
        // Write back clamped total time to parameter
        this->editCurrentSimTimeParam.Param<param::FloatParam>()->SetValue(s, false);
    }
    // editCurrentPosParam ----------------------------------------------------
    if (this->editCurrentPosParam.IsDirty()) {
        this->editCurrentPosParam.ResetDirty();

        if (this->getKeyframeIndex(this->keyframes, this->selectedKeyframe) >= 0) {
            Keyframe tmp_kf = this->selectedKeyframe;
            glm::vec3 pos_v = vislib_vector_to_glm(this->editCurrentPosParam.Param<param::Vector3fParam>()->Value());
            std::array<float, 3> pos = { pos_v.x, pos_v.y, pos_v.z };
            auto cam_state = this->selectedKeyframe.GetCameraState();
            cam_state.position = pos;
            this->selectedKeyframe.SetCameraState(cam_state);
            this->replaceKeyframe(tmp_kf, this->selectedKeyframe, true);
            this->refreshInterpolCamPos(this->interpolSteps);
        }
        else {
            vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [EditCurrentPosParam] No existing keyframe selected.");
        }
    }
    // resetViewParam -------------------------------------------------------
    if (this->resetViewParam.IsDirty()) {
        this->resetViewParam.ResetDirty();

        if (this->getKeyframeIndex(this->keyframes, this->selectedKeyframe) >= 0) {
            Keyframe tmp_kf = this->selectedKeyframe;
  
            megamol::core::view::Camera_2 camera(this->selectedKeyframe.GetCameraState());
            cam_type::snapshot_type snapshot;
            camera.take_snapshot(snapshot, thecam::snapshot_content::all);
            glm::vec4 cam_pos = snapshot.position;
            glm::vec4 cam_up = snapshot.up_vector;

            glm::vec3 up = static_cast<glm::vec3>(cam_up);
            glm::vec3 new_view = this->modelBboxCenter - static_cast<glm::vec3>(cam_pos);
            glm::quat new_orientation = quaternion_from_vectors(new_view, up);
            camera.orientation(new_orientation);

            cam_type::minimal_state_type camera_state;
            camera.get_minimal_state(camera_state);
            this->selectedKeyframe.SetCameraState(camera_state);
            this->replaceKeyframe(tmp_kf, this->selectedKeyframe, true);
        }
        else {
            vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [ResetViewParam] No existing keyframe selected.");
        }
    }
    // editCurrentViewParam -------------------------------------------------
    if (this->editCurrentViewParam.IsDirty()) {
        this->editCurrentViewParam.ResetDirty();

        if (this->getKeyframeIndex(this->keyframes, this->selectedKeyframe) >= 0) {
            Keyframe tmp_kf = this->selectedKeyframe;

            megamol::core::view::Camera_2 camera(this->selectedKeyframe.GetCameraState());
            cam_type::snapshot_type snapshot;
            camera.take_snapshot(snapshot, thecam::snapshot_content::all);
            glm::vec4 cam_up = snapshot.up_vector;
            glm::vec3 up = static_cast<glm::vec3>(cam_up);

            glm::vec3 new_view = vislib_vector_to_glm(this->editCurrentViewParam.Param<param::Vector3fParam>()->Value());
            glm::quat new_orientation = quaternion_from_vectors(new_view, up);
            camera.orientation(new_orientation);

            cam_type::minimal_state_type camera_state;
            camera.get_minimal_state(camera_state);
            this->selectedKeyframe.SetCameraState(camera_state);
            this->replaceKeyframe(tmp_kf, this->selectedKeyframe, true);
        }
        else {
            vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [EditCurrentViewParam] No existing keyframe selected.");
        }
    }
    // editCurrentUpParam -----------------------------------------------------
    if (this->editCurrentUpParam.IsDirty()) {
        this->editCurrentUpParam.ResetDirty();

        if (this->getKeyframeIndex(this->keyframes, this->selectedKeyframe) >= 0) {
            Keyframe tmp_kf = this->selectedKeyframe;

            megamol::core::view::Camera_2 camera(this->selectedKeyframe.GetCameraState());
            cam_type::snapshot_type snapshot;
            camera.take_snapshot(snapshot, thecam::snapshot_content::view_vector);
            glm::vec4 cam_view = snapshot.view_vector;
            glm::vec3 view = static_cast<glm::vec3>(cam_view);

            glm::vec3 new_up= vislib_vector_to_glm(this->editCurrentUpParam.Param<param::Vector3fParam>()->Value());
            glm::quat new_orientation = quaternion_from_vectors(view, new_up);
            camera.orientation(new_orientation);

            cam_type::minimal_state_type camera_state;
            camera.get_minimal_state(camera_state);
            this->selectedKeyframe.SetCameraState(camera_state);
            this->replaceKeyframe(tmp_kf, this->selectedKeyframe, true);
        }
        else {
            vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [EditCurrentUpParam] No existing keyframe selected.");
        }
    }
    // editCurrentApertureParam -----------------------------------------------------
    if (this->editCurrentApertureParam.IsDirty()) {
        this->editCurrentApertureParam.ResetDirty();

        if (this->getKeyframeIndex(this->keyframes, this->selectedKeyframe) >= 0) {
            Keyframe tmp_kf = this->selectedKeyframe;
            float aperture = this->editCurrentApertureParam.Param<param::FloatParam>()->Value();
            auto cam_state = this->selectedKeyframe.GetCameraState();
            cam_state.half_aperture_angle_radians = glm::radians(aperture/2.0f);
            this->selectedKeyframe.SetCameraState(cam_state);
            this->replaceKeyframe(tmp_kf, this->selectedKeyframe, true);
        }
        else {
            vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [EditCurrentApertureParam] No existing keyframe selected.");
        }
    }
    // fileNameParam ----------------------------------------------------------
    if (this->fileNameParam.IsDirty()) {
        this->fileNameParam.ResetDirty();

        this->filename = std::string(this->fileNameParam.Param<param::FilePathParam>()->Value().PeekBuffer());
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

        for (unsigned int i = 0; i < this->keyframes.size(); i++) {
            this->snapKeyframe2AnimFrame(this->keyframes[i]);
        }
        this->snapKeyframe2AnimFrame(this->selectedKeyframe);
    }
    // snapSimFramesParam -----------------------------------------------------
    if (this->snapSimFramesParam.IsDirty()) {
        this->snapSimFramesParam.ResetDirty();

        for (unsigned int i = 0; i < this->keyframes.size(); i++) {
            this->snapKeyframe2SimFrame(this->keyframes[i]);
        }
        this->snapKeyframe2SimFrame(this->selectedKeyframe);      
    }
    // simTangentParam --------------------------------------------------------
    if (this->simTangentParam.IsDirty()) {
        this->simTangentParam.ResetDirty();

        this->simTangentStatus = true;
    }

    // PROPAGATE UPDATED DATA TO CALL -----------------------------------------
    ccc->SetCameraState(std::make_shared<camera_state_type>(this->cameraState));
    ccc->SetKeyframes(std::make_shared<std::vector<Keyframe>>(this->keyframes));
    ccc->SetSelectedKeyframe(this->selectedKeyframe);
    ccc->SetTotalAnimTime(this->totalAnimTime);
    ccc->SetInterpolCamPositions(std::make_shared<std::vector<glm::vec3>>(this->interpolCamPos));
    ccc->SetTotalSimTime(this->totalSimTime);
    ccc->SetControlPointPosition(this->startCtrllPos, this->endCtrllPos);
    ccc->SetFps(this->fps);

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


bool KeyframeKeeper::undoAction(void) {

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


bool KeyframeKeeper::redoAction(void) {

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

    // Linearize tangent between simTangentKf and currently selected keyframe by shifting all inbetween keyframe simulation times
    if (this->simTangentStatus) {
        // Linearize tangent only between existing keyframes
        if (this->getKeyframeIndex(this->keyframes, stkf) < 0) {
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
                newSimTime = m * (this->keyframes[i].GetAnimTime()) + b;

                // ADD UNDO
                // Store old keyframe
                Keyframe tmp_kf = this->keyframes[i];
                // Apply changes to keyframe
                this->keyframes[i].SetSimTime(newSimTime);
                // Add modification to undo queue
                this->addKeyframeUndoAction(KeyframeKeeper::Undo::Action::UNDO_KEYFRAME_MODIFY, this->keyframes[i], tmp_kf);
            }
    
            this->simTangentStatus = false;
        }
        else {
            vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [linearizeSimTangent] Select existing keyframe to finish linearizing the tangent.");
        }
    }
}


void KeyframeKeeper::snapKeyframe2AnimFrame(Keyframe& inout_kf) {

    if (this->fps == 0) {
        vislib::sys::Log::DefaultLog.WriteError("[KEYFRAME KEEPER] [snapKeyframe2AnimFrame] FPS is ZERO.");
        return;
    }

    float fpsFrac = 1.0f / (float)(this->fps);
    float t = std::round(inout_kf.GetAnimTime() / fpsFrac) * fpsFrac;
    if (std::abs(t - std::round(t)) < (fpsFrac / 2.0)) {
        t = std::round(t);
    }
    t = (t < 0.0f) ? (0.0f) : (t);
    t = (t > this->totalAnimTime) ? (this->totalAnimTime) : (t);

    // ADD UNDO
    Keyframe tmp_kf = inout_kf;
    inout_kf.SetAnimTime(t);
    this->addKeyframeUndoAction(KeyframeKeeper::Undo::Action::UNDO_KEYFRAME_MODIFY, inout_kf, tmp_kf);
}


void KeyframeKeeper::snapKeyframe2SimFrame(Keyframe& inout_kf) {

    float s = std::round(inout_kf.GetSimTime() * this->totalSimTime) / this->totalSimTime;

    // ADD UNDO
    Keyframe tmp_kf = inout_kf;
    inout_kf.SetSimTime(s);
    this->addKeyframeUndoAction(KeyframeKeeper::Undo::Action::UNDO_KEYFRAME_MODIFY, inout_kf, tmp_kf);
}


void KeyframeKeeper::setSameSpeed() {

    if (this->keyframes.size() > 2) {
        // Store index of selected keyframe to restore seleection after changing time of keyframes
        int selIndex = this->getKeyframeIndex(this->keyframes, this->selectedKeyframe);

        // Get total values
        float totTime = this->keyframes.back().GetAnimTime() - this->keyframes.front().GetAnimTime();
        if (totTime == 0.0f) {
            vislib::sys::Log::DefaultLog.WriteError("[KEYFRAME KEEPER] [setSameSpeed] totTime is ZERO.");
            return;
        }
        float totDist = 0.0f;
        for (unsigned int i = 0; i < this->interpolCamPos.size() - 1; i++) {
            totDist += glm::length(this->interpolCamPos[i + 1] - this->interpolCamPos[i]);
        }
        float totalVelocity = totDist / totTime; // unit doesn't matter ... it is only relative

        // Get values between two consecutive keyframes and shift remoter keyframe if necessary
        float kfTime = 0.0f;
        float kfDist = 0.0f;
        for (unsigned int i = 0; i < this->interpolCamPos.size() - 2; i++) {
            if ((i > 0) && (i % this->interpolSteps == 0)) {  // skip checking for first keyframe (last keyframe is skipped by prior loop)
                kfTime = kfDist / totalVelocity;

                unsigned int index = static_cast<unsigned int>(floorf(((float)i / (float)this->interpolSteps)));
                float t = this->keyframes[index - 1].GetAnimTime() + kfTime;
                t = (t < 0.0f) ? (0.0f) : (t);
                t = (t > this->totalAnimTime) ? (this->totalAnimTime) : (t);
                kfDist = 0.0f;

                // ADD UNDO
                Keyframe tmp_kf = this->keyframes[index];
                this->keyframes[index].SetAnimTime(t);
                this->addKeyframeUndoAction(KeyframeKeeper::Undo::Action::UNDO_KEYFRAME_MODIFY, this->keyframes[index], tmp_kf);
            }
            // Add distance up to existing keyframe
            kfDist += glm::length(this->interpolCamPos[i + 1] - this->interpolCamPos[i]);
        }

        // Restore previous selected keyframe
        if (selIndex >= 0) {
            this->selectedKeyframe = this->keyframes[selIndex];
        }
    }
}


void KeyframeKeeper::refreshInterpolCamPos(unsigned int s) {

    if (s == 0) {
        vislib::sys::Log::DefaultLog.WriteError("[KEYFRAME KEEPER] [refreshInterpolCamPos] Interpolation step count should be greater than zero.");
        return;
    }

    float startTime;
    float deltaTimeStep;
    Keyframe kf;
    this->interpolCamPos.clear();
    if (this->keyframes.size() > 1) {
        for (unsigned int i = 0; i < this->keyframes.size() - 1; i++) {
            startTime = this->keyframes[i].GetAnimTime();
            deltaTimeStep = (this->keyframes[i + 1].GetAnimTime() - startTime) / (float)s;

            for (unsigned int j = 0; j < s; j++) {
                kf = this->interpolateKeyframe(startTime + deltaTimeStep*(float)j);
                auto p = kf.GetCameraState().position;
                this->interpolCamPos.emplace_back(glm::vec3(p[0], p[1], p[2]));
            }
        }
        // Add last existing camera position
        auto p = this->keyframes.back().GetCameraState().position;
        this->interpolCamPos.emplace_back(glm::vec3(p[0], p[1], p[2]));
    }
}


bool KeyframeKeeper::replaceKeyframe(Keyframe oldkf, Keyframe newkf, bool undo) {

    if (!this->keyframes.empty()) {
        // Both are equal ... nothing to do
        if (oldkf == newkf) {
            return true;
        }
        // Check if old keyframe exists
        if (this->getKeyframeIndex(this->keyframes, oldkf) >= 0) {
            // Delete old keyframe
            this->deleteKeyframe(oldkf, false);
            // Try to add new keyframe
            if (!this->addKeyframe(newkf, false)) {
                // There is alredy a keyframe on the new position ... overwrite existing keyframe.
                float newAnimTime = newkf.GetAnimTime();
                for (unsigned int i = 0; i < this->keyframes.size(); i++) {
                    if (this->keyframes[i].GetAnimTime() == newAnimTime) {
                        this->deleteKeyframe(this->keyframes[i], true);
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

    if (!this->keyframes.empty()) {
        // Get index of keyframe to delete
        int selIndex = this->getKeyframeIndex(this->keyframes, kf);
        // Choose new selected keyframe
        if (selIndex >= 0) {
            // DELETE UNDO
            this->keyframes.erase(this->keyframes.begin() + selIndex);
            if (undo) {
                // ADD UNDO
                this->addKeyframeUndoAction(KeyframeKeeper::Undo::Action::UNDO_KEYFRAME_DELETE, kf, kf);
                // Adjust first/last control point position - ONLY if it is a "real" delete and no replace
                if (this->keyframes.size() > 1) {
                    if (selIndex == 0) {
                        auto p0 = this->keyframes[0].GetCameraState().position;
                        auto p1 = this->keyframes[1].GetCameraState().position;
                        glm::vec3 tmpV = glm::normalize(glm::vec3(p0[0], p0[1], p0[2]) - glm::vec3(p1[0], p1[1], p1[2]));
                        this->startCtrllPos = glm::vec3(p0[0], p0[1], p0[2]) + tmpV;
                    }
                    if (selIndex == this->keyframes.size()) { // Element is already removed so the index is now: (this->keyframes.size() - 1) + 1
                        auto p0 = this->keyframes.back().GetCameraState().position;
                        auto p1 = this->keyframes[(int)this->keyframes.size() - 2].GetCameraState().position;
                        glm::vec3 tmpV = glm::normalize(glm::vec3(p0[0], p0[1], p0[2]) - glm::vec3(p1[0], p1[1], p1[2]));
                        this->endCtrllPos = glm::vec3(p0[0], p0[1], p0[2]) + tmpV;
                    }
                }
            }
            this->refreshInterpolCamPos(this->interpolSteps);
            if (selIndex > 0) {
                this->selectedKeyframe = this->keyframes[selIndex - 1];
            }
            else if (selIndex < this->keyframes.size()) {
                this->selectedKeyframe = this->keyframes[selIndex];
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
    for (unsigned int i = 0; i < this->keyframes.size(); i++) {
        if (this->keyframes[i].GetAnimTime() == time) {
            //vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] [addKeyframe] Keyframe already exists.");
            return false;
        }
    }

    // Sort new keyframe to keyframe array
    if (this->keyframes.empty() || (this->keyframes.back().GetAnimTime() < time)) {
        this->keyframes.emplace_back(kf);
        // Adjust first/last control point position - ONLY if it is a "real" add and no replace
        if (undo && this->keyframes.size() > 1) {
            auto p0 = this->keyframes.back().GetCameraState().position;
            auto p1 = this->keyframes[(int)this->keyframes.size() - 2].GetCameraState().position;
            glm::vec3 tmpV = glm::normalize(glm::vec3(p0[0], p0[1], p0[2]) - glm::vec3(p1[0], p1[1], p1[2]));
            this->endCtrllPos = glm::vec3(p0[0], p0[1], p0[2]) + tmpV;
        }
    }
    else if (time < this->keyframes.front().GetAnimTime()) {
        this->keyframes.insert(this->keyframes.begin(), kf);
        // Adjust first/last control point position - ONLY if it is a "real" add and no replace
        if (undo && this->keyframes.size() > 1) {
            auto p0 = this->keyframes.back().GetCameraState().position;
            auto p1 = this->keyframes[(int)this->keyframes.size() - 2].GetCameraState().position;
            glm::vec3 tmpV = glm::normalize(glm::vec3(p0[0], p0[1], p0[2]) - glm::vec3(p1[0], p1[1], p1[2]));
            this->endCtrllPos = glm::vec3(p0[0], p0[1], p0[2]) + tmpV;
        }
    }
    else { // Insert keyframe in-between existing keyframes
        unsigned int insertIdx = 0;
        for (unsigned int i = 0; i < this->keyframes.size(); i++) {
            if (time < this->keyframes[i].GetAnimTime()) {
                insertIdx = i;
                break;
            }
        }
        this->keyframes.insert(this->keyframes.begin() + insertIdx, kf);
    }

    // ADD UNDO 
    if (undo) {
        this->addKeyframeUndoAction(KeyframeKeeper::Undo::Action::UNDO_KEYFRAME_ADD, kf, kf);
    }

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
    for (SIZE_T i = 0; i < this->keyframes.size(); i++) {
        if (t == this->keyframes[i].GetAnimTime()) {
            return this->keyframes[i];
        }
    }

    if (this->keyframes.empty()) {
        Keyframe kf = Keyframe();
        kf.SetAnimTime(t);
        kf.SetSimTime(0.0f);
        auto state = this->cameraState;
        state.half_aperture_angle_radians = glm::radians(30.0f); // = 60° aperture angle
        kf.SetCameraState(state);
        return kf;
    }
    else if (t < this->keyframes.front().GetAnimTime()) {
        Keyframe kf = this->keyframes.front();
        kf.SetAnimTime(t);
        return kf;

    }
    else if (t > this->keyframes.back().GetAnimTime()) {
        Keyframe kf = this->keyframes.back();
        kf.SetAnimTime(t);
        return kf;
    }
    else { // if ((t > this->keyframes.front().GetAnimTime()) && (t < this->keyframes.back().GetAnimTime())) {

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
        int kfIdxCnt = (int)this->keyframes.size() - 1;
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

        megamol::core::view::Camera_2 c0(this->keyframes[i0].GetCameraState());
        megamol::core::view::Camera_2 c1 = this->keyframes[i1].GetCameraState();
        megamol::core::view::Camera_2 c2 = this->keyframes[i2].GetCameraState();
        megamol::core::view::Camera_2 c3 = this->keyframes[i3].GetCameraState();

        // Interpolate simulation time linear between i1 and i2
        float simT1 = this->keyframes[i1].GetSimTime();
        float simT2 = this->keyframes[i2].GetSimTime();
        float simT = simT1 + (simT2 - simT1)*iT;
        kf.SetSimTime(simT);

        // ! Skip interpolation of camera parameters if they are equal for ?1 and ?2.
        // => Prevent interpolation loops if time of keyframes is different, but cam params are the same.

        //interpolate position ------------------------------------------------
        glm::vec4 ps0 = c0.position();
        glm::vec4 ps1 = c1.position();
        glm::vec4 ps2 = c2.position();
        glm::vec4 ps3 = c3.position();
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
            cam_kf.position(c1.position());
        }
        else {
            glm::vec3 pk = this->vec3_interpolation(iT, p0, p1, p2, p3);
            cam_kf.position(glm::vec4(pk.x, pk.y, pk.z, 1.0f));
        }

        //interpolate aperture angle ------------------------------------------
        float a0 = c0.aperture_angle();
        float a1 = c1.aperture_angle();
        float a2 = c2.aperture_angle();
        float a3 = c3.aperture_angle();
        if (a1 == a2) {
            cam_kf.aperture_angle(a1);
        }
        else {
            float ak = this->float_interpolation(iT, a0, a1, a2, a3);
            cam_kf.aperture_angle(ak);
        }

        //interpolate orientation ---------------------------------------------
        glm::quat c1_orient = c1.orientation();
        glm::quat c2_orient = c2.orientation();
        cam_kf.orientation(this->quaternion_interpolation(iT, c1_orient, c2_orient));

        // Finally set new interpoated camera for keyframe
        cam_type::minimal_state_type camera_state;
        cam_kf.get_minimal_state(camera_state);
        kf.SetCameraState(camera_state);

        return kf;
    }
}


glm::quat KeyframeKeeper::quaternion_interpolation(float u, glm::quat q0, glm::quat q1) {

    /// Slerp - spherical linear interpolation
    // SOURCE: https://en.wikipedia.org/wiki/Slerp and https://web.mit.edu/2.998/www/QuaternionReport1.pdf

    return glm::normalize(glm::slerp(q0, q1, u));
}


float KeyframeKeeper::float_interpolation(float u, float f0, float f1, float f2, float f3) {

    /// Catmull-Rom Interpolation
    // Considering global tangent length
    // SOURCE: https://www.cs.cmu.edu/~462/projects/assn2/assn2/catmullRom.pdf
    float tl = this->splineTangentLength;
    float f = (f1)+
              (-(tl * f0) + (tl* f2)) * u +
              ((2.0f*tl * f0) + ((tl - 3.0f) * f1) + ((3.0f - 2.0f*tl) * f2) - (tl* f3)) * u * u +
              (-(tl * f0) + ((2.0f - tl) * f1) + ((tl - 2.0f) * f2) + (tl* f3)) * u * u * u;

    return f;
}


glm::vec3 KeyframeKeeper::vec3_interpolation(float u, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3) {

    glm::vec3 v;
    v.x = this->float_interpolation(u, v0.x, v1.x, v2.x, v3.x);
    v.y = this->float_interpolation(u, v0.y, v1.y, v2.y, v3.y);
    v.z = this->float_interpolation(u, v0.z, v1.z, v2.z, v3.z);

    return v;
}


bool KeyframeKeeper::saveKeyframes() {

    if (this->filename.empty()) {
        vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] [saveKeyframes] No filename given. Using default filename.");
        time_t t = std::time(0);  // get time now
        struct tm *now = nullptr;
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
        struct tm nowdata;
        now = &nowdata;
        localtime_s(now, &t);
#else /* defined(_WIN32) && (_MSC_VER >= 1400) */
        now = localtime(&t);
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

        std::stringstream stream;
        stream << "keyframes_" << (now->tm_year + 1900) << 
            std::setfill('0') << std::setw(2) << (now->tm_mon + 1) << 
            std::setfill('0') << std::setw(2) << now->tm_mday << "-" << 
            std::setfill('0') << std::setw(2) << now->tm_hour << 
            std::setfill('0') << std::setw(2) << now->tm_min << 
            std::setfill('0') << std::setw(2) << now->tm_sec << ".kf";
        this->filename = stream.str();
        this->fileNameParam.Param<param::FilePathParam>()->SetValue(vislib::StringA(this->filename.c_str()), false);
    } 

    try {
        std::ofstream outfile;
        outfile.open(this->filename.c_str(), std::ios::binary);
        if (!outfile.good()) {
            vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] Failed to create keyframe file.");
            return false;
        }

        try {
            nlohmann::json json;
            // Set general data
            json["total_animation_time"] = this->totalAnimTime;
            json["spline_tangent_length"] = this->splineTangentLength;
            json["first_ctrl_point"]["x"] = this->startCtrllPos.x;
            json["first_ctrl_point"]["y"] = this->startCtrllPos.y;
            json["first_ctrl_point"]["z"] = this->startCtrllPos.z;
            json["last_ctrl_point"]["x"] = this->endCtrllPos.x;
            json["last_ctrl_point"]["y"] = this->endCtrllPos.y;
            json["last_ctrl_point"]["z"] = this->endCtrllPos.z;
            // Set keyframe data
            auto count = this->keyframes.size();
            std::string kf_str;
            for (size_t i = 0; i < count; i++) {
                this->keyframes[i].Serialise(json, i);
            }
            // Dump with indent of 2 spaces and new lines.
            outfile << json.dump(2); 
        }
        catch (nlohmann::json::type_error& e) {
            vislib::sys::Log::DefaultLog.WriteError(
                "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::exception& e) {
            vislib::sys::Log::DefaultLog.WriteError(
                "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::parse_error& e) {
            vislib::sys::Log::DefaultLog.WriteError(
                "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::invalid_iterator& e) {
            vislib::sys::Log::DefaultLog.WriteError(
                "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::out_of_range& e) {
            vislib::sys::Log::DefaultLog.WriteError(
                "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::other_error& e) {
            vislib::sys::Log::DefaultLog.WriteError(
                "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (...) {
            vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: Unknown Error (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
            return false;
        }

        outfile.close();
        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] Successfully stored keyframes to file: %s", this->filename.c_str());
        return true;
    } 
    catch (...) {
    }
    return false;
}


bool KeyframeKeeper::loadKeyframes() {

    if (this->filename.empty()) {
        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] No filename given.");
        return false;
    }
    else {
        std::ifstream infile;
        infile.open(this->filename.c_str());
        if (!infile.good()) {
            vislib::sys::Log::DefaultLog.WriteWarn("[KEYFRAME KEEPER] Failed to open keyframe file.");
            return false;
        }

        try {
            nlohmann::json json;
            std::string content;
            std::string line;
            while (std::getline(infile, line)) {
                content += line;
            }
            json = nlohmann::json::parse(content);
            // Check for valid JSON object
            if (!json.is_object()) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "[KEYFRAME KEEPER] Given string is no valid JSON object.");
                return false;
            }

            // Get general data
            if (json.at("total_animation_time").is_number()) {
                json.at("total_animation_time").get_to(this->totalAnimTime);
            }
            else {
                vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'total_animation_time': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
            }
            if (json.at("spline_tangent_length").is_number()) {
                json.at("spline_tangent_length").get_to(this->splineTangentLength);
            }
            else {
                vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'spline_tangent_length': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
            }
            if (json.at("first_ctrl_point").at("x").is_number()) {
                json.at("first_ctrl_point").at("x").get_to(this->startCtrllPos.x);
            }
            else {
                vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'first_ctrl_point'-'x': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
            }
            if (json.at("first_ctrl_point").at("y").is_number()) {
                json.at("first_ctrl_point").at("y").get_to(this->startCtrllPos.y);
            }
            else {
                vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'first_ctrl_point'-'y': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
            }
            if (json.at("first_ctrl_point").at("z").is_number()) {
                json.at("first_ctrl_point").at("z").get_to(this->startCtrllPos.z);
            }
            else {
                vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'first_ctrl_point'-'z': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
            }
            if (json.at("last_ctrl_point").at("x").is_number()) {
                json.at("last_ctrl_point").at("x").get_to(this->endCtrllPos.x);
            }
            else {
                vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'last_ctrl_point'-'x': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
            }
            if (json.at("last_ctrl_point").at("y").is_number()) {
                json.at("last_ctrl_point").at("y").get_to(this->endCtrllPos.y);
            }
            else {
                vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'last_ctrl_point'-'y': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
            }
            if (json.at("last_ctrl_point").at("z").is_number()) {
                json.at("last_ctrl_point").at("z").get_to(this->endCtrllPos.z);
            }
            else {
                vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'last_ctrl_point'-'z': %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
            }

            // Get keyframe data
            this->keyframes.clear();
            if (json.at("keyframes").is_array()) {
                bool valid = true;
                size_t keyframe_count = json.at("keyframes").size();
                this->keyframes.resize(keyframe_count);
                for (size_t i = 0; i < keyframe_count; ++i) {
                    valid = valid && this->keyframes[i].Deserialise(json.at("keyframes").at(i));
                }
                if (!valid) {
                    vislib::sys::Log::DefaultLog.WriteWarn("JSON ERROR - Could not deserialise keyframes.");
                }
            }
            else {
                vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - Couldn't read 'keyframes' array: %s (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
            }
        }
        catch (nlohmann::json::type_error& e) {
            vislib::sys::Log::DefaultLog.WriteError(
                "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::exception& e) {
            vislib::sys::Log::DefaultLog.WriteError(
                "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::parse_error& e) {
            vislib::sys::Log::DefaultLog.WriteError(
                "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::invalid_iterator& e) {
            vislib::sys::Log::DefaultLog.WriteError(
                "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::out_of_range& e) {
            vislib::sys::Log::DefaultLog.WriteError(
                "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (nlohmann::json::other_error& e) {
            vislib::sys::Log::DefaultLog.WriteError(
                "JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
            return false;
        }
        catch (...) {
            vislib::sys::Log::DefaultLog.WriteError("JSON ERROR - %s: Unknown Error (%s:%d)", __FUNCTION__, __FILE__, __LINE__);
            return false;
        }

        if (!this->keyframes.empty()) {
            this->selectedKeyframe = this->interpolateKeyframe(0.0f);
            this->updateEditParameters(this->selectedKeyframe);
            this->refreshInterpolCamPos(this->interpolSteps);
        }

        infile.close();
        vislib::sys::Log::DefaultLog.WriteInfo("[KEYFRAME KEEPER] Successfully loaded keyframes from file: %s", this->filename.c_str());
    }
    return true;
}


void KeyframeKeeper::updateEditParameters(Keyframe kf) {

    // Set new parameter values of changed selected keyframe
    megamol::core::view::Camera_2 camera(kf.GetCameraState());
    cam_type::snapshot_type snapshot;
    camera.take_snapshot(snapshot, thecam::snapshot_content::up_vector | thecam::snapshot_content::view_vector | thecam::snapshot_content::camera_coordinate_system);
    glm::vec4 cam_pos = snapshot.position;
    glm::vec4 cam_up = snapshot.up_vector;
    glm::vec4 cam_view = snapshot.view_vector;
    glm::vec3 pos = static_cast<glm::vec3>(cam_pos);
    glm::vec3 view = static_cast<glm::vec3>(cam_view);
    glm::vec3 up = static_cast<glm::vec3>(cam_up);
    this->editCurrentAnimTimeParam.Param<param::FloatParam>()->SetValue(kf.GetAnimTime(), false);
    this->editCurrentSimTimeParam.Param<param::FloatParam>()->SetValue(kf.GetSimTime() * this->totalSimTime, false);
    this->editCurrentPosParam.Param<param::Vector3fParam>()->SetValue(glm_to_vislib_vector(pos), false);
    this->editCurrentViewParam.Param<param::Vector3fParam>()->SetValue(glm_to_vislib_vector(view), false);
    this->editCurrentUpParam.Param<param::Vector3fParam>()->SetValue(glm_to_vislib_vector(up), false);
    this->editCurrentApertureParam.Param<param::FloatParam>()->SetValue(camera.aperture_angle(), false);
    
}


int KeyframeKeeper::getKeyframeIndex(std::vector<Keyframe>& keyframes, Keyframe keyframe) {

    int count = keyframes.size();
    for (int i = 0; i < count; ++i) {
        if (keyframes[i] == keyframe) {
            return i;
        }
    }
    return -1;
}
