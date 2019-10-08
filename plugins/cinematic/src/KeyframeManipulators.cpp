/**
* KeyframeManipulators.cpp
*
* Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "KeyframeManipulators.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::cinematic;


KeyframeManipulators::KeyframeManipulators(void)
    : togglevisibleGroupParam("toggleVisibleGroup", "Toggle visibility of different manipulator groups.")
    , visibleGroupParam("visibleGroup", "Select visible manipulator group.")
    , toggleOusideBboxParam("showOutsideBBox", "Show manipulators always outside of model bounding box.")
    , visibleGroup(VisibleGroup::KEYFRAME_AND_CTRPOINT)
    , toggleOusideBbox(false) {

    this->toggleOusideBboxParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_W, core::view::Modifier::CTRL));

    this->togglevisibleGroupParam.SetParameter(new param::ButtonParam(core::view::Key::KEY_Q, core::view::Modifier::CTRL));

    param::EnumParam* vmg = new param::EnumParam(this->visibleGroup);
    vmg->SetTypePair(VisibleGroup::KEYFRAME_AND_CTRPOINT, "Keyframe and Ctrl-Point Positions");
    vmg->SetTypePair(VisibleGroup::LOOKAT_AND_UP, "LookAt-Vector and Up-Vector");
    this->visibleGroupParam << vmg;
}


KeyframeManipulators::~KeyframeManipulators(void) {

    // nothing to do here ...
}


bool KeyframeManipulators::UpdateRendering(std::shared_ptr<CinematicUtils> utils, std::vector<Keyframe> const& keyframes, Keyframe selected_keyframe, glm::vec2 viewport_dim, glm::mat4 mvp,
    camera_state_type snapshot, glm::vec3 first_ctrl_pos, glm::vec3 last_ctrl_pos) {

    if (utils == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("[KeyframeManipulators] [UpdateRendering] Pointer to 'utils' should not be nullptr.");
        return false;
    }

    // Update parameters
    if (this->visibleGroupParam.IsDirty()) {
        this->visibleGroup = static_cast<VisibleGroup>(this->visibleGroupParam.Param<core::param::EnumParam>()->Value());
        this->visibleGroupParam.ResetDirty();
    }

    if (this->togglevisibleGroupParam.IsDirty()) {
        this->visibleGroup = static_cast<VisibleGroup>((this->visibleGroup + 1) % VisibleGroup::GROUP_COUNT);
        this->togglevisibleGroupParam.ResetDirty();
    }

    if (this->toggleOusideBboxParam.IsDirty()) {
        this->toggleOusideBbox = !this->toggleOusideBbox;
        this->toggleOusideBboxParam.ResetDirty();
    }

    // Update current state varaibles
    this->current_utils = utils;
    this->current_viewport = viewport_dim;
    bool worldChanged = false;
    if ((this->current_mvp != mvp) || (this->current_cam_snapshot != snapshot)) {
        worldChanged = true;
    }
    this->current_mvp = mvp;
    this->current_cam_snapshot = snapshot;

    // Update manipulator only if selected keyframe changed
    if ((this->current_selected_keyframe != selected_keyframe) || (this->current_first_ctrl_point != first_ctrl_pos) || (this->current_last_ctrl_point != last_ctrl_pos) || worldChanged) {
        this->current_selected_keyframe = selected_keyframe;
        this->current_first_ctrl_point = first_ctrl_pos;
        this->current_last_ctrl_point = last_ctrl_pos;

        // Check if selected keyframe exists in keyframe array
        int selected_in_keyframe_array = -1;
        for (int i = 0; i < keyframes.size(); ++i) {
            if (keyframes[i] == this->current_selected_keyframe) {
                this->selected_in_keyframe_array = i;
            }
        }

        this->selectedKeyframeIsFirst = (this->selectedKf == kfa->front())?(true):(false);
        this->selectedIsLast  = (this->selectedKf == kfa->back())?(true):(false);

        this->updateManipulatorPositions();
    }

    // Update local keyframe positions ------------------------------------
    unsigned int kfACnt     = static_cast<unsigned int>(kfa->size());
    unsigned int keyframesCnt = static_cast<unsigned int>(this->keyframes.size());

    bool kfAvail = false;
    for (auto m : am) {
        if (m == manipType::KEYFRAME_POS) {
            kfAvail = true;
        }
    }

    if ((kfACnt != keyframesCnt) || worldChanged) {
        this->keyframes.clear();
        this->keyframes.reserve(kfACnt);
        for (unsigned int i = 0; i < kfACnt; i++) {
            ManipPosData tmpkfA;
            tmpkfA.wsPos     = (*kfa)[i].GetCamPosition();
            tmpkfA.ssPos     = this->getScreenSpace(tmpkfA.wsPos);
            tmpkfA.offset    = glm::length(this->getScreenSpace(tmpkfA.wsPos + this->circleVertices[1]) - tmpkfA.ssPos);
            tmpkfA.available = kfAvail;
            this->keyframes.emplace_back(tmpkfA);
        }
    }
    else { // Update positions (which might have changed)
        for (unsigned int i = 0; i < kfACnt; i++) {
            if (this->keyframes[i].wsPos    != (*kfa)[i].GetCamPosition()) { 
                this->keyframes[i].wsPos     = (*kfa)[i].GetCamPosition();
                this->keyframes[i].ssPos     = this->getScreenSpace(this->keyframes[i].wsPos);
                this->keyframes[i].offset    = glm::length(this->getScreenSpace(this->keyframes[i].wsPos + this->circleVertices[1]) - this->keyframes[i].ssPos); 
                this->keyframes[i].available = kfAvail;
            }
        }
    }

    // Update availability of manipulators
    for (unsigned int i = 0; i < static_cast<unsigned int>(this->manipArray.size()); i++) {
        this->manipArray[i].available = false;
    }
    if (this->sKfInArray >= 0) { // Manipulators are only available if selected keyframe exists in keyframe array
        for (unsigned int i = 0; i < static_cast<unsigned int>(am.size()); i++) {
            unsigned int index = static_cast<unsigned int>(am[i]);
            if (index < static_cast<unsigned int>(NUM_OF_SELECTED_MANIP)) {
                this->manipArray[index].available = true;
            }
        }
    }

    this->isDataSet = true;
    return true;
}


bool KeyframeManipulators::updateManipulators(void) {

    glm::vec3 skfPosV = this->selectedKf.GetCamPosition();
    glm::vec3 skfLaV  = this->selectedKf.GetCamLookAt();
    // Update screen space positions
    this->sKfSsPos    = this->getScreenSpace(skfPosV);
    this->sKfSsLookAt = this->getScreenSpace(skfLaV);

    // Fill empty manipulator array
    if (this->manipArray.empty()) {
        this->manipArray.clear();
        this->manipArray.reserve(static_cast<unsigned int>(manipType::NUM_OF_SELECTED_MANIP));
        for (unsigned int i = 0; i < static_cast<unsigned int>(manipType::NUM_OF_SELECTED_MANIP); i++) {
            this->manipArray.emplace_back(ManipPosData());
        }
    }

    // Adaptive axis length of manipulators
    float radius = glm::length(this->circleVertices[1]);
    float length = vislib::math::Max((glm::length(this->worldCamModDir) * this->axisLengthFac), 0.1f);
    glm::vec3 tmpV;
    float len;

    ManipPosData tmpManipData;
    for (unsigned int i = 0; i < static_cast<unsigned int>(manipType::NUM_OF_SELECTED_MANIP); i++) { // skip SELECTED_KF_POS
        tmpV = glm::vec3();
        switch (static_cast<manipType>(i)) {
            case (manipType::CTRL_POINT_POS_X):
                if (this->selectedIsFirst) {
                    tmpV = this->startCtrllPos;
                }
                else if (this->selectedIsLast) {
                    tmpV = this->endCtrllPos;
                }
                tmpManipData.wsPos = tmpV + glm::vec3(length*0.5f, 0.0f, 0.0f);
                if (this->manipOusideBbox && this->modelBbox.Contains(G2P(tmpManipData.wsPos))) {
                    tmpManipData.wsPos.x = this->modelBbox.Right() + radius;
                }
             break;
            case (manipType::CTRL_POINT_POS_Y):
                if (this->selectedIsFirst) {
                    tmpV = this->startCtrllPos;
                }
                else if (this->selectedIsLast) {
                    tmpV = this->endCtrllPos;
                }
                tmpManipData.wsPos = tmpV + glm::vec3(0.0f, length*0.5f, 0.0f);
                if (this->manipOusideBbox && this->modelBbox.Contains(G2P(tmpManipData.wsPos))) {
                    tmpManipData.wsPos.y = this->modelBbox.Top() + radius;
                }
                break;
            case (manipType::CTRL_POINT_POS_Z):
                if (this->selectedIsFirst) {
                    tmpV = this->startCtrllPos;
                }
                else if (this->selectedIsLast) {
                    tmpV = this->endCtrllPos;
                }
                tmpManipData.wsPos = tmpV + glm::vec3(0.0f, 0.0f, length*0.5f);
                if (this->manipOusideBbox && this->modelBbox.Contains(G2P(tmpManipData.wsPos))) {
                    tmpManipData.wsPos.z = this->modelBbox.Front() + radius;
                }
                break;
            case (manipType::SELECTED_KF_POS_X):      
                tmpManipData.wsPos = skfPosV + glm::vec3(length, 0.0f, 0.0f); 
                if (this->manipOusideBbox && this->modelBbox.Contains(G2P(tmpManipData.wsPos))) {
                    tmpManipData.wsPos.x = this->modelBbox.Right() + radius;
                }
               break;
            case (manipType::SELECTED_KF_POS_Y):      
                tmpManipData.wsPos = skfPosV + glm::vec3(0.0f, length, 0.0f); 
                if (this->manipOusideBbox && this->modelBbox.Contains(G2P(tmpManipData.wsPos))) {
                    tmpManipData.wsPos.y = this->modelBbox.Top() + radius;
                }
                break;
            case (manipType::SELECTED_KF_POS_Z):      
                tmpManipData.wsPos = skfPosV + glm::vec3(0.0f, 0.0f, length); 
                if (this->manipOusideBbox && this->modelBbox.Contains(G2P(tmpManipData.wsPos))) {
                    tmpManipData.wsPos.z = this->modelBbox.Front() + radius;
                }
                break;
            case (manipType::SELECTED_KF_LOOKAT_X):  
                tmpManipData.wsPos = skfLaV + glm::vec3(length, 0.0f, 0.0f);
                if (this->manipOusideBbox && this->modelBbox.Contains(G2P(tmpManipData.wsPos))) {
                    tmpManipData.wsPos.x = this->modelBbox.Right() + radius;
                }
                break;
            case (manipType::SELECTED_KF_LOOKAT_Y):   
                tmpManipData.wsPos = skfLaV + glm::vec3(0.0f, length, 0.0f);
                if (this->manipOusideBbox && this->modelBbox.Contains(G2P(tmpManipData.wsPos))) {
                    tmpManipData.wsPos.y = this->modelBbox.Top() + radius;
                }
                break;
            case (manipType::SELECTED_KF_LOOKAT_Z):   
                tmpManipData.wsPos = skfLaV + glm::vec3(0.0f, 0.0f, length);
                if (this->manipOusideBbox && this->modelBbox.Contains(G2P(tmpManipData.wsPos))) {
                    tmpManipData.wsPos.z = this->modelBbox.Front() + radius;
                }
                break;
            case (manipType::SELECTED_KF_UP):         
                tmpV = this->selectedKf.GetCamUp();
                tmpV = glm::normalize(tmpV) * length;
                tmpManipData.wsPos = skfPosV + tmpV;
                if (this->manipOusideBbox && this->modelBbox.Contains(G2P(tmpManipData.wsPos))) {
                    len = glm::length(P2G(this->modelBbox.GetRightTopFront()) - P2G(this->modelBbox.GetLeftBottomBack())) / 3.0f;
					tmpV = glm::normalize(tmpV) * len;
                    tmpManipData.wsPos = skfPosV + tmpV;
                    if (this->modelBbox.Contains(G2P(tmpManipData.wsPos))) {
						tmpV = glm::normalize(tmpV) * (2.0f * len);
                        tmpManipData.wsPos = skfPosV + tmpV;
                        if (this->modelBbox.Contains(G2P(tmpManipData.wsPos))) {
							tmpV = glm::normalize(tmpV) * (3.0f * len);
                            tmpManipData.wsPos = skfPosV + tmpV;
                        }
                    }
                }
                break;
            case (manipType::SELECTED_KF_POS_LOOKAT): 
                tmpV = skfPosV - skfLaV;
				tmpV = glm::normalize(tmpV) * (length);
                tmpManipData.wsPos = skfPosV + tmpV;
                if (this->manipOusideBbox && this->modelBbox.Contains(G2P(tmpManipData.wsPos))) {
					len = glm::length(P2G(this->modelBbox.GetRightTopFront()) - P2G(this->modelBbox.GetLeftBottomBack())) / 3.0f;
					tmpV = glm::normalize(tmpV) * (len);
                    tmpManipData.wsPos = skfPosV + tmpV;
                    if (this->modelBbox.Contains(G2P(tmpManipData.wsPos))) {
						tmpV = glm::normalize(tmpV) * (2.0f * len);
                        tmpManipData.wsPos = skfPosV + tmpV;
                        if (this->modelBbox.Contains(G2P(tmpManipData.wsPos))) {
							tmpV = glm::normalize(tmpV) * (3.0f * len);
                            tmpManipData.wsPos = skfPosV + tmpV;
                        }
                    }
                }
                break;
            default: 
                vislib::sys::Log::DefaultLog.WriteError("[KeyframeManipulators] [updateManipulators] Bug: %i", i); 
                return false;
        }
        tmpManipData.ssPos     = this->getScreenSpace(tmpManipData.wsPos);
        tmpManipData.offset    = glm::length(this->getScreenSpace(tmpManipData.wsPos + this->circleVertices[1]) - tmpManipData.ssPos);

        this->manipArray[i].wsPos  = tmpManipData.wsPos;
        this->manipArray[i].ssPos  = tmpManipData.ssPos;
        this->manipArray[i].offset = tmpManipData.offset;
        // DO NOT change availability ...
    }

    return true;
}


void KeyframeManipulators::UpdateExtents(vislib::math::Cuboid<float>& inout_bbox) {

    // Store hard copy current bounding box of model
    this->current_bbox = inout_bbox;
    // Grow bounding box of model to manipulator positions
    for (unsigned int i = 0; i < this->manipulators.size(); i++) {
        inout_bbox.GrowToPoint(G2P(this->manipulators[i].position));
    }
}


int KeyframeManipulators::GetSelectedKeyframePositionIndex(float mouse_x, float mouse_y) {

    int index = -1;
    for (auto& m : this->manipulators) {
        if ((m.group == Manipulator::Group::SELECTION_KEYFRAME_POSITIONS) && (m.show)) {
            glm::vec2 pos = this->world2ScreenSpace(m.position);
            glm::vec2 mouse = glm::vec2(mouse_x, mouse_y);
            if (glm::length(pos - mouse) <= m.point_radius) {
                return m.keyframe_index;
            }
        }
    }
    return index;
}


bool KeyframeManipulators::CheckForHitManipulator(float mouse_x, float mouse_y) {

    this->current_hit = nullptr;

    for (auto& m : this->manipulators) {
        if ((m.group != Manipulator::Group::SELECTION_KEYFRAME_POSITIONS) && (m.show)) {
            glm::vec2 pos = this->world2ScreenSpace(m.position);
            glm::vec2 mouse = glm::vec2(mouse_x, mouse_y);
            if (glm::length(pos - mouse) <= m.point_radius) {
                this->current_mouse = mouse;
                this->current_hit = std::make_shared<std::vector<Manipulator>>(m); // TUT DAS?
                return true;
            }
        }
    }
    return false;
}


bool KeyframeManipulators::ProcessHitManipulator(float mouse_x, float mouse_y) {

    if (this->current_hit == nullptr) return false;

    unsigned int index = static_cast<unsigned int>(this->activeType);
    if (index >= static_cast<unsigned int>(NUM_OF_SELECTED_MANIP)) {
        vislib::sys::Log::DefaultLog.WriteError("[KeyframeManipulators] [processManipHit] Bug"); 
        return false;
    }

    float lineDiff = 0.0f;

    // Local copies as vectors
    glm::vec3 skfPosV = this->selectedKf.GetCamPosition();
    glm::vec3 skfLaV  = this->selectedKf.GetCamLookAt();
    glm::vec3 skfUpV  = this->selectedKf.GetCamUp();

    glm::vec2 ssVec = this->manipArray[index].ssPos - this->sKfSsPos;
    if ((this->activeType == manipType::SELECTED_KF_LOOKAT_X) || 
        (this->activeType == manipType::SELECTED_KF_LOOKAT_Y) || 
        (this->activeType == manipType::SELECTED_KF_LOOKAT_Z)) {
        ssVec = this->manipArray[index].ssPos - this->sKfSsLookAt;
    }
    if ((this->activeType == manipType::CTRL_POINT_POS_X) ||
        (this->activeType == manipType::CTRL_POINT_POS_Y) ||
        (this->activeType == manipType::CTRL_POINT_POS_Z)) {

        if (this->selectedIsFirst) {
            ssVec = this->manipArray[index].ssPos - this->getScreenSpace(this->startCtrllPos);
        }
        else if (this->selectedIsLast) {
            ssVec = this->manipArray[index].ssPos - this->getScreenSpace(this->endCtrllPos);
        }
    }

    // Select manipulator axis with greatest contribution
    if (vislib::math::Abs(ssVec.x) > vislib::math::Abs(ssVec.y)) {
        lineDiff = (x - this->lastMousePos.x) * sensitivity;
        if (ssVec.x < 0.0f) { // Adjust line changes depending on manipulator axis direction
            lineDiff *= -1.0f;
        }
    }
    else {
        lineDiff = (y - this->lastMousePos.y) * this->sensitivity;
        if (ssVec.y < 0.0f) { // Adjust line changes depending on manipulator axis direction
            lineDiff *= -1.0f;
        }
    }
    lineDiff *= ((glm::length(this->worldCamModDir) * this->axisLengthFac));


    glm::vec3 tmpVec;
    switch (this->activeType) {
        case (manipType::CTRL_POINT_POS_X):                  
            if (this->selectedIsFirst) {
                this->startCtrllPos.x = this->startCtrllPos.x + lineDiff;
            }
            else if (this->selectedIsLast) {
               this->endCtrllPos.x = this->endCtrllPos.x + lineDiff;
            }
            break;
        case (manipType::CTRL_POINT_POS_Y):      
            if (this->selectedIsFirst) {
                this->startCtrllPos.y = this->startCtrllPos.y + lineDiff;
            }
            else if (this->selectedIsLast) {
                this->endCtrllPos.y = this->endCtrllPos.y + lineDiff;
            }
            break;
        case (manipType::CTRL_POINT_POS_Z):      
            if (this->selectedIsFirst) {
                this->startCtrllPos.z = this->startCtrllPos.z + lineDiff;
            }
            else if (this->selectedIsLast) {
                this->endCtrllPos.z = this->endCtrllPos.z + lineDiff;
            }
            break;
        case (manipType::SELECTED_KF_POS_X):      skfPosV.x = skfPosV.x + lineDiff; break;
        case (manipType::SELECTED_KF_POS_Y):      skfPosV.y = skfPosV.y + lineDiff; break;
        case (manipType::SELECTED_KF_POS_Z):      skfPosV.z = skfPosV.z + lineDiff; break;
        case (manipType::SELECTED_KF_LOOKAT_X):   skfLaV.x = skfLaV.x + lineDiff; break;
        case (manipType::SELECTED_KF_LOOKAT_Y):   skfLaV.y = skfLaV.y + lineDiff; break;
        case (manipType::SELECTED_KF_LOOKAT_Z):   skfLaV.z = skfLaV.z + lineDiff; break;
        case (manipType::SELECTED_KF_POS_LOOKAT): tmpVec = (skfPosV - skfLaV);
                                                  tmpVec = glm::normalize(tmpVec) * (lineDiff);
                                                  skfPosV += tmpVec;
                                                  break;
	default: break;
    }

    if (this->activeType == manipType::SELECTED_KF_UP) {
        bool cwRot = glm::length(this->worldCamLaDir - skfLaV) > glm::length(this->worldCamLaDir - skfPosV);
        glm::vec3 tmpSsUp = glm::vec3(0.0f, 0.0f, 1.0f); // up vector for screen space
        if (!cwRot) {
            tmpSsUp.z = -1.0f;
        }
        glm::vec2 tmpM          = this->lastMousePos - this->sKfSsPos;
        glm::vec3 tmpSsMani     = glm::vec3(tmpM.x, tmpM.y, 0.0f);
		tmpSsMani = glm::normalize(tmpSsMani);
        glm::vec3 tmpSsRight    = glm::cross(tmpSsMani, tmpSsUp);
        glm::vec3 tmpDeltaMouse = glm::vec3(x - this->lastMousePos.x, y - this->lastMousePos.y, 0.0f);

        lineDiff = glm::abs(glm::length(tmpDeltaMouse)) * this->sensitivity / 4.0f; // Adjust sensitivity of rotation here ...
        if (glm::dot(tmpSsRight, tmpSsMani + tmpDeltaMouse) < 0.0f) {
            lineDiff *= -1.0f;
        }
        lineDiff /= ((glm::length(this->worldCamModDir) * this->axisLengthFac));

        // rotate up vector aroung lookat vector with the "Rodrigues' rotation formula"
        glm::vec3 k = (skfPosV - skfLaV); // => rotation axis = camera lookat
        skfUpV = skfUpV * glm::cos(lineDiff) + glm::cross(k, skfUpV) * glm::sin(lineDiff) + k * glm::dot(k, skfUpV) * (1.0f - glm::cos(lineDiff));
    }

    // Apply changes to selected keyframe
    this->selectedKf.SetCameraPosition(skfPosV);
    this->selectedKf.SetCameraLookAt(skfLaV);
    glm::normalize(skfUpV);
    this->selectedKf.SetCameraUp(skfUpV);

    // Update manipulators
    this->updateManipulatorPositions();

    this->lastMousePos.x = x;
    this->lastMousePos.y = y;

    return true;
}


glm::vec2 KeyframeManipulators::world2ScreenSpace(glm::vec3 vec) {

	glm::vec4 world = { vec.x, vec.y, vec.z, 1.0f };
    world = this->current_mvp * world;
    world = world / world.w;
    glm::vec2 screen;
    screen.x = (screen.x + 1.0f) / 2.0f * this->current_viewport.x;
    screen.y = glm::abs(screen.y - 1.0f) / 2.0f * this->current_viewport.y; // (flipped y-axis)
    return screen;
}



bool KeyframeManipulators::Draw(void) {

    glm::vec3 skfPosV = this->selectedKf.GetCamPosition();
    glm::vec3 skfLaV  = this->selectedKf.GetCamLookAt();

    GLfloat tmpLw;
    glGetFloatv(GL_LINE_WIDTH, &tmpLw);
    glLineWidth(2.0f);

    // Rest of necessary OpenGl settings are already done in TrackingShotRenderer ...

    // Get the foreground color (inverse background color)
    float bgColor[4];
    float fgColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glGetFloatv(GL_COLOR_CLEAR_VALUE, bgColor);
    for (unsigned int i = 0; i < 4; i++) {
        fgColor[i] -= bgColor[i];
    }
    // COLORS
    float kColor[4]   = { 0.7f, 0.7f, 1.0f, 1.0f }; // Color for KEYFRAME
    float skColor[4]  = {0.2f, 0.2f, 1.0f, 1.0f };  // Color for SELECTED KEYFRAME
    float mlaColor[4] = {0.3f, 0.8f, 0.8f, 1.0f };  // Color for MANIPULATOR LOOKAT
    float muColor[4]  = {0.8f, 0.0f, 0.8f, 1.0f };  // Color for MANIPULATOR UP
    float mxColor[4]  = {0.8f, 0.1f, 0.0f, 1.0f };  // Color for MANIPULATOR X-AXIS
    float myColor[4]  = {0.8f, 0.8f, 0.0f, 1.0f };  // Color for MANIPULATOR Y-AXIS
    float mzColor[4]  = {0.1f, 0.8f, 0.0f, 1.0f };  // Color for MANIPULATOR Z-AXIS
    // Adapt colors depending on  Lightness
    float L = (vislib::math::Max(bgColor[0], vislib::math::Max(bgColor[1], bgColor[2])) + vislib::math::Min(bgColor[0], vislib::math::Min(bgColor[1], bgColor[2]))) / 2.0f;
    if (L < 0.5f) {
        float tmp;
        // Swap keyframe colors
        for (unsigned int i = 0; i < 4; i++) {
            tmp = kColor[i];
            kColor[i] = skColor[i];
            skColor[i] = tmp;
        }
    }

    // Draw keyframe positions
    if (static_cast<int>(this->keyframes.size()) > 0) {
        if (this->keyframes[0].available) {
            for (unsigned int k = 0; k < static_cast<unsigned int>(this->keyframes.size()); k++) {
                if (this->keyframes[k].wsPos == skfPosV) {
                    glColor4fv(skColor);
                }
                else {
                    glColor4fv(kColor);
                }
                this->drawCircle(this->keyframes[k].wsPos, 1.0f);
            }
        }
    }

    // get control point position for first and last keyframe
    glm::vec3 tmpCtrlPos;
    if (this->selectedIsFirst) {
        tmpCtrlPos = this->startCtrllPos;
    }
    else if (this->selectedIsLast) {
        tmpCtrlPos = this->endCtrllPos;
    }

    // Draw manipulators
    if (this->sKfInArray >= 0) {

        // Draw line between current keyframe position and dragged keyframe position
        if (this->keyframes[this->sKfInArray].wsPos != skfPosV) {
            glBegin(GL_LINES);
            glColor4fv(skColor);
                glVertex3fv(glm::value_ptr(skfPosV));
                glVertex3fv(glm::value_ptr(this->keyframes[this->sKfInArray].wsPos));
            glEnd();
        }

        for (unsigned int i = 0; i < static_cast<unsigned int>(this->manipArray.size()); i++) {
            if (this->manipArray[i].available) {
                switch (static_cast<manipType>(i)) {
                case (manipType::CTRL_POINT_POS_X):  
                    if ((this->selectedIsFirst) || (this->selectedIsLast)) {
                        glColor4fv(mlaColor);
                        this->drawManipulator(tmpCtrlPos, this->manipArray[i].wsPos);
                        // Draw additional line only once
                        glBegin(GL_LINES);
                        glColor4fv(mlaColor);
                            glVertex3fv(glm::value_ptr(skfPosV));
                            glVertex3fv(glm::value_ptr(tmpCtrlPos));
                        glEnd();
                    }
                    break;
                case (manipType::CTRL_POINT_POS_Y):    
                    if ((this->selectedIsFirst) || (this->selectedIsLast)) {
                        glColor4fv(mlaColor);
                        this->drawManipulator(tmpCtrlPos, this->manipArray[i].wsPos);
                    }
                    break;
                case (manipType::CTRL_POINT_POS_Z):     
                    if ((this->selectedIsFirst) || (this->selectedIsLast)) {
                        glColor4fv(mlaColor);
                        this->drawManipulator(tmpCtrlPos, this->manipArray[i].wsPos);
                    }
                    break;
                case (manipType::SELECTED_KF_POS_X):      glColor4fv(mxColor);  this->drawManipulator(skfPosV, this->manipArray[i].wsPos); break;
                case (manipType::SELECTED_KF_POS_Y):      glColor4fv(myColor);  this->drawManipulator(skfPosV, this->manipArray[i].wsPos); break;
                case (manipType::SELECTED_KF_POS_Z):      glColor4fv(mzColor);  this->drawManipulator(skfPosV, this->manipArray[i].wsPos); break;
                case (manipType::SELECTED_KF_UP):         glColor4fv(muColor);  this->drawManipulator(skfPosV, this->manipArray[i].wsPos); break;
                case (manipType::SELECTED_KF_POS_LOOKAT): glColor4fv(mlaColor); this->drawManipulator(skfLaV, this->manipArray[i].wsPos); break;
                case (manipType::SELECTED_KF_LOOKAT_X):   glColor4fv(mxColor);  this->drawManipulator(skfLaV, this->manipArray[i].wsPos); break;
                case (manipType::SELECTED_KF_LOOKAT_Y):   glColor4fv(myColor);  this->drawManipulator(skfLaV, this->manipArray[i].wsPos); break;
                case (manipType::SELECTED_KF_LOOKAT_Z):   glColor4fv(mzColor);  this->drawManipulator(skfLaV, this->manipArray[i].wsPos); break;
                default: vislib::sys::Log::DefaultLog.WriteError("[KeyframeManipulators] [draw] Bug.");  break;
                }
            }
        }
    }
    else { // If selected keyframe is not in keyframe array just draw keyframe position, lookat line and up line
        glBegin(GL_LINES);
            // Up
            glColor4fv(muColor);
            glVertex3fv(glm::value_ptr(skfPosV));
            glVertex3fv(glm::value_ptr(this->manipArray[static_cast<int>(manipType::SELECTED_KF_UP)].wsPos));
            // LookAt
            glColor4fv(mlaColor);
            glVertex3fv(glm::value_ptr(skfPosV));
            glVertex3fv(glm::value_ptr(skfLaV));
        glEnd();
        // Keyframe position
        glColor4fv(skColor);
        this->drawCircle(skfPosV, 0.75f);
    }

    // Reset opengl
    glLineWidth(tmpLw);

    return true;
}
