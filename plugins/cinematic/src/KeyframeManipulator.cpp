/**
* KeyframeManipulator.cpp
*
* Copyright (C) 2017 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"

#include "KeyframeManipulator.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::cinematic;


KeyframeManipulator::KeyframeManipulator(void) :
    circleRadiusFac(0.0075f), // const
    axisLengthFac(0.06f),     // const
    circleSubDiv(20),         // const
    lineWidth(2.5),           // const
    sensitivity(0.01f),       // const
    kfArray(),
    selectedKf(),
    manipArray(),
    sKfSsPos(),
    sKfSsLookAt(),
    sKfInArray(-1),
    activeType(manipType::NONE),
    lastMousePos(),
    modelViewProjMatrix(),
    viewportSize(),
    worldCamLaDir(),
    worldCamModDir(),
    isDataSet(false),
    isDataDirty(true),
    modelBbox(),
    manipOusideBbox(false),
    startCtrllPos(),
    endCtrllPos(),
    selectedIsFirst(false),
    selectedIsLast(false),
    circleVertices()
{

}


KeyframeManipulator::~KeyframeManipulator(void) {

    // nothing to do here ...
}


bool KeyframeManipulator::Update(std::vector<KeyframeManipulator::manipType> am, std::shared_ptr<std::vector<Keyframe>> kfa, Keyframe skf,
	float vph, float vpw, glm::mat4 mvpm, glm::vec3 wclad, glm::vec3 wcmd, bool mob, glm::vec3 fcp, glm::vec3 lcp) {

    if (kfa == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("[KeyframeManipulator] [Update] Pointer to keyframe array is nullptr.");
        return false;
    }

    // ...
    this->manipOusideBbox = mob;

    // Update ModelViewPorjectionMatrix ---------------------------------------
    // and position of world camera 
    bool worldChanged = false;
    if ((this->modelViewProjMatrix != mvpm) || (this->worldCamLaDir != wclad) || (this->worldCamModDir != wcmd)) {
        this->calculateCircleVertices();
        worldChanged = true;
    }
    this->modelViewProjMatrix = mvpm;
    this->worldCamLaDir       = wclad;
    this->worldCamModDir      = wcmd;

    // Update viewport --------------------------------------------------------
    this->viewportSize.x = vph;
    this->viewportSize.y = vpw;
    
    // Update slected keyframe ------------------------------------------------
    // Update manipulator only if selected keyframe changed
    if ((this->selectedKf != skf) || (this->startCtrllPos != fcp) || (this->endCtrllPos != lcp) || worldChanged) {
        this->selectedKf = skf;

        // Check if selected keyframe exists in keyframe array
        this->sKfInArray = -1;
        for (int i = 0; i < kfa->size(); ++i) {
            if (kfa->at(i) == this->selectedKf) {
                this->sKfInArray = i;
            }
        }

        this->selectedIsFirst = (this->selectedKf == kfa->front())?(true):(false);
        this->selectedIsLast  = (this->selectedKf == kfa->back())?(true):(false);
        this->startCtrllPos   = fcp;
        this->endCtrllPos     = lcp;

        this->updateManipulatorPositions();
    }

    // Update local keyframe positions ------------------------------------
    unsigned int kfACnt     = static_cast<unsigned int>(kfa->size());
    unsigned int kfArrayCnt = static_cast<unsigned int>(this->kfArray.size());

    bool kfAvail = false;
    for (auto m : am) {
        if (m == manipType::KEYFRAME_POS) {
            kfAvail = true;
        }
    }

    if ((kfACnt != kfArrayCnt) || worldChanged) {
        this->kfArray.clear();
        this->kfArray.reserve(kfACnt);
        for (unsigned int i = 0; i < kfACnt; i++) {
            manipPosData tmpkfA;
            tmpkfA.wsPos     = (*kfa)[i].GetCamPosition();
            tmpkfA.ssPos     = this->getScreenSpace(tmpkfA.wsPos);
            tmpkfA.offset    = glm::length(this->getScreenSpace(tmpkfA.wsPos + this->circleVertices[1]) - tmpkfA.ssPos);
            tmpkfA.available = kfAvail;
            this->kfArray.emplace_back(tmpkfA);
        }
    }
    else { // Update positions (which might have changed)
        for (unsigned int i = 0; i < kfACnt; i++) {
            if (this->kfArray[i].wsPos    != (*kfa)[i].GetCamPosition()) { 
                this->kfArray[i].wsPos     = (*kfa)[i].GetCamPosition();
                this->kfArray[i].ssPos     = this->getScreenSpace(this->kfArray[i].wsPos);
                this->kfArray[i].offset    = glm::length(this->getScreenSpace(this->kfArray[i].wsPos + this->circleVertices[1]) - this->kfArray[i].ssPos); 
                this->kfArray[i].available = kfAvail;
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


bool KeyframeManipulator::updateManipulatorPositions() {

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
            this->manipArray.emplace_back(manipPosData());
        }
    }

    // Adaptive axis length of manipulators
    float radius = glm::length(this->circleVertices[1]);
    float length = vislib::math::Max((glm::length(this->worldCamModDir) * this->axisLengthFac), 0.1f);
    glm::vec3 tmpV;
    float len;

    manipPosData tmpManipData;
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
                vislib::sys::Log::DefaultLog.WriteError("[KeyframeManipulator] [updateManipulators] Bug: %i", i); 
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


void KeyframeManipulator::SetExtents(vislib::math::Cuboid<float>& bb) {

    // Store hard copy current bounding box of model
    this->modelBbox = bb;

    // Grow bounding box of model to manipulators
    if (this->isDataSet) {
        for (unsigned int i = 0; i < this->manipArray.size(); i++) {
            bb.GrowToPoint(G2P(this->manipArray[i].wsPos));
        }
    }
}


int KeyframeManipulator::CheckKeyframePositionHit(float x, float y) {

    if (!this->isDataSet) {
        vislib::sys::Log::DefaultLog.WriteWarn("[KeyframeManipulator] [checkForHit] Data is not set. Please call 'update' first.");
        return false;
    }

    int index = -1;
    if (static_cast<int>(this->kfArray.size()) > 0) {
        if (this->kfArray[0].available) {
            for (int i = 0; i < static_cast<int>(this->kfArray.size()); i++) {
                float offset = this->kfArray[i].offset;
                glm::vec2 pos = this->kfArray[i].ssPos;
                // Check if mouse position lies within offset quad around keyframe position
                if (((pos.x < (x + offset)) && (pos.x > (x - offset))) &&
                    ((pos.y < (y + offset)) && (pos.y > (y - offset)))) {
                    return i;
                }
            }
        }
    }

    return index;
}


bool KeyframeManipulator::CheckManipulatorHit(float x, float y) {

    if (!this->isDataSet) {
        vislib::sys::Log::DefaultLog.WriteWarn("[KeyframeManipulator] [checkForHit] Data is not set. Please call 'update' first.");
        return false;
    }

    this->activeType = manipType::NONE;

    for (int i = 0; i < static_cast<int>(this->manipArray.size()); i++) {
        if (this->manipArray[i].available) {
            float offset = this->manipArray[i].offset;
            glm::vec2 pos = this->manipArray[i].ssPos;
            // Check if mouse position lies within offset quad around manipulator position
            if (((pos.x < (x + offset)) && (pos.x > (x - offset))) &&
                ((pos.y < (y + offset)) && (pos.y > (y - offset)))) {
                this->activeType = static_cast<manipType>(i);
                this->lastMousePos.x = x;
                this->lastMousePos.y = y;
                return true;
            }
        }
    }
    return false;
}


bool KeyframeManipulator::ProcessManipulatorHit(float x, float y) {

    if (!this->isDataSet) {
        vislib::sys::Log::DefaultLog.WriteWarn("[KeyframeManipulator] [processManipHit] Data is not set. Please call 'update' first.");
        return false;
    }

    // Return if currently no manipulator is hit
    if (this->activeType == manipType::NONE) {
        return false;
    }

    unsigned int index = static_cast<unsigned int>(this->activeType);
    if (index >= static_cast<unsigned int>(NUM_OF_SELECTED_MANIP)) {
        vislib::sys::Log::DefaultLog.WriteError("[KeyframeManipulator] [processManipHit] Bug"); 
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


Keyframe KeyframeManipulator::GetManipulatedKeyframe(void) {

    return this->selectedKf;
}


glm::vec3 KeyframeManipulator::GetFirstControlPointPosition() {

    return this->startCtrllPos;
}


glm::vec3 KeyframeManipulator::GetLastControlPointPosition() {

    return this->endCtrllPos;
}


glm::vec2 KeyframeManipulator::getScreenSpace(glm::vec3 wp) {

    // Transforming position from world space to screen space:

    // World space position
	glm::vec4 wsPos = { wp.x, wp.y, wp.z, 1.0f };
    // Screen space position
    glm::vec4 ssTmpPos = this->modelViewProjMatrix * wsPos;
    // Division by 'w'
    ssTmpPos = ssTmpPos / ssTmpPos.w;
    // Transform to viewport coordinates (x,y in [-1,1] -> viewport size)
    glm::vec2 ssPos;
    ssPos.x = (ssTmpPos.x + 1.0f) / 2.0f * this->viewportSize.x;
    ssPos.y = vislib::math::Abs(ssTmpPos.y - 1.0f) / 2.0f * this->viewportSize.y; // flip y-axis

    return ssPos;
}


void KeyframeManipulator::calculateCircleVertices(void) {

    this->circleVertices.clear();
    this->circleVertices.reserve(this->circleSubDiv);

    // Get normal for plane the cirlce lies on
    glm::vec3 normal = this->worldCamLaDir;
    // Check if world camera direction is zero ...
    if ((normal.x == 0.0f) && (normal.y == 0.0f) && (normal.z == 0.0f) ) {
        normal.z = 1.0f;
        //vislib::sys::Log::DefaultLog.WriteWarn("[KeyframeManipulator] [calculateCircleVertices] LookAt direction of world camera shouldn't be zero.");
    }
	normal = glm::normalize(normal);
    // Size of radius depends on the length of the lookat direction vector
    float radius = glm::max((glm::length(this->worldCamModDir) * this->circleRadiusFac), 0.02f);
    // Get arbitary vector vertical to normal
    glm::vec3 rot = glm::vec3(normal.z, 0.0f, -(normal.x));
    rot = glm::normalize(rot) * (radius);
    // rotate up vector aroung lookat vector with the "Rodrigues' rotation formula" 
    float t = 2.0f*(float)(CINEMATIC_PI) / (float)(this->circleSubDiv); // theta angle for rotation   
    // First vertex is center of triangle fan
    this->circleVertices.emplace_back(glm::vec3(0.0f, 0.0f, 0.0f));
    for (unsigned int i = 0; i <= this->circleSubDiv; i++) {
        rot = rot * glm::cos(t) + glm::cross(normal, rot) * glm::sin(t) + normal * glm::dot(normal, rot) * (1.0f - glm::cos(t));
		rot = glm::normalize(rot) * (radius);
        this->circleVertices.emplace_back(rot);
    }
}


bool KeyframeManipulator::Draw(void) {

    if (!this->isDataSet) {
        vislib::sys::Log::DefaultLog.WriteError("[KeyframeManipulator] [draw] Data is not set. Please call 'update' first.");
        return false;
    }

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
    if (static_cast<int>(this->kfArray.size()) > 0) {
        if (this->kfArray[0].available) {
            for (unsigned int k = 0; k < static_cast<unsigned int>(this->kfArray.size()); k++) {
                if (this->kfArray[k].wsPos == skfPosV) {
                    glColor4fv(skColor);
                }
                else {
                    glColor4fv(kColor);
                }
                this->drawCircle(this->kfArray[k].wsPos, 1.0f);
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
        if (this->kfArray[this->sKfInArray].wsPos != skfPosV) {
            glBegin(GL_LINES);
            glColor4fv(skColor);
                glVertex3fv(glm::value_ptr(skfPosV));
                glVertex3fv(glm::value_ptr(this->kfArray[this->sKfInArray].wsPos));
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
                default: vislib::sys::Log::DefaultLog.WriteError("[KeyframeManipulator] [draw] Bug.");  break;
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


void KeyframeManipulator::drawCircle(glm::vec3 pos, float factor) {

    glBegin(GL_TRIANGLE_FAN);
    for (unsigned int i = 0; i < static_cast<unsigned int>(circleVertices.size()); i++) {
        glVertex3fv(glm::value_ptr(pos + (this->circleVertices[i] * factor)));
    }
    glEnd();
}


void KeyframeManipulator::drawManipulator(glm::vec3 kp, glm::vec3 mp) {

    this->drawCircle(mp, 1.0f);
    glBegin(GL_LINES);
        glVertex3fv(glm::value_ptr(kp));
        glVertex3fv(glm::value_ptr(mp));
    glEnd();
}
