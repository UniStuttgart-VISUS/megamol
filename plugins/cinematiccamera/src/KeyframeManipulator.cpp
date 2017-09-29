/**
* KeyframeManipulator.cpp
*
*/

#include "stdafx.h"
#include "KeyframeManipulator.h"

#include "vislib/sys/Log.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::cinematiccamera;

#ifndef CC_PI
    #define CC_PI 3.1415926535897
#endif

/*
* KeyframeManipulator::KeyframeManipulator
*/
KeyframeManipulator::KeyframeManipulator(void) :
    selectedKf()
    {

    // init variables
    this->activeType          = manipType::NONE;
    this->lastMousePos        = vislib::math::Vector<float, 2>();
    this->sKfSsPos            = vislib::math::Vector<float, 2>();
    this->sKfSsLookAt         = vislib::math::Vector<float, 2>();
    this->sKfInArray          = false;;
    this->modelViewProjMatrix = vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR>();
    this->viewportSize        = vislib::math::Dimension<int, 2>();
    this->worldCamDir         = vislib::math::Vector<float, 3>();
    this->isDataDirty         = true;
    this->kfArray.Clear();
    this->manipArray.Clear();
    this->circleVertices.Clear();

    this->isDataSet = false;
}


/*
* KeyframeManipulator::~KeyframeManipulator
*/
KeyframeManipulator::~KeyframeManipulator(void) {
    // intentionally empty
}


/*
* KeyframeManipulator::KeyframeManipulator
*/
bool KeyframeManipulator::update(vislib::Array<KeyframeManipulator::manipType> am, vislib::Array<Keyframe>* kfa, Keyframe skf, float vph, float vpw,
    vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> mvpm, vislib::math::Vector<float, 3> wcd) {

    if (kfa == NULL) {
        vislib::sys::Log::DefaultLog.WriteError("[KeyframeManipulator] [Update] Pointer to keyframe array is NULL.");
        return false;
    }

    // Update ModelViewPorjectionMatrix ---------------------------------------
    // and position of world camera 
    bool recalcKf = false;
    if ((this->modelViewProjMatrix != mvpm) || (this->worldCamDir != wcd)) {
        this->calculateCircleVertices();
        recalcKf = true;
    }
    this->modelViewProjMatrix = mvpm;
    this->worldCamDir         = wcd;

    // Update viewport --------------------------------------------------------
    this->viewportSize.SetHeight(vph);
    this->viewportSize.SetWidth(vpw);
    
    // Update slected keyframe ------------------------------------------------
    // Update manipulator only if selected keyframe changed
    if ((this->selectedKf != skf) || recalcKf) {
        this->selectedKf = skf;

        // Check if selected keyframe exists in keyframe array
        this->sKfInArray = false;
        int selIndex = static_cast<int>(kfa->IndexOf(this->selectedKf));
        if (selIndex >= 0) {
            this->sKfInArray = true;
        }

        this->updateManipulators();
    }

    // Update  keyframe positions of array ------------------------------------
    unsigned int kfACnt     = static_cast<unsigned int>(kfa->Count());
    unsigned int kfArrayCnt = static_cast<unsigned int>(this->kfArray.Count());
    if ((kfACnt != kfArrayCnt) || recalcKf) {
        this->kfArray.Clear();
        this->kfArray.AssertCapacity(kfACnt);
        for (unsigned int i = 0; i < kfACnt; i++) {
            manipPosData tmpkfA;
            tmpkfA.wsPos = (*kfa)[i].getCamPosition().operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>();
            tmpkfA.ssPos = this->getScreenSpace(tmpkfA.wsPos);
            tmpkfA.offset = (this->getScreenSpace(tmpkfA.wsPos + this->circleVertices[1]) - tmpkfA.ssPos).Norm(); // ...
            this->kfArray.Add(tmpkfA);
        }
    }
    else { // Update only different positions
        for (unsigned int i = 0; i < kfACnt; i++) {
            if (this->kfArray[i].wsPos != (*kfa)[i].getCamPosition().operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>()) { 
                this->kfArray[i].wsPos  = (*kfa)[i].getCamPosition().operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>();
                this->kfArray[i].ssPos  = this->getScreenSpace(this->kfArray[i].wsPos);
                this->kfArray[i].offset = (this->getScreenSpace(this->kfArray[i].wsPos + this->circleVertices[1]) - this->kfArray[i].ssPos).Norm(); 
            }
        }
    }

    // Reset and update available manipulators
    if (this->kfArray.Count() > 0) {
        this->kfArray[0].available = false;
        this->kfArray[0].available = am.Contains(manipType::KEYFRAME_POS);
    }
    for (unsigned int i = 0; i < static_cast<unsigned int>(this->manipArray.Count()); i++) {
        this->manipArray[i].available = false;
    }
    if (this->sKfInArray) { // Manipulators are only available if selected keyframe exists in keyframe array
        for (unsigned int i = 0; i < static_cast<unsigned int>(am.Count()); i++) {
            unsigned int index = static_cast<unsigned int>(am[i]);
            if (index < static_cast<unsigned int>(NUM_OF_SELECTED_MANIP)) {
                this->manipArray[index].available = true;
            }
        }
    }

    this->isDataSet = true;
    return true;
}



/*
* KeyframeManipulator::updateSKfManipulators
*/
bool KeyframeManipulator::updateManipulators() {

    vislib::math::Vector<float, 3> skfPosV = this->selectedKf.getCamPosition().operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>();
    vislib::math::Vector<float, 3> skfLaV  = this->selectedKf.getCamLookAt().operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>();
    // Update screen space positions
    this->sKfSsPos    = this->getScreenSpace(skfPosV);
    this->sKfSsLookAt = this->getScreenSpace(skfLaV);

    manipPosData tmpkfS;
    if (this->manipArray.IsEmpty()) {
        this->manipArray.Clear();
        this->manipArray.AssertCapacity(static_cast<unsigned int>(manipType::NUM_OF_SELECTED_MANIP));
        for (unsigned int i = 0; i < static_cast<unsigned int>(manipType::NUM_OF_SELECTED_MANIP); i++) {
            this->manipArray.Add(tmpkfS);
        }
    }

    // Adaptive axis length of manipulators
    float length = vislib::math::Max((this->worldCamDir.Length() * this->axisLengthFac), 1.0f);

    for (unsigned int i = 0; i < static_cast<unsigned int>(manipType::NUM_OF_SELECTED_MANIP); i++) { // skip SELECTED_KF_POS
        switch (static_cast<manipType>(i)) {
            case (manipType::SELECTED_KF_POS_X):      tmpkfS.wsPos = skfPosV + vislib::math::Vector<float, 3>(length, 0.0f, 0.0f); break;
            case (manipType::SELECTED_KF_POS_Y):      tmpkfS.wsPos = skfPosV + vislib::math::Vector<float, 3>(0.0f, length, 0.0f); break;
            case (manipType::SELECTED_KF_POS_Z):      tmpkfS.wsPos = skfPosV + vislib::math::Vector<float, 3>(0.0f, 0.0f, length); break;
            case (manipType::SELECTED_KF_LOOKAT_X):   tmpkfS.wsPos = skfLaV + vislib::math::Vector<float, 3>(length, 0.0f, 0.0f); break;
            case (manipType::SELECTED_KF_LOOKAT_Y):   tmpkfS.wsPos = skfLaV + vislib::math::Vector<float, 3>(0.0f, length, 0.0f); break;
            case (manipType::SELECTED_KF_LOOKAT_Z):   tmpkfS.wsPos = skfLaV + vislib::math::Vector<float, 3>(0.0f, 0.0f, length); break;
            case (manipType::SELECTED_KF_UP):         tmpkfS.wsPos = this->selectedKf.getCamUp(); 
                                                      tmpkfS.wsPos.ScaleToLength(length);
                                                      tmpkfS.wsPos += skfPosV;
                                                      break;
            case (manipType::SELECTED_KF_POS_LOOKAT): tmpkfS.wsPos = skfPosV - skfLaV;
                                                      tmpkfS.wsPos.ScaleToLength(length);
                                                      tmpkfS.wsPos += skfPosV;
                                                      break;
            default: vislib::sys::Log::DefaultLog.WriteError("[KeyframeManipulator] [Update] Bug: %i", i); return false;
        }
        tmpkfS.ssPos     = this->getScreenSpace(tmpkfS.wsPos);
        tmpkfS.offset    = (this->getScreenSpace(tmpkfS.wsPos + this->circleVertices[1]) - tmpkfS.ssPos).Norm();
        this->manipArray[i]  = tmpkfS;
    }

    return true;
}


/*
* KeyframeManipulator::growBbox
*/
void KeyframeManipulator::growBbox(vislib::math::Cuboid<float> *bb) {

    if (bb != NULL) {
        if (this->isDataSet) {
            for (unsigned int i = 0; i < this->manipArray.Count(); i++) {
                bb->GrowToPoint(this->V2P(this->manipArray[i].wsPos));
            }
        }
    }
}


/*
* KeyframeManipulator::checkKfPosHit
*/
int KeyframeManipulator::checkKfPosHit(float x, float y) {

    if (!isDataSet) {
        vislib::sys::Log::DefaultLog.WriteWarn("[KeyframeManipulator] [checkForHit] Data is not set. Please call 'update' first.");
        return false;
    }

    int index = -1;
    if (static_cast<int>(this->kfArray.Count()) > 0) {
        if (this->kfArray[0].available) {
            for (int i = 0; i < static_cast<int>(this->kfArray.Count()); i++) {
                float offset = this->kfArray[i].offset;
                vislib::math::Vector<float, 2> pos = this->kfArray[i].ssPos;
                // Check if mouse position lies within offset quad around keyframe position
                if (((pos.X() < (x + offset)) && (pos.X() > (x - offset))) &&
                    ((pos.Y() < (y + offset)) && (pos.Y() > (y - offset)))) {
                    return i;
                }
            }
        }
    }

    return index;
}


/*
* KeyframeManipulator::checkManipHit
*/
bool KeyframeManipulator::checkManipHit(float x, float y) {

    if (!isDataSet) {
        vislib::sys::Log::DefaultLog.WriteWarn("[KeyframeManipulator] [checkForHit] Data is not set. Please call 'update' first.");
        return false;
    }

    this->activeType = manipType::NONE;

    for (int i = 0; i < static_cast<int>(this->manipArray.Count()); i++) {
        if (this->manipArray[i].available) {
            float offset = this->manipArray[i].offset;
            vislib::math::Vector<float, 2> pos = this->manipArray[i].ssPos;
            // Check if mouse position lies within offset quad around manipulator position
            if (((pos.X() < (x + offset)) && (pos.X() > (x - offset))) &&
                ((pos.Y() < (y + offset)) && (pos.Y() > (y - offset)))) {
                this->activeType = static_cast<manipType>(i);
                this->lastMousePos.SetX(x);
                this->lastMousePos.SetY(y);
                return true;
            }
        }
    }
    return false;
}

/*
* KeyframeManipulator::ProcessManipHit
*/
bool KeyframeManipulator::processManipHit(float x, float y) {

    if (!isDataSet) {
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
    vislib::math::Vector<float, 3> skfPosV = this->selectedKf.getCamPosition().operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>();
    vislib::math::Vector<float, 3> skfLaV  = this->selectedKf.getCamLookAt().operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>();
    vislib::math::Vector<float, 3> skfUpV  = this->selectedKf.getCamUp();

    vislib::math::Vector<float, 2> ssVec = this->manipArray[index].ssPos - this->sKfSsPos;
    if ((this->activeType == manipType::SELECTED_KF_LOOKAT_X) || 
        (this->activeType == manipType::SELECTED_KF_LOOKAT_Y) || 
        (this->activeType == manipType::SELECTED_KF_LOOKAT_Z)) {
        ssVec = this->manipArray[index].ssPos - this->sKfSsLookAt;
    }

    // Select manipulator axis with greatest contribution
    if (vislib::math::Abs(ssVec.X()) > vislib::math::Abs(ssVec.Y())) {
        lineDiff = (x - this->lastMousePos.X()) * sensitivity;
        if (ssVec.X() < 0.0f) { // Adjust line changes depending on manipulator axis direction
            lineDiff *= -1.0f;
        }
    }
    else {
        lineDiff = (y - this->lastMousePos.Y()) * this->sensitivity;
        if (ssVec.Y() < 0.0f) { // Adjust line changes depending on manipulator axis direction
            lineDiff *= -1.0f;
        }
    }
    lineDiff *= ((this->worldCamDir.Length() * this->axisLengthFac));


    vislib::math::Vector<float, 3> tmpVec;
    switch (this->activeType) {
        case (manipType::SELECTED_KF_POS_X):      skfPosV.SetX(skfPosV.X() + lineDiff); break;
        case (manipType::SELECTED_KF_POS_Y):      skfPosV.SetY(skfPosV.Y() + lineDiff); break;
        case (manipType::SELECTED_KF_POS_Z):      skfPosV.SetZ(skfPosV.Z() + lineDiff); break;
        case (manipType::SELECTED_KF_LOOKAT_X):   skfLaV.SetX(skfLaV.X() + lineDiff); break;
        case (manipType::SELECTED_KF_LOOKAT_Y):   skfLaV.SetY(skfLaV.Y() + lineDiff); break;
        case (manipType::SELECTED_KF_LOOKAT_Z):   skfLaV.SetZ(skfLaV.Z() + lineDiff); break;
        case (manipType::SELECTED_KF_POS_LOOKAT): tmpVec = (skfPosV - skfLaV);
                                                  tmpVec.ScaleToLength(lineDiff);
                                                  skfPosV += tmpVec;
                                                  break;
    }

    if (this->activeType == manipType::SELECTED_KF_UP) {
        bool cwRot = ((this->worldCamDir - skfLaV).Norm() > (this->worldCamDir - skfPosV).Norm());
        vislib::math::Vector<float, 3> tmpSsUp = vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f); // up vector for screen space
        if (!cwRot) {
            tmpSsUp.SetZ(-1.0f);
        }
        vislib::math::Vector<float, 2> tmpM          = this->lastMousePos - this->sKfSsPos;
        vislib::math::Vector<float, 3> tmpSsMani     = vislib::math::Vector<float, 3>(tmpM.X(), tmpM.Y(), 0.0f);
        tmpSsMani.Normalise();
        vislib::math::Vector<float, 3> tmpSsRight    = tmpSsMani.Cross(tmpSsUp);
        vislib::math::Vector<float, 3> tmpDeltaMouse = vislib::math::Vector<float, 3>(x - this->lastMousePos.X(), y - this->lastMousePos.Y(), 0.0f);

        lineDiff = vislib::math::Abs(tmpDeltaMouse.Norm()) * this->sensitivity / 4.0f; // Adjust sensitivity of rotation here ...
        if (tmpSsRight.Dot(tmpSsMani + tmpDeltaMouse) < 0.0f) {
            lineDiff *= -1.0f;
        }
        lineDiff /= ((this->worldCamDir.Length() * this->axisLengthFac));

        // rotate up vector aroung lookat vector with the "Rodrigues' rotation formula"
        vislib::math::Vector<float, 3> k = (skfPosV - skfLaV); // => rotation axis = camera lookat
        skfUpV = skfUpV * cos(lineDiff) + k.Cross(skfUpV) * sin(lineDiff) + k * (k.Dot(skfUpV)) * (1.0f - cos(lineDiff));
    }

    // Apply changes to selected keyframe
    this->selectedKf.setCameraPosition(this->V2P(skfPosV));
    this->selectedKf.setCameraLookAt(this->V2P(skfLaV));
    skfUpV.Normalise();
    this->selectedKf.setCameraUp(skfUpV);

    // Update manipulators
    this->updateManipulators();

    this->lastMousePos.SetX(x);
    this->lastMousePos.SetY(y);

    return true;
}


/*
* KeyframeManipulator::getManipulatedKeyframe
*/
Keyframe KeyframeManipulator::getManipulatedKeyframe(void) {

    return this->selectedKf;
}


/*
* KeyframeManipulator::getScreenSpace
*
* Transform position from world space to screen space
*/
vislib::math::Vector<float, 2> KeyframeManipulator::getScreenSpace(vislib::math::Vector<float, 3> wp) {

    // World space position
    vislib::math::Vector<float, 4> wsPos = vislib::math::Vector<float, 4>(wp.X(), wp.Y(), wp.Z(), 1.0f);
    // Screen space position
    vislib::math::Vector<float, 4> ssTmpPos = this->modelViewProjMatrix * wsPos;
    // Division by 'w'
    ssTmpPos = ssTmpPos / ssTmpPos.GetW();
    // Transform to viewport coordinates (x,y in [-1,1] -> viewport size)
    vislib::math::Vector<float, 2> ssPos;
    ssPos.SetX((ssTmpPos.X() + 1.0f) / 2.0f * this->viewportSize.GetWidth());
    ssPos.SetY(vislib::math::Abs(ssTmpPos.Y() - 1.0f) / 2.0f * this->viewportSize.GetHeight()); // flip y-axis

    return ssPos;
}


/*
* KeyframeManipulator::initCircleVertices
*/
void KeyframeManipulator::calculateCircleVertices(void) {

    this->circleVertices.Clear();
    this->circleVertices.AssertCapacity(this->circleSubDiv);

    // Get normal for plane the cirlce lies on
    vislib::math::Vector<float, 3> normal = this->worldCamDir;
    // Check if world camera direction is zero ...
    if (normal.IsNull()) {
        normal.SetZ(1.0f);
        //vislib::sys::Log::DefaultLog.WriteWarn("[KeyframeManipulator] [calculateCircleVertices] LookAt direction of world camera shouldn't be zero.");
    }
    normal.Normalise();
    // Size of radius depends on the length of the lookat direction vector
    float radius = vislib::math::Max((this->worldCamDir.Length() * this->circleRadiusFac), 0.15f);
    // Get arbitary vector vertical to normal
    vislib::math::Vector<float, 3> rot = vislib::math::Vector<float, 3>(normal.Z(), 0.0f, -(normal.X()));
    rot.ScaleToLength(radius);
    // rotate up vector aroung lookat vector with the "Rodrigues' rotation formula" 
    float t = 2.0f*(float)(CC_PI) / (float)(this->circleSubDiv); // theta angle for rotation   
    // First vertex is center of triangle fan
    this->circleVertices.Add(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
    for (unsigned int i = 0; i <= this->circleSubDiv; i++) {
        rot = rot * cos(t) + normal.Cross(rot) * sin(t) + normal * (normal.Dot(rot)) * (1.0f - cos(t));
        rot.ScaleToLength(radius);
        this->circleVertices.Add(rot);
    }
}


/*
* KeyframeManipulator::Draw
*/
bool KeyframeManipulator::draw(void) {

    if (!isDataSet) {
        vislib::sys::Log::DefaultLog.WriteError("[KeyframeManipulator] [draw] Data is not set. Please call 'update' first.");
        return false;
    }

    vislib::math::Vector<float, 3> skfPosV = this->selectedKf.getCamPosition().operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>();
    vislib::math::Vector<float, 3> skfLaV  = this->selectedKf.getCamLookAt().operator vislib::math::Vector<vislib::graphics::SceneSpaceType, 3U>();

    GLfloat tmpLw;
    glGetFloatv(GL_LINE_WIDTH, &tmpLw);
    glLineWidth(2.0f);

    glEnable(GL_LINE_SMOOTH);
    glDisable(GL_POLYGON_SMOOTH);

    // Rest of necessary OpenGl settings are already done in CinematicRenderer ...

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
    if (static_cast<int>(this->kfArray.Count()) > 0) {
        if (this->kfArray[0].available) {
            for (unsigned int k = 0; k < static_cast<unsigned int>(this->kfArray.Count()); k++) {
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

    // Draw manipulators
    if (this->sKfInArray) {
        for (unsigned int i = 0; i < static_cast<unsigned int>(this->manipArray.Count()); i++) {
            if (this->manipArray[i].available) {
                switch (static_cast<manipType>(i)) {
                case (manipType::SELECTED_KF_POS_X):      glColor4fv(mxColor);  this->drawManipulator(skfPosV, this->manipArray[i].wsPos);    break;
                case (manipType::SELECTED_KF_POS_Y):      glColor4fv(myColor);  this->drawManipulator(skfPosV, this->manipArray[i].wsPos);    break;
                case (manipType::SELECTED_KF_POS_Z):      glColor4fv(mzColor);  this->drawManipulator(skfPosV, this->manipArray[i].wsPos);    break;
                case (manipType::SELECTED_KF_UP):         glColor4fv(muColor);  this->drawManipulator(skfPosV, this->manipArray[i].wsPos);    break;
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
            glVertex3fv(skfPosV.PeekComponents());
            glVertex3fv(this->manipArray[static_cast<int>(manipType::SELECTED_KF_UP)].wsPos.PeekComponents());
            // LookAt
            glColor4fv(mlaColor);
            glVertex3fv(skfPosV.PeekComponents());
            glVertex3fv(skfLaV.PeekComponents());
        glEnd();
        // Keyframe position
        glColor4fv(skColor);
        this->drawCircle(skfPosV, 0.75f);
    }

    // Reset opengl
    glLineWidth(tmpLw);
    glDisable(GL_LINE_SMOOTH);

    return true;
}


/*
* KeyframeManipulator::drawCircle
*/
void KeyframeManipulator::drawCircle(vislib::math::Vector<float, 3> pos, float factor) {

    glBegin(GL_TRIANGLE_FAN);
    for (unsigned int i = 0; i < static_cast<unsigned int>(circleVertices.Count()); i++) {
        glVertex3fv((pos + (this->circleVertices[i] * factor)).PeekComponents());
    }
    glEnd();
}


/*
* KeyframeManipulator::drawManipulator
*/
void KeyframeManipulator::drawManipulator(vislib::math::Vector<float, 3> kp, vislib::math::Vector<float, 3> mp) {

    this->drawCircle(mp, 1.0f);
    glBegin(GL_LINES);
        glVertex3fv(kp.PeekComponents());
        glVertex3fv(mp.PeekComponents());
    glEnd();
}
