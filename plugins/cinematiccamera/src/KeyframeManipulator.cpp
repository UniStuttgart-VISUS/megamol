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
    sKeyframe()
    {

    // init variables
    this->activeType          = manipType::NONE;
    this->lastMousePos        = vislib::math::Vector<float, 2>();
    this->skfPos              = vislib::math::Vector<float, 3>();
    this->skfUp               = vislib::math::Vector<float, 3>();
    this->skfLookAt           = vislib::math::Vector<float, 3>();
    this->skfSsPos            = vislib::math::Vector<float, 2>();
    this->skfSsLookAt         = vislib::math::Vector<float, 2>();
    this->sKeyframeInArray    = false;;
    this->modelViewProjMatrix = vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR>();
    this->viewportSize        = vislib::math::Dimension<int, 2>();
    this->worldCamPos         = vislib::math::Vector<float, 3>();
    this->isDataDirty         = true;
    this->kfArray.Clear();
    this->sArray.Clear();
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
bool KeyframeManipulator::update(vislib::Array<KeyframeManipulator::manipType> am, vislib::Array<Keyframe>* kfa, Keyframe skf, vislib::math::Dimension<int, 2> vps,
                           vislib::math::Point<float, 3> wcp, vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> mvpm) {

    if (kfa == NULL) {
        vislib::sys::Log::DefaultLog.WriteError("[KeyframeManipulator] [Update] Pointer to keyframe array is NULL.");
        return false;
    }

    // Update ModelViewPorjectionMatrix ---------------------------------------
    // and position of world camera 
    bool recalcKf = false;
    if ((this->modelViewProjMatrix != mvpm) || (this->worldCamPos != wcp)) {
        this->calculateCircleVertices();
        recalcKf = true;
    }
    this->modelViewProjMatrix = mvpm;
    this->worldCamPos         = vislib::math::Vector<float, 3>(wcp.GetX(), wcp.GetY(), wcp.GetZ());

    // Update viewport --------------------------------------------------------
    this->viewportSize = vps;
    
    // Update slected keyframe ------------------------------------------------
    // Update manipulator only if selected keyframe changed
    if ((this->sKeyframe != skf) || recalcKf) {

        vislib::SmartPtr<vislib::graphics::CameraParameters> p = skf.getCamParameters();
        if (p.IsNull()) {
            throw vislib::Exception("[KEYFRAME MANIPULATOR] [update] Camera parameter pointer is NULL.", __FILE__, __LINE__);
        }

        vislib::graphics::Camera c;
        c.Parameters()->SetPosition(p->Position());
        c.Parameters()->SetLookAt(p->LookAt());
        c.Parameters()->SetUp(p->Up());
        c.Parameters()->SetApertureAngle(p->ApertureAngle());

        this->sKeyframe.setTime(skf.getTime());
        this->sKeyframe.setCamera(c);

        // Check if selected keyframe exists in keyframe array
        this->sKeyframeInArray = false;
        int selIndex = static_cast<int>(kfa->IndexOf(skf));
        if (selIndex >= 0) {
            this->sKeyframeInArray = true;
        }

        this->skfPos    = static_cast<vislib::math::Vector<float, 3> >(skf.getCamPosition());
        this->skfUp     = skf.getCamUp();
        this->skfLookAt = static_cast<vislib::math::Vector<float, 3> >(skf.getCamLookAt());

        this->updateSKfManipulators();
    }

    // Update  keyframe positions of array ------------------------------------
    unsigned int kfACnt     = static_cast<unsigned int>(kfa->Count());
    unsigned int kfArrayCnt = static_cast<unsigned int>(this->kfArray.Count());
    if ((kfACnt != kfArrayCnt) || recalcKf) {
        this->kfArray.Clear();
        this->kfArray.AssertCapacity(kfACnt);
        for (unsigned int i = 0; i < kfACnt; i++) {
            manipPosData tmpkfA;
            tmpkfA.wsPos = static_cast<vislib::math::Vector<float, 3>>((*kfa)[i].getCamPosition());
            tmpkfA.ssPos = this->getScreenSpace(tmpkfA.wsPos);
            tmpkfA.offset = (this->getScreenSpace(tmpkfA.wsPos + this->circleVertices[1]) - tmpkfA.ssPos).Norm(); // ...
            this->kfArray.Add(tmpkfA);
        }
    }
    else { // Update only different positions
        for (unsigned int i = 0; i < kfACnt; i++) {
            if (this->kfArray[i].wsPos != (*kfa)[i].getCamPosition()) { 
                this->kfArray[i].wsPos  = static_cast<vislib::math::Vector<float, 3> >((*kfa)[i].getCamPosition());
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
    for (unsigned int i = 0; i < static_cast<unsigned int>(this->sArray.Count()); i++) {
        this->sArray[i].available = false;
    }
    for (unsigned int i = 0; i < static_cast<unsigned int>(am.Count()); i++) {
        unsigned int index = static_cast<unsigned int>(am[i]);
        if (index < static_cast<unsigned int>(NUM_OF_SELECTED_MANIP)) {
            this->sArray[index].available = true;
        }
    }

    this->isDataSet = true;
    return true;
}



/*
* KeyframeManipulator::updateSKfManipulators
*/
bool KeyframeManipulator::updateSKfManipulators() {

    // Update screen space positions
    this->skfSsPos    = this->getScreenSpace(this->skfPos);
    this->skfSsLookAt = this->getScreenSpace(this->skfLookAt);

    manipPosData tmpkfS;
    if (this->sArray.IsEmpty()) {
        this->sArray.Clear();
        this->sArray.AssertCapacity(static_cast<unsigned int>(manipType::NUM_OF_SELECTED_MANIP));
        for (unsigned int i = 0; i < static_cast<unsigned int>(manipType::NUM_OF_SELECTED_MANIP); i++) {
            this->sArray.Add(tmpkfS);
        }
    }

    for (unsigned int i = 0; i < static_cast<unsigned int>(manipType::NUM_OF_SELECTED_MANIP); i++) { // skip SELECTED_KF_POS
        switch (static_cast<manipType>(i)) {
            case (manipType::SELECTED_KF_POS_X):      tmpkfS.wsPos = this->skfPos + vislib::math::Vector<float, 3>(1.0f, 0.0f, 0.0f); break;
            case (manipType::SELECTED_KF_POS_Y):      tmpkfS.wsPos = this->skfPos + vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f); break;
            case (manipType::SELECTED_KF_POS_Z):      tmpkfS.wsPos = this->skfPos + vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f); break;
            case (manipType::SELECTED_KF_LOOKAT_X):   tmpkfS.wsPos = this->skfLookAt + vislib::math::Vector<float, 3>(1.0f, 0.0f, 0.0f); break;
            case (manipType::SELECTED_KF_LOOKAT_Y):   tmpkfS.wsPos = this->skfLookAt + vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f); break;
            case (manipType::SELECTED_KF_LOOKAT_Z):   tmpkfS.wsPos = this->skfLookAt + vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f); break;
            case (manipType::SELECTED_KF_UP):         tmpkfS.wsPos = this->skfPos + this->skfUp; break;
            case (manipType::SELECTED_KF_POS_LOOKAT): tmpkfS.wsPos = this->skfPos - this->skfLookAt;
                                                      tmpkfS.wsPos.Normalise();
                                                      tmpkfS.wsPos += this->skfPos;
                                                      break;
            default: vislib::sys::Log::DefaultLog.WriteError("[KeyframeManipulator] [Update] Bug: %i", i); return false;
        }
        tmpkfS.ssPos     = this->getScreenSpace(tmpkfS.wsPos);
        tmpkfS.offset    = (this->getScreenSpace(tmpkfS.wsPos + this->circleVertices[1]) - tmpkfS.ssPos).Norm();
        this->sArray[i]  = tmpkfS;
    }

    return true;
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
                if (((pos.GetX() < (x + offset)) && (pos.GetX() > (x - offset))) &&
                    ((pos.GetY() < (y + offset)) && (pos.GetY() > (y - offset)))) {
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
    for (int i = 0; i < static_cast<int>(this->sArray.Count()); i++) {
        if (this->sArray[i].available) {
            float offset = this->sArray[i].offset;
            vislib::math::Vector<float, 2> pos = this->sArray[i].ssPos;
            // Check if mouse position lies within offset quad around keyframe position
            if (((pos.GetX() < (x + offset)) && (pos.GetX() > (x - offset))) &&
                ((pos.GetY() < (y + offset)) && (pos.GetY() > (y - offset)))) {
                this->activeType = static_cast<manipType>(i);
                this->lastMousePos.SetX(x);
                this->lastMousePos.SetY(y);
                break;
            }
        }
    }

    return true;
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

    vislib::math::Vector<float, 2> ssVec = this->sArray[index].ssPos - this->skfSsPos;
    if ((this->activeType == manipType::SELECTED_KF_LOOKAT_X) || 
        (this->activeType == manipType::SELECTED_KF_LOOKAT_Y) || 
        (this->activeType == manipType::SELECTED_KF_LOOKAT_Z)) {
        ssVec = this->sArray[index].ssPos - this->skfSsLookAt;
    }

    // Select manipulator axis with greatest contribution
    if (vislib::math::Abs(ssVec.GetX()) > vislib::math::Abs(ssVec.GetY())) {
        lineDiff = (x - this->lastMousePos.GetX()) * sensitivity;
        if (ssVec.GetX() < 0.0f) { // Adjust line changes depending on manipulator axis direction
            lineDiff *= -1.0f;
        }
    }
    else {
        lineDiff = (y - this->lastMousePos.GetY()) * this->sensitivity;
        if (ssVec.GetY() < 0.0f) { // Adjust line changes depending on manipulator axis direction
            lineDiff *= -1.0f;
        }
    }
    vislib::math::Vector<float, 3> tmpVec;
    switch (this->activeType) {
        case (manipType::SELECTED_KF_POS_X):      this->skfPos.SetX(this->skfPos.GetX() + lineDiff); break;
        case (manipType::SELECTED_KF_POS_Y):      this->skfPos.SetY(this->skfPos.GetY() + lineDiff); break;
        case (manipType::SELECTED_KF_POS_Z):      this->skfPos.SetZ(this->skfPos.GetZ() + lineDiff); break;
        case (manipType::SELECTED_KF_LOOKAT_X):   this->skfLookAt.SetX(this->skfLookAt.GetX() + lineDiff); break;
        case (manipType::SELECTED_KF_LOOKAT_Y):   this->skfLookAt.SetY(this->skfLookAt.GetY() + lineDiff); break;
        case (manipType::SELECTED_KF_LOOKAT_Z):   this->skfLookAt.SetZ(this->skfLookAt.GetZ() + lineDiff); break;
        case (manipType::SELECTED_KF_POS_LOOKAT): tmpVec = (this->skfPos - this->skfLookAt);
                                                  tmpVec.ScaleToLength(lineDiff);
                                                  this->skfPos += tmpVec;
                                                  break;
    }

    if (this->activeType == manipType::SELECTED_KF_UP) {
        bool cwRot = ((this->worldCamPos - this->skfLookAt).Norm() > (this->worldCamPos - this->skfPos).Norm());
        vislib::math::Vector<float, 3> tmpSsUp = vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f); // up vector for screen space
        if (!cwRot) {
            tmpSsUp.SetZ(-1.0f);
        }
        vislib::math::Vector<float, 2> tmpM          = this->lastMousePos - this->skfSsPos;
        vislib::math::Vector<float, 3> tmpSsMani     = vislib::math::Vector<float, 3>(tmpM.GetX(), tmpM.GetY(), 0.0f);
        tmpSsMani.Normalise();
        vislib::math::Vector<float, 3> tmpSsRight    = tmpSsMani.Cross(tmpSsUp);
        vislib::math::Vector<float, 3> tmpDeltaMouse = vislib::math::Vector<float, 3>(x - this->lastMousePos.GetX(), y - this->lastMousePos.GetY(), 0.0f);

        lineDiff = vislib::math::Abs(tmpDeltaMouse.Norm()) * this->sensitivity / 4.0f; // Adjust sensitivity of rotation here ...
        if (tmpSsRight.Dot(tmpSsMani + tmpDeltaMouse) < 0.0f) {
            lineDiff *= -1.0f;
        }
        // rotate up vector aroung lookat vector with the "Rodrigues' rotation formula"
        vislib::math::Vector<float, 3> k = (this->skfPos - this->skfLookAt); // => rotation axis = camera lookat
        this->skfUp = this->skfUp * cos(lineDiff) + k.Cross(this->skfUp) * sin(lineDiff) + k * (k.Dot(this->skfUp)) * (1.0f - cos(lineDiff));
    }

    // Update manipulators
    this->updateSKfManipulators();

    this->lastMousePos.SetX(x);
    this->lastMousePos.SetY(y);

    return true;
}


/*
* KeyframeManipulator::getManipulatedPos
*/
vislib::math::Point<float, 3> KeyframeManipulator::getManipulatedPos(void) {
    return vislib::math::Point<float, 3>(this->skfPos.GetX(), this->skfPos.GetY(), this->skfPos.GetZ());
}


/*
* KeyframeManipulator::getManipulatedUp
*/
vislib::math::Vector<float, 3> KeyframeManipulator::getManipulatedUp(void) {
    return this->skfUp;
}


/*
* KeyframeManipulator::getManipulatedLookAt
*/
vislib::math::Point<float, 3> KeyframeManipulator::getManipulatedLookAt(void) {
    return vislib::math::Point<float, 3>(this->skfLookAt.GetX(), this->skfLookAt.GetY(), this->skfLookAt.GetZ());
}


/*
* KeyframeManipulator::getScreenSpace
*
* Transform position from world space to screen space
*/
vislib::math::Vector<float, 2> KeyframeManipulator::getScreenSpace(vislib::math::Vector<float, 3> wp) {

    // World space position
    vislib::math::Vector<float, 4> wsPos = vislib::math::Vector<float, 4>(wp.GetX(), wp.GetY(), wp.GetZ(), 1.0f);
    // Screen space position
    vislib::math::Vector<float, 4> ssTmpPos = this->modelViewProjMatrix * wsPos;
    // Division by 'w'
    ssTmpPos = ssTmpPos / ssTmpPos.GetW();
    // Transform to viewport coordinates (x,y in [-1,1] -> viewport size)
    vislib::math::Vector<float, 2> ssPos;
    ssPos.SetX((ssTmpPos.GetX() + 1.0f) / 2.0f * this->viewportSize.GetWidth());
    ssPos.SetY(vislib::math::Abs(ssTmpPos.GetY() - 1.0f) / 2.0f * this->viewportSize.GetHeight()); // flip y-axis

    return ssPos;
}


/*
* KeyframeManipulator::initCircleVertices
*/
void KeyframeManipulator::calculateCircleVertices(void) {

    this->circleVertices.Clear();
    this->circleVertices.AssertCapacity(this->circleSubDiv);

    // Get normal for plane the cirlce lies on
    vislib::math::Vector<float, 3> normal = this->worldCamPos;
    normal.Normalise();
    // Get arbitary vector vertical to normal
    vislib::math::Vector<float, 3> rot = vislib::math::Vector<float, 3>(normal.GetZ(), 0.0f, -(normal.GetX()));
    rot.ScaleToLength(this->circleRadius);
    // rotate up vector aroung lookat vector with the "Rodrigues' rotation formula" 
    float t = 2.0f*(float)(CC_PI) / (float)(this->circleSubDiv); // theta angle for rotation   

    // First vertex is center of triangle fan
    this->circleVertices.Add(vislib::math::Vector<float, 3>(0.0f, 0.0f, 0.0f));
    for (unsigned int i = 0; i <= this->circleSubDiv; i++) {
        rot = rot * cos(t) + normal.Cross(rot) * sin(t) + normal * (normal.Dot(rot)) * (1.0f - cos(t));
        rot.ScaleToLength(this->circleRadius);
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

    GLfloat tmpLw;
    glGetFloatv(GL_LINE_WIDTH, &tmpLw);
    glLineWidth(2.0f);

    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);    
    glEnable(GL_LINE_SMOOTH);
    glDisable(GL_POLYGON_SMOOTH);

    // Get the foreground color (inverse background color)
    float bgColor[4];
    float fgColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    glGetFloatv(GL_COLOR_CLEAR_VALUE, bgColor);
    for (unsigned int i = 0; i < 4; i++) {
        fgColor[i] -= bgColor[i];
    }
    // COLORS
    float kColor[4] = { 0.7f, 0.7f, 1.0f, 1.0f }; // Color for KEYFRAME
    float skColor[4] = {0.1f, 0.1f, 1.0f, 1.0f }; // Color for SELECTED KEYFRAME
    float mlaColor[4] = {0.3f, 0.8f, 0.8f, 1.0f }; // Color for MANIPULATOR LOOKAT
    float muColor[4] = {0.8f, 0.0f, 0.8f, 1.0f }; // Color for MANIPULATOR UP
    float mxColor[4] = {0.8f, 0.1f, 0.0f, 1.0f }; // Color for MANIPULATOR X-AXIS
    float myColor[4] = {0.8f, 0.8f, 0.0f, 1.0f }; // Color for MANIPULATOR Y-AXIS
    float mzColor[4] = {0.1f, 0.8f, 0.0f, 1.0f }; // Color for MANIPULATOR Z-AXIS
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
                if (this->kfArray[k].wsPos == this->skfPos) {
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
    if (this->sKeyframeInArray) {
        for (unsigned int i = 0; i < static_cast<unsigned int>(this->sArray.Count()); i++) {
            if (this->sArray[i].available) {
                switch (static_cast<manipType>(i)) {
                case (manipType::SELECTED_KF_POS_X):      glColor4fv(mxColor);  this->drawManipulator(this->skfPos, this->sArray[i].wsPos);    break;
                case (manipType::SELECTED_KF_POS_Y):      glColor4fv(myColor);  this->drawManipulator(this->skfPos, this->sArray[i].wsPos);    break;
                case (manipType::SELECTED_KF_POS_Z):      glColor4fv(mzColor);  this->drawManipulator(this->skfPos, this->sArray[i].wsPos);    break;
                case (manipType::SELECTED_KF_UP):         glColor4fv(muColor);  this->drawManipulator(this->skfPos, this->sArray[i].wsPos);    break;
                case (manipType::SELECTED_KF_POS_LOOKAT): glColor4fv(mlaColor); this->drawManipulator(this->skfLookAt, this->sArray[i].wsPos); break;
                case (manipType::SELECTED_KF_LOOKAT_X):   glColor4fv(mxColor);  this->drawManipulator(this->skfLookAt, this->sArray[i].wsPos); break;
                case (manipType::SELECTED_KF_LOOKAT_Y):   glColor4fv(myColor);  this->drawManipulator(this->skfLookAt, this->sArray[i].wsPos); break;
                case (manipType::SELECTED_KF_LOOKAT_Z):   glColor4fv(mzColor);  this->drawManipulator(this->skfLookAt, this->sArray[i].wsPos); break;
                default: vislib::sys::Log::DefaultLog.WriteError("[KeyframeManipulator] [draw] Bug.");  break;
                }
            }
        }
    }
    else { // If selected keyframe is not in keyframe array just draw keyframe position, lookat line and up line
        glBegin(GL_LINES);
            // Up
            glColor4fv(muColor);
            glVertex3fv(this->skfPos.PeekComponents());
            glVertex3fv(this->sArray[static_cast<int>(manipType::SELECTED_KF_UP)].wsPos.PeekComponents());
            // LookAt
            glColor4fv(mlaColor);
            glVertex3fv(this->skfPos.PeekComponents());
            glVertex3fv(this->skfLookAt.PeekComponents());
        glEnd();
        // Keyframe position
        glColor4fv(skColor);
        this->drawCircle(this->skfPos, 0.75f);
    }

    // Reset opengl
    glLineWidth(tmpLw);
    glDisable(GL_BLEND);
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
