#include "stdafx.h"
#include "View3DSpaceMouse.h"
#include "vislib/sys/Log.h"
#include "vislib/math/Cuboid.h"
#include "param/FloatParam.h"
#include "param/BoolParam.h"
#include "param/EnumParam.h"

using namespace megamol::core;
using namespace megamol::protein;

/*
 * View3DSpaceMouse::View3DSpaceMouse
 */
View3DSpaceMouse::View3DSpaceMouse(void) : view::View3D(),
    translateSpeed3DSlot("translateSpeed3D", "Integer parameter controlling 3D mouse translation speed scalar"),
    rotateSpeed3DSlot("rotateSpeed3D", "Integer parameter controlling 3D mouse rotation speed scalar"),
    translateToggleSlot("translateToggle", "True to allow 3d mouse to translate view"),
    rotateToggleSlot("rotateToggle", "True to allow 3d mouse to rotate view"),
    singleAxisToggleSlot("singleAxisToggle", "True to cause 3d mouse to only use largest translation/rotation value"),
    cameraControlModeSlot("cameraControlMode", "Sets the way in whcih to space mouse manipulates the camera")
{

    this->translateSpeed3DSlot << new param::FloatParam(10.0f, 1.0f, 100.0f);
    this->translateSpeed3DSlot.SetUpdateCallback(&View3DSpaceMouse::update3DSpeed);
    this->MakeSlotAvailable(&this->translateSpeed3DSlot);

    this->rotateSpeed3DSlot << new param::FloatParam(10.0f, 1.0f, 100.0f);
    this->rotateSpeed3DSlot.SetUpdateCallback(&View3DSpaceMouse::update3DSpeed);
    this->MakeSlotAvailable(&this->rotateSpeed3DSlot);

    this->translateToggleSlot << new param::BoolParam(true);
    this->translateToggleSlot.SetUpdateCallback(&View3DSpaceMouse::updateModes);
    this->MakeSlotAvailable(&this->translateToggleSlot);

    this->rotateToggleSlot << new param::BoolParam(true);
    this->rotateToggleSlot.SetUpdateCallback(&View3DSpaceMouse::updateModes);
    this->MakeSlotAvailable(&this->rotateToggleSlot);

    this->singleAxisToggleSlot << new param::BoolParam(false);
    this->singleAxisToggleSlot.SetUpdateCallback(&View3DSpaceMouse::updateModes);
    this->MakeSlotAvailable(&this->singleAxisToggleSlot);

    param::EnumParam *cm = new param::EnumParam(int(vislib::graphics::CameraAdjust3D::CAMERA_MODE));
    cm->SetTypePair( vislib::graphics::CameraAdjust3D::CAMERA_MODE, "Camera Mode");
    cm->SetTypePair( vislib::graphics::CameraAdjust3D::OBJECT_MODE, "Object Mode");
    cm->SetTypePair( vislib::graphics::CameraAdjust3D::TARGET_CAMERA_MODE, "Target Camera Mode");
    this->cameraControlModeSlot << cm;
    this->cameraControlModeSlot.SetUpdateCallback(&View3DSpaceMouse::updateCameraModes);
    this->MakeSlotAvailable( &this->cameraControlModeSlot);
}

/*
 * View3DSpaceMouse::~View3DSpaceMouse
 */
View3DSpaceMouse::~View3DSpaceMouse(void) {
    this->Release();
}

/*
 * View3DSpaceMouse::ResetView
 */
void View3DSpaceMouse::ResetView(void) {
    view::View3D::ResetView(); // parent function call

    this->adjustor.SetObjectCenter(); // set object center as lookat point (which is also the center of the bbox)

}

/*
 * View3DSpaceMouse::On3dMouseMotion
 */
void View3DSpaceMouse::On3DMouseMotion(float tx, float ty, float tz, float rx, float ry, float rz) {
    /* This value taken from the 3dConnexion SDK */   
    // object angular motion per mouse tick 0.008 milliradians per count
    //const float angularMotion = 8.0e-6f; // radians per count

    /* Need to switch to the OpenGL coordinate system (X right, Y up, Z out of screen)
     * from the mouse system (X right, Y out of screen, Z down)
     */
    float temp;
    temp = ty;
    ty = tz * -1.0f;
    tz = temp;
    temp = ry;
    ry = rz * -1.0f;
    rz = temp;

    // set modifiers
    /* TODO: FIX: Can't make this call outside of a glut callback function (none of which I'm using b/c they don't support the 3d mouse) */
    //int modifiers = glutGetModifiers();
    //this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_SHIFT, (modifiers & GLUT_ACTIVE_SHIFT) == GLUT_ACTIVE_SHIFT);
    //this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_CTRL, (modifiers & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL);
    //this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_ALT, (modifiers & GLUT_ACTIVE_ALT) == GLUT_ACTIVE_ALT);

    // Perform default motion procedure
    this->relativeCursor3d.Motion(tx, ty, tz, rx, ry, rz);

    // Check the lookAt Point motion toggle
    //if (!this->lookAtMotion) {
    //    // Perform default motion procedure
    //    this->relCursor3d.Motion(static_cast<vislib::graphics::SceneSpaceType>(tranX), 
    //        static_cast<vislib::graphics::SceneSpaceType>(tranY), 
    //        static_cast<vislib::graphics::SceneSpaceType>(tranZ),
    //        static_cast<float>(rotX), 
    //        static_cast<float>(rotY), 
    //        static_cast<float>(rotZ));
    //} else {
    //    // move the lookAt point
    //    vislib::graphics::SceneSpacePoint3D lookAt = this->camera.Parameters()->LookAt();
    //    vislib::graphics::SceneSpaceVector3D translate;
    //    // don't translate past the current position when bringing the lookAt point toward the camera
    //    vislib::graphics::SceneSpaceVector3D lookVector;
    //    lookVector = this->camera.Parameters()->Position() - 
    //        this->camera.Parameters()->LookAt();
    //    float focalDist = lookVector.Length();
    //    /* scale the X and Y translations by the focal distance (otherwise nearby
    //     * motion will be very fast, and far away motion will be very slow
    //     */
    //    tranX *= focalDist;
    //    tranY *= focalDist;
    //    if ((focalDist - 0.01f) < tranZ) {
    //        tranZ = focalDist - 0.01f;
    //    }
    //    // don't translate further out than the far clipping plane
    //    if ((focalDist + (-1.0f * tranZ)) > (this->camera.Parameters()->FarClip() - 1.0f)) {
    //        tranZ = -1.0f * (this->camera.Parameters()->FarClip() - 1.0f - focalDist);
    //    }
    //    // convert translate vector into camera coordinates
    //    translate = tranX * this->camera.Parameters()->Right() +
    //        tranY * this->camera.Parameters()->Up() +
    //        -1.0f * tranZ * this->camera.Parameters()->Front();
    //    // apply the translation
    //    lookAt += translate;
    //    this->camera.Parameters()->SetLookAt(lookAt);
    //}
}


/*
 * View3DSpaceMouse::On3dMouseButton
 */
void View3DSpaceMouse::On3DMouseButton(unsigned long keyState) {
    unsigned int btn = 0;

    // set modifiers
    /* TODO: FIX: Can't make this call outside of a glut callback function (none of which I'm using b/c they don't support the 3d mouse) */
    //int modifiers = glutGetModifiers();
    //this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_SHIFT, (modifiers & GLUT_ACTIVE_SHIFT) == GLUT_ACTIVE_SHIFT);
    //this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_CTRL, (modifiers & GLUT_ACTIVE_CTRL) == GLUT_ACTIVE_CTRL);
    //this->modkeys.SetModifierState(vislib::graphics::InputModifiers::MODIFIER_ALT, (modifiers & GLUT_ACTIVE_ALT) == GLUT_ACTIVE_ALT);

    // Mask through the keyState unsigned long
    for (btn = 0; btn < this->relativeCursor3d.GetButtonCount(); btn += 1) {
        bool storedBtnState = this->relativeCursor3d.GetButtonState(btn); // get the current state of the button in question
        unsigned long currentBtnState = keyState & 0x0001; // mask for the least significant bit only
        // compare the values of the current stored button state and the current given button state
        if ((currentBtnState == 1) != storedBtnState) {
            // different values means the button has changed - if the current is true it was pressed, otherwise it was released
            if (currentBtnState) {
                // button was pressed
                this->relativeCursor3d.SetButtonState(btn, true); // for cursor events 
                //if (btn == 1) {
                //    // reset view on right mouse button click
                //    this->ResetView();
                //}
            } else {
                // button was released
                this->relativeCursor3d.SetButtonState(btn, false); // for cursor events
                //this->ButtonEvent(btn, false); // custom events
            }
        }
        keyState = keyState >> 1; // right shift the keyState to apply the mask to the next button
    }
    
    if (keyState > 0) {
        // if the keyState is still larger than 0, there are buttons that weren't checked but that exist
        // this likely means an incorrect button count was set (i.e. the 3d mouse used has more than 2 buttons)
        vislib::sys::Log::DefaultLog.WriteMsg( vislib::sys::Log::LEVEL_ERROR,
            "Button press not recorded because button number is larger than button count.");
    }
}

/*
 * View3DSpaceMouse::update3DSpeed
 */
bool View3DSpaceMouse::update3DSpeed(param::ParamSlot& p) {
#ifdef _WIN32
    this->rawInput.SetTranslationSpeed(this->translateSpeed3DSlot.Param<param::FloatParam>()->Value());
    this->rawInput.SetRotationSpeed(this->rotateSpeed3DSlot.Param<param::FloatParam>()->Value());
    return true;
#endif // _WIN32
}



/*
 * View3DSpaceMouse::updateModes
 */
bool View3DSpaceMouse::updateModes(param::ParamSlot& p) {
    this->adjustor.SetNoTranslationMode(!this->translateToggleSlot.Param<param::BoolParam>()->Value());
    this->adjustor.SetNoRotationMode(!this->rotateToggleSlot.Param<param::BoolParam>()->Value());
    this->adjustor.SetSingleAxisMode(this->singleAxisToggleSlot.Param<param::BoolParam>()->Value());
    return true;
}

/*
 * View3DSpaceMouse::updateCameraModes
 */
bool View3DSpaceMouse::updateCameraModes(param::ParamSlot& p) {
    if ( this->cameraControlModeSlot.Param<param::EnumParam>()->Value() ==
        vislib::graphics::CameraAdjust3D::CAMERA_MODE) {

        this->adjustor.SetCameraControlMode();
    } else if (this->cameraControlModeSlot.Param<param::EnumParam>()->Value() ==
        vislib::graphics::CameraAdjust3D::OBJECT_MODE) {

        this->adjustor.SetObjectControlMode();
        // Make sure the object center is set as the center of the bbox.
        this->adjustor.SetObjectCenter( this->bboxs.WorldSpaceBBox().CalcCenter());
    } else {
        this->adjustor.SetTargetCameraControlMode();
        // Make sure the object center is set as the center of the bbox.
        this->adjustor.SetObjectCenter( this->bboxs.WorldSpaceBBox().CalcCenter());
    }
    return true;
}


/*
 * View3DSpaceMouse::create
 */
bool View3DSpaceMouse::create(void) {

    view::View3D::create();

    /* Set up 3d cursor */
    
    this->adjustor.SetCameraParams(this->camParams);
    this->adjustor.SetModifierTestCount(3);
    this->adjustor.SetModifierTestCount(2);
    this->adjustor.SetModifierTest(0,
        vislib::graphics::InputModifiers::MODIFIER_CTRL, false);
    this->adjustor.SetModifierTest(1,
        vislib::graphics::InputModifiers::MODIFIER_ALT, false);
    
#ifdef _WIN32 
    this->rawInput.Initialize(); // create a raw input class to get input from 3d mouse
    // set callback functions for motion and button events from the raw input device
    this->rawInput.SetMotionFunction(Raw3DMotionDelegate(*this, &View3DSpaceMouse::On3DMouseMotion));
    this->rawInput.SetButtonFunction(Raw3DButtonDelegate(*this, &View3DSpaceMouse::On3DMouseButton));
    // Set a speed scalar for both rotation and translation
    this->rawInput.SetTranslationSpeed(9); /* These speeds should be configurable */
    this->rawInput.SetRotationSpeed(9);
#endif

    // set up the relative 3D cursor for use with the SpaceNavigator
    this->relativeCursor3d.SetButtonCount(2); /* This could be configurable. */

    this->relativeCursor3d.SetCameraParams(this->camParams);
    this->relativeCursor3d.RegisterCursorEvent(&this->adjustor);
    this->relativeCursor3d.SetInputModifiers(&this->modkeys);

    this->modkeys.RegisterObserver(&this->relativeCursor3d);

    return true;
}

/*
 * View3DSpaceMouse::release
 */
void View3DSpaceMouse::release(void) {
    view::View3D::release();
    
    this->relativeCursor3d.UnregisterCursorEvent(&this->adjustor);
}
