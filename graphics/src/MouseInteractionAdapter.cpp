/*
 * MouseInteractionAdapter.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/MouseInteractionAdapter.h"

#include "vislib/IllegalParamException.h"
#include "vislib/memutils.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::graphics::MouseInteractionAdapter::MouseInteractionAdapter
 */
vislib::graphics::MouseInteractionAdapter::MouseInteractionAdapter(
        const SmartPtr<CameraParameters>& params, const unsigned int cntButtons) 
        : rotator(NULL), zoomer(NULL) {

    /* Configure the available modifiers, Ctrl, Shift and Alt. */
    this->modifiers.SetModifierCount(3);
    this->modifiers.RegisterObserver(&this->cursor);

    /* Configure the cursor as a mouse. */
    this->cursor.SetButtonCount((cntButtons < 1) ? 1 : cntButtons);
    this->cursor.SetInputModifiers(&this->modifiers);
 
    /* Configure default controller. */
    this->SetCamera(params);    // Must be before Configure* calls!
    this->ConfigureRotation();
    this->ConfigureZoom();
}


/*
 * vislib::graphics::MouseInteractionAdapter::~MouseInteractionAdapter
 */
vislib::graphics::MouseInteractionAdapter::~MouseInteractionAdapter(void) {
    SAFE_DELETE(this->rotator);
    SAFE_DELETE(this->zoomer);
}


/*
 * vislib::graphics::MouseInteractionAdapter::ConfigureRotation
 */
void vislib::graphics::MouseInteractionAdapter::ConfigureRotation(
        const RotationType type, const Button button, 
        const InputModifiers::Modifier altModifier) {
    ASSERT(!this->cursor.CameraParams().IsNull());

    /* Clean up old stuff. */
    if (this->rotator != NULL) {
        this->cursor.UnregisterCursorEvent(this->rotator);
        SAFE_DELETE(this->rotator);
    }
    ASSERT(this->rotator == NULL);

    /* Create and configure new rotation. */
    switch (type) {
        case ROTATION_FREE: {
            CameraRotate2D *r = new CameraRotate2D(this->cursor.CameraParams());
            r->SetAltModifier(altModifier);
            this->rotator = r;
            } break;

        case ROTATION_LOOKAT: {
            CameraRotate2DLookAt *r = new CameraRotate2DLookAt(
                this->cursor.CameraParams());
            r->SetAltModifier(altModifier);
            this->rotator = r;
            } break;

        default:
            throw IllegalParamException("type", __FILE__, __LINE__);
            break;
    }
    ASSERT(this->rotator != NULL);

    this->rotator->SetTestButton(static_cast<const unsigned int>(button));
    this->rotator->SetModifierTestCount(0); // TODO: Do not know what this does.
    
    this->cursor.RegisterCursorEvent(dynamic_cast<AbstractCursor2DEvent *>(
        this->rotator));
}


/*
 * vislib::graphics::MouseInteractionAdapter::ConfigureZoom
 */
void vislib::graphics::MouseInteractionAdapter::ConfigureZoom(
        const ZoomType type, const Button button, const SceneSpaceType speed,
        const CameraZoom2DMove::ZoomBehaviourType behaviour) {
    ASSERT(!this->cursor.CameraParams().IsNull());

    /* Clean up old stuff. */
    if (this->zoomer != NULL) {
        this->cursor.UnregisterCursorEvent(this->zoomer);
        SAFE_DELETE(this->zoomer);
    }
    ASSERT(this->zoomer == NULL);

    /* Create and configure new zoom. */
    switch (type) {
        case ZOOM_ANGLE:
            this->zoomer = new CameraZoom2DAngle(this->cursor.CameraParams());
            break;

        case ZOOM_MOVE: {
            CameraZoom2DMove *z = new CameraZoom2DMove(
                this->cursor.CameraParams());
            z->SetZoomBehaviour(behaviour);
            z->SetSpeed(speed);
            this->zoomer = z;
            } break;

        default:
            throw IllegalParamException("type", __FILE__, __LINE__);
            break;
    }

    this->zoomer->SetTestButton(static_cast<const unsigned int>(button));
    this->zoomer->SetModifierTestCount(0);    // TODO: Do not know what this does.

    this->cursor.RegisterCursorEvent(dynamic_cast<AbstractCursor2DEvent *>(
        this->zoomer));
}


/*
 * vislib::graphics::MouseInteractionAdapter::GetCamera
 */
vislib::SmartPtr<vislib::graphics::CameraParameters> 
vislib::graphics::MouseInteractionAdapter::GetCamera(void) {
    return this->cursor.CameraParams();
}

/*
 * vislib::graphics::MouseInteractionAdapter::SetCamera
 */
void vislib::graphics::MouseInteractionAdapter::SetCamera(
        const SmartPtr<CameraParameters>& params) {
    AbstractCameraController *ctrl = NULL;

    this->cursor.SetCameraParams(params);

    if ((ctrl = this->getRotateCtrl()) != NULL) {
        ctrl->SetCameraParams(params);
    }

    if ((ctrl = this->getZoomCtrl()) != NULL) {
        ctrl->SetCameraParams(params);
    }
}


/*
 * vislib::graphics::MouseInteractionAdapter::MouseInteractionAdapter
 */
vislib::graphics::MouseInteractionAdapter::MouseInteractionAdapter(
        const MouseInteractionAdapter& rhs) {
    throw UnsupportedOperationException("MouseInteractionAdapter", __FILE__, 
        __LINE__);
}


/*
 * vislib::graphics::MouseInteractionAdapter::operator =
 */
vislib::graphics::MouseInteractionAdapter& 
vislib::graphics::MouseInteractionAdapter::operator =(
        const MouseInteractionAdapter& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
