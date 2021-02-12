/*
 * View3DGL.cpp
 *
 * Copyright (C) 2018, 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/View3DGL.h"


#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/view/CallRender3DGL.h"
#include "mmcore/view/CallRenderViewGL.h"

using namespace megamol::core;
using namespace megamol::core::view;

/*
 * View3DGL::View3DGL
 */
View3DGL::View3DGL(void) : view::AbstractView3D(), toggleMouseSelection(false), cursor2d() {
    this->rendererSlot.SetCompatibleCall<CallRender3DGLDescription>();
    this->MakeSlotAvailable(&this->rendererSlot);
    // Override renderSlot behavior
    this->renderSlot.SetCallback(
        view::CallRenderViewGL::ClassName(), InputCall::FunctionName(InputCall::FnOnKey), &AbstractView::OnKeyCallback);
    this->renderSlot.SetCallback(
        view::CallRenderViewGL::ClassName(), InputCall::FunctionName(InputCall::FnOnChar), &AbstractView::OnCharCallback);
    this->renderSlot.SetCallback(view::CallRenderViewGL::ClassName(), InputCall::FunctionName(InputCall::FnOnMouseButton),
        &AbstractView::OnMouseButtonCallback);
    this->renderSlot.SetCallback(view::CallRenderViewGL::ClassName(), InputCall::FunctionName(InputCall::FnOnMouseMove),
        &AbstractView::OnMouseMoveCallback);
    this->renderSlot.SetCallback(view::CallRenderViewGL::ClassName(), InputCall::FunctionName(InputCall::FnOnMouseScroll),
        &AbstractView::OnMouseScrollCallback);
    // AbstractCallRender
    this->renderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        AbstractCallRender::FunctionName(AbstractCallRender::FnRender), &AbstractView::OnRenderView);
    this->renderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        AbstractCallRender::FunctionName(AbstractCallRender::FnGetExtents), &AbstractView::GetExtents);
    // CallRenderViewGL
    this->renderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        view::CallRenderViewGL::FunctionName(view::CallRenderViewGL::CALL_FREEZE), &AbstractView::OnFreezeView);
    this->renderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        view::CallRenderViewGL::FunctionName(view::CallRenderViewGL::CALL_UNFREEZE), &AbstractView::OnUnfreezeView);
    this->renderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        view::CallRenderViewGL::FunctionName(view::CallRenderViewGL::CALL_RESETVIEW), &AbstractView::onResetView);
    this->MakeSlotAvailable(&this->renderSlot);

}

/*
 * View3DGL::~View3DGL
 */
View3DGL::~View3DGL(void) {
    this->Release();
}


/*
 * View3DGL::Render
 */
void View3DGL::Render(const mmcRenderViewContext& context) {

    CallRender3DGL* cr3d = this->rendererSlot.CallAs<CallRender3DGL>();
    this->handleCameraMovement();

    if (cr3d == NULL) {
        return;
    }
    cr3d->SetFramebufferObject(this->fbo);

    AbstractView3D::beforeRender(context);

    if (cr3d != nullptr) {
        cr3d->SetCameraState(this->cam);
        (*cr3d)(view::CallRender3DGL::FnRender);
    }

    AbstractView3D::afterRender(context);
}

/*
 * View3DGL::OnKey
 */
bool view::View3DGL::OnKey(view::Key key, view::KeyAction action, view::Modifiers mods) {
    auto* cr = this->rendererSlot.CallAs<CallRender3DGL>();
    if (cr != nullptr) {
        view::InputEvent evt;
        evt.tag = view::InputEvent::Tag::Key;
        evt.keyData.key = key;
        evt.keyData.action = action;
        evt.keyData.mods = mods;
        cr->SetInputEvent(evt);
        if ((*cr)(CallRender3DGL::FnOnKey)) return true;
    }

    if (action == view::KeyAction::PRESS || action == view::KeyAction::REPEAT) {
        this->pressedKeyMap[key] = true;
    } else if (action == view::KeyAction::RELEASE) {
        this->pressedKeyMap[key] = false;
    }

    if (key == view::Key::KEY_LEFT_ALT || key == view::Key::KEY_RIGHT_ALT) {
        if (action == view::KeyAction::PRESS || action == view::KeyAction::REPEAT) {
            this->modkeys.set(view::Modifier::ALT);
            cameraControlOverrideActive = true;
        } else if (action == view::KeyAction::RELEASE) {
            this->modkeys.reset(view::Modifier::ALT);
            cameraControlOverrideActive = false;
        }
    }
    if (key == view::Key::KEY_LEFT_SHIFT || key == view::Key::KEY_RIGHT_SHIFT) {
        if (action == view::KeyAction::PRESS || action == view::KeyAction::REPEAT) {
            this->modkeys.set(view::Modifier::SHIFT);
        } else if (action == view::KeyAction::RELEASE) {
            this->modkeys.reset(view::Modifier::SHIFT);
        }
    }
    if (key == view::Key::KEY_LEFT_CONTROL || key == view::Key::KEY_RIGHT_CONTROL) {
        if (action == view::KeyAction::PRESS || action == view::KeyAction::REPEAT) {
            this->modkeys.set(view::Modifier::CTRL);
            cameraControlOverrideActive = true;
        } else if (action == view::KeyAction::RELEASE) {
            this->modkeys.reset(view::Modifier::CTRL);
            cameraControlOverrideActive = false;
        }
    }

    if (action == view::KeyAction::PRESS && (key >= view::Key::KEY_0 && key <= view::Key::KEY_9)) {
        int index =
            static_cast<int>(key) - static_cast<int>(view::Key::KEY_0); // ugly hack, maybe this can be done better
        index = (index - 1) % 10;                                       // put key '1' at index 0
        index = index < 0 ? index + 10 : index;                         // wrap key '0' to a positive index '9'

        if (mods.test(view::Modifier::CTRL)) {
            this->savedCameras[index].first = this->cam.get_minimal_state(this->savedCameras[index].first);
            this->savedCameras[index].second = true;
            if (this->autoSaveCamSettingsSlot.Param<param::BoolParam>()->Value()) {
                this->onStoreCamera(this->storeCameraSettingsSlot); // manually trigger the storing
            }
        } else {
            if (this->savedCameras[index].second) {
                // As a change of camera position should not change the display resolution, we actively save and restore
                // the old value of the resolution
                auto oldResolution = this->cam.resolution_gate; // save old resolution
                this->cam = this->savedCameras[index].first;    // override current camera
                this->cam.resolution_gate = oldResolution;      // restore old resolution
            }
        }
    }

    return false;
}

/*
 * View3DGL::OnChar
 */
bool view::View3DGL::OnChar(unsigned int codePoint) {
    auto* cr = this->rendererSlot.CallAs<view::CallRender3DGL>();
    if (cr == NULL) return false;

    view::InputEvent evt;
    evt.tag = view::InputEvent::Tag::Char;
    evt.charData.codePoint = codePoint;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRender3DGL::FnOnChar)) return false;

    return true;
}

/*
 * View3DGL::OnMouseButton
 */
bool view::View3DGL::OnMouseButton(view::MouseButton button, view::MouseButtonAction action, view::Modifiers mods) {

    bool anyManipulatorActive = arcballManipulator.manipulating() || translateManipulator.manipulating() ||
                                rotateManipulator.manipulating() || turntableManipulator.manipulating() ||
                                orbitAltitudeManipulator.manipulating();

    if (!cameraControlOverrideActive && !anyManipulatorActive) {
        auto* cr = this->rendererSlot.CallAs<CallRender3DGL>();
        if (cr != nullptr) {
            view::InputEvent evt;
            evt.tag = view::InputEvent::Tag::MouseButton;
            evt.mouseButtonData.button = button;
            evt.mouseButtonData.action = action;
            evt.mouseButtonData.mods = mods;
            cr->SetInputEvent(evt);
            if ((*cr)(CallRender3DGL::FnOnMouseButton)) return true;
        }
    }

    if (action == view::MouseButtonAction::PRESS) {
        this->pressedMouseMap[button] = true;
    } else if (action == view::MouseButtonAction::RELEASE) {
        this->pressedMouseMap[button] = false;
    }

    // This mouse handling/mapping is so utterly weird and should die!
    auto down = action == view::MouseButtonAction::PRESS;
    bool altPressed = mods.test(view::Modifier::ALT); // this->modkeys.test(view::Modifier::ALT);
    bool ctrlPressed = mods.test(view::Modifier::CTRL); // this->modkeys.test(view::Modifier::CTRL);

    // get window resolution to help computing mouse coordinates
    auto wndSize = this->cam.resolution_gate();

    if (!this->toggleMouseSelection) {
        switch (button) {
        case megamol::core::view::MouseButton::BUTTON_LEFT:
            this->cursor2d.SetButtonState(0, down);

            if (!anyManipulatorActive) {
                if (altPressed ^
                    (this->arcballDefault &&
                        !ctrlPressed)) // Left mouse press + alt/arcDefault+noCtrl -> activate arcball manipluator
                {
                    this->arcballManipulator.setActive(
                        wndSize.width() - static_cast<int>(this->mouseX), static_cast<int>(this->mouseY));
                } else if (ctrlPressed) // Left mouse press + Ctrl -> activate orbital manipluator
                {
                    this->turntableManipulator.setActive(
                        wndSize.width() - static_cast<int>(this->mouseX), static_cast<int>(this->mouseY));
                }
            }

            break;
        case megamol::core::view::MouseButton::BUTTON_RIGHT:
            this->cursor2d.SetButtonState(1, down);

            if (!anyManipulatorActive) {
                if ((altPressed ^ this->arcballDefault) || ctrlPressed) {
                    this->orbitAltitudeManipulator.setActive(
                        wndSize.width() - static_cast<int>(this->mouseX), static_cast<int>(this->mouseY));
                } else {
                    this->rotateManipulator.setActive();
                    this->translateManipulator.setActive(
                        wndSize.width() - static_cast<int>(this->mouseX), static_cast<int>(this->mouseY));
                }
            }

            break;
        case megamol::core::view::MouseButton::BUTTON_MIDDLE:
            this->cursor2d.SetButtonState(2, down);

            if (!anyManipulatorActive) {
                this->translateManipulator.setActive(
                    wndSize.width() - static_cast<int>(this->mouseX), static_cast<int>(this->mouseY));
            }

            break;
        default:
            break;
        }


        if (action == view::MouseButtonAction::RELEASE) // Mouse release + no other mouse button pressed ->
                                                        // deactivate all mouse manipulators
        {
            if (!(this->cursor2d.GetButtonState(0) || this->cursor2d.GetButtonState(1) ||
                    this->cursor2d.GetButtonState(2))) {
                this->arcballManipulator.setInactive();
                this->orbitAltitudeManipulator.setInactive();
                this->rotateManipulator.setInactive();
                this->turntableManipulator.setInactive();
                this->translateManipulator.setInactive();
            }
        }
    }
    return true;
}

/*
 * View3DGL::OnMouseMove
 */
bool view::View3DGL::OnMouseMove(double x, double y) {
    this->mouseX = (float)static_cast<int>(x);
    this->mouseY = (float)static_cast<int>(y);

    bool anyManipulatorActive = arcballManipulator.manipulating() || translateManipulator.manipulating() ||
                                rotateManipulator.manipulating() || turntableManipulator.manipulating() ||
                                orbitAltitudeManipulator.manipulating();

    if (!anyManipulatorActive) {
        auto* cr = this->rendererSlot.CallAs<CallRender3DGL>();
        if (cr != nullptr) {
            view::InputEvent evt;
            evt.tag = view::InputEvent::Tag::MouseMove;
            evt.mouseMoveData.x = x;
            evt.mouseMoveData.y = y;
            cr->SetInputEvent(evt);
            if ((*cr)(CallRender3DGL::FnOnMouseMove)) return true;
        }
    }

    // This mouse handling/mapping is so utterly weird and should die!
    if (!this->toggleMouseSelection) {

        auto wndSize = this->cam.resolution_gate();

        this->cursor2d.SetPosition(x, y, true, wndSize.height());

        glm::vec3 newPos;

        if (this->turntableManipulator.manipulating()) {
            this->turntableManipulator.on_drag(wndSize.width() - static_cast<int>(this->mouseX),
                static_cast<int>(this->mouseY), glm::vec4(rotCenter, 1.0));
        }

        if (this->arcballManipulator.manipulating()) {
            this->arcballManipulator.on_drag(wndSize.width() - static_cast<int>(this->mouseX),
                static_cast<int>(this->mouseY), glm::vec4(rotCenter, 1.0));
        }

        if (this->orbitAltitudeManipulator.manipulating()) {
            this->orbitAltitudeManipulator.on_drag(wndSize.width() - static_cast<int>(this->mouseX),
                static_cast<int>(this->mouseY), glm::vec4(rotCenter, 1.0));
        }

        if (this->translateManipulator.manipulating() && !this->rotateManipulator.manipulating() ) {

            // compute proper step size by computing pixel world size at distance to rotCenter
            glm::vec3 currCamPos(static_cast<glm::vec4>(this->cam.position()));
            float orbitalAltitude = glm::length(currCamPos - rotCenter);
            auto fovy = cam.half_aperture_angle_radians();
            auto vertical_height = 2.0f * std::tan(fovy) * orbitalAltitude;
            auto pixel_world_size = vertical_height / wndSize.height();

            this->translateManipulator.set_step_size(pixel_world_size);

            this->translateManipulator.move_horizontally(wndSize.width() - static_cast<int>(this->mouseX));
            this->translateManipulator.move_vertically(static_cast<int>(this->mouseY));
        }
    }

    return true;
}

/*
 * View3DGL::OnMouseScroll
 */
bool view::View3DGL::OnMouseScroll(double dx, double dy) {
    auto* cr = this->rendererSlot.CallAs<view::CallRender3DGL>();
    if (cr != NULL) {
        view::InputEvent evt;
        evt.tag = view::InputEvent::Tag::MouseScroll;
        evt.mouseScrollData.dx = dx;
        evt.mouseScrollData.dy = dy;
        cr->SetInputEvent(evt);
        if ((*cr)(view::CallRender3DGL::FnOnMouseScroll)) return true;
    }


    // This mouse handling/mapping is so utterly weird and should die!
    if (!this->toggleMouseSelection && (abs(dy) > 0.0)) {
        if (this->rotateManipulator.manipulating()) {
            this->viewKeyMoveStepSlot.Param<param::FloatParam>()->SetValue(
                this->viewKeyMoveStepSlot.Param<param::FloatParam>()->Value() + 
                (dy * 0.1f * this->viewKeyMoveStepSlot.Param<param::FloatParam>()->Value())
            ); 
        } else {
            auto cam_pos = this->cam.eye_position();
            auto rot_cntr = thecam::math::point<glm::vec4>(glm::vec4(this->rotCenter, 0.0f));

            cam_pos.w() = 0.0f;

            auto v = thecam::math::normalise(rot_cntr - cam_pos);

            auto altitude = thecam::math::length(rot_cntr - cam_pos);

            this->cam.position(cam_pos + (v * dy * (altitude / 50.0f)));
        }
    }

    return true;
}


/*
 * View3DGL::create
 */
bool View3DGL::create(void) {

    AbstractView3D::create();

    this->fbo = std::make_shared<vislib::graphics::gl::FramebufferObject>();

    this->cursor2d.SetButtonCount(3);

    return true;
}

void View3DGL::release() {
    AbstractView3D::release();
}
