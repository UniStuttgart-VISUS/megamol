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
View3DGL::View3DGL(void) : view::AbstractView3D(), _cursor2d() {
    this->_rhsRenderSlot.SetCompatibleCall<CallRender3DGLDescription>();
    this->MakeSlotAvailable(&this->_rhsRenderSlot);
    // Override renderSlot behavior
    this->_lhsRenderSlot.SetCallback(
        view::CallRenderViewGL::ClassName(), InputCall::FunctionName(InputCall::FnOnKey), &AbstractView::OnKeyCallback);
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(), InputCall::FunctionName(InputCall::FnOnChar),
        &AbstractView::OnCharCallback);
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        InputCall::FunctionName(InputCall::FnOnMouseButton), &AbstractView::OnMouseButtonCallback);
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        InputCall::FunctionName(InputCall::FnOnMouseMove), &AbstractView::OnMouseMoveCallback);
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        InputCall::FunctionName(InputCall::FnOnMouseScroll), &AbstractView::OnMouseScrollCallback);
    // AbstractCallRender
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        AbstractCallRender::FunctionName(AbstractCallRender::FnRender), &AbstractView::OnRenderView);
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        AbstractCallRender::FunctionName(AbstractCallRender::FnGetExtents), &AbstractView::GetExtents);
    // CallRenderViewGL
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        view::CallRenderViewGL::FunctionName(view::CallRenderViewGL::CALL_FREEZE), &AbstractView::OnFreezeView);
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        view::CallRenderViewGL::FunctionName(view::CallRenderViewGL::CALL_UNFREEZE), &AbstractView::OnUnfreezeView);
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        view::CallRenderViewGL::FunctionName(view::CallRenderViewGL::CALL_RESETVIEW), &AbstractView::OnResetView);
    this->MakeSlotAvailable(&this->_lhsRenderSlot);

    this->_rhsRenderSlot.SetNecessity(megamol::core::AbstractCallSlotPresentation::SLOT_REQUIRED);
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
void View3DGL::Render(const mmcRenderViewContext& context, Call* call) {

    CallRender3DGL* cr3d = this->_rhsRenderSlot.CallAs<CallRender3DGL>();

    if (cr3d == NULL) {
        return;
    }

    AbstractView3D::beforeRender(context);

    auto current_frame_fbo = _fbo;
    auto bgcol = this->BkgndColour();

    if (call == nullptr) {
        bool tgt_res_ok = (_camera.image_tile().width() != 0) && (_camera.image_tile().height() != 0);
        bool fbo_update_needed = (_fbo->GetWidth() != _camera.image_tile().width()) ||
                                 (_fbo->GetHeight() != _camera.image_tile().height()) || (!_fbo->IsValid());

        std::pair<int, int> tgt_res =
            tgt_res_ok ? std::make_pair<int, int>(_camera.image_tile().width(), _camera.image_tile().height())
                       : std::make_pair<int, int>(1, 1);

        if (fbo_update_needed) {
            _fbo->Release();
            if (!_fbo->Create(tgt_res.first, tgt_res.second, GL_RGBA8, GL_RGBA,
                    GL_UNSIGNED_BYTE, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE,
                    GL_DEPTH_COMPONENT)) {
                throw vislib::Exception("[View3DGL] Unable to create image framebuffer object.", __FILE__, __LINE__);
                return;
            }
        }
    } else {
        auto gpu_call = dynamic_cast<view::CallRenderViewGL*>(call);
        current_frame_fbo = gpu_call->GetFramebufferObject();
        bgcol = gpu_call->BackgroundColor();
    }

    current_frame_fbo->Enable();
    glClearColor(bgcol.r, bgcol.g, bgcol.b, bgcol.a);
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, current_frame_fbo->GetWidth(), current_frame_fbo->GetHeight());

    cr3d->SetFramebufferObject(current_frame_fbo);
    cr3d->SetCamera(this->_camera);

    (*cr3d)(view::CallRender3DGL::FnRender);

    current_frame_fbo->Disable();
    if (call == nullptr) {
        current_frame_fbo->DrawColourTexture(); // TODO replace me
    }

    AbstractView3D::afterRender(context);
}

/*
 * View3DGL::OnKey
 */
bool view::View3DGL::OnKey(view::Key key, view::KeyAction action, view::Modifiers mods) {
    auto* cr = this->_rhsRenderSlot.CallAs<CallRender3DGL>();
    if (cr != nullptr) {
        view::InputEvent evt;
        evt.tag = view::InputEvent::Tag::Key;
        evt.keyData.key = key;
        evt.keyData.action = action;
        evt.keyData.mods = mods;
        cr->SetInputEvent(evt);
        if ((*cr)(CallRender3DGL::FnOnKey))
            return true;
    }

    if (action == view::KeyAction::PRESS || action == view::KeyAction::REPEAT) {
        this->_pressedKeyMap[key] = true;
    } else if (action == view::KeyAction::RELEASE) {
        this->_pressedKeyMap[key] = false;
    }

    if (key == view::Key::KEY_LEFT_ALT || key == view::Key::KEY_RIGHT_ALT) {
        if (action == view::KeyAction::PRESS || action == view::KeyAction::REPEAT) {
            this->modkeys.set(view::Modifier::ALT);
            _cameraControlOverrideActive = true;
        } else if (action == view::KeyAction::RELEASE) {
            this->modkeys.reset(view::Modifier::ALT);
            _cameraControlOverrideActive = false;
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
            _cameraControlOverrideActive = true;
        } else if (action == view::KeyAction::RELEASE) {
            this->modkeys.reset(view::Modifier::CTRL);
            _cameraControlOverrideActive = false;
        }
    }

    if (action == view::KeyAction::PRESS && (key >= view::Key::KEY_0 && key <= view::Key::KEY_9)) {
        int index =
            static_cast<int>(key) - static_cast<int>(view::Key::KEY_0); // ugly hack, maybe this can be done better
        index = (index - 1) % 10;                                       // put key '1' at index 0
        index = index < 0 ? index + 10 : index;                         // wrap key '0' to a positive index '9'

        if (mods.test(view::Modifier::CTRL)) {
            this->_savedCameras[index].first = this->_camera.get_minimal_state(this->_savedCameras[index].first);
            this->_savedCameras[index].second = true;
            if (this->_autoSaveCamSettingsSlot.Param<param::BoolParam>()->Value()) {
                this->onStoreCamera(this->_storeCameraSettingsSlot); // manually trigger the storing
            }
        } else {
            if (this->_savedCameras[index].second) {
                // As a change of camera position should not change the display resolution, we actively save and restore
                // the old value of the resolution
                auto oldResolution = this->_camera.resolution_gate; // save old resolution
                this->_camera = this->_savedCameras[index].first;   // override current camera
                this->_camera.resolution_gate = oldResolution;      // restore old resolution
            }
        }
    }

    return false;
}

/*
 * View3DGL::OnChar
 */
bool view::View3DGL::OnChar(unsigned int codePoint) {
    auto* cr = this->_rhsRenderSlot.CallAs<view::CallRender3DGL>();
    if (cr == NULL)
        return false;

    view::InputEvent evt;
    evt.tag = view::InputEvent::Tag::Char;
    evt.charData.codePoint = codePoint;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRender3DGL::FnOnChar))
        return false;

    return true;
}

/*
 * View3DGL::OnMouseButton
 */
bool view::View3DGL::OnMouseButton(view::MouseButton button, view::MouseButtonAction action, view::Modifiers mods) {

    bool anyManipulatorActive = _arcballManipulator.manipulating() || _translateManipulator.manipulating() ||
                                _rotateManipulator.manipulating() || _turntableManipulator.manipulating() ||
                                _orbitAltitudeManipulator.manipulating();

    if (!_cameraControlOverrideActive && !anyManipulatorActive) {
        auto* cr = this->_rhsRenderSlot.CallAs<CallRender3DGL>();
        if (cr != nullptr) {
            view::InputEvent evt;
            evt.tag = view::InputEvent::Tag::MouseButton;
            evt.mouseButtonData.button = button;
            evt.mouseButtonData.action = action;
            evt.mouseButtonData.mods = mods;
            cr->SetInputEvent(evt);
            if ((*cr)(CallRender3DGL::FnOnMouseButton))
                return true;
        }
    }

    if (action == view::MouseButtonAction::PRESS) {
        this->_pressedMouseMap[button] = true;
    } else if (action == view::MouseButtonAction::RELEASE) {
        this->_pressedMouseMap[button] = false;
    }

    // This mouse handling/mapping is so utterly weird and should die!
    auto down = action == view::MouseButtonAction::PRESS;
    bool altPressed = mods.test(view::Modifier::ALT);   // this->modkeys.test(view::Modifier::ALT);
    bool ctrlPressed = mods.test(view::Modifier::CTRL); // this->modkeys.test(view::Modifier::CTRL);

    // get window resolution to help computing mouse coordinates
    auto wndSize = this->_camera.resolution_gate();


    switch (button) {
    case megamol::core::view::MouseButton::BUTTON_LEFT:
        this->_cursor2d.SetButtonState(0, down);

        if (!anyManipulatorActive) {
            if (altPressed ^
                (this->_arcballDefault &&
                    !ctrlPressed)) // Left mouse press + alt/arcDefault+noCtrl -> activate arcball manipluator
            {
                this->_arcballManipulator.setActive(
                    wndSize.width() - static_cast<int>(this->_mouseX), static_cast<int>(this->_mouseY));
            } else if (ctrlPressed) // Left mouse press + Ctrl -> activate orbital manipluator
            {
                this->_turntableManipulator.setActive(
                    wndSize.width() - static_cast<int>(this->_mouseX), static_cast<int>(this->_mouseY));
            }
        }

        break;
    case megamol::core::view::MouseButton::BUTTON_RIGHT:
        this->_cursor2d.SetButtonState(1, down);

        if (!anyManipulatorActive) {
            if ((altPressed ^ this->_arcballDefault) || ctrlPressed) {
                this->_orbitAltitudeManipulator.setActive(
                    wndSize.width() - static_cast<int>(this->_mouseX), static_cast<int>(this->_mouseY));
            } else {
                this->_rotateManipulator.setActive();
                this->_translateManipulator.setActive(
                    wndSize.width() - static_cast<int>(this->_mouseX), static_cast<int>(this->_mouseY));
            }
        }

        break;
    case megamol::core::view::MouseButton::BUTTON_MIDDLE:
        this->_cursor2d.SetButtonState(2, down);

        if (!anyManipulatorActive) {
            this->_translateManipulator.setActive(
                wndSize.width() - static_cast<int>(this->_mouseX), static_cast<int>(this->_mouseY));
        }

        break;
    default:
        break;
    }


    if (action == view::MouseButtonAction::RELEASE) // Mouse release + no other mouse button pressed ->
                                                    // deactivate all mouse manipulators
    {
        if (!(this->_cursor2d.GetButtonState(0) || this->_cursor2d.GetButtonState(1) ||
                this->_cursor2d.GetButtonState(2))) {
            this->_arcballManipulator.setInactive();
            this->_orbitAltitudeManipulator.setInactive();
            this->_rotateManipulator.setInactive();
            this->_turntableManipulator.setInactive();
            this->_translateManipulator.setInactive();
        }
    }
    return true;
}

/*
 * View3DGL::OnMouseMove
 */
bool view::View3DGL::OnMouseMove(double x, double y) {
    this->_mouseX = (float) static_cast<int>(x);
    this->_mouseY = (float) static_cast<int>(y);

    bool anyManipulatorActive = _arcballManipulator.manipulating() || _translateManipulator.manipulating() ||
                                _rotateManipulator.manipulating() || _turntableManipulator.manipulating() ||
                                _orbitAltitudeManipulator.manipulating();

    if (!anyManipulatorActive) {
        auto* cr = this->_rhsRenderSlot.CallAs<CallRender3DGL>();
        if (cr != nullptr) {
            view::InputEvent evt;
            evt.tag = view::InputEvent::Tag::MouseMove;
            evt.mouseMoveData.x = x;
            evt.mouseMoveData.y = y;
            cr->SetInputEvent(evt);
            if ((*cr)(CallRender3DGL::FnOnMouseMove))
                return true;
        }
    }

    auto wndSize = this->_camera.resolution_gate();

    this->_cursor2d.SetPosition(x, y, true, wndSize.height());

    glm::vec3 newPos;

    if (this->_turntableManipulator.manipulating()) {
        this->_turntableManipulator.on_drag(wndSize.width() - static_cast<int>(this->_mouseX),
            static_cast<int>(this->_mouseY), glm::vec4(_rotCenter, 1.0));
    }

    if (this->_arcballManipulator.manipulating()) {
        this->_arcballManipulator.on_drag(wndSize.width() - static_cast<int>(this->_mouseX),
            static_cast<int>(this->_mouseY), glm::vec4(_rotCenter, 1.0));
    }

    if (this->_orbitAltitudeManipulator.manipulating()) {
        this->_orbitAltitudeManipulator.on_drag(wndSize.width() - static_cast<int>(this->_mouseX),
            static_cast<int>(this->_mouseY), glm::vec4(_rotCenter, 1.0));
    }

    if (this->_translateManipulator.manipulating() && !this->_rotateManipulator.manipulating()) {

        // compute proper step size by computing pixel world size at distance to rotCenter
        glm::vec3 currCamPos(static_cast<glm::vec4>(this->_camera.position()));
        float orbitalAltitude = glm::length(currCamPos - _rotCenter);
        auto fovy = _camera.half_aperture_angle_radians();
        auto vertical_height = 2.0f * std::tan(fovy) * orbitalAltitude;
        auto pixel_world_size = vertical_height / wndSize.height();

        this->_translateManipulator.set_step_size(pixel_world_size);

        this->_translateManipulator.move_horizontally(wndSize.width() - static_cast<int>(this->_mouseX));
        this->_translateManipulator.move_vertically(static_cast<int>(this->_mouseY));
    }


    return true;
}

/*
 * View3DGL::OnMouseScroll
 */
bool view::View3DGL::OnMouseScroll(double dx, double dy) {
    auto* cr = this->_rhsRenderSlot.CallAs<view::CallRender3DGL>();
    if (cr != NULL) {
        view::InputEvent evt;
        evt.tag = view::InputEvent::Tag::MouseScroll;
        evt.mouseScrollData.dx = dx;
        evt.mouseScrollData.dy = dy;
        cr->SetInputEvent(evt);
        if ((*cr)(view::CallRender3DGL::FnOnMouseScroll))
            return true;
    }


    // This mouse handling/mapping is so utterly weird and should die!
    if ((abs(dy) > 0.0)) {
        if (this->_rotateManipulator.manipulating()) {
            this->_viewKeyMoveStepSlot.Param<param::FloatParam>()->SetValue(
                this->_viewKeyMoveStepSlot.Param<param::FloatParam>()->Value() +
                (dy * 0.1f * this->_viewKeyMoveStepSlot.Param<param::FloatParam>()->Value()));
        } else {
            auto cam_pos = this->_camera.eye_position();
            auto rot_cntr = thecam::math::point<glm::vec4>(glm::vec4(this->_rotCenter, 0.0f));

            cam_pos.w() = 0.0f;

            auto v = thecam::math::normalise(rot_cntr - cam_pos);

            auto altitude = thecam::math::length(rot_cntr - cam_pos);

            this->_camera.position(cam_pos + (v * dy * (altitude / 50.0f)));
        }
    }

    return true;
}


/*
 * View3DGL::create
 */
bool View3DGL::create(void) {

    AbstractView3D::create();

    this->_fbo = std::make_shared<vislib::graphics::gl::FramebufferObject>();

    this->_cursor2d.SetButtonCount(3);

    return true;
}

void View3DGL::release() {
    AbstractView3D::release();
}
