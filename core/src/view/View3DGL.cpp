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
    this->_lhsRenderSlot.SetCallback(
        view::CallRenderViewGL::ClassName(), InputCall::FunctionName(InputCall::FnOnChar), &AbstractView::OnCharCallback);
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(), InputCall::FunctionName(InputCall::FnOnMouseButton),
        &AbstractView::OnMouseButtonCallback);
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(), InputCall::FunctionName(InputCall::FnOnMouseMove),
        &AbstractView::OnMouseMoveCallback);
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(), InputCall::FunctionName(InputCall::FnOnMouseScroll),
        &AbstractView::OnMouseScrollCallback);
    // AbstractCallRender
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        AbstractCallRender::FunctionName(AbstractCallRender::FnRender), &AbstractView::OnRenderView);
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        AbstractCallRender::FunctionName(AbstractCallRender::FnGetExtents), &AbstractView::GetExtents);
    // CallRenderViewGL
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        view::CallRenderViewGL::FunctionName(view::CallRenderViewGL::CALL_RESETVIEW), &AbstractView::onResetView);
    this->MakeSlotAvailable(&this->_lhsRenderSlot);
}

/*
 * View3DGL::~View3DGL
 */
View3DGL::~View3DGL(void) {
    this->Release();
}

void megamol::core::view::View3DGL::Render(double time, double instanceTime) {
    CallRender3DGL* cr3d = this->_rhsRenderSlot.CallAs<CallRender3DGL>();

    if (cr3d == NULL) {
        return;
    }

    AbstractView3D::beforeRender(time, instanceTime);

    // clear fbo before sending it down the rendering call
    // the view is the owner of this fbo and therefore responsible
    // for clearing it at the beginning of a render frame
    _fbo->bind();
    auto bgcol = this->BkgndColour();
    glClearColor(bgcol.r, bgcol.g, bgcol.b, bgcol.a);
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // set camera and fbo in rendering call
    cr3d->SetFramebufferObject(_fbo);
    cr3d->SetCamera(this->_camera);

    // call the rendering call
    (*cr3d)(view::CallRender3DGL::FnRender);
    
    AbstractView3D::afterRender();

    // Blit the final image to the default framebuffer of the window.
    // Technically, the view's fbo should always match the size of the window so a blit is fine.
    // Eventually, presenting the fbo will become the frontends job.
    // Bind and blit framebuffer.
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    _fbo->bindToRead(0);
    glBlitFramebuffer(0, 0, _fbo->getWidth(), _fbo->getHeight(), 0, 0, _fbo->getWidth(), _fbo->getHeight(),
        GL_COLOR_BUFFER_BIT, GL_NEAREST);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}

void megamol::core::view::View3DGL::ResetView() {
    AbstractView3D::ResetView(static_cast<float>(_fbo->getWidth())/static_cast<float>(_fbo->getHeight()));
}

void megamol::core::view::View3DGL::Resize(unsigned int width, unsigned int height) {
    if ( (_fbo->getWidth() != width) || (_fbo->getHeight() != height) ) {

        glBindFramebuffer(GL_FRAMEBUFFER, 0); // better safe then sorry, "unbind" fbo before delting one
        try {
            _fbo = std::make_shared<glowl::FramebufferObject>(width, height);
            _fbo->createColorAttachment(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);

            //TODO: check completness and throw if not?
        }
        catch (glowl::FramebufferObjectException const& exc) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[View3DGL] Unable to create framebuffer object: %s\n", exc.what());
        }

        if (_cameraIsMutable) { // view seems to be in control of the camera
            auto cam_pose = _camera.get<Camera::Pose>();
            if (_camera.get<Camera::ProjectionType>() == Camera::ProjectionType::PERSPECTIVE) {
                auto cam_intrinsics = _camera.get<Camera::PerspectiveParameters>();
                cam_intrinsics.aspect = static_cast<float>(width) / static_cast<float>(height);
                _camera = Camera(cam_pose, cam_intrinsics);
            } else if (_camera.get<Camera::ProjectionType>() == Camera::ProjectionType::ORTHOGRAPHIC) {
                auto cam_intrinsics = _camera.get<Camera::OrthographicParameters>();
                cam_intrinsics.aspect = static_cast<float>(width) / static_cast<float>(height);
                _camera = Camera(cam_pose, cam_intrinsics);
            }
        }
    }
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
        if ((*cr)(CallRender3DGL::FnOnKey)) return true;
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
            this->_savedCameras[index].first = this->_camera;
            this->_savedCameras[index].second = true;
            if (this->_autoSaveCamSettingsSlot.Param<param::BoolParam>()->Value()) {
                this->onStoreCamera(this->_storeCameraSettingsSlot); // manually trigger the storing
            }
        } else {
            if (this->_savedCameras[index].second) {
                // As a change of camera position should not change the display resolution, we actively save and restore
                // the old value of the resolution
                this->_camera = this->_savedCameras[index].first;    // override current camera
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
            if ((*cr)(CallRender3DGL::FnOnMouseButton)) return true;
        }
    }

    if (action == view::MouseButtonAction::PRESS) {
        this->_pressedMouseMap[button] = true;
    } else if (action == view::MouseButtonAction::RELEASE) {
        this->_pressedMouseMap[button] = false;
    }

    // This mouse handling/mapping is so utterly weird and should die!
    auto down = action == view::MouseButtonAction::PRESS;
    bool altPressed = mods.test(view::Modifier::ALT); // this->modkeys.test(view::Modifier::ALT);
    bool ctrlPressed = mods.test(view::Modifier::CTRL); // this->modkeys.test(view::Modifier::CTRL);

    // get window resolution to help computing mouse coordinates
    int wndWidth;
    int wndHeight;
    auto projType = _camera.get<Camera::ProjectionType>();
    if (projType == Camera::ProjectionType::PERSPECTIVE) {
        auto tile_end = _camera.get<Camera::PerspectiveParameters>().image_plane_tile.tile_end;
        auto tile_start = _camera.get<Camera::PerspectiveParameters>().image_plane_tile.tile_start;
        auto tile_size = tile_end - tile_start;

        wndWidth = static_cast<int>(static_cast<float>(_fbo->getWidth()) / tile_size.x);
        wndHeight = static_cast<int>(static_cast<float>(_fbo->getHeight()) / tile_size.y);
    } else if (projType == Camera::ProjectionType::ORTHOGRAPHIC) {
        auto tile_end = _camera.get<Camera::OrthographicParameters>().image_plane_tile.tile_end;
        auto tile_start = _camera.get<Camera::OrthographicParameters>().image_plane_tile.tile_start;
        auto tile_size = tile_end - tile_start;

        wndWidth = static_cast<int>(static_cast<float>(_fbo->getWidth()) / tile_size.x);
        wndHeight = static_cast<int>(static_cast<float>(_fbo->getHeight()) / tile_size.y);
    } else {
        return false; // Oh bother...
    }

    switch (button) {
    case megamol::core::view::MouseButton::BUTTON_LEFT:
        this->_cursor2d.SetButtonState(0, down);

        if (!anyManipulatorActive) {
            if (altPressed ^
                (this->_arcballDefault &&
                    !ctrlPressed)) // Left mouse press + alt/arcDefault+noCtrl -> activate arcball manipluator
            {
                this->_arcballManipulator.setActive(
                    wndWidth - static_cast<int>(this->_mouseX), static_cast<int>(this->_mouseY));
            } else if (ctrlPressed) // Left mouse press + Ctrl -> activate orbital manipluator
            {
                this->_turntableManipulator.setActive(
                    wndWidth - static_cast<int>(this->_mouseX), static_cast<int>(this->_mouseY));
            }
        }

        break;
    case megamol::core::view::MouseButton::BUTTON_RIGHT:
        this->_cursor2d.SetButtonState(1, down);

        if (!anyManipulatorActive) {
            if ((altPressed ^ this->_arcballDefault) || ctrlPressed) {
                this->_orbitAltitudeManipulator.setActive(
                    wndWidth - static_cast<int>(this->_mouseX), static_cast<int>(this->_mouseY));
            } else {
                this->_rotateManipulator.setActive();
                this->_translateManipulator.setActive(
                    wndWidth - static_cast<int>(this->_mouseX), static_cast<int>(this->_mouseY));
            }
        }

        break;
    case megamol::core::view::MouseButton::BUTTON_MIDDLE:
        this->_cursor2d.SetButtonState(2, down);

        if (!anyManipulatorActive) {
            this->_translateManipulator.setActive(
                wndWidth - static_cast<int>(this->_mouseX), static_cast<int>(this->_mouseY));
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
    this->_mouseX = (float)static_cast<int>(x);
    this->_mouseY = (float)static_cast<int>(y);

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
            if ((*cr)(CallRender3DGL::FnOnMouseMove)) return true;
        }
    }

    // get window resolution to help computing mouse coordinates
    int wndWidth;
    int wndHeight;
    auto projType = _camera.get<Camera::ProjectionType>();
    if (projType == Camera::ProjectionType::PERSPECTIVE) {
        auto tile_end = _camera.get<Camera::PerspectiveParameters>().image_plane_tile.tile_end;
        auto tile_start = _camera.get<Camera::PerspectiveParameters>().image_plane_tile.tile_start;
        auto tile_size = tile_end - tile_start;

        wndWidth = static_cast<int>(static_cast<float>(_fbo->getWidth()) / tile_size.x);
        wndHeight = static_cast<int>(static_cast<float>(_fbo->getHeight()) / tile_size.y);
    } else if (projType == Camera::ProjectionType::ORTHOGRAPHIC) {
        auto tile_end = _camera.get<Camera::OrthographicParameters>().image_plane_tile.tile_end;
        auto tile_start = _camera.get<Camera::OrthographicParameters>().image_plane_tile.tile_start;
        auto tile_size = tile_end - tile_start;

        wndWidth = static_cast<int>(static_cast<float>(_fbo->getWidth()) / tile_size.x);
        wndHeight = static_cast<int>(static_cast<float>(_fbo->getHeight()) / tile_size.y);
    } else {
        return false; // Oh bother...
    }

    this->_cursor2d.SetPosition(x, y, true, wndHeight);

    glm::vec3 newPos;

    if (this->_turntableManipulator.manipulating()) {
        this->_turntableManipulator.on_drag(
            wndWidth - static_cast<int>(this->_mouseX),
            static_cast<int>(this->_mouseY),
            glm::vec4(_rotCenter, 1.0),
            wndWidth,
            wndHeight
        );
    }

    if (this->_arcballManipulator.manipulating()) {
        this->_arcballManipulator.on_drag(wndWidth - static_cast<int>(this->_mouseX),
            static_cast<int>(this->_mouseY), glm::vec4(_rotCenter, 1.0));
    }

    if (this->_orbitAltitudeManipulator.manipulating()) {
        this->_orbitAltitudeManipulator.on_drag(wndWidth - static_cast<int>(this->_mouseX),
            static_cast<int>(this->_mouseY), glm::vec4(_rotCenter, 1.0));
    }

    if (this->_translateManipulator.manipulating() && !this->_rotateManipulator.manipulating() ) {

        // compute proper step size by computing pixel world size at distance to rotCenter
        glm::vec3 currCamPos = this->_camera.get<Camera::Pose>().position;
        float orbitalAltitude = glm::length(currCamPos - _rotCenter);
        float pixel_world_size;
        if (projType == Camera::ProjectionType::PERSPECTIVE) {
            auto fovy = _camera.get<Camera::PerspectiveParameters>().fovy;
            auto vertical_height = std::tan(fovy) * orbitalAltitude;
            pixel_world_size = vertical_height / static_cast<float>(wndHeight);
        } else if (projType == Camera::ProjectionType::ORTHOGRAPHIC) {
            auto vertical_height = _camera.get<Camera::OrthographicParameters>().frustrum_height;
            pixel_world_size = vertical_height / static_cast<float>(wndHeight);
        } else {
            return false; // Oh bother...
        }

        this->_translateManipulator.set_step_size(pixel_world_size);

        this->_translateManipulator.move_horizontally(wndWidth - static_cast<int>(this->_mouseX));
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
        if ((*cr)(view::CallRender3DGL::FnOnMouseScroll)) return true;
    }


    // This mouse handling/mapping is so utterly weird and should die!
    if ((abs(dy) > 0.0)) {
        if (this->_rotateManipulator.manipulating()) {
            this->_viewKeyMoveStepSlot.Param<param::FloatParam>()->SetValue(
                this->_viewKeyMoveStepSlot.Param<param::FloatParam>()->Value() + 
                (dy * 0.1f * this->_viewKeyMoveStepSlot.Param<param::FloatParam>()->Value())
            ); 
        } else {
            auto cam_pose = _camera.get<Camera::Pose>();
            auto v = glm::normalize(_rotCenter - cam_pose.position);
            auto altitude = thecam::math::length(_rotCenter - cam_pose.position);
            cam_pose.position = cam_pose.position + (v * static_cast<float>(dy) * (altitude / 50.0f));
            _camera.setPose(cam_pose);
        }
    }

    return true;
}


/*
 * View3DGL::create
 */
bool View3DGL::create(void) {

    AbstractView3D::create();

    // intialize fbo with dummy size until the actual size is set during first call to Resize
    this->_fbo = std::make_shared<glowl::FramebufferObject>(1,1);

    this->_cursor2d.SetButtonCount(3);

    return true;
}

void View3DGL::release() {
    AbstractView3D::release();
}
