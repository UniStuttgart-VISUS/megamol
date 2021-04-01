/*
 * View2DGL.cpp
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/View2DGL.h"
#include "json.hpp"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/view/CallRender2DGL.h"
#include "mmcore/view/CallRenderViewGL.h"
#include "vislib/Trace.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/math/Matrix4.h"
#include "vislib/math/Rectangle.h"


using namespace megamol::core;


/*
 * view::View2DGL::View2DGL
 */
view::View2DGL::View2DGL(void)
        : view::AbstractView()
        , _ctrlDown(false)
        , _mouseMode(MouseMode::Propagate)
        , _mouseX(0.0f)
        , _mouseY(0.0f)
        , _viewUpdateCnt(0) {

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
        view::CallRenderViewGL::FunctionName(view::CallRenderViewGL::CALL_RESETVIEW), &AbstractView::OnResetView);
    this->MakeSlotAvailable(&this->_lhsRenderSlot);

    this->_rhsRenderSlot.SetCompatibleCall<CallRender2DGLDescription>();
    this->MakeSlotAvailable(&this->_rhsRenderSlot);
}


/*
 * view::View2DGL::~View2DGL
 */
view::View2DGL::~View2DGL(void) {
    this->Release();
}


/*
 * view::View2DGL::GetCameraSyncNumber
 */
unsigned int view::View2DGL::GetCameraSyncNumber(void) const {
    return this->_viewUpdateCnt;
}


/*
 * view::View2DGL::Render
 */
void view::View2DGL::Render(double time, double instanceTime, bool present_fbo) {

    AbstractView::beforeRender(time,instanceTime);

    CallRender2DGL* cr2d = this->_rhsRenderSlot.CallAs<CallRender2DGL>();

    if (cr2d == NULL) {
        return;
    }

    // clear fbo before sending it down the rendering call
    // the view is the owner of this fbo and therefore responsible
    // for clearing it at the beginning of a render frame
    this->_fbo->bind();
    auto bgcol = this->BkgndColour();
    glClearColor(bgcol.r, bgcol.g, bgcol.b, bgcol.a);
    glClearDepth(1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER,0);

    cr2d->SetFramebufferObject(_fbo);
    cr2d->SetCamera(_camera);

    (*cr2d)(AbstractCallRender::FnRender);

    // after render
    AbstractView::afterRender();

    if (present_fbo) {
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
}


/*
 * view::View2DGL::ResetView
 */
void view::View2DGL::ResetView(void) {
    if (_cameraIsMutable) { // check if view is in control of the camera
        CallRender2DGL* cr2d = this->_rhsRenderSlot.CallAs<CallRender2DGL>();
        if ((cr2d != nullptr) && (_fbo != nullptr) && ((*cr2d)(AbstractCallRender::FnGetExtents))) {
            Camera::OrthographicParameters cam_intrinsics;
            cam_intrinsics.near_plane = 0.1f;
            cam_intrinsics.far_plane = 100.0f;
            cam_intrinsics.frustrum_height = cr2d->GetBoundingBoxes().BoundingBox().Height();
            cam_intrinsics.aspect = static_cast<float>(_fbo->getWidth()) / static_cast<float>(_fbo->getHeight());
            cam_intrinsics.image_plane_tile =
                Camera::ImagePlaneTile(); // view is in control -> no tiling -> use default tile values

            if ((static_cast<float>(_fbo->getWidth()) / static_cast<float>(_fbo->getHeight())) <
                (static_cast<float>(cr2d->GetBoundingBoxes().BoundingBox().Width()) / cr2d->GetBoundingBoxes().BoundingBox().Height()))
            {
                cam_intrinsics.frustrum_height = cr2d->GetBoundingBoxes().BoundingBox().Width() / cam_intrinsics.aspect;
            }

            Camera::Pose cam_pose;
            cam_pose.position = glm::vec3(
                0.5f * (cr2d->GetBoundingBoxes().BoundingBox().Right() + cr2d->GetBoundingBoxes().BoundingBox().Left()),
                 0.5f * (cr2d->GetBoundingBoxes().BoundingBox().Top() + cr2d->GetBoundingBoxes().BoundingBox().Bottom()), 1.0f);
            cam_pose.direction = glm::vec3(0.0, 0.0, -1.0);
            cam_pose.up = glm::vec3(0.0, 1.0, 0.0);

            _camera = Camera(cam_pose, cam_intrinsics);

        } else {
            Camera::OrthographicParameters cam_intrinsics;
            cam_intrinsics.near_plane = 0.1f;
            cam_intrinsics.far_plane = 100.0f;
            cam_intrinsics.frustrum_height = 1.0f;
            cam_intrinsics.aspect = 1.0f;
            cam_intrinsics.image_plane_tile =
                Camera::ImagePlaneTile(); // view is in control -> no tiling -> use default tile values

            Camera::Pose cam_pose;
            cam_pose.position = glm::vec3(0.0f, 0.0f, 1.0f);
            cam_pose.direction = glm::vec3(0.0, 0.0, -1.0);
            cam_pose.up = glm::vec3(0.0, 1.0, 0.0);

            _camera = Camera(cam_pose, cam_intrinsics);
        }

        this->_viewUpdateCnt++;

    } else {
        // TODO print warning
    }
}


/*
 * view::View2DGL::Resize
 */
void view::View2DGL::Resize(unsigned int width, unsigned int height) {
    if ((this->_fbo->getWidth() != width) || (this->_fbo->getHeight() != height)) {

        glBindFramebuffer(GL_FRAMEBUFFER, 0); // better safe then sorry, "unbind" fbo before delting one
        try {
            _fbo = std::make_shared<glowl::FramebufferObject>(width, height);
            _fbo->createColorAttachment(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);

            // TODO: check completness and throw if not?
        } catch (glowl::FramebufferObjectException const& exc) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[View3DGL] Unable to create framebuffer object: %s\n", exc.what());
        }
    }
}


/*
 * view::View2DGL::OnRenderView
 */
bool view::View2DGL::OnRenderView(Call& call) {
    view::CallRenderViewGL* crv = dynamic_cast<view::CallRenderViewGL*>(&call);
    if (crv == NULL) {
        return false;
    }

    // get time from incoming call
    double time = crv->Time();
    if (time < 0.0f) time = this->DefaultTime(crv->InstanceTime());
    double instanceTime = crv->InstanceTime();

    auto fbo = _fbo;
    _fbo = crv->GetFramebufferObject();

    auto cam_cpy = _camera;
    auto cam_pose = _camera.get<Camera::Pose>();
    auto cam_intrinsics = _camera.get<Camera::OrthographicParameters>();
    cam_intrinsics.aspect = static_cast<float>(_fbo->getWidth()) / static_cast<float>(_fbo->getHeight());
    _camera = Camera(cam_pose, cam_intrinsics);

    this->Render(time, instanceTime, false);

    _fbo = fbo;
    _camera = cam_cpy;

    return true;
}


bool view::View2DGL::OnKey(Key key, KeyAction action, Modifiers mods) {
    auto* cr = this->_rhsRenderSlot.CallAs<view::CallRender2DGL>();
    if (cr == NULL)
        return false;

    if (key == Key::KEY_HOME) {
        OnResetView(this->_resetViewSlot);
    }
    _ctrlDown = mods.test(core::view::Modifier::CTRL);

    InputEvent evt;
    evt.tag = InputEvent::Tag::Key;
    evt.keyData.key = key;
    evt.keyData.action = action;
    evt.keyData.mods = mods;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRender2DGL::FnOnKey))
        return false;

    return true;
}


bool view::View2DGL::OnChar(unsigned int codePoint) {
    auto* cr = this->_rhsRenderSlot.CallAs<view::CallRender2DGL>();
    if (cr == NULL)
        return false;

    InputEvent evt;
    evt.tag = InputEvent::Tag::Char;
    evt.charData.codePoint = codePoint;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRender2DGL::FnOnChar))
        return false;

    return true;
}


bool view::View2DGL::OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) {
    this->_mouseMode = MouseMode::Propagate;

    auto* cr = this->_rhsRenderSlot.CallAs<view::CallRender2DGL>();
    if (cr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseButton;
        evt.mouseButtonData.button = button;
        evt.mouseButtonData.action = action;
        evt.mouseButtonData.mods = mods;
        cr->SetInputEvent(evt);
        if ((*cr)(view::CallRender2DGL::FnOnMouseButton))
            return true;
    }

    if (_ctrlDown) {
        auto down = action == MouseButtonAction::PRESS;
        if (button == MouseButton::BUTTON_LEFT && down) {
            this->_mouseMode = MouseMode::Pan;
        } else if (button == MouseButton::BUTTON_MIDDLE && down) {
            this->_mouseMode = MouseMode::Zoom;
        }
    } else {
        if (button == MouseButton::BUTTON_MIDDLE && action == MouseButtonAction::PRESS) {
            this->_mouseMode = MouseMode::Pan;
        }
    }
    
    return true;
}


bool view::View2DGL::OnMouseMove(double x, double y) {
    if (this->_mouseMode == MouseMode::Propagate) {
        auto* cr = this->_rhsRenderSlot.CallAs<view::CallRender2DGL>();
        if (cr) {
            InputEvent evt;
            evt.tag = InputEvent::Tag::MouseMove;
            evt.mouseMoveData.x = x;
            evt.mouseMoveData.y = y;
            cr->SetInputEvent(evt);
            if ((*cr)(view::CallRender2DGL::FnOnMouseMove))
                return true;
        }
    } else if (this->_mouseMode == MouseMode::Pan) {

        if (_cameraIsMutable) { // check if view is in control of the camera
            // compute size of a pixel in world space
            float stepSize = _camera.get<Camera::OrthographicParameters>().frustrum_height / _fbo->getHeight();
            auto dx = (this->_mouseX - x) * stepSize;
            auto dy = (this->_mouseY - y) * stepSize;

            auto cam_pose = _camera.get<Camera::Pose>();
            cam_pose.position += glm::vec3(dx,-dy,0.0f);

            _camera.setPose(cam_pose);

            if (dx > 0.0f || dy > 0.0f) {
                this->_viewUpdateCnt++;
            }
        }

    } else if (this->_mouseMode == MouseMode::Zoom) {

        if (_cameraIsMutable) {
            auto dy = (this->_mouseY - y);

            auto cam_pose = _camera.get<Camera::Pose>();
            auto cam_intrinsics = _camera.get<Camera::OrthographicParameters>();

            float bbox_height = cam_intrinsics.frustrum_height;
            CallRender2DGL* cr2d = this->_rhsRenderSlot.CallAs<CallRender2DGL>();
            if ((cr2d != NULL) && ((*cr2d)(AbstractCallRender::FnGetExtents))) {
                bbox_height = cr2d->GetBoundingBoxes().BoundingBox().Height();
            }
            cam_intrinsics.frustrum_height -= (dy / _fbo->getHeight()) * (cam_intrinsics.frustrum_height);

            _camera = Camera(cam_pose, cam_intrinsics);
        }
    }

    this->_mouseX = x;
    this->_mouseY = y;

    return true;
}


bool view::View2DGL::OnMouseScroll(double dx, double dy) {
    auto* cr = this->_rhsRenderSlot.CallAs<view::CallRender2DGL>();
    if (cr == NULL)
        return false;

    InputEvent evt;
    evt.tag = InputEvent::Tag::MouseScroll;
    evt.mouseScrollData.dx = dx;
    evt.mouseScrollData.dy = dy;
    cr->SetInputEvent(evt);
    if ((*cr)(view::CallRender2DGL::FnOnMouseScroll)) return true;

    if (_cameraIsMutable) {
        auto cam_pose = _camera.get<Camera::Pose>();
        auto cam_intrinsics = _camera.get<Camera::OrthographicParameters>();
        float bbox_height = cam_intrinsics.frustrum_height;
        CallRender2DGL* cr2d = this->_rhsRenderSlot.CallAs<CallRender2DGL>();
        if ((cr2d != NULL) && ((*cr2d)(AbstractCallRender::FnGetExtents))) {
            bbox_height = cr2d->GetBoundingBoxes().BoundingBox().Height();
        }
        cam_intrinsics.frustrum_height -= (dy/10.0) * (cam_intrinsics.frustrum_height);

        _camera = Camera(cam_pose, cam_intrinsics);
    }

    return true;
}


/*
 * view::View2DGL::create
 */
bool view::View2DGL::create(void) {

    this->_firstImg = true;

    // intialize fbo with dummy size until the actual size is set during first call to Resize
    this->_fbo = std::make_shared<glowl::FramebufferObject>(1,1);

    return true;
}


/*
 * view::View2DGL::release
 */
void view::View2DGL::release(void) {
    // intentionally empty
}


/*
 * view::View2DGL::GetExtents
 */
bool view::View2DGL::GetExtents(Call& call) {
    view::CallRenderViewGL* crv = dynamic_cast<view::CallRenderViewGL*>(&call);
    if (crv == nullptr)
        return false;

    CallRender2DGL* cr2d = this->_rhsRenderSlot.CallAs<CallRender2DGL>();
    if (cr2d == nullptr) {
        return false;
    }
    cr2d->SetCamera(this->_camera);

    if (!(*cr2d)(CallRender2DGL::FnGetExtents))
        return false;

    crv->SetTimeFramesCount(cr2d->TimeFramesCount());
    crv->SetIsInSituTime(cr2d->IsInSituTime());
    return true;
}
