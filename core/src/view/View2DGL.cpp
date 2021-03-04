/*
 * View2DGL.cpp
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/view/View2DGL.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/view/CallRenderViewGL.h"
#include "mmcore/view/CallRender2DGL.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"
#include "vislib/Trace.h"
#include "vislib/math/Matrix4.h"
#include "json.hpp"
#include "mmcore/utility/log/Log.h"
#include "vislib/math/Rectangle.h"


using namespace megamol::core;


/*
 * view::View2DGL::View2DGL
 */
view::View2DGL::View2DGL(void)
        : view::AbstractView()
    , _height(1.0f)
    , _mouseMode(MouseMode::Propagate)
    , _mouseX(0.0f)
    , _mouseY(0.0f)
    , _viewX(0.0f)
    , _viewY(0.0f)
    , _viewZoom(1.0f)
    , _viewUpdateCnt(0)
    , _width(1.0f) {

    // Override renderSlot behavior
    this->_lhsRenderSlot.SetCallback(
        view::CallRenderViewGL::ClassName(), InputCall::FunctionName(InputCall::FnOnKey), &AbstractView::OnKeyCallback);
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(), InputCall::FunctionName(InputCall::FnOnChar),
        &AbstractView::OnCharCallback);
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(),
        InputCall::FunctionName(InputCall::FnOnMouseButton), &AbstractView::OnMouseButtonCallback);
    this->_lhsRenderSlot.SetCallback(view::CallRenderViewGL::ClassName(), InputCall::FunctionName(InputCall::FnOnMouseMove),
        &AbstractView::OnMouseMoveCallback);
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

    this->_rhsRenderSlot.SetCompatibleCall<CallRender2DGLDescription>();
    this->MakeSlotAvailable(&this->_rhsRenderSlot);

    this->ResetView();
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
void view::View2DGL::Render(const mmcRenderViewContext& context, Call* call) {

    AbstractView::beforeRender(context);

    CallRender2DGL* cr2d = this->_rhsRenderSlot.CallAs<CallRender2DGL>();

    if (cr2d == NULL) {
        return;
    }
    cr2d->SetCamera(this->_camera);
    ::glMatrixMode(GL_PROJECTION);
    ::glLoadIdentity();
    float w = this->_width;
    float h = this->_height;
    float asp = h / w;
    //::glScalef(asp, 1.0f, 1.0f);
    //float aMatrix[16];
    vislib::math::Matrix<float, 4, vislib::math::MatrixLayout::COLUMN_MAJOR> m;
    //glGetFloatv(GL_PROJECTION_MATRIX, aMatrix);

    m.SetIdentity();
    m.SetAt(0,0,asp);
    glLoadMatrixf(m.PeekComponents());

    float vx = this->_viewX;
    float vy = this->_viewY;
    float vz = this->_viewZoom;

    ::glMatrixMode(GL_MODELVIEW);
    ::glLoadIdentity();
    m.SetIdentity();
    m.SetAt(0, 0, vz);
    m.SetAt(1, 1, vz);
    m.SetAt(0, 3, vx * vz);
    m.SetAt(1, 3, vy * vz);
    //::glScalef(vz, vz, 1.0f);
    //::glTranslatef(vx, vy, 0.0f);
    //glGetFloatv(GL_MODELVIEW_MATRIX, aMatrix);
    glLoadMatrixf(m.PeekComponents());

    asp = 1.0f / asp;
    vislib::math::Rectangle<float> vr(
        (-asp / vz - vx),
        (-1.0f / vz - vy),
        (asp / vz - vx),
        (1.0f / vz - vy));
    cr2d->AccessBoundingBoxes().SetBoundingBox(vr.Left(),vr.Bottom(),vr.Right(),vr.Top());

    if (call == nullptr) {
        if ((this->_fbo->GetWidth() != w) ||
            (this->_fbo->GetHeight() != h) ||
            !this->_fbo->IsValid() ) {
            this->_fbo->Release();
            if (!this->_fbo->Create(w, h, GL_RGBA8, GL_RGBA,
                    GL_UNSIGNED_BYTE, vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE)) {
                throw vislib::Exception(
                    "[View2DGL] Unable to create image framebuffer object.", __FILE__, __LINE__);
                return;
            }
        }
    } else {
        auto gl_call = dynamic_cast<view::CallRenderViewGL*>(call);
        this->_fbo = gl_call->GetFramebufferObject();
    }

    this->_fbo->Enable();
    auto bgcol = this->BkgndColour();
    glClearColor(bgcol.r, bgcol.g, bgcol.b, bgcol.a);
    glClear(GL_COLOR_BUFFER_BIT);
    cr2d->SetFramebufferObject(_fbo);

    (*cr2d)(AbstractCallRender::FnRender);

    this->_fbo->Disable();
    if (call == nullptr) {
        this->_fbo->DrawColourTexture();
    }

    //after render
    AbstractView::afterRender(context);
}


/*
 * view::View2DGL::ResetView
 */
void view::View2DGL::ResetView(void) {
    // using namespace vislib::graphics;
    VLTRACE(VISLIB_TRCELVL_INFO, "View2DGL::ResetView\n");

    CallRender2DGL *cr2d = this->_rhsRenderSlot.CallAs<CallRender2DGL>();
    if ((cr2d != NULL) && ((*cr2d)(AbstractCallRender::FnGetExtents))) {
        this->_viewX =
            -0.5f * (cr2d->GetBoundingBoxes().BoundingBox().Left() + cr2d->GetBoundingBoxes().BoundingBox().Right());
        this->_viewY =
            -0.5f * (cr2d->GetBoundingBoxes().BoundingBox().Bottom() + cr2d->GetBoundingBoxes().BoundingBox().Top());
        if ((this->_width / this->_height) > static_cast<float>(cr2d->GetBoundingBoxes().BoundingBox().Width() /
                                                              cr2d->GetBoundingBoxes().BoundingBox().Height())) {
            this->_viewZoom = 2.0f / cr2d->GetBoundingBoxes().BoundingBox().Height();
        } else {
            this->_viewZoom = (2.0f * this->_width) / (this->_height * cr2d->GetBoundingBoxes().BoundingBox().Width());
        }
        this->_viewZoom *= 0.99f;

    } else {
        this->_viewX = 0.0f;
        this->_viewY = 0.0f;
        this->_viewZoom = 1.0f;
    }

    this->_viewUpdateCnt++;
}


/*
 * view::View2DGL::Resize
 */
void view::View2DGL::Resize(unsigned int width, unsigned int height) {

    this->_width = static_cast<float>(width);
    this->_height = static_cast<float>(height);

    AbstractView::Resize(width, height);

    //TODO resize/recreate FBO
}


/*
 * view::View2DGL::OnRenderView
 */
bool view::View2DGL::OnRenderView(Call& call) {
    float overBC[3];
    view::CallRenderViewGL *crv = dynamic_cast<view::CallRenderViewGL*>(&call);
    if (crv == NULL) return false;

    float time = crv->Time();
    if (time < 0.0f) time = this->DefaultTime(crv->InstanceTime());
    mmcRenderViewContext context;
    ::ZeroMemory(&context, sizeof(context));
    context.Time = time;
    context.InstanceTime = crv->InstanceTime();
    // TODO: Affinity
    this->Render(context, &call);

    return true;
}


/*
 * view::View2DGL::UpdateFreeze
 */
void view::View2DGL::UpdateFreeze(bool freeze) {
    // currently not supported
}


bool view::View2DGL::OnKey(Key key, KeyAction action, Modifiers mods) {
    auto* cr = this->_rhsRenderSlot.CallAs<view::CallRender2DGL>();
    if (cr == NULL) return false;

    if (key == Key::KEY_HOME) {
        OnResetView(this->_resetViewSlot);
    }

    InputEvent evt;
    evt.tag = InputEvent::Tag::Key;
    evt.keyData.key = key;
    evt.keyData.action = action;
    evt.keyData.mods = mods;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRender2DGL::FnOnKey)) return false;

    return true;
}


bool view::View2DGL::OnChar(unsigned int codePoint) {
    auto* cr = this->_rhsRenderSlot.CallAs<view::CallRender2DGL>();
    if (cr == NULL) return false;

    InputEvent evt;
    evt.tag = InputEvent::Tag::Char;
    evt.charData.codePoint = codePoint;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRender2DGL::FnOnChar)) return false;

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
        if ((*cr)(view::CallRender2DGL::FnOnMouseButton)) return true;
    }

    auto down = action == MouseButtonAction::PRESS;
    if (button == MouseButton::BUTTON_LEFT && down) {
        this->_mouseMode = MouseMode::Pan;
    } else if (button == MouseButton::BUTTON_MIDDLE && down) {
        this->_mouseMode = MouseMode::Zoom;
    }

    return true;
}


bool view::View2DGL::OnMouseMove(double x, double y) {
    if (this->_mouseMode == MouseMode::Propagate) {
        float mx, my;
        mx = ((x * 2.0f / this->_width) - 1.0f) * this->_width / this->_height;
        my = 1.0f - (y * 2.0f / this->_height);
        mx /= this->_viewZoom;
        my /= this->_viewZoom;
        mx -= this->_viewX;
        my -= this->_viewY;

        auto* cr = this->_rhsRenderSlot.CallAs<view::CallRender2DGL>();
        if (cr) {
            InputEvent evt;
            evt.tag = InputEvent::Tag::MouseMove;
            evt.mouseMoveData.x = mx;
            evt.mouseMoveData.y = my;
            cr->SetInputEvent(evt);
            if ((*cr)(view::CallRender2DGL::FnOnMouseMove)) return true;
        }
    } else if (this->_mouseMode == MouseMode::Pan) {
        float movSpeed = 2.0f / (this->_viewZoom * this->_height);
        this->_viewX -= (this->_mouseX - x) * movSpeed;
        this->_viewY += (this->_mouseY - y) * movSpeed;
        if (((this->_mouseX - x) > 0.0f) || ((this->_mouseY - y) > 0.0f)) {
            this->_viewUpdateCnt++;
        }
    } else if (this->_mouseMode == MouseMode::Zoom) {
        const double spd = 2.0;
        const double logSpd = log(spd);
        float base = 1.0f;

        CallRender2DGL* cr2d = this->_rhsRenderSlot.CallAs<CallRender2DGL>();
        if ((cr2d != NULL) && ((*cr2d)(AbstractCallRender::FnGetExtents))) {
            base = cr2d->GetBoundingBoxes().BoundingBox().Height();
        }

        float newZoom =
            static_cast<float>(pow(spd, log(static_cast<double>(this->_viewZoom / base)) / logSpd +
                                            static_cast<double>(((this->_mouseY - y) * 1.0f / this->_height)))) *
            base;

        if (!vislib::math::IsEqual(newZoom, this->_viewZoom)) {
            this->_viewUpdateCnt++;
        }
        this->_viewZoom = newZoom;
    }

    this->_mouseX = x;
    this->_mouseY = y;

    return true;
}


bool view::View2DGL::OnMouseScroll(double dx, double dy) {
    auto* cr = this->_rhsRenderSlot.CallAs<view::CallRender2DGL>();
    if (cr == NULL) return false;

    InputEvent evt;
    evt.tag = InputEvent::Tag::MouseScroll;
    evt.mouseScrollData.dx = dx;
    evt.mouseScrollData.dy = dy;
    cr->SetInputEvent(evt);
    if (!(*cr)(view::CallRender2DGL::FnOnMouseScroll)) return false;

    return true;
}


/*
 * view::View2DGL::unpackMouseCoordinates
 */
void view::View2DGL::unpackMouseCoordinates(float &x, float &y) {
    x *= this->_width;
    y *= this->_height;
    y -= 1.0f;
}


/*
 * view::View2DGL::create
 */
bool view::View2DGL::create(void) {
 
    this->_firstImg = true;

    this->_fbo = std::make_shared<vislib::graphics::gl::FramebufferObject>();

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
    if (crv == nullptr) return false;

    CallRender2DGL* cr2d = this->_rhsRenderSlot.CallAs<CallRender2DGL>();
    if (cr2d == nullptr) {
        return false;
    }
    cr2d->SetCamera(this->_camera);

    if (!(*cr2d)(CallRender2DGL::FnGetExtents)) return false;

    crv->SetTimeFramesCount(cr2d->TimeFramesCount());
    crv->SetIsInSituTime(cr2d->IsInSituTime());
    return true;
}
