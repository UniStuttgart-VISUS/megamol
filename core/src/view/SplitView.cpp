/*
 * SplitView.cpp
 *
 * Copyright (C) 2012 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/view/SplitView.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/view/CallRenderView.h"

#include "vislib/Trace.h"
#include "vislib/sys/Log.h"


using namespace megamol;
using namespace megamol::core;
using vislib::sys::Log;

enum Orientation { HORIZONTAL = 0, VERTICAL = 1 };

view::SplitView::SplitView()
    : AbstractView()
    , render1Slot("render1", "Connects to the view 1 (left or top)")
    , render2Slot("render2", "Connects to the view 2 (right or bottom)")
    , splitOrientationSlot("split.orientation", "Splitter orientation")
    , splitPositionSlot("split.pos", "Splitter position")
    , splitWidthSlot("split.width", "Splitter width")
    , splitColourSlot("split.colour", "Splitter colour")
    , overrideCall(nullptr)
    , clientArea()
    , clientArea1()
    , clientArea2()
    , fbo1()
    , fbo2()
    , focus(0)
    , mouseX(0.0f)
    , mouseY(0.0f)
    , dragSplitter(false) {
    this->render1Slot.SetCompatibleCall<CallRenderViewDescription>();
    this->MakeSlotAvailable(&this->render1Slot);

    this->render2Slot.SetCompatibleCall<CallRenderViewDescription>();
    this->MakeSlotAvailable(&this->render2Slot);

    auto* orientations = new param::EnumParam(0);
    orientations->SetTypePair(HORIZONTAL, "Horizontal (side by side)");
    orientations->SetTypePair(VERTICAL, "Vertical");
    this->splitOrientationSlot << orientations;
    this->MakeSlotAvailable(&this->splitOrientationSlot);

    this->splitPositionSlot << new param::FloatParam(0.5f, 0.0f, 1.0f);
    this->MakeSlotAvailable(&this->splitPositionSlot);

    this->splitWidthSlot << new param::FloatParam(4.0f, 0.0f, 100.0f);
    this->MakeSlotAvailable(&this->splitWidthSlot);

    this->splitColourSlot << new param::ColorParam(0.75f, 0.75f, 0.75f, 1.0f);
    this->MakeSlotAvailable(&this->splitColourSlot);
}

view::SplitView::~SplitView() { this->Release(); }

float view::SplitView::DefaultTime(double instTime) const {
    // This view does not do any time control
    return 0.0f;
}

unsigned int view::SplitView::GetCameraSyncNumber() const {
    Log::DefaultLog.WriteWarn("SplitView::GetCameraSyncNumber unsupported");
    return 0u;
}

void view::SplitView::SerialiseCamera(vislib::Serialiser& serialiser) const {
    Log::DefaultLog.WriteWarn("SplitView::SerialiseCamera unsupported");
}

void view::SplitView::DeserialiseCamera(vislib::Serialiser& serialiser) {
    Log::DefaultLog.WriteWarn("SplitView::DeserialiseCamera unsupported");
}

void view::SplitView::Render(const mmcRenderViewContext& context) {
    // TODO: Affinity

    if (this->doHookCode()) {
        this->doBeforeRenderHook();
    }

    unsigned int vpw = 0;
    unsigned int vph = 0;

    if (this->overrideCall == nullptr) {
        GLint vp[4];
        ::glGetIntegerv(GL_VIEWPORT, vp);
        vpw = vp[2];
        vph = vp[3];
    } else {
        vpw = this->overrideCall->ViewportWidth();
        vph = this->overrideCall->ViewportHeight();
    }

    if (this->splitPositionSlot.IsDirty() || this->splitOrientationSlot.IsDirty() || this->splitWidthSlot.IsDirty() ||
        !this->fbo1.IsValid() || !this->fbo2.IsValid() ||
        !vislib::math::IsEqual(this->clientArea.Width(), static_cast<float>(vpw)) ||
        !vislib::math::IsEqual(this->clientArea.Height(), static_cast<float>(vph))) {

        this->updateSize(vpw, vph);

        if (this->overrideCall != nullptr) {
            this->overrideCall->EnableOutputBuffer();
        }
    }
    auto renderAndBlit = [&](vislib::graphics::gl::FramebufferObject& fbo, CallRenderView* crv,
                             const vislib::math::Rectangle<float>& ca) {
        if (crv == nullptr) {
            return;
        }
        crv->SetOutputBuffer(&fbo);
        crv->SetInstanceTime(context.InstanceTime);
        crv->SetTime(-1.0f);

#if defined(DEBUG) || defined(_DEBUG)
        unsigned int otl = vislib::Trace::GetInstance().GetLevel();
        vislib::Trace::GetInstance().SetLevel(0);
#endif /* DEBUG || _DEBUG */
        fbo.Enable();
#if defined(DEBUG) || defined(_DEBUG)
        vislib::Trace::GetInstance().SetLevel(otl);
#endif /* DEBUG || _DEBUG */

        // Defer render call to subview that should clear (if it does not,
        // non-splitview rendering will be broken as well).
        (*crv)(CallRenderView::CALL_RENDER);

#if defined(DEBUG) || defined(_DEBUG)
        vislib::Trace::GetInstance().SetLevel(0);
#endif /* DEBUG || _DEBUG */
        fbo.Disable();
#if defined(DEBUG) || defined(_DEBUG)
        vislib::Trace::GetInstance().SetLevel(otl);
#endif /* DEBUG || _DEBUG */

        if (this->overrideCall != nullptr) {
            this->overrideCall->EnableOutputBuffer();
        }

        // Bind and blit framebuffer.
        GLint binding, readBuffer;
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &binding);
        glGetIntegerv(GL_READ_BUFFER, &readBuffer);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo.GetID());
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBlitFramebuffer(0, 0, fbo.GetWidth(), fbo.GetHeight(), ca.Left(), this->clientArea.Height() - ca.Top(),
            ca.Right(), this->clientArea.Height() - ca.Bottom(), GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, binding);
        glReadBuffer(readBuffer);
    };

    // Draw the splitter through clearing without overplotting.
    auto splitColour = this->splitColourSlot.Param<param::ColorParam>()->Value();
    ::glClearColor(splitColour[0], splitColour[1], splitColour[2], 1.0f);
    ::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    renderAndBlit(this->fbo1, this->render1(), this->clientArea1);
    renderAndBlit(this->fbo2, this->render2(), this->clientArea2);
}

void view::SplitView::ResetView() {
    for (auto crv : {this->render1(), this->render2()}) {
        if (crv != nullptr) (*crv)(CallRenderView::CALL_RESETVIEW);
    }
}

void view::SplitView::Resize(unsigned int width, unsigned int height) {
    if (!vislib::math::IsEqual(this->clientArea.Width(), static_cast<float>(width)) ||
        !vislib::math::IsEqual(this->clientArea.Height(), static_cast<float>(height))) {
        this->updateSize(width, height);
    }
}

bool view::SplitView::OnRenderView(Call& call) {
    auto* crv = dynamic_cast<view::CallRenderView*>(&call);
    if (crv == nullptr) return false;

    this->overrideCall = crv;

    mmcRenderViewContext context;
    ::ZeroMemory(&context, sizeof(context));
    context.Time = crv->Time();
    context.InstanceTime = crv->InstanceTime();
    this->Render(context);

    this->overrideCall = nullptr;

    return true;
}

void view::SplitView::UpdateFreeze(bool freeze) {
    for (auto crv : {this->render1(), this->render2()}) {
        if (crv != nullptr) (*crv)(freeze ? CallRenderView::CALL_FREEZE : CallRenderView::CALL_UNFREEZE);
    }
}

bool view::SplitView::OnKey(Key key, KeyAction action, Modifiers mods) {
    bool consumed = false;

    for (auto crv : {this->render1(), this->render2()}) {
        if (crv != nullptr) {
            InputEvent evt;
            evt.tag = InputEvent::Tag::Key;
            evt.keyData.key = key;
            evt.keyData.action = action;
            evt.keyData.mods = mods;
            crv->SetInputEvent(evt);
            if ((*crv)(view::CallRenderView::FnOnKey)) consumed = true;
        }
    }

    return consumed;
}

bool view::SplitView::OnChar(unsigned int codePoint) {
    bool consumed = false;

    for (auto crv : {this->render1(), this->render2()}) {
        if (crv != nullptr) {
            InputEvent evt;
            evt.tag = InputEvent::Tag::Char;
            evt.charData.codePoint = codePoint;
            crv->SetInputEvent(evt);
            if ((*crv)(view::CallRenderView::FnOnChar)) consumed = true;
        }
    }

    return consumed;
}

bool view::SplitView::OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) {
    auto* crv = this->renderHovered();
    auto* crv1 = this->render1();
    auto* crv2 = this->render2();

    this->dragSplitter = false;

    if (action == MouseButtonAction::PRESS) {
        if (crv == crv1) {
            this->focus = 1;
        } else if (crv == crv2) {
            this->focus = 2;
        } else {
            this->focus = 0;
            this->dragSplitter = true;
        }
    }

    if (crv) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseButton;
        evt.mouseButtonData.button = button;
        evt.mouseButtonData.action = action;
        evt.mouseButtonData.mods = mods;
        crv->SetInputEvent(evt);
        if (!(*crv)(view::CallRenderView::FnOnMouseButton)) return false;
    }

    return true;
}


bool view::SplitView::OnMouseMove(double x, double y) {
    // x, y are coordinates in pixel
    this->mouseX = x;
    this->mouseY = y;

    if (this->dragSplitter) {
        if (this->splitOrientationSlot.Param<param::EnumParam>()->Value() == HORIZONTAL) {
            this->splitPositionSlot.Param<param::FloatParam>()->SetValue(x / this->clientArea.Width());
        } else {
            this->splitPositionSlot.Param<param::FloatParam>()->SetValue(y / this->clientArea.Height());
        }
    }

    auto* crv = this->renderHovered();
    auto* crv1 = this->render1();
    auto* crv2 = this->render2();

    float mx;
    float my;

    if (crv == crv1) {
        mx = this->mouseX - this->clientArea1.Left();
        my = this->mouseY - this->clientArea1.Bottom();
    } else if (crv == crv2) {
        mx = this->mouseX - this->clientArea2.Left();
        my = this->mouseY - this->clientArea2.Bottom();
    } else {
        return false;
    }

    if (crv != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseMove;
        evt.mouseMoveData.x = mx;
        evt.mouseMoveData.y = my;
        crv->SetInputEvent(evt);
        if (!(*crv)(view::CallRenderView::FnOnMouseMove)) return false;
    }

    return true;
}


bool view::SplitView::OnMouseScroll(double dx, double dy) {
    auto* crv = this->renderHovered();
    if (crv == nullptr) return false;

    InputEvent evt;
    evt.tag = InputEvent::Tag::MouseScroll;
    evt.mouseScrollData.dx = dx;
    evt.mouseScrollData.dy = dy;
    crv->SetInputEvent(evt);
    if (!(*crv)(view::CallRenderView::FnOnMouseScroll)) return false;

    return true;
}

bool view::SplitView::create() {
    // nothing to do
    return true;
}

void view::SplitView::release() {
    this->overrideCall = nullptr; // do not delete
    if (this->fbo1.IsValid()) this->fbo1.Release();
    if (this->fbo2.IsValid()) this->fbo2.Release();
}

void view::SplitView::unpackMouseCoordinates(float& x, float& y) {
    x *= this->clientArea.Width();
    y *= this->clientArea.Height();
}

void view::SplitView::updateSize(size_t width, size_t height) {
    this->clientArea.SetWidth(static_cast<float>(width));
    this->clientArea.SetHeight(static_cast<float>(height));
    this->adjustClientAreas();

#if defined(DEBUG) || defined(_DEBUG)
    unsigned int otl = vislib::Trace::GetInstance().GetLevel();
    vislib::Trace::GetInstance().SetLevel(0);
#endif /* DEBUG || _DEBUG */
    if (this->fbo1.IsValid()) this->fbo1.Release();
    this->fbo1.Create(
        static_cast<unsigned int>(this->clientArea1.Width()), static_cast<unsigned int>(this->clientArea1.Height()));
    this->fbo1.Disable();

    if (this->fbo2.IsValid()) this->fbo2.Release();
    this->fbo2.Create(
        static_cast<unsigned int>(this->clientArea2.Width()), static_cast<unsigned int>(this->clientArea2.Height()));
    this->fbo2.Disable();
#if defined(DEBUG) || defined(_DEBUG)
    vislib::Trace::GetInstance().SetLevel(otl);
#endif /* DEBUG || _DEBUG */

    // Propagate viewport changes to connected views.
    auto propagateViewport = [](CallRenderView* crv, vislib::math::Rectangle<float>& clientArea) {
        if (crv == nullptr) {
            return;
        }
        // der ganz ganz dicke "because-i-know"-Knueppel
        auto* crvView = const_cast<AbstractView*>(
            dynamic_cast<const AbstractView*>(static_cast<const Module*>(crv->PeekCalleeSlot()->Owner())));
        if (crvView != nullptr) {
            crvView->Resize(
                static_cast<unsigned int>(clientArea.Width()), static_cast<unsigned int>(clientArea.Height()));
        }
    };
    propagateViewport(this->render1(), this->clientArea1);
    propagateViewport(this->render2(), this->clientArea2);
}

void view::SplitView::adjustClientAreas() {
    float sp = this->splitPositionSlot.Param<param::FloatParam>()->Value();
    float shw = this->splitWidthSlot.Param<param::FloatParam>()->Value() * 0.5f;
    auto so = static_cast<Orientation>(this->splitOrientationSlot.Param<param::EnumParam>()->Value());
    this->splitPositionSlot.ResetDirty();
    this->splitWidthSlot.ResetDirty();
    this->splitOrientationSlot.ResetDirty();

    if (so == HORIZONTAL) {
        this->clientArea1.Set(this->clientArea.Left(), this->clientArea.Bottom(),
            this->clientArea.Left() + this->clientArea.Width() * sp - shw, this->clientArea.Top());
        this->clientArea2.Set(this->clientArea.Left() + this->clientArea.Width() * sp + shw, this->clientArea.Bottom(),
            this->clientArea.Right(), this->clientArea.Top());
    } else {
        this->clientArea1.Set(this->clientArea.Left(), this->clientArea.Bottom(), this->clientArea.Right(),
            this->clientArea.Bottom() + this->clientArea.Height() * sp - shw);
        this->clientArea2.Set(this->clientArea.Left(), this->clientArea.Bottom() + this->clientArea.Height() * sp + shw,
            this->clientArea.Right(), this->clientArea.Top());
    }

    this->clientArea1.EnforcePositiveSize();
    this->clientArea2.EnforcePositiveSize();
}
