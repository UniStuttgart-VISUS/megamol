/*
 * SplitView.cpp
 *
 * Copyright (C) 2012 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/SplitView.h"
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

/*
 * view::SplitView::SplitView
 */
view::SplitView::SplitView(void)
    : AbstractView()
    , render1Slot("render1", "Connects to the view 1 (left or top)")
    , render2Slot("render2", "Connects to the view 2 (right or bottom)")
    , splitOriSlot("split.orientation", "Splitter orientation")
    , splitSlot("split.pos", "Splitter position")
    , splitWidthSlot("split.width", "Splitter width")
    , splitColourSlot("split.colour", "Splitter colour")
    , overrideCall(NULL)
    , clientArea()
    , client1Area()
    , client2Area()
    , fbo1()
    , fbo2()
    , focus(0)
    , mouseX(0.0f)
    , mouseY(0.0f)
    , dragSlider(false) {

    this->render1Slot.SetCompatibleCall<CallRenderViewDescription>();
    this->MakeSlotAvailable(&this->render1Slot);

    this->render2Slot.SetCompatibleCall<CallRenderViewDescription>();
    this->MakeSlotAvailable(&this->render2Slot);

    param::EnumParam* ori = new param::EnumParam(0);
    ori->SetTypePair(HORIZONTAL, "Horizontal");
    ori->SetTypePair(VERTICAL, "Vertical");
    this->splitOriSlot << ori;
    this->MakeSlotAvailable(&this->splitOriSlot);

    this->splitSlot << new param::FloatParam(0.5f, 0.0f, 1.0f);
    this->MakeSlotAvailable(&this->splitSlot);

    this->splitWidthSlot << new param::FloatParam(4.0f, 0.0f, 100.0f);
    this->MakeSlotAvailable(&this->splitWidthSlot);

    this->splitColour = {0.75f, 0.75f, 0.75f, 1.0f};
    this->splitColourSlot << new param::ColorParam(this->splitColour);
    this->splitColourSlot.SetUpdateCallback(&SplitView::splitColourUpdated);
    this->MakeSlotAvailable(&this->splitColourSlot);
}


/*
 * view::SplitView::~SplitView
 */
view::SplitView::~SplitView(void) { this->Release(); }


/*
 * view::SplitView::DefaultTime
 */
float view::SplitView::DefaultTime(double instTime) const {
    // This view does not do any time control
    return 0.0f;
}


/*
 * view::SplitView::GetCameraSyncNumber
 */
unsigned int view::SplitView::GetCameraSyncNumber(void) const {
    Log::DefaultLog.WriteWarn("SplitView::GetCameraSyncNumber unsupported");
    return 0u;
}


/*
 * view::SplitView::SerialiseCamera
 */
void view::SplitView::SerialiseCamera(vislib::Serialiser& serialiser) const {
    Log::DefaultLog.WriteWarn("SplitView::SerialiseCamera unsupported");
}


/*
 * view::SplitView::DeserialiseCamera
 */
void view::SplitView::DeserialiseCamera(vislib::Serialiser& serialiser) {
    Log::DefaultLog.WriteWarn("SplitView::DeserialiseCamera unsupported");
}


/*
 * view::SplitView::Render
 */
void view::SplitView::Render(const mmcRenderViewContext& context) {
    float time = static_cast<float>(context.Time);
    double instTime = context.InstanceTime;
    // TODO: Affinity

    if (this->doHookCode()) {
        this->doBeforeRenderHook();
    }

    unsigned int vpw = 0;
    unsigned int vph = 0;

    if (this->overrideCall == NULL) {
        GLint vp[4];
        ::glGetIntegerv(GL_VIEWPORT, vp);
        vpw = vp[2];
        vph = vp[3];
    } else {
        vpw = this->overrideCall->ViewportWidth();
        vph = this->overrideCall->ViewportHeight();
    }

    if (this->splitSlot.IsDirty() || this->splitOriSlot.IsDirty() || this->splitWidthSlot.IsDirty() ||
        !this->fbo1.IsValid() || !this->fbo2.IsValid() ||
        !vislib::math::IsEqual(this->clientArea.Width(), static_cast<float>(vpw)) ||
        !vislib::math::IsEqual(this->clientArea.Height(), static_cast<float>(vph))) {

        this->clientArea.SetWidth(static_cast<float>(vpw));
        this->clientArea.SetHeight(static_cast<float>(vph));

        this->validate();

        if (this->overrideCall != NULL) {
            this->overrideCall->EnableOutputBuffer();
        }
    }

    auto renderAndBlit = [&](CallRenderView* crv, vislib::math::Rectangle<float>* car,
                             vislib::graphics::gl::FramebufferObject* fbo) {
        assert((crv != NULL) && (fbo != NULL) && "Behold of the null silliness");
        crv->SetOutputBuffer(fbo);
        crv->SetInstanceTime(instTime);
        crv->SetTime(-1.0f);

#if defined(DEBUG) || defined(_DEBUG)
        unsigned int otl = vislib::Trace::GetInstance().GetLevel();
        vislib::Trace::GetInstance().SetLevel(0);
#endif /* DEBUG || _DEBUG */
        fbo->Enable();
#if defined(DEBUG) || defined(_DEBUG)
        vislib::Trace::GetInstance().SetLevel(otl);
#endif /* DEBUG || _DEBUG */

        // Defer render call to subview that should clear (if it does not,
        // non-splitview rendering will be broken as well).
        (*crv)(CallRenderView::CALL_RENDER);

#if defined(DEBUG) || defined(_DEBUG)
        vislib::Trace::GetInstance().SetLevel(0);
#endif /* DEBUG || _DEBUG */
        fbo->Disable();
#if defined(DEBUG) || defined(_DEBUG)
        vislib::Trace::GetInstance().SetLevel(otl);
#endif /* DEBUG || _DEBUG */

        if (this->overrideCall != NULL) {
            this->overrideCall->EnableOutputBuffer();
        }

        // Bind and blit framebuffer.
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo->GetID());
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBlitFramebuffer(0, 0, fbo->GetWidth(), fbo->GetHeight(), car->Left(), this->clientArea.Height() - car->Top(),
            car->Right(), this->clientArea.Height() - car->Bottom(), GL_COLOR_BUFFER_BIT, GL_NEAREST);
    };

    // Draw a splitter by clearing with the right color(tm).
    ::glClearColor(this->splitColour[0], this->splitColour[1], this->splitColour[2], 1.0f);
    ::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    renderAndBlit(this->render1(), &this->client1Area, &this->fbo1);
    renderAndBlit(this->render2(), &this->client2Area, &this->fbo2);
}


/*
 * view::SplitView::ResetView
 */
void view::SplitView::ResetView(void) {
    CallRenderView* crv = this->render1();
    if (crv != NULL) (*crv)(CallRenderView::CALL_RESETVIEW);
    crv = this->render2();
    if (crv != NULL) (*crv)(CallRenderView::CALL_RESETVIEW);
}


/*
 * view::SplitView::Resize
 */
void view::SplitView::Resize(unsigned int width, unsigned int height) {
    if (!vislib::math::IsEqual(this->clientArea.Width(), static_cast<float>(width)) ||
        !vislib::math::IsEqual(this->clientArea.Height(), static_cast<float>(height))) {
        this->clientArea.SetWidth(static_cast<float>(width));
        this->clientArea.SetHeight(static_cast<float>(height));

        this->validate();
    }
}


/*
 * view::SplitView::OnRenderView
 */
bool view::SplitView::OnRenderView(Call& call) {
    view::CallRenderView* crv = dynamic_cast<view::CallRenderView*>(&call);
    if (crv == NULL) return false;

    this->overrideCall = crv;

    mmcRenderViewContext context;
    ::ZeroMemory(&context, sizeof(context));
    context.Time = crv->Time();
    context.InstanceTime = crv->InstanceTime();
    // TODO: Affinity
    this->Render(context);

    this->overrideCall = NULL;

    return true;
}


/*
 * view::SplitView::UpdateFreeze
 */
void view::SplitView::UpdateFreeze(bool freeze) {
    CallRenderView* crv = this->render1();
    if (crv != NULL) (*crv)(freeze ? CallRenderView::CALL_FREEZE : CallRenderView::CALL_UNFREEZE);
    crv = this->render2();
    if (crv != NULL) (*crv)(freeze ? CallRenderView::CALL_FREEZE : CallRenderView::CALL_UNFREEZE);
}


bool view::SplitView::OnKey(Key key, KeyAction action, Modifiers mods) {

    bool consumed = false;

    auto* crv = this->render1();
    if (crv != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::Key;
        evt.keyData.key = key;
        evt.keyData.action = action;
        evt.keyData.mods = mods;
        crv->SetInputEvent(evt);
        if ((*crv)(view::CallRenderView::FnOnKey)) consumed = true;
    }

    crv = this->render2();
    if (crv != nullptr) {

        InputEvent evt;
        evt.tag = InputEvent::Tag::Key;
        evt.keyData.key = key;
        evt.keyData.action = action;
        evt.keyData.mods = mods;
        crv->SetInputEvent(evt);
        if ((*crv)(view::CallRenderView::FnOnKey)) consumed = true;
    }

    return consumed;
}


bool view::SplitView::OnChar(unsigned int codePoint) {

    bool consumed = false;

    auto* crv = this->render1();
    if (crv != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::Char;
        evt.charData.codePoint = codePoint;
        crv->SetInputEvent(evt);
        if ((*crv)(view::CallRenderView::FnOnChar)) consumed = true;
    }

    crv = this->render2();
    if (crv != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::Char;
        evt.charData.codePoint = codePoint;
        crv->SetInputEvent(evt);
        if ((*crv)(view::CallRenderView::FnOnChar)) consumed = true;
    }

    return consumed;
}


bool view::SplitView::OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) {

    auto* crv = this->renderHovered();
    auto* crv1 = this->render1();
    auto* crv2 = this->render2();

    this->dragSlider = false;

    auto down = (action == MouseButtonAction::PRESS);
    if (down) {
        if (crv == crv1) {
            this->focus = 1;
        } else if (crv == crv2) {
            this->focus = 2;
        } else {
            this->focus = 0;
            this->dragSlider = true;
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

    if (this->dragSlider) {
        if (this->splitOriSlot.Param<param::EnumParam>()->Value() == HORIZONTAL) {
            this->splitSlot.Param<param::FloatParam>()->SetValue(x / this->clientArea.Width());
        } else {
            this->splitSlot.Param<param::FloatParam>()->SetValue(y / this->clientArea.Height());
        }
    }

    auto* crv = this->renderHovered();
    auto* crv1 = this->render1();
    auto* crv2 = this->render2();

    float mx;
    float my;

    if (crv == crv1) {
        mx = this->mouseX - this->client1Area.Left();
        my = this->mouseY - this->client1Area.Bottom();
    } else if (crv == crv2) {
        mx = this->mouseX - this->client2Area.Left();
        my = this->mouseY - this->client2Area.Bottom();
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
    if (crv == NULL) return false;

    InputEvent evt;
    evt.tag = InputEvent::Tag::MouseScroll;
    evt.mouseScrollData.dx = dx;
    evt.mouseScrollData.dy = dy;
    crv->SetInputEvent(evt);
    if (!(*crv)(view::CallRenderView::FnOnMouseScroll)) return false;

    return true;
}


/*
 * view::SplitView::create
 */
bool view::SplitView::create(void) {
    // nothing to do
    return true;
}


/*
 * view::SplitView::release
 */
void view::SplitView::release(void) {
    this->overrideCall = NULL; // do not delete
    if (this->fbo1.IsValid()) this->fbo1.Release();
    if (this->fbo2.IsValid()) this->fbo2.Release();
}


/*
 * view::SplitView::unpackMouseCoordinates
 */
void view::SplitView::unpackMouseCoordinates(float& x, float& y) {
    x *= this->clientArea.Width();
    y *= this->clientArea.Height();
}


void view::SplitView::validate() {
    this->calcClientAreas();

#if defined(DEBUG) || defined(_DEBUG)
    unsigned int otl = vislib::Trace::GetInstance().GetLevel();
    vislib::Trace::GetInstance().SetLevel(0);
#endif /* DEBUG || _DEBUG */
    if (this->fbo1.IsValid()) this->fbo1.Release();
    this->fbo1.Create(
        static_cast<unsigned int>(this->client1Area.Width()), static_cast<unsigned int>(this->client1Area.Height()));
    this->fbo1.Disable();

    if (this->fbo2.IsValid()) this->fbo2.Release();
    this->fbo2.Create(
        static_cast<unsigned int>(this->client2Area.Width()), static_cast<unsigned int>(this->client2Area.Height()));
    this->fbo2.Disable();
#if defined(DEBUG) || defined(_DEBUG)
    vislib::Trace::GetInstance().SetLevel(otl);
#endif /* DEBUG || _DEBUG */

    // Propagate viewport changes to connected views.
    auto propagateViewport = [](CallRenderView* crv, vislib::math::Rectangle<float>& clientArea) {
        if (crv == NULL) {
            return;
        }
        // der ganz ganz dicke "because-i-know"-Knueppel
        AbstractView* crvView = const_cast<AbstractView*>(
            dynamic_cast<const AbstractView*>(static_cast<const Module*>(crv->PeekCalleeSlot()->Owner())));
        if (crvView != NULL) {
            crvView->Resize(
                static_cast<unsigned int>(clientArea.Width()), static_cast<unsigned int>(clientArea.Height()));
        }
    };
    propagateViewport(this->render1(), this->client1Area);
    propagateViewport(this->render2(), this->client2Area);
}

/*
 * view::SplitView::splitColourUpdated
 */
bool view::SplitView::splitColourUpdated(param::ParamSlot& sender) {
    try {
        this->splitColour = this->splitColourSlot.Param<param::ColorParam>()->Value();
    } catch (...) {
        Log::DefaultLog.WriteError("Unable to parse splitter colour");
    }
    return true;
}


/*
 * view::SplitView::calcClientAreas
 */
void view::SplitView::calcClientAreas(void) {
    float sp = this->splitSlot.Param<param::FloatParam>()->Value();
    float shw = this->splitWidthSlot.Param<param::FloatParam>()->Value() * 0.5f;
    auto so = static_cast<Orientation>(this->splitOriSlot.Param<param::EnumParam>()->Value());
    this->splitSlot.ResetDirty();
    this->splitWidthSlot.ResetDirty();
    this->splitOriSlot.ResetDirty();

    if (so == HORIZONTAL) {
        this->client1Area.Set(this->clientArea.Left(), this->clientArea.Bottom(),
            this->clientArea.Left() + this->clientArea.Width() * sp - shw, this->clientArea.Top());
        this->client2Area.Set(this->clientArea.Left() + this->clientArea.Width() * sp + shw, this->clientArea.Bottom(),
            this->clientArea.Right(), this->clientArea.Top());
    } else {
        this->client1Area.Set(this->clientArea.Left(), this->clientArea.Bottom(), this->clientArea.Right(),
            this->clientArea.Bottom() + this->clientArea.Height() * sp - shw);
        this->client2Area.Set(this->clientArea.Left(), this->clientArea.Bottom() + this->clientArea.Height() * sp + shw,
            this->clientArea.Right(), this->clientArea.Top());
    }

    this->client1Area.EnforcePositiveSize();
    this->client2Area.EnforcePositiveSize();
}
