/*
 * SplitView.cpp
 *
 * Copyright (C) 2012 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/SplitView.h"
#include "mmcore/view/CallRenderView.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/view/CallSplitViewOverlay.h"
#include "vislib/sys/Log.h"
#include "vislib/Trace.h"


using namespace megamol;
using namespace megamol::core;
using vislib::sys::Log;


/*
 * view::SplitView::SplitView
 */
view::SplitView::SplitView(void) : AbstractView(),
render1Slot("render1", "Connects to the view 1 (left or top)"),
render2Slot("render2", "Connects to the view 2 (right or bottom)"),
overlaySlot("overlay", "Connects to a special overlay renderer (e.g. gui::GUIRenderer)"),
splitOriSlot("split.orientation", "The split orientation"),
splitSlot("split.pos", "The split position"),
splitWidthSlot("split.width", "The split border width"),
splitColourSlot("split.colour", "The split border colour"),
overrideCall(NULL),
clientArea(), client1Area(), client2Area(), fbo1(), fbo2(),
focus(0), mouseX(0.0f), mouseY(0.0f) {

    this->overlaySlot.SetCompatibleCall<view::CallSplitViewOverlayDescription>();
    this->MakeSlotAvailable(&this->overlaySlot);

    this->render1Slot.SetCompatibleCall<CallRenderViewDescription>();
    this->MakeSlotAvailable(&this->render1Slot);

    this->render2Slot.SetCompatibleCall<CallRenderViewDescription>();
    this->MakeSlotAvailable(&this->render2Slot);

    param::EnumParam *ori = new param::EnumParam(0);
    ori->SetTypePair(0, "horizontal");
    ori->SetTypePair(1, "vertical");
    this->splitOriSlot << ori;
    this->MakeSlotAvailable(&this->splitOriSlot);

    this->splitSlot << new param::FloatParam(0.5f, 0.0f, 1.0f);
    this->MakeSlotAvailable(&this->splitSlot);

    this->splitWidthSlot << new param::FloatParam(4.0f, 0.0f, 100.0f);
    this->MakeSlotAvailable(&this->splitWidthSlot);

    this->splitColour = { 0.75f, 0.75f, 0.75f, 1.0f };
    this->splitColourSlot << new param::ColorParam(this->splitColour);
    this->splitColourSlot.SetUpdateCallback(&SplitView::splitColourUpdated);
    this->MakeSlotAvailable(&this->splitColourSlot);

}


/*
 * view::SplitView::~SplitView
 */
view::SplitView::~SplitView(void) {
    this->Release();
}


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
    }
    else {
        vpw = this->overrideCall->ViewportWidth();
        vph = this->overrideCall->ViewportHeight();
    }

    if (this->splitSlot.IsDirty()
        || this->splitOriSlot.IsDirty()
        || this->splitWidthSlot.IsDirty()
        || !this->fbo1.IsValid()
        || !this->fbo2.IsValid()
        || !vislib::math::IsEqual(this->clientArea.Width(), static_cast<float>(vpw))
        || !vislib::math::IsEqual(this->clientArea.Height(), static_cast<float>(vph))) {

        this->clientArea.SetWidth(static_cast<float>(vpw));
        this->clientArea.SetHeight(static_cast<float>(vph));

        this->calcClientAreas();

#if defined(DEBUG) || defined(_DEBUG)
        unsigned int otl = vislib::Trace::GetInstance().GetLevel();
        vislib::Trace::GetInstance().SetLevel(0);
#endif /* DEBUG || _DEBUG */
        if (this->fbo1.IsValid()) this->fbo1.Release();
        this->fbo1.Create(static_cast<unsigned int>(this->client1Area.Width()),
            static_cast<unsigned int>(this->client1Area.Height()));
        this->fbo1.Disable();

        if (this->fbo2.IsValid()) this->fbo2.Release();
        this->fbo2.Create(static_cast<unsigned int>(this->client2Area.Width()),
            static_cast<unsigned int>(this->client2Area.Height()));
        this->fbo2.Disable();
#if defined(DEBUG) || defined(_DEBUG)
        vislib::Trace::GetInstance().SetLevel(otl);
#endif /* DEBUG || _DEBUG */

        // Propagate viewport changes to connected views.
        CallRenderView* crv = this->render1();
        if (crv != NULL) {
            // der ganz ganz dicke "because-i-know"-Knüppel
            AbstractView* crvView = const_cast<AbstractView*>(
                dynamic_cast<const AbstractView*>(static_cast<const Module*>(crv->PeekCalleeSlot()->Owner())));
            if (crvView != NULL) {
                crvView->Resize(static_cast<unsigned int>(this->client1Area.Width()),
                    static_cast<unsigned int>(this->client1Area.Height()));
            }
        }
        crv = this->render2();
        if (crv != NULL) {
            // der ganz ganz dicke "because-i-know"-Knüppel
            AbstractView* crvView = const_cast<AbstractView*>(
                dynamic_cast<const AbstractView*>(static_cast<const Module*>(crv->PeekCalleeSlot()->Owner())));
            if (crvView != NULL) {
                crvView->Resize(static_cast<unsigned int>(this->client2Area.Width()),
                    static_cast<unsigned int>(this->client2Area.Height()));
            }
        }

        if (this->overrideCall != NULL) {
            this->overrideCall->EnableOutputBuffer();
        }
    }

    ::glMatrixMode(GL_PROJECTION);
    ::glLoadIdentity();
    ::glTranslatef(-1.0f, 1.0f, 0.0f);
    ::glScalef(2.0f / this->clientArea.Width(), -2.0f / this->clientArea.Height(), 1.0f);
    ::glPushMatrix();
    ::glMatrixMode(GL_MODELVIEW);
    ::glLoadIdentity();
    ::glPushMatrix();

    ::glViewport(
        static_cast<int>(this->clientArea.Left()),
        static_cast<int>(this->clientArea.Bottom()),
        static_cast<int>(this->clientArea.Width()),
        static_cast<int>(this->clientArea.Height()));

    ::glClearColor(0.0f, 0.0f, 0.0, 1.0f);
    ::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    ::glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    ::glDisable(GL_LIGHTING);
    ::glDisable(GL_CULL_FACE);
    ::glDisable(GL_DEPTH_TEST);
    ::glDisable(GL_TEXTURE_2D);
    ::glBegin(GL_QUADS);
    ::glColor4fv(this->splitColour.data());
    float sx1, sx2, sy1, sy2;
    float sp = this->splitSlot.Param<param::FloatParam>()->Value();
    float shw = this->splitWidthSlot.Param<param::FloatParam>()->Value() * 0.5f;
    if (this->splitOriSlot.Param<param::EnumParam>()->Value() == 0) { // horizontal
        sy1 = 0.0f;
        sy2 = this->clientArea.Height();
        sx1 = this->clientArea.Width() * sp;
        sx2 = sx1 + shw;
        sx1 -= shw;
    }
    else { // vertical
        sx1 = 0.0f;
        sx2 = this->clientArea.Width();
        sy1 = this->clientArea.Height() * sp;
        sy2 = sy1 + shw;
        sy1 -= shw;
    }
    ::glVertex2f(sx1, sy1);
    ::glVertex2f(sx2, sy1);
    ::glVertex2f(sx2, sy2);
    ::glVertex2f(sx1, sy2);
    ::glEnd();

    CallRenderView *crv = this->render1();
    vislib::math::Rectangle<float>* car = &this->client1Area;
    vislib::graphics::gl::FramebufferObject* fbo = &this->fbo1;
    for (unsigned int ri = 0; ri < 2; ri++) {
        if ((crv != NULL) && (fbo != NULL)) {
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

            ::glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            ::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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

            ::glEnable(GL_TEXTURE_2D);
            ::glDisable(GL_LIGHTING);
            ::glDisable(GL_CULL_FACE);
            ::glDisable(GL_DEPTH_TEST);
            ::glDisable(GL_BLEND);

            ::glMatrixMode(GL_PROJECTION);
            ::glPopMatrix();
            ::glPushMatrix();
            ::glMatrixMode(GL_MODELVIEW);
            ::glPopMatrix();
            ::glPushMatrix();

            fbo->BindColourTexture();

            ::glColor4ub(255, 255, 255, 255);
            ::glBegin(GL_QUADS);
            ::glTexCoord2f(0.0f, 1.0f); ::glVertex2f(car->Left(), car->Bottom());
            ::glTexCoord2f(0.0f, 0.0f); ::glVertex2f(car->Left(), car->Top());
            ::glTexCoord2f(1.0f, 0.0f); ::glVertex2f(car->Right(), car->Top());
            ::glTexCoord2f(1.0f, 1.0f); ::glVertex2f(car->Right(), car->Bottom());
            ::glEnd();

            ::glBindTexture(GL_TEXTURE_2D, 0);

        }
        crv = this->render2();
        car = &this->client2Area;
        fbo = &this->fbo2;
    }

    // Render overlay renderer as overlay at last.
    auto* cgr = this->overlaySlot.CallAs<view::CallSplitViewOverlay>();
    if (cgr != nullptr) {
        cgr->SetInstanceTime(instTime);
        cgr->Resize(this->clientArea.Width(), this->clientArea.Height());
        (*cgr)(view::CallSplitViewOverlay::FnOverlay);
    }

    ::glMatrixMode(GL_PROJECTION);
    ::glPopMatrix();
    ::glMatrixMode(GL_MODELVIEW);
    ::glPopMatrix();

}


/*
 * view::SplitView::ResetView
 */
void view::SplitView::ResetView(void) {
    CallRenderView *crv = this->render1();
    if (crv != NULL) (*crv)(CallRenderView::CALL_RESETVIEW);
    crv = this->render2();
    if (crv != NULL) (*crv)(CallRenderView::CALL_RESETVIEW);
}


/*
 * view::SplitView::Resize
 */
void view::SplitView::Resize(unsigned int width, unsigned int height) {
    if (!vislib::math::IsEqual(this->clientArea.Width(), static_cast<float>(width))
        || !vislib::math::IsEqual(this->clientArea.Height(), static_cast<float>(height))) {
        this->clientArea.SetWidth(static_cast<float>(width));
        this->clientArea.SetHeight(static_cast<float>(height));

        this->calcClientAreas();

#if defined(DEBUG) || defined(_DEBUG)
        unsigned int otl = vislib::Trace::GetInstance().GetLevel();
        vislib::Trace::GetInstance().SetLevel(0);
#endif /* DEBUG || _DEBUG */
        if (this->fbo1.IsValid()) this->fbo1.Release();
        this->fbo1.Create(static_cast<unsigned int>(this->client1Area.Width()),
            static_cast<unsigned int>(this->client1Area.Height()));

        if (this->fbo2.IsValid()) this->fbo2.Release();
        this->fbo2.Create(static_cast<unsigned int>(this->client2Area.Width()),
            static_cast<unsigned int>(this->client2Area.Height()));
#if defined(DEBUG) || defined(_DEBUG)
        vislib::Trace::GetInstance().SetLevel(otl);
#endif /* DEBUG || _DEBUG */

        CallRenderView *crv = this->render1();
        if (crv != NULL) {
            // der ganz ganz dicke "because-i-know"-Knüppel
            AbstractView *crvView =
                const_cast<AbstractView*>(dynamic_cast<const AbstractView *>(
                    static_cast<const Module*>(crv->PeekCalleeSlot()->Owner())));
            if (crvView != NULL) {
                crvView->Resize(
                    static_cast<unsigned int>(this->client1Area.Width()),
                    static_cast<unsigned int>(this->client1Area.Height()));
            }
        }
        crv = this->render2();
        if (crv != NULL) {
            // der ganz ganz dicke "because-i-know"-Knüppel
            AbstractView *crvView =
                const_cast<AbstractView*>(dynamic_cast<const AbstractView *>(
                    static_cast<const Module*>(crv->PeekCalleeSlot()->Owner())));
            if (crvView != NULL) {
                crvView->Resize(
                    static_cast<unsigned int>(this->client2Area.Width()),
                    static_cast<unsigned int>(this->client2Area.Height()));
            }
        }

        auto* cgr = this->overlaySlot.CallAs<view::CallSplitViewOverlay>();
        if (cgr != nullptr) {
            cgr->Resize(this->clientArea.Width(), this->clientArea.Height());
        }
    }
}


/*
 * view::SplitView::OnRenderView
 */
bool view::SplitView::OnRenderView(Call& call) {
    view::CallRenderView *crv = dynamic_cast<view::CallRenderView *>(&call);
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
    CallRenderView *crv = this->render1();
    if (crv != NULL) (*crv)(freeze ? CallRenderView::CALL_FREEZE : CallRenderView::CALL_UNFREEZE);
    crv = this->render2();
    if (crv != NULL) (*crv)(freeze ? CallRenderView::CALL_FREEZE : CallRenderView::CALL_UNFREEZE);
}


bool view::SplitView::OnKey(Key key, KeyAction action, Modifiers mods) {

    bool consumed = false;

    // Prioritize overlay renderer
    auto* cgr = this->overlaySlot.CallAs<view::CallSplitViewOverlay>();
    if (cgr != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::Key;
        evt.keyData.key = key;
        evt.keyData.action = action;
        evt.keyData.mods = mods;
        cgr->SetInputEvent(evt);
        if ((*cgr)(view::CallSplitViewOverlay::FnOnKey)) return true;
    }

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

    // Prioritize overlay renderer
    auto* cgr = this->overlaySlot.CallAs<view::CallSplitViewOverlay>();
    if (cgr != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::Char;
        evt.charData.codePoint = codePoint;
        cgr->SetInputEvent(evt);
        if ((*cgr)(view::CallSplitViewOverlay::FnOnChar)) return true;
    }

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

    // Prioritize overlay renderer
    auto* cgr = this->overlaySlot.CallAs<view::CallSplitViewOverlay>();
    if (cgr != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseButton;
        evt.mouseButtonData.button = button;
        evt.mouseButtonData.action = action;
        evt.mouseButtonData.mods = mods;
        cgr->SetInputEvent(evt);
        if ((*cgr)(view::CallSplitViewOverlay::FnOnMouseButton)) return true;
    }

    auto* crv = this->renderHovered();
    auto* crv1 = this->render1();
    auto* crv2 = this->render2();

    auto down = (action == MouseButtonAction::PRESS);
    if (down) {
        if (crv == crv1) {
            this->focus = 1;
        }
        else if (crv == crv2) {
            this->focus = 2;
        }
        else {
            this->focus = 0;
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

    // Prioritize overlay renderer
    auto* cgr = this->overlaySlot.CallAs<view::CallSplitViewOverlay>();
    if (cgr != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseMove;
        evt.mouseMoveData.x = x;
        evt.mouseMoveData.y = y;
        cgr->SetInputEvent(evt);
        if ((*cgr)(view::CallSplitViewOverlay::FnOnMouseMove)) return true;
    }

    auto* crv = this->renderHovered();
    auto* crv1 = this->render1();
    auto* crv2 = this->render2();

    float mx;
    float my;

    if (crv == crv1) {
        mx = x - this->client1Area.Left();
        my = y - this->client1Area.Bottom();
    }
    else if (crv == crv2) {
        mx = x - this->client2Area.Left();
        my = y - this->client2Area.Bottom();
    }
    else {
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

    // Prioritize overlay renderer
    auto* cgr = this->overlaySlot.CallAs<view::CallSplitViewOverlay>();
    if (cgr != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseScroll;
        evt.mouseScrollData.dx = dx;
        evt.mouseScrollData.dy = dy;
        cgr->SetInputEvent(evt);
        if ((*cgr)(view::CallSplitViewOverlay::FnOnMouseScroll)) return true;
    }

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
void view::SplitView::unpackMouseCoordinates(float &x, float &y) {
    x *= this->clientArea.Width();
    y *= this->clientArea.Height();
}


/*
 * view::SplitView::splitColourUpdated
 */
bool view::SplitView::splitColourUpdated(param::ParamSlot& sender) {
    try {
        this->splitColour = this->splitColourSlot.Param<param::ColorParam>()->Value();
    }
    catch (...) {
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
    int so = this->splitOriSlot.Param<param::EnumParam>()->Value();
    this->splitSlot.ResetDirty();
    this->splitWidthSlot.ResetDirty();
    this->splitOriSlot.ResetDirty();

    if (so == 0) { // horizontal
        this->client1Area.Set(
            this->clientArea.Left(),
            this->clientArea.Bottom(),
            this->clientArea.Left() + this->clientArea.Width() * sp - shw,
            this->clientArea.Top());
        this->client2Area.Set(
            this->clientArea.Left() + this->clientArea.Width() * sp + shw,
            this->clientArea.Bottom(),
            this->clientArea.Right(),
            this->clientArea.Top());
    }
    else { // vertical
        this->client1Area.Set(
            this->clientArea.Left(),
            this->clientArea.Bottom(),
            this->clientArea.Right(),
            this->clientArea.Bottom() + this->clientArea.Height() * sp - shw);
        this->client2Area.Set(
            this->clientArea.Left(),
            this->clientArea.Bottom() + this->clientArea.Height() * sp + shw,
            this->clientArea.Right(),
            this->clientArea.Top());
    }

    this->client1Area.EnforcePositiveSize();
    this->client2Area.EnforcePositiveSize();

}