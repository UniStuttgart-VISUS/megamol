/*
 * SplitViewGL.cpp
 *
 * Copyright (C) 2012 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "mmcore/view/SplitViewGL.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/view/CallRenderViewGL.h"

#include "vislib/Trace.h"
#include "mmcore/utility/log/Log.h"


using namespace megamol;
using namespace megamol::core;
using megamol::core::utility::log::Log;

enum Orientation { HORIZONTAL = 0, VERTICAL = 1 };

view::SplitViewGL::SplitViewGL()
    : AbstractView()
    , _render1Slot("render1", "Connects to the view 1 (left or top)")
    , _render2Slot("render2", "Connects to the view 2 (right or bottom)")
    , _splitOrientationSlot("split.orientation", "Splitter orientation")
    , _splitPositionSlot("split.pos", "Splitter position")
    , _splitWidthSlot("split.width", "Splitter width")
    , _splitColourSlot("split.colour", "Splitter colour")
    , _enableTimeSyncSlot("timeLord",
          "Enables time synchronization between the connected views. The time of this view is then used instead")
    , _inputToBothSlot("inputToBoth", "Forward input to both child views")
    , _clientArea()
    , _clientArea1()
    , _clientArea2()
    , _fbo1()
    , _fbo2()
    , _focus(0)
    , _mouseX(0.0f)
    , _mouseY(0.0f)
    , _dragSplitter(false) {

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

    this->_render1Slot.SetCompatibleCall<CallRenderViewGLDescription>();
    this->MakeSlotAvailable(&this->_render1Slot);

    this->_render2Slot.SetCompatibleCall<CallRenderViewGLDescription>();
    this->MakeSlotAvailable(&this->_render2Slot);

    auto* orientations = new param::EnumParam(0);
    orientations->SetTypePair(HORIZONTAL, "Horizontal (side by side)");
    orientations->SetTypePair(VERTICAL, "Vertical");
    this->_splitOrientationSlot << orientations;
    this->MakeSlotAvailable(&this->_splitOrientationSlot);

    this->_splitPositionSlot << new param::FloatParam(0.5f, 0.0f, 1.0f);
    this->MakeSlotAvailable(&this->_splitPositionSlot);

    this->_splitWidthSlot << new param::FloatParam(4.0f, 0.0f, 100.0f);
    this->MakeSlotAvailable(&this->_splitWidthSlot);

    this->_splitColourSlot << new param::ColorParam(0.75f, 0.75f, 0.75f, 1.0f);
    this->MakeSlotAvailable(&this->_splitColourSlot);

    this->_enableTimeSyncSlot << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->_enableTimeSyncSlot);

    this->_inputToBothSlot << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->_inputToBothSlot);
}

view::SplitViewGL::~SplitViewGL(void) { this->Release(); }

float view::SplitViewGL::DefaultTime(double instTime) const { return this->_timeCtrl.Time(instTime); }

unsigned int view::SplitViewGL::GetCameraSyncNumber() const {
    Log::DefaultLog.WriteWarn("SplitViewGL::GetCameraSyncNumber unsupported");
    return 0u;
}

void view::SplitViewGL::Render(const mmcRenderViewContext& context, Call* call) {
    // TODO: Affinity
    float time = static_cast<float>(context.Time);

    if (this->doHookCode()) {
        this->doBeforeRenderHook();
    }

    unsigned int vpw = 0;
    unsigned int vph = 0;

    if (call == nullptr) {
        vpw = _camera.image_tile().width();
        vph = _camera.image_tile().height();
    } else {
        auto gpu_call = dynamic_cast<view::CallRenderViewGL*>(call);
        vpw = gpu_call->GetFramebufferObject()->GetWidth();
        vph = gpu_call->GetFramebufferObject()->GetHeight();
    }

    if (this->_enableTimeSyncSlot.Param<param::BoolParam>()->Value()) {
        auto cr = this->render1();
        (*cr)(CallRenderViewGL::CALL_EXTENTS);
        auto fcount = cr->TimeFramesCount();
        auto insitu = cr->IsInSituTime();
        cr = this->render2();
        (*cr)(CallRenderViewGL::CALL_EXTENTS);
        fcount = std::min(fcount, cr->TimeFramesCount());
        insitu = insitu && cr->IsInSituTime();

        this->_timeCtrl.SetTimeExtend(fcount, insitu);
        if (time > static_cast<float>(fcount)) {
            time = static_cast<float>(fcount);
        }
    }

    //float sp = this->splitPositionSlot.Param<param::FloatParam>()->Value();
    //float shw = this->splitWidthSlot.Param<param::FloatParam>()->Value() * 0.5f;
    //auto so = static_cast<Orientation>(this->splitOrientationSlot.Param<param::EnumParam>()->Value());
    //if (so == HORIZONTAL) {
    //    auto oc = this->overrideCall;
    //    float splitpos = oc->VirtualWidth() * sp;

    //    auto left1 = oc->TileX();
    //    auto right1 = std::max(std::min(oc->TileX() + oc->TileWidth(), splitpos), oc->TileX());
    //    if (left1 == right1) {
    //        // skip client 1
    //        // draw no handle at all
    //    }
    //    // or the other way round?
    //    auto top1 = oc->TileY();
    //    auto bottom1 = oc->TileY() + oc->TileHeight();

    //    auto left2 = std::min(std::max(oc->TileX(), splitpos), oc->TileX() + oc->TileWidth());
    //    auto right2 = oc->TileX() + oc->TileWidth();
    //    if (left2 == right2) {
    //        // skip client 2
    //        // draw no handle at all
    //    }
    //    auto top2 = top1;
    //    auto bottom2 = bottom1;
    //} else {
    //}

    if (this->_splitPositionSlot.IsDirty() || this->_splitOrientationSlot.IsDirty() || this->_splitWidthSlot.IsDirty() ||
        !this->_fbo1->IsValid() || !this->_fbo2->IsValid() ||
        !vislib::math::IsEqual(this->_clientArea.Width(), static_cast<float>(vpw)) ||
        !vislib::math::IsEqual(this->_clientArea.Height(), static_cast<float>(vph))) {
        this->updateSize(vpw, vph);

        // is the following even needed here?
        if (call != nullptr) {
            auto gpu_call = dynamic_cast<view::CallRenderViewGL*>(call);
            gpu_call->GetFramebufferObject()->Enable();
        }
    }

    // Propagate viewport changes to connected views.
    // this cannot be done in a smart way currently since reconnects and early initialization
    // would skip propagating the data when called in updateSize
    auto propagateViewport = [](CallRenderViewGL* crv, vislib::math::Rectangle<float>& clientArea) {
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
    propagateViewport(this->render1(), this->_clientArea1);
    propagateViewport(this->render2(), this->_clientArea2);

    auto renderAndBlit = [&](std::shared_ptr<vislib::graphics::gl::FramebufferObject> fbo, CallRenderViewGL* crv,
                             const vislib::math::Rectangle<float>& ca) {
        if (crv == nullptr) {
            return;
        }
        crv->SetFramebufferObject(fbo);
        crv->SetInstanceTime(context.InstanceTime);
        crv->SetTime(-1.0f);

        if (this->_enableTimeSyncSlot.Param<param::BoolParam>()->Value()) {
            crv->SetTime(static_cast<float>(time));
        }

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
        (*crv)(CallRenderViewGL::CALL_RENDER);

#if defined(DEBUG) || defined(_DEBUG)
        vislib::Trace::GetInstance().SetLevel(0);
#endif /* DEBUG || _DEBUG */
        fbo->Disable();
#if defined(DEBUG) || defined(_DEBUG)
        vislib::Trace::GetInstance().SetLevel(otl);
#endif /* DEBUG || _DEBUG */

        if (call != nullptr) {
            auto gpu_call = dynamic_cast<view::CallRenderViewGL*>(call);
            gpu_call->GetFramebufferObject()->Enable();
        }

        // Bind and blit framebuffer.
        GLint binding, readBuffer;
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &binding);
        glGetIntegerv(GL_READ_BUFFER, &readBuffer);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo->GetID());
        glReadBuffer(GL_COLOR_ATTACHMENT0);
        glBlitFramebuffer(0, 0, fbo->GetWidth(), fbo->GetHeight(), ca.Left(), this->_clientArea.Height() - ca.Top(),
            ca.Right(), this->_clientArea.Height() - ca.Bottom(), GL_COLOR_BUFFER_BIT, GL_NEAREST);

        glBindFramebuffer(GL_READ_FRAMEBUFFER, binding);
        glReadBuffer(readBuffer);
    };

    // Draw the splitter through clearing without overplotting.
    auto splitColour = this->_splitColourSlot.Param<param::ColorParam>()->Value();
    ::glClearColor(splitColour[0], splitColour[1], splitColour[2], 1.0f);
    ::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    renderAndBlit(this->_fbo1, this->render1(), this->_clientArea1);
    renderAndBlit(this->_fbo2, this->render2(), this->_clientArea2);
}

bool view::SplitViewGL::GetExtents(core::Call& call) {
    if (this->_enableTimeSyncSlot.Param<param::BoolParam>()->Value()) {
        auto cr = this->render1();
        if (!(*cr)(CallRenderViewGL::CALL_EXTENTS)) return false;
        auto time = cr->TimeFramesCount();
        auto insitu = cr->IsInSituTime();
        cr = this->render2();
        if (!(*cr)(CallRenderViewGL::CALL_EXTENTS)) return false;
        time = std::min(time, cr->TimeFramesCount());
        insitu = insitu && cr->IsInSituTime();

        CallRenderViewGL* crv = dynamic_cast<CallRenderViewGL*>(&call);
        if (crv == nullptr) return false;
        crv->SetTimeFramesCount(time);
        crv->SetIsInSituTime(insitu);
    }
    return true;
}

void view::SplitViewGL::ResetView() {
    for (auto crv : {this->render1(), this->render2()}) {
        if (crv != nullptr) (*crv)(CallRenderViewGL::CALL_RESETVIEW);
    }
}

void view::SplitViewGL::Resize(unsigned int width, unsigned int height) {
    AbstractView::Resize(width,height);

    if (!vislib::math::IsEqual(this->_clientArea.Width(), static_cast<float>(width)) ||
        !vislib::math::IsEqual(this->_clientArea.Height(), static_cast<float>(height))) {
        this->updateSize(width, height);
    }
}

bool view::SplitViewGL::OnRenderView(Call& call) {
    auto* crv = dynamic_cast<view::CallRenderViewGL*>(&call);
    if (crv == nullptr) return false;

    mmcRenderViewContext context;
    ::ZeroMemory(&context, sizeof(context));
    context.Time = crv->Time();
    if (this->_enableTimeSyncSlot.Param<param::BoolParam>()->Value() && context.Time < 0.0) {
        context.Time = this->DefaultTime(crv->InstanceTime());
    }
    context.InstanceTime = crv->InstanceTime();
    this->Render(context, &call);

    return true;
}

void view::SplitViewGL::UpdateFreeze(bool freeze) {
    for (auto crv : {this->render1(), this->render2()}) {
        if (crv != nullptr) (*crv)(freeze ? CallRenderViewGL::CALL_FREEZE : CallRenderViewGL::CALL_UNFREEZE);
    }
}

bool view::SplitViewGL::OnKey(Key key, KeyAction action, Modifiers mods) {
    auto* crv = this->renderHovered();
    auto* crv1 = this->render1();
    auto* crv2 = this->render2();

    if (crv != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::Key;
        evt.keyData.key = key;
        evt.keyData.action = action;
        evt.keyData.mods = mods;

        if (this->_inputToBothSlot.Param<param::BoolParam>()->Value()) {
            crv1->SetInputEvent(evt);
            auto consumed = (*crv1)(view::CallRenderViewGL::FnOnKey);

            crv2->SetInputEvent(evt);
            consumed |= (*crv2)(view::CallRenderViewGL::FnOnKey);

            return consumed;
        } else {
            crv->SetInputEvent(evt);
            if (!(*crv)(view::CallRenderViewGL::FnOnKey)) return false;
        }
    }

    return false;
}

bool view::SplitViewGL::OnChar(unsigned int codePoint) {
    auto* crv = this->renderHovered();
    auto* crv1 = this->render1();
    auto* crv2 = this->render2();

    if (crv != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::Char;
        evt.charData.codePoint = codePoint;

        if (this->_inputToBothSlot.Param<param::BoolParam>()->Value()) {
            crv1->SetInputEvent(evt);
            auto consumed = (*crv1)(view::CallRenderViewGL::FnOnChar);

            crv2->SetInputEvent(evt);
            consumed |= (*crv2)(view::CallRenderViewGL::FnOnChar);

            return consumed;
        } else {
            crv->SetInputEvent(evt);
            if (!(*crv)(view::CallRenderViewGL::FnOnChar)) return false;
        }
    }

    return false;
}

bool view::SplitViewGL::OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) {
    auto* crv = this->renderHovered();
    auto* crv1 = this->render1();
    auto* crv2 = this->render2();

    this->_dragSplitter = false;

    auto down = (action == MouseButtonAction::PRESS);
    if (down && crv != crv1 && crv != crv2) {
        this->_dragSplitter = true;
    }

    if (crv != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseButton;
        evt.mouseButtonData.button = button;
        evt.mouseButtonData.action = action;
        evt.mouseButtonData.mods = mods;

        if (this->_inputToBothSlot.Param<param::BoolParam>()->Value()) {
            crv1->SetInputEvent(evt);
            auto consumed = (*crv1)(view::CallRenderViewGL::FnOnMouseButton);

            crv2->SetInputEvent(evt);
            consumed |= (*crv2)(view::CallRenderViewGL::FnOnMouseButton);

            return consumed;
        } else {
            crv->SetInputEvent(evt);
            if (!(*crv)(view::CallRenderViewGL::FnOnMouseButton)) return false;
        }
    }

    return false;
}


bool view::SplitViewGL::OnMouseMove(double x, double y) {
    // x, y are coordinates in pixel
    this->_mouseX = x;
    this->_mouseY = y;

    if (this->_dragSplitter) {
        if (this->_splitOrientationSlot.Param<param::EnumParam>()->Value() == HORIZONTAL) {
            this->_splitPositionSlot.Param<param::FloatParam>()->SetValue(x / this->_clientArea.Width());
        } else {
            this->_splitPositionSlot.Param<param::FloatParam>()->SetValue(y / this->_clientArea.Height());
        }
    }

    auto* crv = this->renderHovered();
    auto* crv1 = this->render1();
    auto* crv2 = this->render2();

    float mx;
    float my;

    if (crv == crv1) {
        mx = this->_mouseX - this->_clientArea1.Left();
        my = this->_mouseY - this->_clientArea1.Bottom();
    } else if (crv == crv2) {
        mx = this->_mouseX - this->_clientArea2.Left();
        my = this->_mouseY - this->_clientArea2.Bottom();
    } else {
        return false;
    }

    if (crv != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseMove;
        evt.mouseMoveData.x = mx;
        evt.mouseMoveData.y = my;

        if (this->_inputToBothSlot.Param<param::BoolParam>()->Value()) {
            crv1->SetInputEvent(evt);
            auto consumed = (*crv1)(view::CallRenderViewGL::FnOnMouseMove);

            crv2->SetInputEvent(evt);
            consumed |= (*crv2)(view::CallRenderViewGL::FnOnMouseMove);

            return consumed;
        } else {
            crv->SetInputEvent(evt);
            if (!(*crv)(view::CallRenderViewGL::FnOnMouseMove)) return false;
        }
    }

    return false;
}


bool view::SplitViewGL::OnMouseScroll(double dx, double dy) {
    auto* crv = this->renderHovered();
    auto* crv1 = this->render1();
    auto* crv2 = this->render2();

    if (crv != nullptr) {
        InputEvent evt;
        evt.tag = InputEvent::Tag::MouseScroll;
        evt.mouseScrollData.dx = dx;
        evt.mouseScrollData.dy = dy;

        if (this->_inputToBothSlot.Param<param::BoolParam>()->Value()) {
            crv1->SetInputEvent(evt);
            auto consumed = (*crv1)(view::CallRenderViewGL::FnOnMouseScroll);

            crv2->SetInputEvent(evt);
            consumed |= (*crv2)(view::CallRenderViewGL::FnOnMouseScroll);

            return consumed;
        } else {
            crv->SetInputEvent(evt);
            if (!(*crv)(view::CallRenderViewGL::FnOnMouseScroll)) return false;
        }
    }

    return false;
}

bool view::SplitViewGL::create() {
    this->_fbo1 = std::make_shared<vislib::graphics::gl::FramebufferObject>();
    this->_fbo2 = std::make_shared<vislib::graphics::gl::FramebufferObject>();
    return true;
}

void view::SplitViewGL::release() {
    if (this->_fbo1->IsValid()) this->_fbo1->Release();
    if (this->_fbo2->IsValid()) this->_fbo2->Release();
}

void view::SplitViewGL::unpackMouseCoordinates(float& x, float& y) {
    x *= this->_clientArea.Width();
    y *= this->_clientArea.Height();
}

void view::SplitViewGL::updateSize(size_t width, size_t height) {
    this->_clientArea.SetWidth(static_cast<float>(width));
    this->_clientArea.SetHeight(static_cast<float>(height));
    this->adjustClientAreas();

#if defined(DEBUG) || defined(_DEBUG)
    unsigned int otl = vislib::Trace::GetInstance().GetLevel();
    vislib::Trace::GetInstance().SetLevel(0);
#endif /* DEBUG || _DEBUG */
    if (this->_fbo1->IsValid()) this->_fbo1->Release();
    this->_fbo1->Create(
        static_cast<unsigned int>(this->_clientArea1.Width()), static_cast<unsigned int>(this->_clientArea1.Height()));
    this->_fbo1->Disable();

    if (this->_fbo2->IsValid()) this->_fbo2->Release();
    this->_fbo2->Create(
        static_cast<unsigned int>(this->_clientArea2.Width()), static_cast<unsigned int>(this->_clientArea2.Height()));
    this->_fbo2->Disable();
#if defined(DEBUG) || defined(_DEBUG)
    vislib::Trace::GetInstance().SetLevel(otl);
#endif /* DEBUG || _DEBUG */

}

void view::SplitViewGL::adjustClientAreas() {
    float sp = this->_splitPositionSlot.Param<param::FloatParam>()->Value();
    float shw = this->_splitWidthSlot.Param<param::FloatParam>()->Value() * 0.5f;
    auto so = static_cast<Orientation>(this->_splitOrientationSlot.Param<param::EnumParam>()->Value());
    this->_splitPositionSlot.ResetDirty();
    this->_splitWidthSlot.ResetDirty();
    this->_splitOrientationSlot.ResetDirty();

    if (so == HORIZONTAL) {
        this->_clientArea1.Set(this->_clientArea.Left(), this->_clientArea.Bottom(),
            this->_clientArea.Left() + this->_clientArea.Width() * sp - shw, this->_clientArea.Top());
        this->_clientArea2.Set(this->_clientArea.Left() + this->_clientArea.Width() * sp + shw, this->_clientArea.Bottom(),
            this->_clientArea.Right(), this->_clientArea.Top());
    } else {
        this->_clientArea1.Set(this->_clientArea.Left(), this->_clientArea.Bottom(), this->_clientArea.Right(),
            this->_clientArea.Bottom() + this->_clientArea.Height() * sp - shw);
        this->_clientArea2.Set(this->_clientArea.Left(), this->_clientArea.Bottom() + this->_clientArea.Height() * sp + shw,
            this->_clientArea.Right(), this->_clientArea.Top());
    }

    this->_clientArea1.EnforcePositiveSize();
    this->_clientArea2.EnforcePositiveSize();
}
