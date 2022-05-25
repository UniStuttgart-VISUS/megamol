/**
 * MegaMol
 * Copyright (c) 2012, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd_gl/view/SplitViewGL.h"

#include "OpenGL_Context.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmstd_gl/renderer/CallRenderViewGL.h"
#include "vislib/Trace.h"
#include "vislib_gl/graphics/gl/FramebufferObject.h"

using namespace megamol;
using namespace megamol::mmstd_gl;
using megamol::core::utility::log::Log;

enum Orientation { HORIZONTAL = 0, VERTICAL = 1 };

view::SplitViewGL::SplitViewGL()
        : _render1Slot("render1", "Connects to the view 1 (left or top)")
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
        , _fboFull(nullptr)
        , _fbo1(nullptr)
        , _fbo2(nullptr)
        , _focus(0)
        , _mouseX(0.0)
        , _mouseY(0.0)
        , _dragSplitter(false) {

    _lhsRenderSlot.SetCallback(CallRenderViewGL::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnKey), &AbstractView::OnKeyCallback);
    _lhsRenderSlot.SetCallback(CallRenderViewGL::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnChar), &AbstractView::OnCharCallback);
    _lhsRenderSlot.SetCallback(CallRenderViewGL::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseButton),
        &AbstractView::OnMouseButtonCallback);
    _lhsRenderSlot.SetCallback(CallRenderViewGL::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseMove), &AbstractView::OnMouseMoveCallback);
    _lhsRenderSlot.SetCallback(CallRenderViewGL::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseScroll),
        &AbstractView::OnMouseScrollCallback);
    // AbstractCallRender
    _lhsRenderSlot.SetCallback(CallRenderViewGL::ClassName(),
        core::view::AbstractCallRender::FunctionName(core::view::AbstractCallRender::FnRender),
        &AbstractView::OnRenderView);
    _lhsRenderSlot.SetCallback(CallRenderViewGL::ClassName(),
        core::view::AbstractCallRender::FunctionName(core::view::AbstractCallRender::FnGetExtents),
        &AbstractView::GetExtents);
    // CallRenderViewGL
    _lhsRenderSlot.SetCallback(CallRenderViewGL::ClassName(),
        CallRenderViewGL::FunctionName(CallRenderViewGL::CALL_RESETVIEW), &AbstractView::OnResetView);
    MakeSlotAvailable(&_lhsRenderSlot);

    _render1Slot.SetCompatibleCall<CallRenderViewGLDescription>();
    MakeSlotAvailable(&_render1Slot);

    _render2Slot.SetCompatibleCall<CallRenderViewGLDescription>();
    MakeSlotAvailable(&_render2Slot);

    auto* orientations = new core::param::EnumParam(0);
    orientations->SetTypePair(HORIZONTAL, "Horizontal (side by side)");
    orientations->SetTypePair(VERTICAL, "Vertical");
    _splitOrientationSlot << orientations;
    MakeSlotAvailable(&_splitOrientationSlot);

    _splitPositionSlot << new core::param::FloatParam(0.5f, 0.0f, 1.0f);
    MakeSlotAvailable(&_splitPositionSlot);

    _splitWidthSlot << new core::param::IntParam(4, 0, 100);
    MakeSlotAvailable(&_splitWidthSlot);

    _splitColourSlot << new core::param::ColorParam(0.75f, 0.75f, 0.75f, 1.0f);
    MakeSlotAvailable(&_splitColourSlot);

    _enableTimeSyncSlot << new core::param::BoolParam(false);
    MakeSlotAvailable(&_enableTimeSyncSlot);

    _inputToBothSlot << new core::param::BoolParam(false);
    MakeSlotAvailable(&_inputToBothSlot);
}

view::SplitViewGL::~SplitViewGL(void) {
    Release();
}

float view::SplitViewGL::DefaultTime(double instTime) const {
    return _timeCtrl.Time(instTime);
}


core::view::ImageWrapper view::SplitViewGL::Render(double time, double instanceTime) {

    if (doHookCode()) {
        doBeforeRenderHook();
    }

    if (_enableTimeSyncSlot.Param<core::param::BoolParam>()->Value()) {
        auto cr = render1();
        (*cr)(CallRenderViewGL::CALL_EXTENTS);
        auto fcount = cr->TimeFramesCount();
        auto insitu = cr->IsInSituTime();
        cr = render2();
        (*cr)(CallRenderViewGL::CALL_EXTENTS);
        fcount = std::min(fcount, cr->TimeFramesCount());
        insitu = insitu && cr->IsInSituTime();

        _timeCtrl.SetTimeExtend(fcount, insitu);
        if (time > static_cast<float>(fcount)) {
            time = static_cast<float>(fcount);
        }
    }

    // float sp = splitPositionSlot.Param<param::FloatParam>()->Value();
    // float shw = splitWidthSlot.Param<param::FloatParam>()->Value() * 0.5f;
    // auto so = static_cast<Orientation>(splitOrientationSlot.Param<param::EnumParam>()->Value());
    // if (so == HORIZONTAL) {
    //    auto oc = overrideCall;
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

    if (_splitPositionSlot.IsDirty() || _splitOrientationSlot.IsDirty() || _splitWidthSlot.IsDirty() ||
        _fbo1 == nullptr || _fbo2 == nullptr || !vislib::math::IsEqual(_clientArea.Width(), _fboFull->getWidth()) ||
        !vislib::math::IsEqual(_clientArea.Height(), _fboFull->getHeight())) {
        updateSize(_fboFull->getWidth(), _fboFull->getHeight());
    }

    // Propagate viewport changes to connected views.
    // this cannot be done in a smart way currently since reconnects and early initialization
    // would skip propagating the data when called in updateSize
    auto propagateViewport = [](CallRenderViewGL* crv, vislib::math::Rectangle<int>& clientArea) {
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
    propagateViewport(render1(), _clientArea1);
    propagateViewport(render2(), _clientArea2);

    auto renderAndBlit = [&](std::shared_ptr<glowl::FramebufferObject> view_fbo,
                             std::shared_ptr<glowl::FramebufferObject> subview_fbo, CallRenderViewGL* crv,
                             const vislib::math::Rectangle<float>& ca) {
        if (crv == nullptr) {
            return;
        }
        crv->SetFramebuffer(subview_fbo);
        crv->SetInstanceTime(instanceTime);
        crv->SetTime(-1.0f);

        if (_enableTimeSyncSlot.Param<core::param::BoolParam>()->Value()) {
            crv->SetTime(static_cast<float>(time));
        }

        // Defer render call to subview that should clear (if it does not,
        // non-splitview rendering will be broken as well).
        (*crv)(CallRenderViewGL::CALL_RENDER);

        // TODO SplitViewGL need its own full view fbo

        // Bind and blit framebuffer.
        view_fbo->bind();

        subview_fbo->bindToRead(0);
        glBlitFramebuffer(0, 0, subview_fbo->getWidth(), subview_fbo->getHeight(), ca.Left(),
            _clientArea.Height() - ca.Top(), ca.Right(), _clientArea.Height() - ca.Bottom(), GL_COLOR_BUFFER_BIT,
            GL_NEAREST);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    };

    _fboFull->bind();
    // Draw the splitter through clearing without overplotting.
    auto splitColour = _splitColourSlot.Param<core::param::ColorParam>()->Value();
    ::glClearColor(splitColour[0], splitColour[1], splitColour[2], splitColour[3]);
    ::glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    renderAndBlit(_fboFull, _fbo1, render1(), _clientArea1);
    renderAndBlit(_fboFull, _fbo2, render2(), _clientArea2);

    return GetRenderingResult();
}

core::view::ImageWrapper view::SplitViewGL::GetRenderingResult() const {
    ImageWrapper::DataChannels channels =
        ImageWrapper::DataChannels::RGBA8; // vislib_gl::graphics::gl::FramebufferObject seems to use RGBA8
    unsigned int fbo_color_buffer_gl_handle =
        _fboFull->getColorAttachment(0)->getName(); // IS THIS SAFE?? IS THIS THE COLOR BUFFER??
    size_t fbo_width = _fboFull->getWidth();
    size_t fbo_height = _fboFull->getHeight();

    return frontend_resources::wrap_image({fbo_width, fbo_height}, fbo_color_buffer_gl_handle, channels);
}

bool view::SplitViewGL::GetExtents(core::Call& call) {
    if (_enableTimeSyncSlot.Param<core::param::BoolParam>()->Value()) {
        auto cr = render1();
        if (!(*cr)(CallRenderViewGL::CALL_EXTENTS))
            return false;
        auto time = cr->TimeFramesCount();
        auto insitu = cr->IsInSituTime();
        cr = render2();
        if (!(*cr)(CallRenderViewGL::CALL_EXTENTS))
            return false;
        time = std::min(time, cr->TimeFramesCount());
        insitu = insitu && cr->IsInSituTime();

        CallRenderViewGL* crv = dynamic_cast<CallRenderViewGL*>(&call);
        if (crv == nullptr)
            return false;
        crv->SetTimeFramesCount(time);
        crv->SetIsInSituTime(insitu);
    }
    return true;
}

void view::SplitViewGL::ResetView() {
    for (auto crv : {render1(), render2()}) {
        if (crv != nullptr)
            (*crv)(CallRenderViewGL::CALL_RESETVIEW);
    }
}

void view::SplitViewGL::Resize(unsigned int width, unsigned int height) {

    if ((width != _fboFull->getWidth() || height != _fboFull->getHeight() && width != 0 && height != 0)) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0); // better safe then sorry, "unbind" fbo before delting one
        try {
            _fboFull = std::make_shared<glowl::FramebufferObject>(width, height);
            _fboFull->createColorAttachment(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);

        } catch (glowl::BaseException const& exc) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[SplitView] Unable to create framebuffer object: %s\n", exc.what());
        }
    }

    if ((!vislib::math::IsEqual(_clientArea.Width(), static_cast<int>(width)) ||
            !vislib::math::IsEqual(_clientArea.Height(), static_cast<int>(height))) &&
        width != 0 && height != 0) {
        updateSize(width, height);
    }
}

bool view::SplitViewGL::OnRenderView(core::Call& call) {
    auto* crv = dynamic_cast<CallRenderViewGL*>(&call);
    if (crv == nullptr)
        return false;

    auto time = crv->Time();
    if (_enableTimeSyncSlot.Param<core::param::BoolParam>()->Value() && time < 0.0) {
        time = DefaultTime(crv->InstanceTime());
    }
    auto instanceTime = crv->InstanceTime();

    auto fbo = _fboFull;
    _fboFull = crv->GetFramebuffer();
    Render(time, instanceTime);
    _fboFull = fbo;

    return true;
}

bool view::SplitViewGL::OnKey(
    frontend_resources::Key key, frontend_resources::KeyAction action, frontend_resources::Modifiers mods) {
    auto* crv = renderHovered();
    auto* crv1 = render1();
    auto* crv2 = render2();

    if (crv != nullptr) {
        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::Key;
        evt.keyData.key = key;
        evt.keyData.action = action;
        evt.keyData.mods = mods;

        if (_inputToBothSlot.Param<core::param::BoolParam>()->Value()) {
            crv1->SetInputEvent(evt);
            auto consumed = (*crv1)(CallRenderViewGL::FnOnKey);

            crv2->SetInputEvent(evt);
            consumed |= (*crv2)(CallRenderViewGL::FnOnKey);

            return consumed;
        } else {
            crv->SetInputEvent(evt);
            if (!(*crv)(CallRenderViewGL::FnOnKey))
                return false;
        }
    }

    return false;
}

bool view::SplitViewGL::OnChar(unsigned int codePoint) {
    auto* crv = renderHovered();
    auto* crv1 = render1();
    auto* crv2 = render2();

    if (crv != nullptr) {
        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::Char;
        evt.charData.codePoint = codePoint;

        if (_inputToBothSlot.Param<core::param::BoolParam>()->Value()) {
            crv1->SetInputEvent(evt);
            auto consumed = (*crv1)(CallRenderViewGL::FnOnChar);

            crv2->SetInputEvent(evt);
            consumed |= (*crv2)(CallRenderViewGL::FnOnChar);

            return consumed;
        } else {
            crv->SetInputEvent(evt);
            if (!(*crv)(CallRenderViewGL::FnOnChar))
                return false;
        }
    }

    return false;
}

bool view::SplitViewGL::OnMouseButton(frontend_resources::MouseButton button,
    frontend_resources::MouseButtonAction action, frontend_resources::Modifiers mods) {
    auto* crv = renderHovered();
    auto* crv1 = render1();
    auto* crv2 = render2();

    _dragSplitter = false;

    auto down = (action == frontend_resources::MouseButtonAction::PRESS);
    if (down && crv != crv1 && crv != crv2) {
        _dragSplitter = true;
    }

    if (crv != nullptr) {
        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::MouseButton;
        evt.mouseButtonData.button = button;
        evt.mouseButtonData.action = action;
        evt.mouseButtonData.mods = mods;

        if (_inputToBothSlot.Param<core::param::BoolParam>()->Value()) {
            crv1->SetInputEvent(evt);
            auto consumed = (*crv1)(CallRenderViewGL::FnOnMouseButton);

            crv2->SetInputEvent(evt);
            consumed |= (*crv2)(CallRenderViewGL::FnOnMouseButton);

            return consumed;
        } else {
            crv->SetInputEvent(evt);
            if (!(*crv)(CallRenderViewGL::FnOnMouseButton))
                return false;
        }
    }

    return false;
}


bool view::SplitViewGL::OnMouseMove(double x, double y) {
    // x, y are coordinates in pixel
    _mouseX = x;
    _mouseY = y;

    if (_dragSplitter) {
        if (_splitOrientationSlot.Param<core::param::EnumParam>()->Value() == HORIZONTAL) {
            _splitPositionSlot.Param<core::param::FloatParam>()->SetValue(
                static_cast<float>(x) / static_cast<float>(_clientArea.Width()));
        } else {
            _splitPositionSlot.Param<core::param::FloatParam>()->SetValue(
                static_cast<float>(y) / static_cast<float>(_clientArea.Height()));
        }
    }

    auto* crv = renderHovered();
    auto* crv1 = render1();
    auto* crv2 = render2();

    double mx;
    double my;

    if (crv == crv1) {
        mx = _mouseX - _clientArea1.Left();
        my = _mouseY - _clientArea1.Bottom();
    } else if (crv == crv2) {
        mx = _mouseX - _clientArea2.Left();
        my = _mouseY - _clientArea2.Bottom();
    } else {
        return false;
    }

    if (crv != nullptr) {
        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::MouseMove;
        evt.mouseMoveData.x = mx;
        evt.mouseMoveData.y = my;

        if (_inputToBothSlot.Param<core::param::BoolParam>()->Value()) {
            crv1->SetInputEvent(evt);
            auto consumed = (*crv1)(CallRenderViewGL::FnOnMouseMove);

            crv2->SetInputEvent(evt);
            consumed |= (*crv2)(CallRenderViewGL::FnOnMouseMove);

            return consumed;
        } else {
            crv->SetInputEvent(evt);
            if (!(*crv)(CallRenderViewGL::FnOnMouseMove))
                return false;
        }
    }

    return false;
}


bool view::SplitViewGL::OnMouseScroll(double dx, double dy) {
    auto* crv = renderHovered();
    auto* crv1 = render1();
    auto* crv2 = render2();

    if (crv != nullptr) {
        core::view::InputEvent evt;
        evt.tag = core::view::InputEvent::Tag::MouseScroll;
        evt.mouseScrollData.dx = dx;
        evt.mouseScrollData.dy = dy;

        if (_inputToBothSlot.Param<core::param::BoolParam>()->Value()) {
            crv1->SetInputEvent(evt);
            auto consumed = (*crv1)(CallRenderViewGL::FnOnMouseScroll);

            crv2->SetInputEvent(evt);
            consumed |= (*crv2)(CallRenderViewGL::FnOnMouseScroll);

            return consumed;
        } else {
            crv->SetInputEvent(evt);
            if (!(*crv)(CallRenderViewGL::FnOnMouseScroll))
                return false;
        }
    }

    return false;
}

bool view::SplitViewGL::create() {
    auto const& ogl_ctx = frontend_resources.get<frontend_resources::OpenGL_Context>();
    if (!ogl_ctx.areExtAvailable(vislib_gl::graphics::gl::FramebufferObject::RequiredExtensions()))
        return false;

    _fboFull = std::make_shared<glowl::FramebufferObject>(1, 1);
    _fbo1 = std::make_shared<glowl::FramebufferObject>(1, 1);
    _fbo2 = std::make_shared<glowl::FramebufferObject>(1, 1);
    return true;
}

void view::SplitViewGL::release() {
    _fbo1.reset();
    _fbo2.reset();
}

void view::SplitViewGL::updateSize(size_t width, size_t height) {
    _clientArea.SetWidth(static_cast<int>(width));
    _clientArea.SetHeight(static_cast<int>(height));
    adjustClientAreas();

#if defined(DEBUG) || defined(_DEBUG)
    unsigned int otl = vislib::Trace::GetInstance().GetLevel();
    vislib::Trace::GetInstance().SetLevel(0);
#endif /* DEBUG || _DEBUG */

    if (width != 0 && height != 0) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0); // better safe then sorry, "unbind" fbo before delting one
        try {
            _fbo1 = std::make_shared<glowl::FramebufferObject>(
                static_cast<unsigned int>(_clientArea1.Width()), static_cast<unsigned int>(_clientArea1.Height()));
            _fbo1->createColorAttachment(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);

            _fbo2 = std::make_shared<glowl::FramebufferObject>(
                static_cast<unsigned int>(_clientArea2.Width()), static_cast<unsigned int>(_clientArea2.Height()));
            _fbo2->createColorAttachment(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);

            // TODO: check completness and throw if not?
        } catch (glowl::BaseException const& exc) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[SplitView] Unable to create framebuffer object: %s\n", exc.what());
        }
    }

#if defined(DEBUG) || defined(_DEBUG)
    vislib::Trace::GetInstance().SetLevel(otl);
#endif /* DEBUG || _DEBUG */
}

void view::SplitViewGL::adjustClientAreas() {
    const float splitPos = _splitPositionSlot.Param<core::param::FloatParam>()->Value();
    const int splitWidth = _splitWidthSlot.Param<core::param::IntParam>()->Value();
    const auto splitOrientation =
        static_cast<Orientation>(_splitOrientationSlot.Param<core::param::EnumParam>()->Value());
    _splitPositionSlot.ResetDirty();
    _splitWidthSlot.ResetDirty();
    _splitOrientationSlot.ResetDirty();

    if (splitOrientation == HORIZONTAL) {
        const int client1Width =
            std::clamp(static_cast<int>(std::lround(static_cast<float>(_clientArea.Width() - splitWidth) * splitPos)),
                0, _clientArea.Width() - splitWidth);
        _clientArea1.Set(
            _clientArea.Left(), _clientArea.Bottom(), _clientArea.Left() + client1Width, _clientArea.Top());
        _clientArea2.Set(_clientArea.Left() + client1Width + splitWidth, _clientArea.Bottom(), _clientArea.Right(),
            _clientArea.Top());
    } else {
        int client1Height =
            std::clamp(static_cast<int>(std::lround(static_cast<float>(_clientArea.Height() - splitWidth) * splitPos)),
                0, _clientArea.Height() - splitWidth);
        _clientArea1.Set(
            _clientArea.Left(), _clientArea.Bottom(), _clientArea.Right(), _clientArea.Bottom() + client1Height);
        _clientArea2.Set(_clientArea.Left(), _clientArea.Bottom() + client1Height + splitWidth, _clientArea.Right(),
            _clientArea.Top());
    }

    _clientArea1.EnforcePositiveSize();
    _clientArea2.EnforcePositiveSize();
}
