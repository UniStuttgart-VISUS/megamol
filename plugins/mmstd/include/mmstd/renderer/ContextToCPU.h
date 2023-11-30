/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmstd/renderer/CallRender3D.h"
#include "mmstd/renderer/Renderer3DModule.h"

namespace megamol::mmstd {

template<typename FBO>
using INITFUNC = void(std::shared_ptr<core::view::CPUFramebuffer> const&, std::shared_ptr<FBO>&, int, int);

template<typename FBO>
using RENFUNC = void(std::shared_ptr<core::view::CPUFramebuffer>&, std::shared_ptr<FBO>&, int, int);

template<typename CALL, INITFUNC<typename CALL::FBO_TYPE> init_func, RENFUNC<typename CALL::FBO_TYPE> ren_func,
    char const* CN, char const* DESC>
class ContextToCPU : public core::view::Renderer3DModule {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return CN;
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return DESC;
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    ContextToCPU() : core::view::Renderer3DModule(), _getContextSlot("getContext", "Slot for non-GL context") {

        this->_getContextSlot.template SetCompatibleCall<core::factories::CallAutoDescription<CALL>>();
        this->MakeSlotAvailable(&this->_getContextSlot);
    }

    /** Dtor. */
    ~ContextToCPU() override {
        this->Release();
    }

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override {
        return true;
    }

    /**
     * Implementation of 'Release'.
     */
    void release() override {}

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(core::view::CallRender3D& call) override;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(core::view::CallRender3D& call) override;

    bool OnChar(unsigned codePoint) override {
        auto* ci = this->_getContextSlot.template CallAs<core::view::InputCall>();
        if (ci != nullptr) {
            core::view::InputEvent evt;
            evt.tag = core::view::InputEvent::Tag::Char;
            evt.charData.codePoint = codePoint;
            ci->SetInputEvent(evt);
            return (*ci)(core::view::InputCall::FnOnChar);
        }
        return false;
    }

    bool OnKey(frontend_resources::Key key, frontend_resources::KeyAction action,
        frontend_resources::Modifiers mods) override {
        auto* ci = this->_getContextSlot.template CallAs<core::view::InputCall>();
        if (ci != nullptr) {
            core::view::InputEvent evt;
            evt.tag = core::view::InputEvent::Tag::Key;
            evt.keyData.key = key;
            evt.keyData.action = action;
            evt.keyData.mods = mods;
            ci->SetInputEvent(evt);
            return (*ci)(core::view::InputCall::FnOnKey);
        }
        return false;
    }

    bool OnMouseButton(frontend_resources::MouseButton button, frontend_resources::MouseButtonAction action,
        frontend_resources::Modifiers mods) override {
        auto* ci = this->chainRenderSlot.template CallAs<core::view::InputCall>();
        if (ci != nullptr) {
            core::view::InputEvent evt;
            evt.tag = core::view::InputEvent::Tag::MouseButton;
            evt.mouseButtonData.button = button;
            evt.mouseButtonData.action = action;
            evt.mouseButtonData.mods = mods;
            ci->SetInputEvent(evt);
            return (*ci)(core::view::InputCall::FnOnMouseButton);
        }
        return false;
    }

    bool OnMouseMove(double x, double y) override {
        auto* ci = this->chainRenderSlot.template CallAs<core::view::InputCall>();
        if (ci != nullptr) {
            core::view::InputEvent evt;
            evt.tag = core::view::InputEvent::Tag::MouseMove;
            evt.mouseMoveData.x = x;
            evt.mouseMoveData.y = y;
            ci->SetInputEvent(evt);
            return (*ci)(core::view::InputCall::FnOnMouseMove);
        }
        return false;
    }

    bool OnMouseScroll(double dx, double dy) override {
        auto* ci = this->chainRenderSlot.template CallAs<core::view::InputCall>();
        if (ci != nullptr) {
            core::view::InputEvent evt;
            evt.tag = core::view::InputEvent::Tag::MouseScroll;
            evt.mouseScrollData.dx = dx;
            evt.mouseScrollData.dy = dy;
            ci->SetInputEvent(evt);
            return (*ci)(core::view::InputCall::FnOnMouseScroll);
        }
        return false;
    }

private:
    core::CallerSlot _getContextSlot;

    std::shared_ptr<typename CALL::FBO_TYPE> _framebuffer;

    glm::uvec2 viewport = {0, 0};
};

template<typename CALL, INITFUNC<typename CALL::FBO_TYPE> init_func, RENFUNC<typename CALL::FBO_TYPE> ren_func,
    char const* CN, char const* DESC>
bool ContextToCPU<CALL, init_func, ren_func, CN, DESC>::GetExtents(core::view::CallRender3D& call) {

    auto cr = _getContextSlot.CallAs<CALL>();
    if (cr == nullptr)
        return false;
    // no copy constructor available
    auto cast_in = dynamic_cast<core::view::AbstractCallRender*>(&call);
    auto cast_out = dynamic_cast<core::view::AbstractCallRender*>(cr);
    *cast_out = *cast_in;

    (*cr)(CALL::FnGetExtents);

    call.AccessBoundingBoxes() = cr->AccessBoundingBoxes();
    call.SetTimeFramesCount(cr->TimeFramesCount());

    return true;
}

template<typename CALL, INITFUNC<typename CALL::FBO_TYPE> init_func, RENFUNC<typename CALL::FBO_TYPE> ren_func,
    char const* CN, char const* DESC>
bool ContextToCPU<CALL, init_func, ren_func, CN, DESC>::Render(core::view::CallRender3D& call) {

    auto cr = _getContextSlot.CallAs<CALL>();
    if (cr == nullptr)
        return false;
    // no copy constructor available
    auto cast_in = dynamic_cast<core::view::AbstractCallRender*>(&call);
    auto cast_out = dynamic_cast<core::view::AbstractCallRender*>(cr);
    *cast_out = *cast_in;

    auto lhs_fbo = call.GetFramebuffer();
    auto width = lhs_fbo->getWidth();
    auto height = lhs_fbo->getHeight();

    if (!_framebuffer || width != viewport.x || height != viewport.y) {
        init_func(lhs_fbo, _framebuffer, width, height);
        viewport = {width, height};
    }
    cr->SetFramebuffer(_framebuffer);

    (*cr)(CALL::FnRender);

    if (lhs_fbo != nullptr) {

        ren_func(lhs_fbo, _framebuffer, width, height);

    } else {
        return false;
    }
    return true;
}

} // namespace megamol::mmstd
