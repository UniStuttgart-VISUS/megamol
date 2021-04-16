/*
 * ContextToGL.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once
#include "mmcore/view/CallRender3D.h"
#include "mmcore/view/RenderUtils.h"
#include "mmcore/view/Renderer3DModuleGL.h"
#include "vislib/graphics/gl/GLSLShader.h"

namespace megamol::core::view {

template<typename FBO>
using INITFUNC = void(std::shared_ptr<vislib::graphics::gl::FramebufferObject>&, std::shared_ptr<FBO>&, int, int);

template<typename FBO>
using RENFUNC = void(
    std::shared_ptr<vislib::graphics::gl::FramebufferObject>&, std::shared_ptr<FBO>&, RenderUtils&, int, int);

template<typename CALL, INITFUNC<typename CALL::FBO_TYPE> init_func, RENFUNC<typename CALL::FBO_TYPE> ren_func,
    char const* CN, char const* DESC>
class ContextToGL : public Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return CN;
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return DESC;
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable();
    }

    /** Ctor. */
    ContextToGL(void) : Renderer3DModuleGL(), _getContextSlot("getContext", "Slot for non-GL context") {

        this->_getContextSlot. template SetCompatibleCall<core::factories::CallAutoDescription<CALL>>();
        this->MakeSlotAvailable(&this->_getContextSlot);
    }

    /** Dtor. */
    virtual ~ContextToGL(void) {
        this->Release();
    }

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void) {
        if (!_utils.isInitialized()) {
            if (!_utils.InitPrimitiveRendering(this->GetCoreInstance()->ShaderSourceFactory())) {
                core::utility::log::Log::DefaultLog.WriteError("[ContextToGL] Unable to initialize RenderUtility.");
            }
        }

        return true;
    }

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void) {}

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(CallRender3DGL& call);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(CallRender3DGL& call);

private:
    core::CallerSlot _getContextSlot;

    std::shared_ptr<typename CALL::FBO_TYPE> _framebuffer;

    RenderUtils _utils;

    glm::uvec2 viewport = {0, 0};
};

template<typename CALL, INITFUNC<typename CALL::FBO_TYPE> init_func, RENFUNC<typename CALL::FBO_TYPE> ren_func,
    char const* CN, char const* DESC>
bool ContextToGL<CALL, init_func, ren_func, CN, DESC>::GetExtents(CallRender3DGL& call) {

    auto cr = _getContextSlot.CallAs<CALL>();
    if (cr == nullptr)
        return false;
    // no copy constructor available
    auto cast_in = dynamic_cast<AbstractCallRender*>(&call);
    auto cast_out = dynamic_cast<AbstractCallRender*>(cr);
    *cast_out = *cast_in;

    (*cr)(CALL::FnGetExtents);

    call.AccessBoundingBoxes() = cr->AccessBoundingBoxes();
    call.SetTimeFramesCount(cr->TimeFramesCount());

    return true;
}

template<typename CALL, INITFUNC<typename CALL::FBO_TYPE> init_func, RENFUNC<typename CALL::FBO_TYPE> ren_func,
    char const* CN, char const* DESC>
bool ContextToGL<CALL, init_func, ren_func, CN, DESC>::Render(CallRender3DGL& call) {

    auto cr = _getContextSlot.CallAs<CALL>();
    if (cr == nullptr)
        return false;
    // no copy constructor available
    auto cast_in = dynamic_cast<AbstractCallRender*>(&call);
    auto cast_out = dynamic_cast<AbstractCallRender*>(cr);
    *cast_out = *cast_in;

    Camera_2 cam;
    call.GetCamera(cam);

    auto width = cam.resolution_gate().width();
    auto height = cam.resolution_gate().height();

    auto lhs_fbo = call.GetFramebufferObject();

    if (!_framebuffer || width != viewport.x || height != viewport.y) {
        init_func(lhs_fbo, _framebuffer, width, height);
        viewport = {width, height};
    }
    cr->SetFramebuffer(_framebuffer);
    cr->SetInputEvent(call.GetInputEvent());

    auto const& ie = cr->GetInputEvent();
    switch (ie.tag) {
    case InputEvent::Tag::Char: {
        (*cr)(CALL::FnOnChar);
    } break;
    case InputEvent::Tag::Key: {
        (*cr)(CALL::FnOnKey);
    } break;
    case InputEvent::Tag::MouseButton: {
        (*cr)(CALL::FnOnMouseButton);
    } break;
    case InputEvent::Tag::MouseMove: {
        (*cr)(CALL::FnOnMouseMove);
    } break;
    case InputEvent::Tag::MouseScroll: {
        (*cr)(CALL::FnOnMouseScroll);
    } break;
    }

    (*cr)(CALL::FnRender);

    if (lhs_fbo != nullptr) {

        ren_func(lhs_fbo, _framebuffer, _utils, width, height);

    } else {
        return false;
    }
    return true;
}

} // namespace megamol::core::view
