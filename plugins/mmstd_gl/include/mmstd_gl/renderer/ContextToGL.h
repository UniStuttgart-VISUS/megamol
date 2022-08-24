/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <glowl/FramebufferObject.hpp>

#include "mmcore/CoreInstance.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmstd/renderer/CallRender3D.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"

namespace megamol::mmstd_gl {

template<typename FBO>
using INITFUNC = void(std::shared_ptr<glowl::FramebufferObject>&, std::shared_ptr<FBO>&, int, int);

template<typename FBO>
using RENFUNC = void(
    std::shared_ptr<glowl::GLSLProgram>&, std::shared_ptr<glowl::FramebufferObject>&, std::shared_ptr<FBO>&, int, int);

template<typename CALL, INITFUNC<typename CALL::FBO_TYPE> init_func, RENFUNC<typename CALL::FBO_TYPE> ren_func,
    char const* CN, char const* DESC>
class ContextToGL : public mmstd_gl::Renderer3DModuleGL {
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
    ContextToGL() : mmstd_gl::Renderer3DModuleGL(), _getContextSlot("getContext", "Slot for non-GL context") {

        this->_getContextSlot.template SetCompatibleCall<core::factories::CallAutoDescription<CALL>>();
        this->MakeSlotAvailable(&this->_getContextSlot);
        _getContextSlot.SetNecessity(megamol::core::AbstractCallSlotPresentation::SLOT_REQUIRED);
    }

    /** Dtor. */
    ~ContextToGL() override {
        this->Release();
    }

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override {
        try {
            auto const shdr_cp_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());

            shader_ = core::utility::make_shared_glowl_shader("simple_compositing", shdr_cp_options,
                std::filesystem::path("core/simple_compositing.vert.glsl"),
                std::filesystem::path("core/simple_compositing.frag.glsl"));
        } catch (glowl::GLSLProgramException const& ex) {
            megamol::core::utility::log::Log::DefaultLog.WriteError("[ContextToGL] %s", ex.what());
        } catch (...) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[ContextToGL] Unable to compile shader: Unknown exception");
            return false;
        }

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
    bool GetExtents(CallRender3DGL& call) override;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(CallRender3DGL& call) override;

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

    std::shared_ptr<glowl::GLSLProgram> shader_;
};

template<typename CALL, INITFUNC<typename CALL::FBO_TYPE> init_func, RENFUNC<typename CALL::FBO_TYPE> ren_func,
    char const* CN, char const* DESC>
bool ContextToGL<CALL, init_func, ren_func, CN, DESC>::GetExtents(CallRender3DGL& call) {

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
bool ContextToGL<CALL, init_func, ren_func, CN, DESC>::Render(CallRender3DGL& call) {

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

        ren_func(shader_, lhs_fbo, _framebuffer, width, height);

    } else {
        return false;
    }
    return true;
}

inline void renderToFBO(std::shared_ptr<glowl::GLSLProgram>& shader, std::shared_ptr<glowl::FramebufferObject>& lhs_fbo,
    GLuint color_tex, GLuint depth_tex, int width, int height) {
    // draw into lhs fbo
    if ((lhs_fbo->getWidth() != width) || (lhs_fbo->getHeight() != height)) {
        lhs_fbo->resize(width, height);
    }
    lhs_fbo->bind();
    shader->use();
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, color_tex);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, depth_tex);
    shader->setUniform("viewport", glm::vec2(width, height));
    shader->setUniform("color_tex", 0);
    shader->setUniform("depth_tex", 1);
    glm::mat4 ortho = glm::ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -1.0f, 1.0f);
    shader->setUniform("mvp", ortho);

    glDrawArrays(GL_QUADS, 0, 4);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);

    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

} // namespace megamol::mmstd_gl
