#pragma once

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore_gl/view/Renderer2DModuleGL.h"

#include "glowl/GLSLProgram.hpp"

namespace megamol::benchmark_gl {
class DrawScalarTexture : public core_gl::view::Renderer2DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() {
        return "DrawScalarTexture";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() {
        return "DrawScalarTexture";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable() {
        return true;
    }

    /**
     * Initialises a new instance.
     */
    DrawScalarTexture();

    /**
     * Finalises an instance.
     */
    virtual ~DrawScalarTexture();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create() override;

    /**
     * Implementation of 'Release'.
     */
    virtual void release() override;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool Render(core_gl::view::CallRender2DGL& call) override;

    /**
     * The extent callback.
     *
     * @param call The calling call.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool GetExtents(core_gl::view::CallRender2DGL& call) override;

    /**
     * Forwards key events.
     */
    virtual bool OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) override;

    /**
     * Forwards character events.
     */
    virtual bool OnChar(unsigned int codePoint) override;

    /**
     * Forwards character events.
     */
    virtual bool OnMouseButton(
        core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) override;

    /**
     * Forwards character events.
     */
    virtual bool OnMouseMove(double x, double y) override;

    /**
     * Forwards scroll events.
     */
    virtual bool OnMouseScroll(double dx, double dy) override;

private:
    bool updateTransferFunction();

    core::CallerSlot tex_in_slot_;

    core::CallerSlot tf_slot_;

    std::unique_ptr<glowl::GLSLProgram> shader_;

    GLuint tf_texture;
};
} // namespace megamol::benchmark_gl
