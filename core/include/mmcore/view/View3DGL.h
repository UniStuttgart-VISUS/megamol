/*
 * View3DGL.h
 *
 * Copyright (C) 2018, 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once
#include "mmcore/view/AbstractView3D.h"
#include "vislib/graphics/Cursor2D.h"

#include "mmcore/view/CameraControllers.h"
#include "mmcore/view/CameraParameterSlots.h"

#define GLOWL_OPENGL_INCLUDE_GLAD
#include "glowl/FramebufferObject.hpp"

namespace megamol {
namespace core {
namespace view {

    inline constexpr auto gl3D_fbo_create_or_resize = [](std::shared_ptr<glowl::FramebufferObject>& fbo, int width,
                                              int height) -> void {
        bool create_fbo = false;
        if (fbo == nullptr) {
            create_fbo = true;
        } else if ((fbo->getWidth() != width) || (fbo->getHeight() != height)) {
            create_fbo = true;
        }

        if (create_fbo) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0); // better safe then sorry, "unbind" fbo before delting one
            try {
                fbo = std::make_shared<glowl::FramebufferObject>(width, height);
                fbo->createColorAttachment(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
                // TODO: check completness and throw if not?
            } catch (glowl::FramebufferObjectException const& exc) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[View3DGL] Unable to create framebuffer object: %s\n", exc.what());
            }
        }
    };

class MEGAMOLCORE_API View3DGL : public view::AbstractView3D<glowl::FramebufferObject, gl3D_fbo_create_or_resize, Camera3DController, Camera3DParameters> {

public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "View3DGL"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "New and improved 3D View Module"; }

    /** Ctor. */
    View3DGL(void);

    /** Dtor. */
    virtual ~View3DGL(void);

    virtual ImageWrapper Render(double time, double instanceTime, bool present_fbo) override;

    ImageWrapper GetRenderingResult() const override;

    /**
     * Resets the view. This normally sets the camera parameters to
     * default values.
     */
    void ResetView();

    /**
     * Callback requesting a rendering of this view
     *
     * @param call The calling call
     *
     * @return The return value
     */
    virtual bool OnRenderView(Call& call) override;

protected:

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);
};

} // namespace view
} /* end namespace core */
} /* end namespace megamol */

