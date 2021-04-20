/*
 * View2DGL.h
 *
 * Copyright (C) 2008 - 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_VIEW2DGL_H_INCLUDED
#define MEGAMOLCORE_VIEW2DGL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/BaseView.h"
#include "mmcore/view/TimeControl.h"

#include "mmcore/view/CameraControllers.h"
#include "mmcore/view/CameraParameterSlots.h"

#define GLOWL_OPENGL_INCLUDE_GLAD
#include <glowl/FramebufferObject.hpp>

namespace megamol {
namespace core {
namespace view {

/*
 * Forward declaration of incoming render calls
 */
class CallRenderViewGL;

//TODO share this function with View3DGL
inline constexpr auto gl2D_fbo_create_or_resize = [](std::shared_ptr<glowl::FramebufferObject>& fbo, int width,
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
                "[View2DGL] Unable to create framebuffer object: %s\n", exc.what());
        }
    }
};

/**
 * Base class of rendering graph calls
 */
class View2DGL : public BaseView<glowl::FramebufferObject, gl2D_fbo_create_or_resize, Camera2DController, Camera2DParameters> {
public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "View2DGL";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "2D View Module";
    }

    /** Ctor. */
    View2DGL(void);

    /** Dtor. */
    virtual ~View2DGL(void);

    /**
     * Answer the camera synchronization number.
     *
     * @return The camera synchronization number
     */
    virtual unsigned int GetCameraSyncNumber(void) const;

    /**
     * ...
     */
    virtual ImageWrapper Render(double time, double instanceTime, bool present_fbo) override;

    /**
     * Resets the view. This normally sets the camera parameters to
     * default values.
     */
    virtual void ResetView(void);

    /**
     * Resizes the View2DGl framebuffer object.
     *
     * @param width The new width.
     * @param height The new height.
     */
    virtual void Resize(unsigned int width, unsigned int height) override;

    /**
     * Callback requesting a rendering of this view
     *
     * @param call The calling call
     *
     * @return The return value
     */
    virtual bool OnRenderView(Call& call);

    virtual bool GetExtents(Call& call) override;

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

private:

    /** the update counter for the view settings */
    unsigned int _viewUpdateCnt;
};
} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_VIEW2DGL_H_INCLUDED */
