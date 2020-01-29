/*
 * draw_to_texture.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "vislib/math/Rectangle.h"

#include "glowl/FramebufferObject.hpp"

#include <memory>

namespace megamol {
namespace flowvis {

/**
 * Module for drawing to a two-dimensional texture.
 *
 * @author Alexander Straub
 */
class draw_to_texture : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() { return "draw_to_texture"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() { return "Draw into a 2D texture"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable() { return true; }

    /**
     * Initialises a new instance.
     */
    draw_to_texture();

    /**
     * Finalises an instance.
     */
    virtual ~draw_to_texture();

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

private:
    /** Functions for getting extents and data from input calls */
    bool get_input_extent();

    /** Callbacks for the texture */
    bool get_data(core::Call& call);
    bool get_extent(core::Call& call);

    /** Callbacks for the model matrix */
    bool get_matrix(core::Call& call);

    /** Output slot for the texture */
    core::CalleeSlot texture_slot;

    /** Output slot for the matrix defining the position in world space */
    core::CalleeSlot model_matrix_slot;

    /** Input slot for getting the rendering */
    core::CallerSlot rendering_slot;

    /** Parameter for setting up the texture */
    core::param::ParamSlot width;
    core::param::ParamSlot height;
    core::param::ParamSlot keep_aspect_ratio;

    /** Bounding rectangle */
    vislib::math::Rectangle<float> bounding_rectangle;

    /** FBO */
    std::unique_ptr<glowl::FramebufferObject> fbo;

    /** Hash dummy */
    SIZE_T hash;
};

} // namespace flowvis
} // namespace megamol
