/*
 * extract_model_matrix.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/Renderer3DModule_2.h"

#include "glm/mat4x4.hpp"

namespace megamol {
namespace flowvis {

/**
 * Module for drawing a two-dimensional texture on a quad in 3D space.
 *
 * @author Alexander Straub
 */
class extract_model_matrix : public core::view::Renderer3DModule_2 {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() { return "extract_model_matrix"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() { return "Extract the model matrix from a view"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable() { return true; }

    /**
     * Initialises a new instance.
     */
    extract_model_matrix();

    /**
     * Finalises an instance.
     */
    virtual ~extract_model_matrix();

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

    /** Callbacks for the computed streamlines */
    virtual bool GetExtents(core::view::CallRender3D_2& call) override;
    virtual bool Render(core::view::CallRender3D_2& call) override;

private:
    /** Provide the model matrix to incoming call */
    bool get_matrix_callback(core::Call& call);

    /** Output slot for providing the model matrix */
    core::CalleeSlot model_matrix_slot;

    /** Model matrix */
    glm::mat4 model_matrix;
    glm::mat4 inverse_initial_model_matrix;

    /** Struct for storing data needed for rendering */
    struct render_data_t {
        GLuint vs, fs, prog;
    } render_data;

    /** Initialization status */
    bool initialized;
};

} // namespace flowvis
} // namespace megamol
