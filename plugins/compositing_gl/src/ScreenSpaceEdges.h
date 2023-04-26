/*
 * EdgeOutline.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include <memory>

#include <glowl/glowl.h>

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_gl/ModuleGL.h"

namespace megamol::compositing_gl {

class ScreenSpaceEdges : public mmstd_gl::ModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "ScreenSpaceEdges";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "...";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    ScreenSpaceEdges();
    ~ScreenSpaceEdges() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * TODO
     */
    bool getDataCallback(core::Call& caller);

    /**
     * TODO
     */
    bool getMetaDataCallback(core::Call& caller);

    /**
     * Callback for mode switch that sets parameter visibility
     */
    bool modeCallback(core::param::ParamSlot& slot);

private:
    uint32_t m_version;

    /** Shader program for texture add */
    std::unique_ptr<glowl::GLSLProgram> m_edge_outline_prgm;

    /** Texture that the combination result will be written to */
    std::shared_ptr<glowl::Texture2D> m_output_texture;

    /** Parameter for setting a weight in additive mode */
    megamol::core::param::ParamSlot m_depth_threshold, m_normal_threshold;

    /** Slot for requesting the output textures from this module, i.e. lhs connection */
    megamol::core::CalleeSlot m_output_tex_slot;

    /** Slot for querying primary input texture, i.e. a rhs connection */
    megamol::core::CallerSlot m_depth_tex_slot;

    /** Slot for querying secondary input texture, i.e. a rhs connection */
    megamol::core::CallerSlot m_normal_tex_slot;

    /** Slot for querying camera, i.e. a rhs connection */
    megamol::core::CallerSlot camera_slot_;

    /** Parameter for choosing texture formats used within this module*/
    megamol::core::param::ParamSlot out_texture_format_slot_;

    /**
     * \brief Sets Texture format variables and recompiles shaders.
     */
    bool setTextureFormatCallback(core::param::ParamSlot& slot);

    /** Texture format variables*/
    int out_tex_internal_format_ = GL_RGBA32F;
    int out_tex_format_ = GL_RGBA;
    int out_tex_type_ = GL_FLOAT;
};

} // namespace megamol::compositing_gl
