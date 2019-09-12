/*
 * ScreenSpaceEffect.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef SCREEN_SPACE_EFFECT_H_INCLUDED
#define SCREEN_SPACE_EFFECT_H_INCLUDED

#include "compositing/compositing_gl.h"

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/graphics/gl/GLSLComputeShader.h"

#include "glowl/BufferObject.hpp"
#include "glowl/Texture2D.hpp"

namespace megamol {
namespace compositing {

class ScreenSpaceEffect : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() { return "ScreenSpaceEffect"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() { return "Compositing module that compute a screen space effect"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() { return true; }

    ScreenSpaceEffect();
    ~ScreenSpaceEffect();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create();

    /**
     * Implementation of 'Release'.
     */
    void release();

    /**
     * TODO
     */
    bool getDataCallback(core::Call& caller);

    /**
     * TODO
     */
    bool getMetaDataCallback(core::Call& caller);

private:
    typedef vislib::graphics::gl::GLSLComputeShader GLSLComputeShader;

    /** Shader program for texture ssao */
    std::unique_ptr<GLSLComputeShader> m_ssao_prgm;

    /** Shader program for texture ssao */
    std::unique_ptr<GLSLComputeShader> m_ssao_blur_prgm;

    /** Shader program for texture ssao */
    std::unique_ptr<GLSLComputeShader> m_fxaa_prgm;

    /** GPU buffer object for making active (point)lights available in during shading pass */
    std::unique_ptr<glowl::BufferObject> m_ssao_samples;

    /** Texture with random ssao kernel rotation */
    std::shared_ptr<glowl::Texture2D> m_ssao_kernelRot_texture;

    /** Texture that the combination result will be written to */
    std::shared_ptr<glowl::Texture2D> m_output_texture;

    /** Texture that can store intermediate results for multi-pass effect, e.g. ssao with blur */
    std::shared_ptr<glowl::Texture2D> m_intermediate_texture;

    /** Hash value to keep track of update to the output texture */
    size_t m_output_texture_hash;

    /** Parameter for selecting the screen space effect that is computed, e.g. ssao, fxaa,... */
    megamol::core::param::ParamSlot m_mode;

    /** Parameter for selecting the ssao radius */
    megamol::core::param::ParamSlot m_ssao_radius;

    /** Parameter for selecting the ssao sample count */
    megamol::core::param::ParamSlot m_ssao_sample_cnt;

    /** Slot for requesting the output textures from this module, i.e. lhs connection */
    megamol::core::CalleeSlot m_output_tex_slot;

    /** Slot for optionally querying an input texture, i.e. a rhs connection */
    megamol::core::CallerSlot m_input_tex_slot;

    /** Slot for querying normals render target texture, i.e. a rhs connection */
    megamol::core::CallerSlot m_normals_tex_slot;

    /** Slot for querying depth render target texture, i.e. a rhs connection */
    megamol::core::CallerSlot m_depth_tex_slot;

    /** Slot for querying camera, i.e. a rhs connection */
    megamol::core::CallerSlot m_camera_slot;
};

} // namespace compositing
} // namespace megamol

#endif // !SCREEN_SPACE_EFFECT_H_INCLUDED