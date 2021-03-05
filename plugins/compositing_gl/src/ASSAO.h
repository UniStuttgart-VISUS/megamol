/*
 * ASSAO.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef ASSAO_H_INCLUDED
#define ASSAO_H_INCLUDED

#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/graphics/gl/GLSLComputeShader.h"

#define GLOWL_OPENGL_INCLUDE_GLAD
#include "glowl/BufferObject.hpp"
#include "glowl/Texture2D.hpp"

namespace megamol {
namespace compositing {

class ASSAO : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() { return "ASSAO"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() { return "Compositing module that compute adaptive screen space ambient occlusion"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() { return true; }

    ASSAO();
    ~ASSAO();

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
    void updateTextures();

    typedef vislib::graphics::gl::GLSLComputeShader GLSLComputeShader;

    uint32_t m_version;

    /////////////////////////////////////////////////////////////////////////
    // COMPUTE SHADER BATTERY
    /////////////////////////////////////////////////////////////////////////
    std::unique_ptr<GLSLComputeShader> m_prepapre_depths_prgm;
    std::unique_ptr<GLSLComputeShader> m_prepare_depths_half_prgm;
    std::unique_ptr<GLSLComputeShader> m_prepare_depths_and_normals_prgm;
    std::unique_ptr<GLSLComputeShader> m_prepare_depths_and_normals_half_prgm;
    std::unique_ptr<GLSLComputeShader> m_prepare_depth_mip1_prgm;
    std::unique_ptr<GLSLComputeShader> m_prepare_depth_mip2_prgm;
    std::unique_ptr<GLSLComputeShader> m_prepare_depth_mip3_prgm;
    std::unique_ptr<GLSLComputeShader> m_generate_q0_prgm;
    std::unique_ptr<GLSLComputeShader> m_generate_q1_prgm;
    std::unique_ptr<GLSLComputeShader> m_generate_q2_prgm;
    std::unique_ptr<GLSLComputeShader> m_smart_blur_prgm;
    std::unique_ptr<GLSLComputeShader> m_smart_blur_wide_prgm;
    std::unique_ptr<GLSLComputeShader> m_apply_prgm;
    std::unique_ptr<GLSLComputeShader> m_non_smart_blur_prgm;
    std::unique_ptr<GLSLComputeShader> m_non_smart_apply_prgm;
    std::unique_ptr<GLSLComputeShader> m_non_smart_half_apply_prgm;
    /////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////
    // TEXTURE BATTERY
    /////////////////////////////////////////////////////////////////////////
    std::shared_ptr<glowl::Texture2D> m_halfDepths[4];
    std::vector<std::vector<std::shared_ptr<glowl::Texture2D>>> m_halfDepthsMipViews;
    std::shared_ptr<glowl::Texture2D> m_pingPongHalfResultA;
    std::shared_ptr<glowl::Texture2D> m_pingPongHalfResultB;
    std::shared_ptr<glowl::Texture2D> m_finalResults;
    std::shared_ptr<glowl::Texture2D> m_finalResultsArrayViews[4];
    std::shared_ptr<glowl::Texture2D> m_normals;

    glowl::TextureLayout m_tx_layout_samplerStatePointClamp;
    glowl::TextureLayout m_tx_layout_samplerStatePointMirror;
    glowl::TextureLayout m_tx_layout_samplerStateLinearClamp;
    glowl::TextureLayout m_tx_layout_samplerStateViewspaceDepthTap;
    /////////////////////////////////////////////////////////////////////////

    /** Hash value to keep track of update to the output texture */
    size_t m_output_texture_hash;

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

#endif // !ASSAO_H_INCLUDED
