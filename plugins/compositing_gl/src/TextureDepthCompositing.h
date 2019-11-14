/*
 * TextureDepthCompositing.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef TEXTURE_DEPTH_COMPOSITING_H_INCLUDED
#define TEXTURE_DEPTH_COMPOSITING_H_INCLUDED

#include <memory>

#include "compositing/compositing_gl.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/graphics/gl/GLSLComputeShader.h"

#include "glowl/Texture2D.hpp"

namespace megamol {
namespace compositing {

class COMPOSITING_GL_API TextureDepthCompositing : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "TextureDepthCompositing";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Compositing module that combines two texture using depth aware alpha compositing.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() { 
        return true;
    }

    TextureDepthCompositing();
    ~TextureDepthCompositing();

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

    /** Shader program for texture add */
    std::unique_ptr<GLSLComputeShader> m_depthComp_prgm;

    /** Texture that the combination result will be written to */
    std::shared_ptr<glowl::Texture2D>  m_output_texture;

    /** Slot for requesting the output textures from this module, i.e. lhs connection */
    megamol::core::CalleeSlot          m_output_tex_slot;

    /** Slot for querying primary input texture, i.e. a rhs connection */
    megamol::core::CallerSlot          m_input_tex_0_slot;

    /** Slot for querying secondary input texture, i.e. a rhs connection */
    megamol::core::CallerSlot          m_input_tex_1_slot;

    /** Slot for querying primary depth texture, i.e. a rhs connection */
    megamol::core::CallerSlot          m_depth_tex_0_slot;

    /** Slot for querying secondary depth texture, i.e. a rhs connection */
    megamol::core::CallerSlot          m_depth_tex_1_slot;
};

} // namespace compositing
} // namespace megamol

#endif // !TEXTURE_DEPTH_COMPOSITING_H_INCLUDED
