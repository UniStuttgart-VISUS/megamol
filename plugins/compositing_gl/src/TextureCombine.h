/*
 * TextureCombine.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef TEXTURE_COMBINE_H_INCLUDED
#define TEXTURE_COMBINE_H_INCLUDED

#include <memory>

#include "compositing/compositing_gl.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/graphics/gl/GLSLComputeShader.h"

#include "glowl/Texture2D.hpp"

namespace megamol {
namespace compositing {

class COMPOSITING_GL_API TextureCombine : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "TextureCombine";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Compositing module that combines two texture with the selected function";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() { 
        return true;
    }

    TextureCombine();
    ~TextureCombine();

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
    std::unique_ptr<GLSLComputeShader> m_add_prgm;

    /** Shader program for texture multiply */
    std::unique_ptr<GLSLComputeShader> m_mult_prgm;

    /** Texture that the combination result will be written to */
    std::shared_ptr<glowl::Texture2D>  m_output_texture;

    /** Hash value to keep track of update to the output texture */
    size_t                             m_output_texture_hash;

    /** Parameter for selecting the texture combination mode, e.g. add, multiply */
    megamol::core::param::ParamSlot    m_mode;

    /** Slot for requesting the output textures from this module, i.e. lhs connection */
    megamol::core::CalleeSlot          m_output_tex_slot;

    /** Slot for querying primary input texture, i.e. a rhs connection */
    megamol::core::CallerSlot          m_input_tex_0_slot;

    /** Slot for querying secondary input texture, i.e. a rhs connection */
    megamol::core::CallerSlot          m_input_tex_1_slot;
};

} // namespace compositing
} // namespace megamol

#endif // !TEXTURE_COMBINE_H_INCLUDED
