/*
 * CallGetTransferFunction.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <array>
#include <memory>

#include "glowl/glowl.h"

#include "mmcore/view/AbstractCallGetTransferFunction.h"

#include "vislib_gl/graphics/gl/GLSLShader.h"
#include "vislib_gl/graphics/gl/IncludeAllGL.h"

namespace megamol {
namespace core_gl {
namespace view {


/**
 * Call for accessing a transfer function.
 *
 * To use this in a shader:
 * - Add `<include file="core_utils" />` and the respective snippet
 * - Add `<snippet name="::core_utils::tflookup" />`
 * - Add `<snippet name="::core_utils::tfconvenience" />` (optional)
 * - Use `vec color = tflookup(tfTexture, tfRange, value);`
 * - Or, conveniently `vec color = tflookup(value);`
 */
class CallGetTransferFunctionGL : public core::view::AbstractCallGetTransferFunction {
public:
    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) {
        return "CallGetTransferFunctionGL";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "Call for a 1D transfer function";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
        return 1;
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case 0:
            return "GetTexture";
        default:
            return NULL;
        }
    }

    /** Ctor. */
    CallGetTransferFunctionGL(void);

    /** Dtor. */
    virtual ~CallGetTransferFunctionGL(void);

    ///// CALLER Interface Functions //////////////////////////////////////////

    // TEXTURE ----------------------------------------------------------------

    /**
     * Bind convenience (to be used with tfconvenience snippet). Usually, one
     * wants to set `activeTexture` to `GL_TEXTURE0` and `textureUniform` to `0`.
     */
    void BindConvenience(vislib_gl::graphics::gl::GLSLShader& shader, GLenum activeTexture, int textureUniform);

    void BindConvenience(std::unique_ptr<glowl::GLSLProgram>& shader, GLenum activeTexture, int textureUniform);

    /**
     * Unbinds convenience.
     */
    void UnbindConvenience();

    // CHANGES ----------------------------------------------------------------

    /** ----- DEPRECATED ----- (use BindConvenience and tfconvenience snippet)
     * Answer the OpenGL texture object id of the transfer function 1D
     * texture.
     *
     * @return The OpenGL texture object id
     */
    inline unsigned int OpenGLTexture(void) const {
        return this->texID;
    }

    /** ----- DEPRECATED ----- (use BindConvenience and tfconvenience snippet)
     * Answer the OpenGL format of the texture.
     *
     * @return The OpenGL format of the texture
     */
    inline int OpenGLTextureFormat(void) const {
        if (this->texFormat == TEXTURE_FORMAT_RGBA) {
            return GL_RGBA;
        }
        return GL_RGB;
    }

    ///// CALLEE Interface Functions //////////////////////////////////////////

    // SET --------------------------------------------------------------------

    /**
     * Sets the 1D texture information
     *
     * @param id The OpenGL texture object id
     * @param size The size of the texture
     * @param tex The float RGBA texture data, i.e. size*4 floats holding
     *            the RGBA color data. The data is not copied. The caller
     *            is responsible for keeping the memory alive.
     * @param format The texture format
     */
    inline void SetTexture(unsigned int id, unsigned int size, float const* tex, TextureFormat format,
        std::array<float, 2> range, uint32_t version) {
        this->texID = id;
        this->texSize = size;
        this->texFormat = format;
        this->texData = tex;
        if (this->texSize == 0) {
            this->texSize = 1;
        }
        this->range = range;
        this->availableTFVersion = version;
    }

private:
    /** The OpenGL texture object id */
    unsigned int texID;
};


/** Description class typedef */
typedef core::factories::CallAutoDescription<CallGetTransferFunctionGL> CallGetTransferFunctionGLDescription;


} /* end namespace view */
} // namespace core_gl
} /* end namespace megamol */
