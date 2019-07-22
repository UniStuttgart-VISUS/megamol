/*
 * CallGetTransferFunction.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CALLGETTRANSFERFUNCTION_H_INCLUDED
#define MEGAMOLCORE_CALLGETTRANSFERFUNCTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/api/MegaMolCore.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/graphics/gl/IncludeAllGL.h"

#include <array>


namespace megamol {
namespace core {
namespace view {


/**
 * Base class of rendering graph calls
 */
class MEGAMOLCORE_API CallGetTransferFunction : public Call {
public:
    /** possible texture formats */
    enum TextureFormat { TEXTURE_FORMAT_RGB = GL_RGB, TEXTURE_FORMAT_RGBA = GL_RGBA };

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) { return "CallGetTransferFunction"; }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) { return "Call for a 1D transfer function"; }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) { return 3; }

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
        case 1:
            return "GetDirty";
        case 2:
            return "ResetDirty";
        default:
            return NULL;
        }
    }

    /** Ctor. */
    CallGetTransferFunction(void);

    /** Dtor. */
    virtual ~CallGetTransferFunction(void);

    /**
     * Answer the OpenGL texture object id of the transfer function 1D
     * texture.
     *
     * @return The OpenGL texture object id
     */
    inline unsigned int OpenGLTexture(void) const { return this->texID; }

    /**
     * Answer the size of the 1D texture in texel.
     *
     * @return The size of the texture
     */
    inline unsigned int TextureSize(void) const { return this->texSize; }

    /**
     * Answer the correct texture coordinates from the first texel center to
     * the last to ensure proper interpolation (see docs/volumes/Readme.md).
     * Note this returns the offset and the reduced range!
     * You can upload this to a uniform vec2 transferFunctionTexCoords; // offset (min), range (max - min)
     * and include the convenience function float transform_to_TF_coordinates(float value, vec2 texcoords)
     * via <include file="core_utils" /> in a btf. Remember to insert the snippet
     * <snippet name="::core_utils::transform_to_TF_coordinates" /> in the respective shader!
     * 
     * @return the texture coordinates: (min, max - min)
     */
    inline std::array<float, 2> TextureCoordinates(void) const { return this->texCoords; }

    /**
     * Answer the OpenGL format of the texture
     *
     * @return The OpenGL format of the texture
     */
    inline TextureFormat OpenGLTextureFormat(void) const { return this->texFormat; }

    /**
     * Answer the OpenGL texture data. This is always an RGBA float color
     * array, regardless the TextureFormat returned. If TextureFormat is
     * RGB the A values stored, are simply meaningless. Thus, this pointer
     * always points to TextureSize*4 floats.
     *
     * @return The OpenGL texture data
     */
    inline float const* GetTextureData(void) const { return this->texData; }

    /**
     * Answer the range the texure lies within.
     *
     * @return The range
     */
    inline std::array<float, 2> Range(void) const { return this->range; }

    /**
     * Answer if the interface of the transferfunction module is dirty
     *
     * @return dirty flag
     */
    inline bool isDirty() {
        (*this)(1);
        return this->dirty;
    }

    /**
     * Resets dirtyness of the interface of the transferfunction module is dirty
     *
     */
    inline void resetDirty() {
        (*this)(2);
        this->dirty = false;
    }

    /**
     * Sets the dirty flag in the call
     *
     */
    inline void setDirty(bool dty) { this->dirty = dty; }


    /**
     * Sets the 1D texture information
     *
     * @param id The OpenGL texture object id
     * @param size The size of the texture
     * @param format The texture format
     */
    inline void SetTexture(unsigned int id, unsigned int size, TextureFormat format = TEXTURE_FORMAT_RGB) {
        this->texID = id;
        this->texSize = size;
        this->texFormat = format;
        this->texData = nullptr;
        if (this->texSize == 0) {
            this->texSize = 1;
        }
        this->updateTexCoords();
    }

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
    inline void SetTexture(
        unsigned int id, unsigned int size, float const* tex, TextureFormat format = TEXTURE_FORMAT_RGB) {
        this->texID = id;
        this->texSize = size;
        this->texFormat = format;
        this->texData = tex;
        if (this->texSize == 0) {
            this->texSize = 1;
        }
        this->updateTexCoords();
    }

    /**
     * Set range the texture lies within
     *
     * @param range The range.
     */
    inline void SetRange(std::array<float, 2> range) { this->range = range; }

    /**
     * Copies a color from the transfer function
     *
     * @param index The n-th color to copy.
     * @param color A pointer to copy the color to.
     * @param colorSize The size of the color in bytes.
     */
    inline void CopyColor(size_t index, float* color, size_t colorSize) {
        assert(index > 0 && index < this->texSize && "Invalid index");
        assert(colorSize == 3 * sizeof(float) || colorSize == 4 * sizeof(float) && "Not a RGB(A) color");
        memcpy(color, &this->texData[index * 4], colorSize);
    }

private:
    inline void updateTexCoords() {
        float half_tex = 1.0f / static_cast<float>(this->texSize);
        half_tex *= 0.5f;
        this->texCoords[0] = half_tex;
        this->texCoords[1] = 1.0f - 2.0f * half_tex;
    }

    /** The OpenGL texture object id */
    unsigned int texID;

    /** The size of the texture in texel */
    unsigned int texSize;

    /** The texture data */
    float const* texData;

    /** The texture format */
    TextureFormat texFormat;

    /** The range the texture lies within */
    std::array<float, 2> range;

    /** for convenience, tex coords with the half texel removed at both ends */
    std::array<float, 2> texCoords;

    /** Dirty flag */
    bool dirty = false;
};


/** Description class typedef */
typedef factories::CallAutoDescription<CallGetTransferFunction> CallGetTransferFunctionDescription;


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CALLGETTRANSFERFUNCTION_H_INCLUDED */
