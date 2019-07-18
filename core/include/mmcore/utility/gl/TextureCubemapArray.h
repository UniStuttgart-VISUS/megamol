/*
 * TextureCubemapArray.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCORE_TEXTURECUBEMAPARRAY_H_INCLUDED
#define MEGAMOLCORE_TEXTURECUBEMAPARRAY_H_INCLUDED

#include "mmcore/utility/gl/Texture.h"

namespace megamol {
namespace core {
namespace utility {
namespace gl {

/**
 * @class TextureCubemapArray
 *
 * @brief Encapsulates cubemap texture array functionality.
 *
 * @author Michael Becher
 */
class TextureCubemapArray : public Texture {
public:
    /**
     * Constructor for a cubemap array
     *
     * @param id The identifier of the cubemap array
     * @param internal_format The textures internal format
     * @param width The width of the texture
     * @param height The height of the texture
     * @param layers Number of texture layers
     * @param format The format of the texture
     * @param type The texture type
     * @param levels Number of mip map levels
     * @param data Pointer to the data for the texture
     * @param generateMipMap True if a mip map should be generated, false otherwise
     */
    TextureCubemapArray(std::string id, GLint internal_format, unsigned int width, unsigned int height,
        unsigned int layers, GLenum format, GLenum type, GLsizei levels, GLvoid* data, bool generateMipmap = false);

    /** Deleted copy constructor */
    TextureCubemapArray(const TextureCubemapArray&) = delete;

    /** Deleted move constructor */
    TextureCubemapArray(TextureCubemapArray&& other) = delete;

    /** Deleted assignment operator */
    TextureCubemapArray& operator=(const TextureCubemapArray& rhs) = delete;

    /** Deleted move operator */
    TextureCubemapArray& operator=(TextureCubemapArray&& rhs) = delete;

    //~Texture2DArray();

    /**
     * Reload the texture with a new size but unchanged format and type.
     *
     * @param width Specifies the new width of the texture in pixels.
     * @param height Specifies the new height of the texture in pixels.
     * @param layers Specifies the new number of layers in the texture array.
     * @param data Pointer to the new texture data.
     * @return Returns true if the texture was succesfully created, false otherwise
     */
    bool reload(
        unsigned int width, unsigned int height, unsigned int layers, GLvoid* data, bool generateMipmap = false);

    /**
     * Bind the texture.
     */
    void bindTexture() const;

    /**
     * Update all mip map levels of the texture
     */
    void updateMipmaps();

    /**
     * Sets a parameter of the cubemap array
     *
     * @param pname The name of the parameter
     * @param param The value to set
     */
    void texParameteri(GLenum pname, GLenum param);

    /**
     * Get the layout of the texture
     *
     * @return The layout of the texture
     */
    TextureLayout getTextureLayout() const;

    /**
     * Get the width of the texture
     *
     * @return The textures width
     */
    unsigned int getWidth() const;

    /**
     * Get the height of the texture
     *
     * @return The textures height
     */
    unsigned int getHeight() const;

    /**
     * Get the number of layer of the texture
     *
     * @return The textures layer count
     */
    unsigned int getLayers() const;

private:
    /** The width of the texture */
    unsigned int m_width;

    /** The height of the texture */
    unsigned int m_height;

    /** The number of layers of the texture */
    unsigned int m_layers;
};

} // namespace gl
} // namespace utility
} // namespace core
} // namespace megamol

#endif // !MEGAMOLCORE_TEXTURECUBEMAPARRAY_H_INCLUDED
