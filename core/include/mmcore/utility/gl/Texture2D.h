/*
 * Texture2D.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCORE_TEXTURE2D_H_INCLUDED
#define MEGAMOLCORE_TEXTURE2D_H_INCLUDED

#include "mmcore/utility/gl/Texture.h"

namespace megamol {
namespace core {
namespace utility {
namespace gl {

/**
 * @class Texture2D
 *
 * @brief Encapsulates 2D texture functionality.
 *
 * @author Michael Becher
 */
class Texture2D : public Texture {
public:
    /**
     * Constructor that creates and loads a 2D texture.
     *
     * @param id A identifier given to the texture object
     * @param layout A TextureLayout struct that specifies size, format and parameters for the texture
     * @param data Pointer to the actual texture data.
     * @param generateMipmap Specifies whether a mipmap will be created for the texture
     */
    Texture2D(std::string id, TextureLayout const& layout, GLvoid* data, bool generateMipmap = false);

    /** Deleted copy constructor */
    Texture2D(const Texture2D&) = delete;

    /** Deleted move constructor */
    Texture2D(Texture2D&& other) = delete;

    /** Deleted move operator */
    Texture2D& operator=(const Texture2D& rhs) = delete;

    /** Deleted assignment operator */
    Texture2D& operator=(Texture2D&& rhs) = delete;

    /**
     * Bind the texture.
     */
    void bindTexture() const;

    /**
     * Updates all mip map levels
     */
    void updateMipmaps();

    /**
     * Reload the texture with any new format, type and size.
     *
     * @param layout A TextureLayout struct that specifies size, format and parameters for the texture
     * @param data Pointer to the actual texture data.
     * @param generateMipmap Specifies whether a mipmap will be created for the texture
     * @return True on success, false otherwise
     */
    bool reload(TextureLayout const& layout, GLvoid* data, bool generateMipmap = false);

    /**
     * Get the layout of the texture
     *
     * @return The texture layout
     */
    TextureLayout getTextureLayout() const;

    /**
     * Get the width of the texture
     *
     * @return The texture width
     */
    unsigned int getWidth() const;

    /**
     * Get the height of the texture
     *
     * @return The texture height
     */
    unsigned int getHeight() const;

private:
    /** Width of the texture */
    unsigned int m_width;

    /** Height of the texture */
    unsigned int m_height;
};

} // namespace gl
} // namespace utility
} // namespace core
} // namespace megamol

#endif // !MEGAMOLCORE_TEXTURE2D_H_INCLUDED
