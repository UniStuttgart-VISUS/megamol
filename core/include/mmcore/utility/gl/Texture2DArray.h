/*
 * Texture2DArray.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCORE_TEXTURE2DARRAY_H_INCLUDED
#define MEGAMOLCORE_TEXTURE2DARRAY_H_INCLUDED

#include "mmcore/utility/gl/Texture.h"

namespace megamol {
namespace core {
namespace utility {
namespace gl {

/**
 * @class Texture2DArray
 *
 * @brief Encapsulates 2D texture array functionality.
 *
 * @author Michael Becher
 */
class Texture2DArray : public Texture {
public:
    /**
     * Constructor for a texture array
     *
     * @param id Identifier for the array
     * @param layout The layout of the stored texture
     * @param data Pointer to the data that is copied to the texture
     * @param generateMipmap True if mipmaps should be generated, false otherwise
     */
    Texture2DArray(std::string id, TextureLayout const& layout, GLvoid* data, bool generateMipmap = false);

    /** Deleted copy constructor */
    Texture2DArray(const Texture2DArray&) = delete;

    /** Deleted move constructor */
    Texture2DArray(Texture2DArray&& other) = delete;

    /** Deleted assignment operator */
    Texture2DArray& operator=(const Texture2DArray& rhs) = delete;

    /** Deleted move operator */
    Texture2DArray& operator=(Texture2DArray&& rhs) = delete;

    //~Texture2DArray();

    /**
     * Binds the texture array
     */
    void bindTexture() const;

    /**
     * Updates all mip map levels of the texture
     */
    void updateMipmaps();

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
    unsigned int getHeigth() const;

    /**
     * Get the number of layers of the texture
     *
     * @return The texture layer count
     */
    unsigned int getLayers() const;

private:
    /** Width of the texture */
    unsigned int m_width;

    /** Height of the texture */
    unsigned int m_height;

    /** Number of texture layers */
    unsigned int m_layers;
};

} // namespace gl
} // namespace utility
} // namespace core
} // namespace megamol

#endif // !MEGAMOLCORE_TEXTURE2DARRAY_H_INCLUDED
