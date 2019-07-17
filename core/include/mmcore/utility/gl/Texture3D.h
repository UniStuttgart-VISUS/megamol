/*
 * Texture3D.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCORE_TEXTURE3D_H_INCLUDED
#define MEGAMOLCORE_TEXTURE3D_H_INCLUDED

#include "mmcore/utility/gl/Texture.h"

namespace megamol {
namespace core {
namespace utility {
namespace gl {

/**
 * @class Texture3D
 *
 * @brief Encapsulates basic 3D texure functionality.
 *
 * This class encapsulates basic 3D functionality including creation of a 3D texture,
 * texture updates and texture binding.
 *
 * @author Michael Becher
 */
class Texture3D : public Texture {
public:
    /**
     * Constructor
     *
     * @param id Identifier of the texture
     * @param layout The layout of the texture
     * @param data Pointer to the data uploaded to the texture
     */
    Texture3D(std::string id, TextureLayout const& layout, GLvoid* data);

    /** Deleted copy constructor */
    Texture3D(const Texture3D&) = delete;

    /** Deleted move constructor */
    Texture3D(Texture3D&& other) = delete;

    /** Deleted assignment operator */
    Texture3D& operator=(const Texture3D& rhs) = delete;

    /** Deleted move operator */
    Texture3D& operator=(Texture3D&& rhs) = delete;

    /**
     * Bind the texture.
     */
    void bindTexture() const;

    /**
     * Update all mip map levels of the texture
     */
    void updateMipmaps();

    /**
     * Reload the texture.
     *
     * @param layout The layout of the texture
     * @param data Pointer to the new texture data
     * @return True on success, false otherwise
     */
    bool reload(TextureLayout const& layout, GLvoid* data);

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
    unsigned int getWidth();

    /**
     * Get the height of the texture
     *
     * @return The textures height
     */
    unsigned int getHeight();

    /**
     * Get the depth of the texture
     *
     * @return The textures depth
     */
    unsigned int getDepth();

private:
    /** The width of the texture */
    unsigned int m_width;

    /** The height of the texture */
    unsigned int m_height;

    /** The depth of the texture */
    unsigned int m_depth;
};

} // namespace gl
} // namespace utility
} // namespace core
} // namespace megamol

#endif // !MEGAMOLCORE_TEXTURE3D_H_INCLUDED
