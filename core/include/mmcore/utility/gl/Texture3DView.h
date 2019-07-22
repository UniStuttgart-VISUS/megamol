/*
 * Texture3DView.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCORE_TEXTURE3DVIEW_H_INCLUDED
#define MEGAMOLCORE_TEXTURE3DVIEW_H_INCLUDED

#include "mmcore/utility/gl/Texture.h"
#include "mmcore/utility/gl/Texture3D.h"

namespace megamol {
namespace core {
namespace utility {
namespace gl {

/**
 * \class TextureCubemapArray
 *
 * \brief Encapsulates 3D texture view functionality.
 *
 * \author Michael Becher
 */
class Texture3DView : public Texture {
public:
    /**
     * Constructor for a view of a 3D texture
     *
     * @param id The identifier of the view
     * @param source_texture The texture to create a view for
     * @param layout The layout of the texture view
     * @param minlevel The minimum mip map level
     * @param numlevels The number of mip map levels
     * @param minlayer Minimal texture layer id
     * @param numlayers Number of represented texture layers
     */
    Texture3DView(std::string id, Texture3D const& source_texture, TextureLayout const& layout, GLuint minlevel,
        GLuint numlevels, GLuint minlayer, GLuint numlayers);

    /**
     * Bind the texture.
     */
    void bindTexture() const;

    /**
     * Binds the texture view to a specific location
     *
     * @param location The location to bind to
     * @param access Texture access modifiers
     */
    void bindImage(GLuint location, GLenum access) const;

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

#endif // !MEGAMOLCORE_TEXTURE3DVIEW_H_INCLUDED
