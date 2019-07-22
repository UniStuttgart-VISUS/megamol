/*
 * Texture.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */
#ifndef MEGAMOLCORE_TEXTURE_H_INCLUDED
#define MEGAMOLCORE_TEXTURE_H_INCLUDED

#include <string>
#include <vector>
#include "vislib/graphics/gl/IncludeAllGL.h"

namespace megamol {
namespace core {
namespace utility {
namespace gl {

struct TextureLayout {
    /**
     * Constructor
     */
    TextureLayout() : width(0), internal_format(0), height(0), depth(0), format(0), type(0), levels(0) {}

    /**
     * Constructor
     *
     * @param internal_format Specifies the (sized) internal format of a texture (e.g. GL_RGBA32F)
     * @param width Specifies the width of the texture in pixels.
     * @param height Specifies the height of the texture in pixels. Will be ignored by Texture1D.
     * @param depth Specifies the depth of the texture in pixels. Will be ignored by Texture1D and Texture2D.
     * @param format Specifies the format of the texture (e.g. GL_RGBA)
     * @param type Specifies the type of the texture (e.g. GL_FLOAT)
     */
    TextureLayout(GLint internal_format, int width, int height, int depth, GLenum format, GLenum type, GLsizei levels)
        : internal_format(internal_format)
        , width(width)
        , height(height)
        , depth(depth)
        , format(format)
        , type(type)
        , levels(levels) {}

    /**
     * Constructor
     *
     * @param internal_format Specifies the (sized) internal format of a texture (e.g. GL_RGBA32F)
     * @param width Specifies the width of the texture in pixels.
     * @param height Specifies the height of the texture in pixels. Will be ignored by Texture1D.
     * @param depth Specifies the depth of the texture in pixels. Will be ignored by Texture1D and Texture2D.
     * @param format Specifies the format of the texture (e.g. GL_RGBA)
     * @param type Specifies the type of the texture (e.g. GL_FLOAT)
     * @param int_parameters A list of integer texture parameters, each given by a pair of name and value (e.g.
     * {{GL_TEXTURE_SPARSE_ARB,GL_TRUE},{...},...}
     * @param float_parameters A list of float texture parameters, each given
     * by a pair of name and value (e.g. {{GL_TEXTURE_MAX_ANISOTROPY_EX,4.0f},{...},...}
     */
    TextureLayout(GLint internal_format, int width, int height, int depth, GLenum format, GLenum type, GLsizei levels,
        std::vector<std::pair<GLenum, GLint>> const& int_parameters,
        std::vector<std::pair<GLenum, GLfloat>> const& float_parameters)
        : internal_format(internal_format)
        , width(width)
        , height(height)
        , depth(depth)
        , format(format)
        , type(type)
        , levels(levels)
        , int_parameters(int_parameters)
        , float_parameters(float_parameters) {}

    /**
     * Constructor
     *
     * @param internal_format Specifies the (sized) internal format of a texture (e.g. GL_RGBA32F)
     * @param width Specifies the width of the texture in pixels.
     * @param height Specifies the height of the texture in pixels. Will be ignored by Texture1D.
     * @param depth Specifies the depth of the texture in pixels. Will be ignored by Texture1D and Texture2D.
     * @param format Specifies the format of the texture (e.g. GL_RGBA)
     * @param type Specifies the type of the texture (e.g. GL_FLOAT)
     * @param levels Number of mim map levels of the texture
     * @param int_parameters A list of integer texture parameters, each given by a pair of name and value (e.g.
     * {{GL_TEXTURE_SPARSE_ARB,GL_TRUE},{...},...}
     * @param float_parameters A list of float texture parameters, each given
     * by a pair of name and value (e.g. {{GL_TEXTURE_MAX_ANISOTROPY_EX,4.0f},{...},...}
     */
    TextureLayout(GLint internal_format, int width, int height, int depth, GLenum format, GLenum type, GLsizei levels,
        std::vector<std::pair<GLenum, GLint>>&& int_parameters,
        std::vector<std::pair<GLenum, GLfloat>>&& float_parameters)
        : internal_format(internal_format)
        , width(width)
        , height(height)
        , depth(depth)
        , format(format)
        , type(type)
        , levels(levels)
        , int_parameters(int_parameters)
        , float_parameters(float_parameters) {}

    /** The internal format of the texture */
    GLint internal_format;

    /** The width of the texture */
    int width;

    /** The height of the texture */
    int height;

    /** The depth of the texture (only needed for 3D textures) */
    int depth;

    /** The texture format */
    GLenum format;

    /** The texture type */
    GLenum type;

    /** The number of mip map levels */
    GLsizei levels;

    /** The int parameters of the texture */
    std::vector<std::pair<GLenum, GLint>> int_parameters;

    /** The float parameters of the texture */
    std::vector<std::pair<GLenum, GLfloat>> float_parameters;
};

/**
 * @class Texture
 *
 * @brief Abstract base class for various OpenGL texture Objects (2D,3D,2DArray).
 *
 * @author Michael Becher
 */
class Texture {
public:
    /**
     * Constructor
     *
     * @param id The identifier of the texture
     * @param internal_format The textures internal format
     * @param format The texture format
     * @param type The texture type
     * @param levels The number of mip map levels
     */
    Texture(std::string id, GLint internal_format, GLenum format, GLenum type, GLsizei levels)
        : m_id(id), m_internal_format(internal_format), m_format(format), m_type(type), m_levels(levels) {}

    /**
     * Destructor
     */
    virtual ~Texture() { glDeleteTextures(1, &m_name); }

    /** deleted copy constructor */
    Texture(const Texture&) = delete;

    /**
     * Binds the texture
     */
    virtual void bindTexture() const = 0;

    /**
     * Binds the texture to a specific image unit
     *
     * @param location The location of the image unit
     * @param access The access modifier
     */
    void bindImage(GLuint location, GLenum access) const {
        glBindImageTexture(location, m_name, 0, GL_TRUE, 0, access, m_internal_format);
    }

    /**
     * Enforce that the texture lies on the GPU
     */
    void makeResident() { glMakeTextureHandleResidentARB(m_texture_handle); }

    /**
     * Removes the forced GPU residency
     */
    void makeNonResident() { glMakeTextureHandleNonResidentARB(m_texture_handle); }

    /**
     * Recompute all mip map levels
     */
    virtual void updateMipmaps() = 0;

    /**
     * Get the texture layout
     *
     * @return The layout of the texture
     */
    virtual TextureLayout getTextureLayout() const = 0;

    /**
     * Get the identifier of the texture
     *
     * @return The textures identifier
     */
    std::string getId() const { return m_id; }

    /**
     * Get the name of the texture
     *
     * @return The name of the texture
     */
    GLuint getName() const { return m_name; }

    /**
     * Get the OpenGL texture handle
     *
     * @param The handle of the texture
     */
    GLuint64 getTextureHandle() const { return m_texture_handle; }

    /**
     * Get the handle of a specific texture image
     *
     * @param level The mip map level to retrieve
     * @param layered True, if this is a layered texture, false otherwise
     * @param layer The index of the texture layer
     * @return The handle of the wanted texture
     */
    GLuint64 getImageHandle(GLint level, GLboolean layered, GLint layer) const {
        return glGetImageHandleARB(m_name, level, layered, layer, m_internal_format);
    }

    /**
     * Get the textures internal format
     *
     * @return The internal format
     */
    GLenum getInternalFormat() const { return m_internal_format; }

    /**
     * Get the texture format
     *
     * @return The texture format
     */
    GLenum getFormat() const { return m_format; }

    /**
     * Returns the type of the texture
     *
     * @return Type of the texture
     */
    GLenum getType() const { return m_type; }

protected:
    /** Identifier set by application to help identifying textures */
    std::string m_id;

    /** OpenGL texture name given by glGenTextures */
    GLuint m_name;

    /** Actual OpenGL texture handle (used for bindless) */
    GLuint64 m_texture_handle;

    /** Target of the texture, i.e. it's type (e.g. GL_TEXTURE_2D) */
    // GLenum		m_target;

    /** Internal texture format */
    GLenum m_internal_format;

    /** Texture formate */
    GLenum m_format;

    /** Type of the texture */
    GLenum m_type;

    /** Number of mip map levels */
    GLsizei m_levels;

    // TODO: Store texture parameters as well ?
};

} // namespace gl
} // namespace utility
} // namespace core
} // namespace megamol

#endif //! MEGAMOLCORE_TEXTURE_H_INCLUDED
