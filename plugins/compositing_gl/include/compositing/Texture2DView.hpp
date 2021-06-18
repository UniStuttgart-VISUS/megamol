/*
 * Texture2DView.hpp
 *
 * MIT License
 * Copyright (c) 2021 Michael Becher
 */

#ifndef GLOWL_TEXTURE2DVIEW_HPP
#define GLOWL_TEXTURE2DVIEW_HPP

#include "glowl/Exceptions.hpp"
#include "glowl/Texture.hpp"
#include "glowl/Texture2D.hpp"

#include "glowl/glinclude.h"

namespace glowl
{

    /**
     * \class TextureCubemapArray
     *
     * \brief Encapsulates 2D texture view functionality.
     *
     * \author Michael Becher
     */
    class Texture2DView : public Texture
    {
    public:
        /**
         * \brief Texture2DView constructor.
         *
         * Note: Active OpenGL context required for construction.
         * Use std::unqiue_ptr (or shared_ptr) for delayed construction of class member variables of this type.
         */
        template<typename T>
        Texture2DView(std::string          id,
                      T const&     source_texture,
                      TextureLayout const& layout,
                      GLuint               minlevel,
                      GLuint               numlevels,
                      GLuint               minlayer,
                      GLuint               numlayers);
        ~Texture2DView();

        void bindTexture() const;

        void updateMipmaps();

        TextureLayout getTextureLayout() const;

        /**
         * \brief Reload the texture view with the given source texture.
         *
         * \param source_texture The texture which the view references
         * \param layout    A TextureLayout struct that specifies size, format and parameters for the texture
         * \param minlevel  The mipmap level of the source texture (which is the base level of the view)
         * \param numlevels The number of mipmap levels of the view
         * \param minlayer  The layer number (in e.g. a TEXTURE_2D_ARRAY) of the source texture
         * \param numlayers The number of layers of the view
         */
        template<class T>
        void reload(
            T const& source_texture,
            TextureLayout const& layout,
            GLuint minlevel,
            GLuint numlevels,
            GLuint minlayer,
            GLuint numlayers);

        unsigned int getWidth();
        unsigned int getHeight();
        unsigned int getDepth();

    private:
        unsigned int m_width;
        unsigned int m_height;
        unsigned int m_depth;
    };

    // texturelayout not really needed since the view inherits (nearly) everything from the
    // original texture, but it's still nice to have them available
    template<typename T>
    inline Texture2DView::Texture2DView(std::string          id,
                                        T const&     source_texture,
                                        TextureLayout const& layout,
                                        GLuint               minlevel,
                                        GLuint               numlevels,
                                        GLuint               minlayer,
                                        GLuint               numlayers)
        : Texture(id, layout.internal_format, layout.format, layout.type, layout.levels)
    {
        glGenTextures(1, &m_name);

        glTextureView(m_name,
                      GL_TEXTURE_2D,
                      source_texture.getName(),
                      m_internal_format,
                      minlevel,
                      numlevels,
                      minlayer,
                      numlayers);

        glBindTexture(GL_TEXTURE_2D, m_name);

        GLint w, h, d;
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_DEPTH, &d);
        m_width = (unsigned int) w;
        m_height = (unsigned int) h;
        m_depth = (unsigned int) d;

        GLenum err = glGetError();
        if (err != GL_NO_ERROR)
        {
            throw TextureException("Texture2DView::Texture2DView - texture id: " + m_id + " - OpenGL error " +
                                     std::to_string(err));
        }
    }

    inline Texture2DView::~Texture2DView()
    {
        glDeleteTextures(1, &m_name);
    }

    inline void Texture2DView::bindTexture() const
    {
        glBindTexture(GL_TEXTURE_2D, m_name);
    }

    inline void Texture2DView::updateMipmaps() {
        glGenerateTextureMipmap(m_name);
    }

    inline TextureLayout Texture2DView::getTextureLayout() const
    {
        return TextureLayout(m_internal_format, m_width, m_height, m_depth, m_format, m_type, m_levels);
    }

    template<class T>
    inline void Texture2DView::reload(
        T const& source_texture,
        TextureLayout const& layout,
        GLuint minlevel,
        GLuint numlevels,
        GLuint minlayer,
        GLuint numlayers) {

        m_width = layout.width;
        m_height = layout.height;
        m_internal_format = layout.internal_format;
        m_format = layout.format;
        m_type = layout.type;
        m_levels = layout.levels;

        glDeleteTextures(1, &m_name);

        glGenTextures(1, &m_name);

        glTextureView(m_name, GL_TEXTURE_2D, source_texture.getName(), m_internal_format, minlevel, numlevels, minlayer,
            numlayers);

        glBindTexture(GL_TEXTURE_2D, m_name);

        GLint w, h, d;
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_DEPTH, &d);
        m_width = (unsigned int) w;
        m_height = (unsigned int) h;
        m_depth = (unsigned int) d;

        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            throw TextureException(
                "Texture2DView::Texture2DView - texture id: " + m_id + " - OpenGL error " + std::to_string(err));
        }

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    inline unsigned int Texture2DView::getWidth()
    {
        return m_width;
    }
    
    inline unsigned int Texture2DView::getHeight()
    {
        return m_height;
    }

    inline unsigned int Texture2DView::getDepth()
    {
        return m_depth;
    }

} // namespace glowl

#endif // GLOWL_TEXTURE2DVIEW_HPP
