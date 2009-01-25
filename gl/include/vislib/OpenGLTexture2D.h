/*
 * OpenGLTexture2D.h
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_OPENGLTEXTURE2D_H_INCLUDED
#define VISLIB_OPENGLTEXTURE2D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractOpenGLTexture.h"


namespace vislib {
namespace graphics {
namespace gl {


    /**
     * This class wraps a two-dimensional OpenGL texture object.
     */
    class OpenGLTexture2D : public AbstractOpenGLTexture {

    public:

        /**
         * Set the minification and magnification filter for the active texture.
         * 
         * @param minFilter Specifies the minification filter.
         * @param magFilter Specifies the magnification filter.
         *
         * @return GL_NO_ERROR in case of success, an OpenGL error code 
         *         otherwise. 
         */
        static GLenum SetFilter(const GLint minFilter, const GLint magFilter);

        /**
         * Set the texture wrapping mode for the active texture.
         *
         * @param s The wrap parameter for texture coordinate s.
         * @param t The wrap parameter for texture coordinate t.
         *
         * @return GL_NO_ERROR in case of success, an OpenGL error code 
         *         otherwise. 
         */
        static GLenum SetWrap(const GLint s, const GLint t);

        /** Ctor. */
        OpenGLTexture2D(void);

        /** Dtor. */
        virtual ~OpenGLTexture2D(void);

        /**
         * Bind the texture on the active texture unit.
         *
         * @return GL_NO_ERROR in case of success, an OpenGL error code 
         *         otherwise.
         */
        virtual GLenum Bind(void);

        /**
         * Create and initialise the texture object.
         *
         * The newly created texture will be bound to the active texture unit.
         *
         * If the texture object was already created, the exisiting texture is 
         * overwritten.
         *
         * If 'pixels' is not NULL, these data are used at texture data for 
         * mipmap level 0. Otherwise, the texture will not be initialised with
         * image data.
         *
         * @param width          Specifies the width of the texture image. 
         * @param height         Specifies the height of the texture image. 
         * @param pixels         Specifies a pointer to the image data in 
         *                       memory. This defaults to NULL.
         * @param format         Specifies the format of the pixel data 
         *                       designated by 'pixels'. This defaults to
         *                       GL_RGBA.
         * @param type           Specifies the data type of the pixel data 
         *                       designatey by 'pixels'. This defaults to
         *                       GL_UNSIGNED_BYTE.
         * @param internalFormat Specifies the colour components in the 
         *                       texture. This defaults to GL_RGBA.
         * @param border         Specifies the width of the border. This 
         *                       defaults to 0.
         *
         * @return GL_NO_ERROR in case of success, an OpenGL error code 
         *         otherwise.
         */
        GLenum Create(const UINT width, const UINT height, 
            const void *pixels = NULL, 
            const GLenum format = GL_RGBA, const GLenum type = GL_UNSIGNED_BYTE,
            const GLint internalFormat = GL_RGBA, const GLint border = 0);

        /**
         * Create and initialise the texture object.
         *
         * The newly created texture will be bound to the active texture unit.
         *
         * If the texture object was already created, the exisiting texture is 
         * overwritten.
         *
         * If 'pixels' is not NULL, these data are used at texture data for 
         * mipmap level 0. Otherwise, the texture will not be initialised with
         * image data.
         *
         * @param width           Specifies the width of the texture image. 
         * @param height          Specifies the height of the texture image. 
         * @param forcePowerOfTwo If true, the texture size will be forced to
         *                        the next larger power of two (width and 
         *                        height).
         * @param pixels          Specifies a pointer to the image data in 
         *                        memory. This defaults to NULL.
         * @param format          Specifies the format of the pixel data 
         *                        designated by 'pixels'. This defaults to
         *                        GL_RGBA.
         * @param type            Specifies the data type of the pixel data 
         *                        designatey by 'pixels'. This defaults to
         *                        GL_UNSIGNED_BYTE.
         * @param internalFormat  Specifies the colour components in the 
         *                        texture. This defaults to GL_RGBA.
         * @param border          Specifies the width of the border. This 
         *                        defaults to 0.
         *
         * @return GL_NO_ERROR in case of success, an OpenGL error code 
         *         otherwise.
         */
        GLenum Create(const UINT width, const UINT height, 
            const bool forcePowerOfTwo, const void *pixels = NULL,
            const GLenum format = GL_RGBA, const GLenum type = GL_UNSIGNED_BYTE,
            const GLint internalFormat = GL_RGBA, const GLint border = 0);

        /**
         * Update the texture image.
         *
         * For updating, the texture will be bound to the active texture unit. 
         * The texture binding is not reset after the operation completed unless
         * 'resetBind' is set true.
         *
         * @param pixels    Specifies a pointer to the image data in 
         *                  memory.
         * @param width     Specifies the width of the image data 
         *                  designated  by 'pixels.
         * @param height    Specifies the height of the image data 
         *                  designated  by 'pixels.
         * @param format    Specifies the format of the pixel data 
         *                  designated by 'pixels'. This defaults to
         *                  GL_RGBA.
         * @param type      Specifies the data type of the pixel data 
         *                  designatey by 'pixels'. This defaults to
         *                  GL_UNSIGNED_BYTE.
         * @param offsetX   The offset of the pixels on the abscissa.
         * @param offsetY   The offset of the pixels on the ordinate.
         * @param resetBind If true, the previously bound texture will be 
         *                  bound to the active texture unit after the operation
         *                  completed. Otherwise, the texture will remain bound.
         *
         * @return GL_NO_ERROR in case of success, an OpenGL error code 
         *         otherwise.
         */
        GLenum Update(const void *pixels, const UINT width, const UINT height,
            const GLenum format = GL_RGBA, const GLenum type = GL_UNSIGNED_BYTE,
            const UINT offsetX = 0, const UINT offsetY = 0, 
            const GLint level = 0, const bool resetBind = false);

    protected:

        /** Superclass typedef. */
        typedef AbstractOpenGLTexture Super;
    };
    
} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_OPENGLTEXTURE2D_H_INCLUDED */
