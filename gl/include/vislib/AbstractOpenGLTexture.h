/*
 * AbstractOpenGLTexture.h
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTOPENGLTEXTURE_H_INCLUDED
#define VISLIB_ABSTRACTOPENGLTEXTURE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#ifdef _WIN32
#include <windows.h>
#endif /* _WIN32 */
#include "glh/glh_genext.h"

#include "vislib/ExtensionsDependent.h"
#include "vislib/StackTrace.h"


namespace vislib {
namespace graphics {
namespace gl {


    /**
     * This class provides the interface and common functionality of OpenGL
     * texture object in the VISlib.
     */
    class AbstractOpenGLTexture {

    public:

        /**
         * Answer the extensions that are required for texture objects as
         * space-separated ANSI strings. 
         *
         * Implementation note: The AbstractOpenGLTexture does not inherit from
         * ExtensionsDependent. Only the last class in the inheritance hierarchy
         * should do so and include the required extensions of the base class
         * in its list.
         *
         * @return The extensions that are requiered for framebuffer objects.
         */
        static const char *RequiredExtensions(void);

        /** Dtor. */
        virtual ~AbstractOpenGLTexture(void);

        /**
         * Bind the texture on the active texture unit.
         *
         * @return GL_NO_ERROR in case of success, an OpenGL error code 
         *         otherwise.
         */
        virtual GLenum Bind(void) = 0;

        /**
         * Bind the texture on the specified texture unit. If 'isReset' is true,
         * the active texture unit will be reset to its original value after 
         * the operation.
         *
         * @param textureUnit The texture unit to bind the texture to.
         * @param isReset     Reset the active texture unit to the previous unit
         *                    after binding the texture.
         *
         * @return GL_NO_ERROR in case of success, an OpenGL error code 
         *         otherwise. Best efforts are made to reset the texture unit 
         *         in case of an error.
         */
        GLenum Bind(GLenum textureUnit, const bool isReset = false);

        /**
         * Answer the texture ID of the wrapped OpenGL texture. 
         *
         * It is stronly discouraged to manipulate the texture via this ID as 
         * this might tamper with the object state.
         *
         * @return The ID of the texture.
         */
        inline GLuint GetId(void) const {
            VLSTACKTRACE("AbstractOpenGLTexture::GetId", __FILE__, __LINE__);
            return this->id;
        }

        /**
         * Answer whether the texture is valid, i. e. the OpenGL resources have
         * been successfully allocated.
         *
         * @return true if the texture is valid, false otherwise.
         */
        virtual bool IsValid(void) const throw();

        /**
         * Release the OpenGL resources of the texture object. The texture 
         * cannot be used after calling this method and must be recreated.
         *
         * It is safe to call this method on invalid texture objects.
         *
         * @throws OpenGLException If the operation failed (This should never
         *                         happen.).
         */
        virtual void Release(void);

    protected:

        /**
         * Set the minification and magnification filter for the specified 
         * texture target.
         * 
         * @param target    Specifies the target texture.
         * @param minFilter Specifies the minification filter.
         * @param magFilter Specifies the magnification filter.
         *
         * @return GL_NO_ERROR in case of success, an OpenGL error code 
         *         otherwise. 
         */
        static GLenum setFilter(const GLenum target, const GLint minFilter,
            const GLint magFilter);

        /** Ctor. */
        AbstractOpenGLTexture(void);

        /**
         * Create a texture wrapper object for an exisiting texture ID. This 
         * should only be performed once for every texture ID to avoid aliasing.
         *
         * @param id The ID of the texture to wrap.
         */
        explicit AbstractOpenGLTexture(const GLuint id);

        /**
         * Forbidden copy ctor. We cannot allow aliases of the texture ID as 
         * this would have unexpected results if an aliased object is destroyed.
         *
         * @param rhs The object to be cloned.
         *
         * @throws UnsupportedOperationException Unconditionally.
         */
        AbstractOpenGLTexture(AbstractOpenGLTexture& rhs);

        /**
         * Create a texture ID for this texture. A copy of the ID is returned
         * for immediate use.
         *
         * @return this->id.
         *
         * @throws OpenGLException If the operation fails.
         */
        GLuint createId(void);

        /**
         * Forbidden assignment. We cannot allow aliases of the texture ID as 
         * this would have unexpected results if an aliased object is destroyed.
         *
         * @param rhs The right hand side operand.
         *
         * @return *this.
         *
         * @throws IllegalParamException If (this != &rhs).
         */
        AbstractOpenGLTexture& operator =(const AbstractOpenGLTexture& rhs);

    private:

        /** The ID of the texture. */
        GLuint id;
    };
    
} /* end namespace gl */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTOPENGLTEXTURE_H_INCLUDED */
