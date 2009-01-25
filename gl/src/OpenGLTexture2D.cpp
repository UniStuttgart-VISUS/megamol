/*
 * OpenGLTexture2D.cpp
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/OpenGLTexture2D.h"

#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"
#include "vislib/glverify.h"
#include "vislib/mathfunctions.h"
#include "vislib/Trace.h"
#include "vislib/UnsupportedOperationException.h"



/*
 * vislib::graphics::gl::OpenGLTexture2D::SetFilter
 */
GLenum vislib::graphics::gl::OpenGLTexture2D::SetFilter(const GLint minFilter, 
                                                        const GLint magFilter) {
    VLSTACKTRACE("OpenGLTexture2D::SetFilter", __FILE__, __LINE__);
    return Super::setFilter(GL_TEXTURE_2D, minFilter, magFilter);
}


/*
 * vislib::graphics::gl::OpenGLTexture2D::SetWrap
 */
GLenum vislib::graphics::gl::OpenGLTexture2D::SetWrap(const GLint s, 
                                                      const GLint t) {
    VLSTACKTRACE("OpenGLTexture2D::SetWrap", __FILE__, __LINE__);
    USES_GL_VERIFY;

    GL_VERIFY_RETURN(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, s));
    GL_VERIFY_RETURN(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, t));

    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::OpenGLTexture2D::OpenGLTexture2D
 */
vislib::graphics::gl::OpenGLTexture2D::OpenGLTexture2D(void) : Super() {
    VLSTACKTRACE("OpenGLTexture2D::OpenGLTexture2D", __FILE__, __LINE__);
    // Nothing to do.
}


/*
 * vislib::graphics::gl::OpenGLTexture2D::~OpenGLTexture2D
 */
vislib::graphics::gl::OpenGLTexture2D::~OpenGLTexture2D(void) {
    VLSTACKTRACE("OpenGLTexture2D::~OpenGLTexture2D", __FILE__, __LINE__);
    // Nothing to do.
}


/*
 * vislib::graphics::gl::OpenGLTexture2D::Bind
 */
GLenum vislib::graphics::gl::OpenGLTexture2D::Bind(void) {
    VLSTACKTRACE("OpenGLTexture2D::Bind", __FILE__, __LINE__);
    USES_GL_VERIFY;
    GL_VERIFY_RETURN(::glBindTexture(GL_TEXTURE_2D, this->GetId()));
    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::OpenGLTexture2D::Create
 */
GLenum vislib::graphics::gl::OpenGLTexture2D::Create(const UINT width, 
        const UINT height, const void *pixels, const GLenum format, 
        const GLenum type, const GLint internalFormat, const GLint border) {
    VLSTACKTRACE("OpenGLTexture2D::Create", __FILE__, __LINE__);
    USES_GL_VERIFY;

    /* Create new texture. */
    if (!this->IsValid()) {
        try {
            this->createId();
        } catch (OpenGLException e) {
            return e.GetErrorCode();
        }   
    }

    GL_VERIFY_EXPR_RETURN(this->Bind());
    GL_VERIFY_RETURN(::glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, 
        height, border, format, type, pixels));

    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::OpenGLTexture2D::Create
 */
GLenum vislib::graphics::gl::OpenGLTexture2D::Create(const UINT width, 
        const UINT height, const bool forcePowerOfTwo, const void *pixels,
        const GLenum format, const GLenum type, const GLint internalFormat, 
        const GLint border) {
    VLSTACKTRACE("OpenGLTexture2D::Create", __FILE__, __LINE__);
    USES_GL_VERIFY;

    UINT w = math::NextPowerOfTwo(width);
    UINT h = math::NextPowerOfTwo(height);

    if (!forcePowerOfTwo || ((w == width) && (h == height))) {
        return this->Create(width, height, pixels, format, type, 
            internalFormat, border);
    } else {
        GL_VERIFY_EXPR_RETURN(this->Create(w, h, NULL, format, type, 
            internalFormat, border));
        GL_VERIFY_EXPR_RETURN(this->Update(pixels, width, height, format, type, 
            0, 0, 0));
    }

    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::OpenGLTexture2D::Update
 */
GLenum vislib::graphics::gl::OpenGLTexture2D::Update(const void *pixels, 
        const UINT width, const UINT height, const GLenum format, 
        const GLenum type, const UINT offsetX, const UINT offsetY, 
        const GLint level, const bool resetBind) {
    VLSTACKTRACE("OpenGLTexture2D::Update", __FILE__, __LINE__);
    USES_GL_VERIFY;
    GLint oldTex;
    GLenum retval = GL_NO_ERROR;

    if (resetBind) {
        GL_VERIFY_RETURN(::glGetIntegerv(GL_TEXTURE_BINDING_2D, &oldTex));
    }
    
    if ((retval = this->Bind()) != GL_NO_ERROR) {
        goto cleanup;
    }
    ::glTexSubImage2D(GL_TEXTURE_2D, level, offsetX, offsetY, width, height, 
        format, type, pixels);
    retval = ::glGetError();

cleanup:
    if (resetBind) {
        ::glGetError();
        ::glBindTexture(GL_TEXTURE_2D, oldTex);
        if (retval == GL_NO_ERROR) {
            retval = ::glGetError();
        }
    }

    return retval;
}
