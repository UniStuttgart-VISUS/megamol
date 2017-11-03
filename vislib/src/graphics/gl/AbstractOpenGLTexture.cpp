/*
 * AbstractOpenGLTexture.cpp
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/graphics/gl/AbstractOpenGLTexture.h"

#include <climits>

#include "vislib/assert.h"
#include "vislib/IllegalParamException.h"
#include "vislib/graphics/gl/glverify.h"
#include "vislib/Trace.h"
#include "vislib/UnsupportedOperationException.h"


/*
 * vislib::graphics::gl::AbstractOpenGLTexture::RequiredExtensions
 */
const char *vislib::graphics::gl::AbstractOpenGLTexture::RequiredExtensions(
        void) {
    return " GL_VERSION_1_3 ";
}


/*
 * vislib::graphics::gl::AbstractOpenGLTexture::~AbstractOpenGLTexture
 */
vislib::graphics::gl::AbstractOpenGLTexture::~AbstractOpenGLTexture(void) {
    this->Release();
}


/*
 * vislib::graphics::gl::AbstractOpenGLTexture::Bind
 */
GLenum vislib::graphics::gl::AbstractOpenGLTexture::Bind(GLenum textureUnit,
                                                         const bool isReset) {
    USES_GL_VERIFY;
    int oldTexUnit = GL_TEXTURE0;
    GLenum retval = GL_NO_ERROR;

    GL_VERIFY_RETURN(::glGetIntegerv(GL_ACTIVE_TEXTURE, &oldTexUnit));
    GL_VERIFY_RETURN(::glActiveTexture(textureUnit));

    if ((retval = this->Bind()) == GL_NO_ERROR) {
        GL_VERIFY_RETURN(::glActiveTexture(oldTexUnit));
        return GL_NO_ERROR;
    } else {
        // Report the original error.
        ::glActiveTexture(oldTexUnit);
        return retval;
    }
}


/*
 * vislib::graphics::gl::AbstractOpenGLTexture::IsValid
 */
bool vislib::graphics::gl::AbstractOpenGLTexture::IsValid(void) const throw() {
    return (id != UINT_MAX) && (::glIsTexture(this->id) != GL_FALSE);
}


/*
 * vislib::graphics::gl::AbstractOpenGLTexture::Release
 */
void vislib::graphics::gl::AbstractOpenGLTexture::Release(void) {
    USES_GL_VERIFY;
    
    if (this->IsValid()) {
        GL_VERIFY_THROW(::glDeleteTextures(1, &this->id));
    }
    ASSERT(!this->IsValid());
}


/*
 * vislib::graphics::gl::AbstractOpenGLTexture::setFilter
 */
GLenum vislib::graphics::gl::AbstractOpenGLTexture::setFilter(
        const GLenum target, const GLint minFilter, const GLint magFilter) {
    USES_GL_VERIFY;

    GL_VERIFY_RETURN(::glTexParameteri(target, GL_TEXTURE_MIN_FILTER, 
        minFilter));
    GL_VERIFY_RETURN(::glTexParameteri(target, GL_TEXTURE_MAG_FILTER, 
        magFilter));

    return GL_NO_ERROR;
}


/*
 * vislib::graphics::gl::AbstractOpenGLTexture::AbstractOpenGLTexture
 */
vislib::graphics::gl::AbstractOpenGLTexture::AbstractOpenGLTexture(void) 
        : id(UINT_MAX) {
}


/*
 * vislib::graphics::gl::AbstractOpenGLTexture::AbstractOpenGLTexture
 */
vislib::graphics::gl::AbstractOpenGLTexture::AbstractOpenGLTexture(
        const GLuint id) : id(id) {
}


/*
 * vislib::graphics::gl::AbstractOpenGLTexture::AbstractOpenGLTexture
 */
vislib::graphics::gl::AbstractOpenGLTexture::AbstractOpenGLTexture(
        AbstractOpenGLTexture& rhs) {
    throw UnsupportedOperationException(
        "AbstractOpenGLTexture::AbstractOpenGLTexture", __FILE__, __LINE__);
}


/*
 * vislib::graphics::gl::AbstractOpenGLTexture::createId
 */
GLuint vislib::graphics::gl::AbstractOpenGLTexture::createId(void) {
    USES_GL_VERIFY;

    GL_VERIFY_THROW(::glGenTextures(1, &this->id));
    return this->id;
}


/*
 * vislib::graphics::gl::AbstractOpenGLTexture::operator =
 */
vislib::graphics::gl::AbstractOpenGLTexture&
vislib::graphics::gl::AbstractOpenGLTexture::operator =(
        const AbstractOpenGLTexture& rhs) {
    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
