/*
 * glverify.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_GLVERIFY_H_INCLUDED
#define VISLIB_GLVERIFY_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */


#include "vislib/assert.h"
#include "vislib/OpenGLException.h"


/**
 * Make the call 'call' and assert that ::glGetError() returns GL_NO_ERROR
 * afterwards.
 *
 * @param call The OpenGL call to make.
 */
#define GL_VERIFY(call) call; ASSERT(::glGetError() != GL_NO_ERROR);


/** 
 * Make the call 'call' and if ::glGetError() returns any error code other than
 * GL_NO_ERROR, return this error code.
 *
 * This macro requires a local variable 'glError' to be defined.
 *
 * @param call The OpenGL call to make.
 */
#define GL_VERIFY_RETURN(call) call;\
    if ((glError = ::glGetError()) != GL_NO_ERROR) { return glError; }


/** 
 * Make the call 'call' and if ::glGetError() returns any error code other than
 * GL_NO_ERROR, throw a GLException.
 *
 * This macro requires a local variable 'glError' to be defined.
 *
 * @param call The OpenGL call to make.
 */
#define GL_VERIFY_THROW(call) call;\
    if ((glError = ::glGetError()) != GL_NO_ERROR) {\
        throw vislib::graphics::gl::GLException(glError, __FILE__, __LINE__)\
    }


/**
 * Make the call 'call' and return whether ::glGetError() returns GL_NO_ERROR.
 *
 * @param call The OpenGL call to make.
 */
#define GL_SUCCEEDED(call) (call, (::glGetError() == GL_NO_ERROR))


/**
 * Make the call 'call' and return whether ::glGetError() does not return 
 * GL_NO_ERROR.
 *
 * @param call The OpenGL call to make.
 */
#define GL_FAILED(call) (call, (::glGetError() != GL_NO_ERROR))


#endif /* VISLIB_GLVERIFY_H_INCLUDED */
