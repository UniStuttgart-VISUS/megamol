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
#define GL_VERIFY(call) ::glGetError(); call;\
    ASSERT(::glGetError() == GL_NO_ERROR)


/**
 * In the debug version, assert that 'expr' evaluates to GL_NO_ERROR. In the
 * release version, just execute 'expr'.
 *
 * @param expr The expression to execute and evaluate.
 */
#if defined(DEBUG) || defined(_DEBUG)
#define GL_VERIFY_EXPR(expr) ASSERT(expr == GL_NO_ERROR) 
#else /* defined(DEBUG) || defined(_DEBUG) */
#define GL_VERIFY_EXPR(expr)
#endif /* defined(DEBUG) || defined(_DEBUG) */


/** 
 * Make the call 'call' and if ::glGetError() returns any error code other than
 * GL_NO_ERROR, return this error code.
 *
 * This macro requires a local variable '__glv_glError' to be defined. Use 
 * USES_GL_VERIFY at begin of the enclosing function.
 *
 * @param call The OpenGL call to make.
 */
#define GL_VERIFY_RETURN(call) ::glGetError(); call;\
    if ((__glv_glError = ::glGetError()) != GL_NO_ERROR) {\
        return __glv_glError; }


/** 
 * Make the call 'call' and if ::glGetError() returns any error code other than
 * GL_NO_ERROR, throw a GLException.
 *
 * This macro requires a local variable '__glv_glError' to be defined. Use 
 * USES_GL_VERIFY at begin of the enclosing function.
 *
 * @param call The OpenGL call to make.
 */
#define GL_VERIFY_THROW(call) ::glGetError(); call;\
    if ((__glv_glError = ::glGetError()) != GL_NO_ERROR) {\
        throw vislib::graphics::gl::OpenGLException(__glv_glError, __FILE__,\
            __LINE__);\
    }


/**
 * Make the call 'call' and return whether ::glGetError() returns GL_NO_ERROR.
 *
 * @param call The OpenGL call to make.
 */
#define GL_SUCCEEDED(call) (::glGetError(), (call), \
    (::glGetError() == GL_NO_ERROR))


/**
 * Make the call 'call' and return whether ::glGetError() does not return 
 * GL_NO_ERROR.
 *
 * @param call The OpenGL call to make.
 */
#define GL_FAILED(call) (::glGetError(), (call), \
    (::glGetError() != GL_NO_ERROR))


/** 
 * Declare the variable '__glv_glError' for use in the GL_VERIFY_* macros. Add 
 * this macro at the begin of functions that use these macros.
 */
#define USES_GL_VERIFY GLenum __glv_glError; __glv_glError = GL_NO_ERROR;
// Note: Extra assignment prevent "unused variable" warning.


#endif /* VISLIB_GLVERIFY_H_INCLUDED */
