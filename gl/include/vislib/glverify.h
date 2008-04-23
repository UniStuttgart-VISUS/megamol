/*
 * glverify.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_GLVERIFY_H_INCLUDED
#define VISLIB_GLVERIFY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


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
#define GL_VERIFY_EXPR(expr) (expr)
#endif /* defined(DEBUG) || defined(_DEBUG) */


/** 
 * Check whether 'expr' is an OpenGL error code other than GL_NO_ERROR and
 * return this error code ('expr') in this case. Note, that 'expr' is 
 * guaranteed to be evaluated only once.
 *
 * This macro requires a local variable '__glv_glError' to be defined. Use 
 * USES_GL_VERIFY at begin of the enclosing function.
 *
 * @param expr The OpenGL call to make.
 */
#define GL_VERIFY_EXPR_RETURN(expr)\
    if ((__glv_glError = (expr)) != GL_NO_ERROR) {\
        return __glv_glError; }


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


/**
 * Declare the '__glvDeferred_glError' and '__glvDeferred_Line' variables for 
 * use with GL_DEFERRED_* macros. Add this macro at the begin of functions that
 * use these macros.
 */
#define USES_GL_DEFERRED_VERIFY \
    GLenum __glvDeferred_glError; __glvDeferred_glError = GL_NO_ERROR; \
    int __glvDeferred_Line; __glvDeferred_Line = 0;


/**
 * Make OpenGL call 'call' at line 'line' (which always should be the builtin
 * __LINE__ macro) and if no previous OpenGL error was stored and 'call' failed,
 * store the error code and the line for throwing an exception later on.
 *
 * This macro requires local variables '__glvDeferred_glError' and 
 * '__glvDeferred_Line' to be defined. Use USES_GL_DEFERRED_VERIFY at begin of 
 * the enclosing function.
 *
 * @param call The OpenGL call to make.
 * @param line The current line, which should be __LINE__.
 */
#define GL_DEFERRED_VERIFY(call, line) ::glGetError(); call;\
    if ((__glvDeferred_glError == GL_NO_ERROR)\
            && ((__glvDeferred_glError = ::glGetError()) != GL_NO_ERROR)) {\
        __glvDeferred_Line = line;\
    }


/**
 * Make the call 'call' and catch all OpenGLExceptions. If an exception was
 * catched and the '__glvDeferred_glError' is not yet set, the error code and
 * line of the exception will be preserved.
 *
 * This macro requires local variables '__glvDeferred_glError' and 
 * '__glvDeferred_Line' to be defined. Use USES_GL_DEFERRED_VERIFY at begin of 
 * the enclosing function.
 */
#define GL_DEFERRED_VERIFY_TRY(call)\
    try {\
        call;\
    } catch (vislib::graphics::gl::OpenGLException __glvDeferredOGLe) {\
        if (__glvDeferred_glError == GL_NO_ERROR) {\
            __glvDeferred_glError = __glvDeferredOGLe.GetErrorCode();\
            __glvDeferred_Line = __glvDeferredOGLe.GetLine();\
        }\
    }


/**
 * Return the first OpenGL error that was catched by a GL_DEFERRED_VERIFY call.
 *
 * This macro requires local variables '__glvDeferred_glError' and 
 * '__glvDeferred_Line' to be defined. Use USES_GL_DEFERRED_VERIFY at begin of 
 * the enclosing function.
 */
#define GL_DEFERRED_VERIFY_RETURN() return __glvDeferred_glError


/**
 * If an OpenGL error was stored to the local '__glvDeferred_glError' variable,
 * throw an exception.
 *
 * Note that GL_DEFERRED* is only handling the first error that occurred in the
 * current scope.
 *
 * This macro requires local variables '__glvDeferred_glError' and 
 * '__glvDeferred_Line' to be defined. Use USES_GL_DEFERRED_VERIFY at begin of 
 * the enclosing function.
 *
 * @param file The curren file, which should be __FILE__.
 */
#define GL_DEFERRED_VERIFY_THROW(file)\
    if (__glvDeferred_glError != GL_NO_ERROR) {\
        throw vislib::graphics::gl::OpenGLException(__glvDeferred_glError, \
            file, __glvDeferred_Line);\
    }


/**
 * Answers whether no error occurred during GL_DEFERRED* calls in the enclosing
 * block.
 *
 * This macro requires local variables '__glvDeferred_glError' and 
 * '__glvDeferred_Line' to be defined. Use USES_GL_DEFERRED_VERIFY at begin of 
 * the enclosing function.
 *
 * @return true if no error was captured, false otherwise.
 */
#define GL_DEFERRED_SUCCEEDED() (__glvDeferred_glError == GL_NO_ERROR)


/**
 * Answers whether an error occurred during GL_DEFERRED* calls in the enclosing
 * block.
 *
 * This macro requires local variables '__glvDeferred_glError' and 
 * '__glvDeferred_Line' to be defined. Use USES_GL_DEFERRED_VERIFY at begin of 
 * the enclosing function.
 *
 * @return true if an error was captured, false otherwise.
 */
#define GL_DEFERRED_FAILED() (__glvDeferred_glError != GL_NO_ERROR)


#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_GLVERIFY_H_INCLUDED */
