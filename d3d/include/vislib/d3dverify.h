/*
 * d3dverify.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_D3DVERIFY_H_INCLUDED
#define VISLIB_D3DVERIFY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/assert.h"
#include "vislib/D3DException.h"


/** 
 * Declare the variable 'hr' for use in the D3D_VERIFY_* macros. Add
 * this macro at the begin of functions that use these macros.
 */
#define USES_D3D_VERIFY HRESULT __d3dv_hr; __d3dv_hr = D3D_OK;
// Note: Extra assignment prevent "unused variable" warning.


/**
 * Determine whether the expression 'hr' represents success.
 *
 * @param hr An expression that yields a HRESULT.
 *
 * @return true in case 'hr' represents success, false otherwise.
 */
#ifdef SUCCEEDED
#define D3D_SUCCEEDED(hr) SUCCEEDED(hr)
#else /* SUCCEEDED */
#define D3D_SUCCEEDED(hr) (static_cast<HRESULT>(hr) >= 0)
#endif /* SUCCEEDED */


/**
 * Determine whether the expression 'hr' represents failure.
 *
 * @param hr An expression that yields a HRESULT.
 *
 * @return true in case 'hr' represents failure, false otherwise.
 */
#ifdef FAILED
#define D3D_FAILED(hr) FAILED(hr)
#else /* FAILED */
#define D3D_FAILED(hr) !D3D_SUCCEEDED(hr)
#endif /* FAILED */


/**
 * Assert that a call 'call' yields a successful HRESULT.
 *
 * @param call The call to test.
 */
//#ifdef V
//#define D3D_VERIFY(call) V(call)
//#else /* V */
#define D3D_VERIFY(call) VERIFY(D3D_SUCCEEDED(__d3dv_hr = (call)))
//#endif /* V */



//#ifdef V_RETURN
//#define D3D_VERIFY_RETURN(call) V_RETURN(call)
//#else /* V_RETURN */
#define D3D_VERIFY_RETURN(call) if (D3D_FAILED(__d3dv_hr = (call))) {          \
    return __d3dv_hr;                                                          \
}
//#endif /* V_RETURN */


#define D3D_VERIFY_THROW(call) if (D3D_FAILED(__d3dv_hr = (call))) {           \
    throw vislib::graphics::d3d::D3DException(__d3dv_hr, __FILE__, __LINE__);  \
}

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_D3DVERIFY_H_INCLUDED */
