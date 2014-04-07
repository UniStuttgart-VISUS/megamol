/*
 * D3DException.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/D3DException.h"

#ifdef HAVE_LEGACY_DIRECTX_SDK
#include <dxerr.h>
#endif /* HAVE_LEGACY_DIRECTX_SDK */


/*
 * vislib::graphics::d3d::D3DException::D3DException
 */
vislib::graphics::d3d::D3DException::D3DException(const HRESULT result, 
        const char *file, const int line) 
#ifdef HAVE_LEGACY_DIRECTX_SDK
        : the::exception(::DXGetErrorString(result), file, line),
#else /* HAVE_LEGACY_DIRECTX_SDK */
        : the::exception(file, line),
#endif /* HAVE_LEGACY_DIRECTX_SDK */
        result(result) {
}


/*
 * vislib::graphics::d3d::D3DException::D3DException
 */
vislib::graphics::d3d::D3DException::D3DException(const D3DException& rhs) 
        : the::exception(rhs), result(rhs.result) {
}


/*
 * vislib::graphics::d3d::D3DException::~D3DException
 */
vislib::graphics::d3d::D3DException::~D3DException(void) {
}


/*
 * vislib::graphics::d3d::D3DException::operator =
 */
vislib::graphics::d3d::D3DException& 
vislib::graphics::d3d::D3DException::operator =(const D3DException& rhs) {
    if (this != &rhs) {
        the::exception::operator =(rhs);
        this->result = rhs.result;
    }
    return *this;
}
