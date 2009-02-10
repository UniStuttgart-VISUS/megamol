/*
 * D3DException.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/D3DException.h"

#include <dxerr.h>


/*
 * vislib::graphics::d3d::D3DException::D3DException
 */
vislib::graphics::d3d::D3DException::D3DException(const HRESULT result, 
        const char *file, const int line) 
        : vislib::Exception(::DXGetErrorString(result), file, line), 
        result(result) {
}


/*
 * vislib::graphics::d3d::D3DException::D3DException
 */
vislib::graphics::d3d::D3DException::D3DException(const D3DException& rhs) 
        : vislib::Exception(rhs), result(rhs.result) {
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
        Exception::operator =(rhs);
        this->result = rhs.result;
    }
    return *this;
}
