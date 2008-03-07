/*
 * D3DCamera.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/D3DCamera.h"


/*
 * vislib::graphics::d3d::D3DCamera::D3DCamera
 */
vislib::graphics::d3d::D3DCamera::D3DCamera(void) : Camera() {
}


/*
 * vislib::graphics::d3d::D3DCamera::D3DCamera
 */
vislib::graphics::d3d::D3DCamera::D3DCamera(
        const SmartPtr<CameraParameters>& params) : Camera(params) {
}


/*
 * vislib::graphics::d3d::D3DCamera::~D3DCamera
 */
vislib::graphics::d3d::D3DCamera::~D3DCamera(void) {
}


/*
 * vislib::graphics::d3d::D3DCamera::operator =
 */
vislib::graphics::d3d::D3DCamera&
vislib::graphics::d3d::D3DCamera::operator =(const D3DCamera& rhs) {
    Camera::operator =(rhs);
    return *this;
}


/*
 * vislib::graphics::d3d::D3DCamera::operator ==
 */
bool vislib::graphics::d3d::D3DCamera::operator ==(const D3DCamera& rhs) const {
    return Camera::operator ==(rhs);
}