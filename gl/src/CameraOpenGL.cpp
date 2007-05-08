/*
 * CameraOpenGL.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#ifdef _WIN32
#include <windows.h>
#else
#endif
#include "vislib/CameraOpenGL.h"


/*
 * vislib::graphics::gl::CameraOpenGL::CameraOpenGL
 */
vislib::graphics::gl::CameraOpenGL::CameraOpenGL(void) 
    : Camera(), viewNeedsUpdate(true), projNeedsUpdate(true) {
}


/*
 * vislib::graphics::gl::CameraOpenGL::CameraOpenGL
 */
vislib::graphics::gl::CameraOpenGL::CameraOpenGL(const vislib::graphics::gl::CameraOpenGL& rhs) 
    : Camera(rhs), viewNeedsUpdate(true), projNeedsUpdate(true) {
}


/*
 * vislib::graphics::gl::CameraOpenGL::CameraOpenGL
 */

vislib::graphics::gl::CameraOpenGL::CameraOpenGL(vislib::graphics::Beholder* beholder) 
    : Camera(beholder), viewNeedsUpdate(true), projNeedsUpdate(true) {
}


/*
 * vislib::graphics::gl::CameraOpenGL::~CameraOpenGL
 */
vislib::graphics::gl::CameraOpenGL::~CameraOpenGL(void) {
}


/*
 * 
 */
void vislib::graphics::gl::CameraOpenGL::glMultProjectionMatrix(void) {
    if (this->NeedUpdate()) {
        this->viewNeedsUpdate = this->projNeedsUpdate = true;
        this->ClearUpdateFlaggs();
    }
    
    if (this->projNeedsUpdate) {
        Camera::CalcFrustumParameters(left, right, bottom, top, nearClip, farClip);
        this->projNeedsUpdate = false;
    }

    if (this->GetProjectionType() != Camera::MONO_ORTHOGRAPHIC) {
        ::glFrustum(left, right, bottom, top, nearClip, farClip);
    } else {
        ::glOrtho(left, right, bottom, top, nearClip, farClip);
    }
}


/*
 * 
 */
void vislib::graphics::gl::CameraOpenGL::glMultViewMatrix(void) {
    if (this->NeedUpdate()) {
        this->viewNeedsUpdate = this->projNeedsUpdate = true;
        this->ClearUpdateFlaggs();
    }

    if (this->viewNeedsUpdate) {
        pos = this->EyePosition();
        lookDir = this->EyeFrontVector();
        up = this->EyeUpVector();
        this->viewNeedsUpdate = false;
    }

    ::gluLookAt(pos.X(), pos.Y(), pos.Z(), 
        pos.X() + lookDir.X(), pos.Y() + lookDir.Y(), pos.Z() + lookDir.Z(), 
        up.X(), up.Y(), up.Z());
}


/*
 * 
 */
vislib::graphics::gl::CameraOpenGL& vislib::graphics::gl::CameraOpenGL::operator=(const vislib::graphics::gl::CameraOpenGL& rhs) {
    Camera::operator=(rhs);
    return *this;
}
