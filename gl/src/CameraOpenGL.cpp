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
#include "vislib/memutils.h"
#include "vislib/ShallowMatrix.h"


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
 * vislib::graphics::gl::CameraOpenGL::GetProjectionMatrix
 */
void vislib::graphics::gl::CameraOpenGL::GetProjectionMatrix(float *mat) {
    if (this->NeedUpdate()) {
        this->viewNeedsUpdate = this->projNeedsUpdate = true;
        this->ClearUpdateFlaggs();
    }
    
    if (this->projNeedsUpdate) {
        Camera::CalcFrustumParameters(left, right, bottom, top, nearClip, farClip);
        this->projNeedsUpdate = false;
    }

    ZeroMemory(mat, sizeof(float) * 16);
    vislib::math::ShallowMatrix<float, 4, vislib::math::COLUMN_MAJOR> matrix(mat);

    matrix.SetAt(0, 0, (2.0f * this->nearClip) / (this->right - this->left));
    matrix.SetAt(1, 1, (2.0f * this->nearClip) / (this->top - this->bottom));
    matrix.SetAt(0, 2, (this->right + this->left) / (this->right - this->left));
    matrix.SetAt(1, 2, (this->top + this->bottom) / (this->top - this->bottom));
    matrix.SetAt(2, 2, - (this->farClip + this->nearClip) / (this->farClip - this->nearClip));
    matrix.SetAt(3, 2, -1.0f);
    matrix.SetAt(2, 3, - (2.0f * this->farClip * this->nearClip) / (this->farClip - this->nearClip));
}


/*
 * vislib::graphics::gl::CameraOpenGL::GetViewMatrix
 */
void vislib::graphics::gl::CameraOpenGL::GetViewMatrix(float *mat) {
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

    vislib::math::Vector<vislib::graphics::SceneSpaceType, 3> right = this->EyeRightVector();

    ZeroMemory(mat, sizeof(float) * 16);
    vislib::math::ShallowMatrix<float, 4, vislib::math::COLUMN_MAJOR> matrix(mat);

    matrix.SetAt(0, 0, right.GetX());
    matrix.SetAt(0, 1, right.GetY());
    matrix.SetAt(0, 2, right.GetZ());
    matrix.SetAt(1, 0, up.GetX());
    matrix.SetAt(1, 1, up.GetY());
    matrix.SetAt(1, 2, up.GetZ());
    matrix.SetAt(2, 0, -lookDir.GetX());
    matrix.SetAt(2, 1, -lookDir.GetY());
    matrix.SetAt(2, 2, -lookDir.GetZ());

}


/*
 * 
 */
vislib::graphics::gl::CameraOpenGL& vislib::graphics::gl::CameraOpenGL::operator=(const vislib::graphics::gl::CameraOpenGL& rhs) {
    Camera::operator=(rhs);
    return *this;
}
