/*
 * CameraOpenGL.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 */

#include "vislib/CameraOpenGL.h"

#include "glh/glh_genext.h"
#include <GL/glu.h>

#include "vislib/assert.h"
#include "vislib/ShallowMatrix.h"


/*
 * vislib::graphics::gl::CameraOpenGL::CameraOpenGL
 */
vislib::graphics::gl::CameraOpenGL::CameraOpenGL(void) : Camera() {
    this->updateMembers();
}


/*
 * vislib::graphics::gl::CameraOpenGL::CameraOpenGL
 */
vislib::graphics::gl::CameraOpenGL::CameraOpenGL(
        const vislib::SmartPtr<vislib::graphics::CameraParameters>& params) 
        : Camera(params) {
    this->updateMembers();
}


/*
 * vislib::graphics::gl::CameraOpenGL::CameraOpenGL
 */
vislib::graphics::gl::CameraOpenGL::CameraOpenGL(
        const vislib::graphics::Camera& rhs) : Camera(rhs) {
    *this = rhs;
}


/*
 * vislib::graphics::gl::CameraOpenGL::~CameraOpenGL
 */
vislib::graphics::gl::CameraOpenGL::~CameraOpenGL(void) {
}


/*
 * vislib::graphics::gl::CameraOpenGL::glMultProjectionMatrix
 */
void vislib::graphics::gl::CameraOpenGL::glMultProjectionMatrix(void) const {
    if (this->needUpdate()) {
        this->updateMembers();
    }

    if (this->Parameters()->Projection() != CameraParameters::MONO_ORTHOGRAPHIC) {
        ::glFrustum(left, right, bottom, top, nearClip, farClip);
    } else {
        // TODO: write alternative ortho to be more compatible with normal projection
        ::glOrtho(left, right, bottom, top, nearClip, farClip);
    }
}


/*
 * vislib::graphics::gl::CameraOpenGL::glMultViewMatrix
 */
void vislib::graphics::gl::CameraOpenGL::glMultViewMatrix(void) const {
    if (this->needUpdate()) {
        this->updateMembers();
    }

    ::gluLookAt(pos.X(), pos.Y(), pos.Z(), 
        pos.X() + lookDir.X(), pos.Y() + lookDir.Y(), pos.Z() + lookDir.Z(), 
        up.X(), up.Y(), up.Z());
}


/*
 * vislib::graphics::gl::CameraOpenGL::glSetProjectionMatrix
 */
void vislib::graphics::gl::CameraOpenGL::glSetProjectionMatrix(void) const {
    ::glMatrixMode(GL_PROJECTION);
    ::glLoadIdentity();
    this->glMultProjectionMatrix();
}


/*
 * vislib::graphics::gl::CameraOpenGL::glSetViewMatrix
 */
void vislib::graphics::gl::CameraOpenGL::glSetViewMatrix(void) const {
    ::glMatrixMode(GL_MODELVIEW);
    ::glLoadIdentity();
    this->glMultViewMatrix();
}


/*
 * vislib::graphics::gl::CameraOpenGL::ProjectionMatrix
 */
void vislib::graphics::gl::CameraOpenGL::ProjectionMatrix(float *mat) const {
    if (this->needUpdate()) {
        this->updateMembers();
    }

    ZeroMemory(mat, sizeof(float) * 16);
    vislib::math::ShallowMatrix<float, 4, vislib::math::COLUMN_MAJOR> matrix(mat);

    if (this->Parameters()->Projection() != CameraParameters::MONO_ORTHOGRAPHIC) {
        matrix.SetAt(0, 0, (2.0f * this->nearClip) / (this->right - this->left));
        matrix.SetAt(1, 1, (2.0f * this->nearClip) / (this->top - this->bottom));
        matrix.SetAt(0, 2, (this->right + this->left) / (this->right - this->left));
        matrix.SetAt(1, 2, (this->top + this->bottom) / (this->top - this->bottom));
        matrix.SetAt(2, 2, - (this->farClip + this->nearClip) / (this->farClip - this->nearClip));
        matrix.SetAt(3, 2, -1.0f);
        matrix.SetAt(2, 3, - (2.0f * this->farClip * this->nearClip) / (this->farClip - this->nearClip));
    } else {
        // TODO: write alternative ortho to be more compatible with normal projection
        matrix.SetAt(0, 0, 2.0f / (this->right - this->left));
        matrix.SetAt(1, 1, 2.0f / (this->top - this->bottom));
        matrix.SetAt(2, 2, -2.0f / (this->farClip - this->nearClip));
        matrix.SetAt(0, 3, (this->right + this->left) / (this->right - this->left));
        matrix.SetAt(1, 3, (this->top + this->bottom) / (this->top - this->bottom));
        matrix.SetAt(2, 3, (this->farClip + this->nearClip) / (this->farClip - this->nearClip));
        matrix.SetAt(3, 3, -1.0f);
    }
}


/*
 * vislib::graphics::gl::CameraOpenGL::ViewMatrix
 */
void vislib::graphics::gl::CameraOpenGL::ViewMatrix(float *mat) const {
    if (this->needUpdate()) {
        this->updateMembers();
    }

    SceneSpacePoint3D eye = this->Parameters()->EyePosition();

    SceneSpaceVector3D zAxis(eye - this->Parameters()->LookAt());
    zAxis.Normalise();

    SceneSpaceVector3D xAxis = this->Parameters()->EyeUpVector().Cross(zAxis);
    xAxis.Normalise();

    SceneSpaceVector3D yAxis = zAxis.Cross(xAxis);
    yAxis.Normalise();

    ::ZeroMemory(mat, sizeof(float) * 16);
    math::ShallowMatrix<float, 4, math::COLUMN_MAJOR> matrix(mat);

    matrix.SetAt(0, 0, xAxis.GetX());
    matrix.SetAt(0, 1, xAxis.GetY());
    matrix.SetAt(0, 2, xAxis.GetZ());
    matrix.SetAt(1, 0, yAxis.GetX());
    matrix.SetAt(1, 1, yAxis.GetY());
    matrix.SetAt(1, 2, yAxis.GetZ());
    matrix.SetAt(2, 0, zAxis.GetX());
    matrix.SetAt(2, 1, zAxis.GetY());
    matrix.SetAt(2, 2, zAxis.GetZ());
    matrix.SetAt(3, 3, 1.0f);

    matrix.SetAt(0, 3, -xAxis.Dot(eye));
    matrix.SetAt(1, 3, -yAxis.Dot(eye));
    matrix.SetAt(2, 3, -zAxis.Dot(eye));
}


/*
 * vislib::graphics::gl::CameraOpenGL::operator=
 */
vislib::graphics::gl::CameraOpenGL& 
vislib::graphics::gl::CameraOpenGL::operator=(
        const vislib::graphics::Camera &rhs) {
    Camera::operator=(rhs);
    this->updateMembers();
    return *this;
}


/*
 * vislib::graphics::gl::CameraOpenGL::operator==
 */
bool vislib::graphics::gl::CameraOpenGL::operator==(
        const vislib::graphics::Camera &rhs) const {
    return Camera::operator==(rhs);
}


/*
 * vislib::graphics::gl::CameraOpenGL::updateMembers
 */
void vislib::graphics::gl::CameraOpenGL::updateMembers(void) const {
    SceneSpaceType w, h;

    // view
    this->pos = this->Parameters()->EyePosition();
    this->lookDir = this->Parameters()->EyeDirection();
    this->up = this->Parameters()->EyeUpVector();

    // clipping distances
    this->nearClip = this->Parameters()->NearClip();
    this->farClip = this->Parameters()->FarClip();

    switch(this->Parameters()->Projection()) {
        case CameraParameters::MONO_PERSPECTIVE: // no break
        case CameraParameters::STEREO_PARALLEL: // no break
        case CameraParameters::STEREO_TOE_IN: {
            // symmetric main frustum
            h = tan(this->Parameters()->HalfApertureAngle()) * this->nearClip;
            w = h * this->Parameters()->VirtualViewSize().Width() 
                / this->Parameters()->VirtualViewSize().Height();

            // recalc tile rect on near clipping plane
            this->left = this->Parameters()->TileRect().GetLeft() 
                * w / (this->Parameters()->VirtualViewSize().Width() * 0.5f);
            this->right = this->Parameters()->TileRect().GetRight() 
                * w / (this->Parameters()->VirtualViewSize().Width() * 0.5f);
            this->bottom = this->Parameters()->TileRect().GetBottom() 
                * h / (this->Parameters()->VirtualViewSize().Height() * 0.5f);
            this->top = this->Parameters()->TileRect().GetTop() 
                * h / (this->Parameters()->VirtualViewSize().Height() * 0.5f);
          
            // cut out local frustum for tile rect
            this->left -= w;
            this->right -= w;
            this->bottom -= h;
            this->top -= h;
        } break;
        case CameraParameters::STEREO_OFF_AXIS: {
            // symmetric main frustum
            h = tan(this->Parameters()->HalfApertureAngle()) * this->nearClip;
            w = h * this->Parameters()->VirtualViewSize().Width() 
                / this->Parameters()->VirtualViewSize().Height();

            // recalc tile rect on near clipping plane
            this->left = this->Parameters()->TileRect().GetLeft() 
                * w / (this->Parameters()->VirtualViewSize().Width() * 0.5f);
            this->right = this->Parameters()->TileRect().GetRight() 
                * w / (this->Parameters()->VirtualViewSize().Width() * 0.5f);
            this->bottom = this->Parameters()->TileRect().GetBottom() 
                * h / (this->Parameters()->VirtualViewSize().Height() * 0.5f);
            this->top = this->Parameters()->TileRect().GetTop() 
                * h / (this->Parameters()->VirtualViewSize().Height() * 0.5f);

            // shear frustum
            w += static_cast<SceneSpaceType>(((this->Parameters()->Eye() == CameraParameters::LEFT_EYE) ? -1.0 : 1.0))
                * (this->nearClip * this->Parameters()->StereoDisparity() * 0.5f) 
                / this->Parameters()->FocalDistance();

            // cut out local frustum for tile rect
            this->left -= w;
            this->right -= w;
            this->bottom -= h;
            this->top -= h;
        } break;
        case CameraParameters::MONO_ORTHOGRAPHIC:
            // return shifted tile
            this->left = this->Parameters()->TileRect().GetLeft() 
                - this->Parameters()->VirtualViewSize().Width() * 0.5f;
            this->right = this->Parameters()->TileRect().GetRight() 
                - this->Parameters()->VirtualViewSize().Width() * 0.5f;
            this->bottom = this->Parameters()->TileRect().GetBottom() 
                - this->Parameters()->VirtualViewSize().Height() * 0.5f;
            this->top = this->Parameters()->TileRect().GetTop() 
                - this->Parameters()->VirtualViewSize().Height() * 0.5f;
            break;
        default:
            // projection parameter calculation still not implemeneted
            ASSERT(false);
    }
    const_cast<CameraOpenGL*>(this)->markAsUpdated();
}
