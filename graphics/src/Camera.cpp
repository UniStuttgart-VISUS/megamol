/*
 * Camera.cpp
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2007, Sebastian Grottel. All rights reserved.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/Camera.h"
#include "vislib/assert.h"
#include "vislib/CameraParamsStore.h"
#include "vislib/IllegalStateException.h"


/*
 * vislib::graphics::Camera::Camera
 */
vislib::graphics::Camera::Camera(void) : syncNumber(), 
        parameters(new CameraParamsStore()) {
    this->syncNumber = this->parameters->SyncNumber() - 1; // force update
}


/*
 * vislib::graphics::Camera::Camera
 */
vislib::graphics::Camera::Camera(
        const vislib::SmartPtr<vislib::graphics::CameraParameters>& params) 
        : syncNumber(params->SyncNumber() - 1), parameters(params) {
    ASSERT(!this->parameters.IsNull());
}


/*
 * vislib::graphics::Camera::Camera
 */
vislib::graphics::Camera::Camera(const vislib::graphics::Camera &rhs) 
        : parameters() {
    *this = rhs;
}


/*
 * vislib::graphics::Camera::~Camera
 */
vislib::graphics::Camera::~Camera(void) {
}


/*
 * vislib::graphics::Camera::CalcViewFrustum
 */
vislib::graphics::SceneSpaceFrustum& 
vislib::graphics::Camera::CalcViewFrustum(SceneSpaceFrustum& outFrustum) {
    if (this->Parameters()->Projection() 
            == CameraParameters::MONO_ORTHOGRAPHIC) {
        throw vislib::IllegalStateException("Computing frustums for "
            "MONO_ORTHOGRAPHIC is currently unsupported.", __FILE__, __LINE__);
    }

    SceneSpaceViewFrustum tmp;
    outFrustum.Set(this->Parameters()->EyePosition(),
        this->Parameters()->EyeDirection(),
        this->Parameters()->EyeUpVector(),
        this->CalcViewFrustum(tmp));
    return outFrustum;
}


/*
 * vislib::graphics::Camera::CalcViewFrustum
 */
vislib::graphics::SceneSpaceViewFrustum& 
vislib::graphics::Camera::CalcViewFrustum(
        SceneSpaceViewFrustum& outFrustum) const {
    SceneSpaceType h;   // Height of the frustum.
    SceneSpaceType w;   // Width of the frustum.
    SceneSpaceType l;   // Left clipping plane.
    SceneSpaceType r;   // Right clipping plane.
    SceneSpaceType b;   // Bottom clipping plane.
    SceneSpaceType t;   // Top clipping plane.
    SceneSpaceType n;   // Near clipping plane.
    SceneSpaceType f;   // Far clipping plane.

    n = this->Parameters()->NearClip();
    f = this->Parameters()->FarClip();

    // TODO: This computation is taken from the OpenGL camera. I think this 
    // step should be identical for D3D, but check this.
    switch(this->Parameters()->Projection()) {

        case CameraParameters::MONO_PERSPECTIVE:
            /* falls through. */
        case CameraParameters::STEREO_PARALLEL: 
            /* falls through. */
        case CameraParameters::STEREO_TOE_IN: {
            // symmetric main frustum
            h = tan(this->Parameters()->HalfApertureAngle()) * n;
            w = h * this->Parameters()->VirtualViewSize().Width() 
                / this->Parameters()->VirtualViewSize().Height();

            // recalc tile rect on near clipping plane
            l = this->Parameters()->TileRect().GetLeft() 
                * w / (this->Parameters()->VirtualViewSize().Width() * 0.5f);
            r = this->Parameters()->TileRect().GetRight() 
                * w / (this->Parameters()->VirtualViewSize().Width() * 0.5f);
            b = this->Parameters()->TileRect().GetBottom() 
                * h / (this->Parameters()->VirtualViewSize().Height() * 0.5f);
            t = this->Parameters()->TileRect().GetTop() 
                * h / (this->Parameters()->VirtualViewSize().Height() * 0.5f);
          
            // cut out local frustum for tile rect
            l -= w;
            r -= w;
            b -= h;
            t -= h;
        } break;

        case CameraParameters::STEREO_OFF_AXIS: {
            // symmetric main frustum
            h = tan(this->Parameters()->HalfApertureAngle()) * n;
            w = h * this->Parameters()->VirtualViewSize().Width() 
                / this->Parameters()->VirtualViewSize().Height();

            // recalc tile rect on near clipping plane
            l = this->Parameters()->TileRect().GetLeft() 
                * w / (this->Parameters()->VirtualViewSize().Width() * 0.5f);
            r = this->Parameters()->TileRect().GetRight() 
                * w / (this->Parameters()->VirtualViewSize().Width() * 0.5f);
            b = this->Parameters()->TileRect().GetBottom() 
                * h / (this->Parameters()->VirtualViewSize().Height() * 0.5f);
            t = this->Parameters()->TileRect().GetTop() 
                * h / (this->Parameters()->VirtualViewSize().Height() * 0.5f);

            // shear frustum
            w += static_cast<SceneSpaceType>((
                (this->Parameters()->Eye() == CameraParameters::LEFT_EYE) 
                ? -1.0 : 1.0)
                * (n * this->Parameters()->StereoDisparity() * 0.5)
                / this->Parameters()->FocalDistance());

            // cut out local frustum for tile rect
            l -= w;
            r -= w;
            b -= h;
            t -= h;
        } break;

        case CameraParameters::MONO_ORTHOGRAPHIC:
            // return shifted tile
            l = this->Parameters()->TileRect().GetLeft() 
                - this->Parameters()->VirtualViewSize().Width() * 0.5f;
            r = this->Parameters()->TileRect().GetRight() 
                - this->Parameters()->VirtualViewSize().Width() * 0.5f;
            b = this->Parameters()->TileRect().GetBottom() 
                - this->Parameters()->VirtualViewSize().Height() * 0.5f;
            t = this->Parameters()->TileRect().GetTop() 
                - this->Parameters()->VirtualViewSize().Height() * 0.5f;
            break;

        default:
            throw IllegalStateException("The specified projection type is not "
                "supported.", __FILE__, __LINE__);
    }

    outFrustum.Set(l, r, b, t, n, f);
    return outFrustum;
}


/*
 * vislib::graphics::CameraParameters::Parameters
 */
vislib::SmartPtr<vislib::graphics::CameraParameters>& 
vislib::graphics::Camera::Parameters(void) {
    return this->parameters;
}


/*
 * vislib::graphics::CameraParameters::Parameters
 */
const vislib::SmartPtr<vislib::graphics::CameraParameters>&
vislib::graphics::Camera::Parameters(void) const {
    return this->parameters;
}


/*
 * vislib::graphics::Camera::SetParameters
 */
void vislib::graphics::Camera::SetParameters(const 
        vislib::SmartPtr<vislib::graphics::CameraParameters>& params) {
    this->parameters = params;
    this->syncNumber = this->parameters->SyncNumber() - 1; // force update
}


/*
 * vislib::graphics::Camera::Camera
 */
vislib::graphics::Camera& vislib::graphics::Camera::operator=(
        const vislib::graphics::Camera &rhs) {
    this->parameters = rhs.parameters;
    this->syncNumber = this->parameters->SyncNumber() - 1; // force update
    return *this;
}


/*
 * vislib::graphics::Camera::Camera
 */
bool vislib::graphics::Camera::operator==(
        const vislib::graphics::Camera &rhs) const {
    return ((this->parameters == rhs.parameters));
}
