/*
 * D3DCamera.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/D3DCamera.h"

#include "d3dx9math.h"


/*
 * vislib::graphics::d3d::D3DCamera::D3DCamera
 */
vislib::graphics::d3d::D3DCamera::D3DCamera(void) : Camera() {
    this->Parameters()->SetCoordSystemType(math::COORD_SYS_LEFT_HANDED);
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
 * vislib::graphics::d3d::D3DCamera::ApplyProjectionTransform
 */
HRESULT vislib::graphics::d3d::D3DCamera::ApplyProjectionTransform(
        IDirect3DDevice9 *device) const {
    D3DMatrix m;
    return device->SetTransform(D3DTS_PROJECTION, 
        static_cast<D3DXMATRIX *>(this->CalcProjectionMatrix(m)));
}


/*
 * vislib::graphics::d3d::D3DCamera::ApplyViewTransform
 */
HRESULT vislib::graphics::d3d::D3DCamera::ApplyViewTransform(
        IDirect3DDevice9 *device) const {
    D3DMatrix m;
    return device->SetTransform(D3DTS_VIEW, 
        static_cast<D3DXMATRIX *>(this->CalcViewMatrix(m)));
}


/*
 * vislib::graphics::d3d::D3DCamera::CalcProjectionMatrix
 */
vislib::graphics::d3d::D3DMatrix& 
vislib::graphics::d3d::D3DCamera::CalcProjectionMatrix(
        D3DMatrix& outMatrix, const bool isLeftHanded) const {
    this->updateCache();

    if (isLeftHanded) {
        ::D3DXMatrixPerspectiveOffCenterLH(
            static_cast<D3DXMATRIX *>(outMatrix),
            this->cacheFrustum.GetLeftDistance(),
            this->cacheFrustum.GetRightDistance(),
            this->cacheFrustum.GetBottomDistance(),
            this->cacheFrustum.GetTopDistance(),
            this->cacheFrustum.GetNearDistance(),
            this->cacheFrustum.GetFarDistance());

    } else {
        ::D3DXMatrixPerspectiveOffCenterRH(
            static_cast<D3DXMATRIX *>(outMatrix),
            this->cacheFrustum.GetLeftDistance(),
            this->cacheFrustum.GetRightDistance(),
            this->cacheFrustum.GetBottomDistance(),
            this->cacheFrustum.GetTopDistance(),
            this->cacheFrustum.GetNearDistance(),
            this->cacheFrustum.GetFarDistance());
    }
    return outMatrix;
}


/*
 * vislib::graphics::d3d::D3DCamera::CalcViewMatrix
 */
vislib::graphics::d3d::D3DMatrix& 
vislib::graphics::d3d::D3DCamera::CalcViewMatrix(D3DMatrix& outMatrix,
        const bool isLeftHanded) const {
    this->updateCache();
    SceneSpaceVector3D right = this->Parameters()->EyeRightVector();

    if (isLeftHanded) {
        ::D3DXMatrixLookAtLH(
            static_cast<D3DXMATRIX *>(outMatrix),
            static_cast<D3DXVECTOR3 *>(this->cacheEye),
            static_cast<D3DXVECTOR3 *>(this->cacheAt),
            static_cast<D3DXVECTOR3 *>(this->cacheUp));

    } else {
        ::D3DXMatrixLookAtRH(
            static_cast<D3DXMATRIX *>(outMatrix),
            static_cast<D3DXVECTOR3 *>(this->cacheEye),
            static_cast<D3DXVECTOR3 *>(this->cacheAt),
            static_cast<D3DXVECTOR3 *>(this->cacheUp));
    }

    return outMatrix;
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


/*
 * vislib::graphics::d3d::D3DCamera::updateCache
 */
bool vislib::graphics::d3d::D3DCamera::updateCache(void) const {
    if (this->needUpdate()) {
        this->cacheEye = this->Parameters()->EyePosition();
        this->cacheAt = this->cacheEye + this->Parameters()->EyeDirection();
        this->cacheUp = this->Parameters()->EyeUpVector();
        this->CalcViewFrustum(this->cacheFrustum);

        const_cast<D3DCamera*>(this)->markAsUpdated();
        return true;

    } else {
        return false;
    }
}
