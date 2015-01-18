/*
 * D3D9TestBoxGeometry.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "DXUT.h"
#include "D3D9TestBoxGeometry.h"

#include "vislib/d3dverify.h"


/*
 * D3D9TestBoxGeometry::D3D9TestBoxGeometry
 */
D3D9TestBoxGeometry::D3D9TestBoxGeometry(void) 
        : boxX(NULL), boxY(NULL), boxZ(NULL) {
}


/*
 * D3D9TestBoxGeometry::~D3D9TestBoxGeometry
 */
D3D9TestBoxGeometry::~D3D9TestBoxGeometry(void) {
    this->Release();
}


/*
 * D3D9TestBoxGeometry::Create
 */
void D3D9TestBoxGeometry::Create(IDirect3DDevice9 *device) {
    USES_D3D_VERIFY;
    D3D_VERIFY_THROW(::D3DXCreateBox(device, 1.0f, 0.1f, 0.1f, &this->boxX, 
        NULL));
    D3D_VERIFY_THROW(::D3DXCreateBox(device, 0.1f, 1.0f, 0.1f, &this->boxY,
        NULL));
    D3D_VERIFY_THROW(::D3DXCreateBox(device, 0.1f, 0.1f, 1.0f, &this->boxZ,
        NULL));
}


/*
 * D3D9TestBoxGeometry::Draw
 */
void D3D9TestBoxGeometry::Draw(void) {
    USES_D3D_VERIFY;
    LPDIRECT3DDEVICE9 device = NULL;
    D3DXMATRIX m;
    
    D3D_VERIFY_THROW(this->boxX->GetDevice(&device));
    device->SetTextureStageState(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1);
	device->SetTextureStageState(0, D3DTSS_COLORARG1, D3DTA_CONSTANT);

    device->SetTextureStageState(0, D3DTSS_CONSTANT, D3DCOLOR_XRGB(255, 0, 0));
    device->SetTransform(D3DTS_WORLD, D3DXMatrixTranslation(&m, 
        0.6f, 0.0f, 0.0f));
    D3D_VERIFY_THROW(this->boxX->DrawSubset(0));

    device->SetTextureStageState(0, D3DTSS_CONSTANT, D3DCOLOR_XRGB(0, 255, 0));
    device->SetTransform(D3DTS_WORLD, D3DXMatrixTranslation(&m, 
        0.0f, 0.6f, 0.0f));
    D3D_VERIFY_THROW(this->boxY->DrawSubset(0));

    device->SetTextureStageState(0, D3DTSS_CONSTANT, D3DCOLOR_XRGB(0, 0, 255));
    device->SetTransform(D3DTS_WORLD, D3DXMatrixTranslation(&m, 
        0.0f, 0.0f, 0.6f));
    D3D_VERIFY_THROW(this->boxZ->DrawSubset(0));

    device->SetTextureStageState(0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
    device->SetTextureStageState(0, D3DTSS_CONSTANT, D3DCOLOR_XRGB(255, 255,
        255));
    device->SetTransform(D3DTS_WORLD, D3DXMatrixIdentity(&m)); 
    SAFE_RELEASE(device);
}


/*
 * D3D9TestBoxGeometry::Release
 */
void D3D9TestBoxGeometry::Release(void) {
    SAFE_RELEASE(this->boxX);
    SAFE_RELEASE(this->boxY);
    SAFE_RELEASE(this->boxZ);
}
