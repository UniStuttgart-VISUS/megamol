/*
 * CallUpdateDirect3D.cpp
 *
 * Copyright (C) 2012 by Visualisierungsinstitut der Universit�t Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/view/CallUpdateDirect3D.h"

#ifdef MEGAMOLCORE_WITH_DIRECT3D11
#include "vislib/graphics/d3d/d3dutils.h"
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */

/*
 * megamol::core::view::CallUpdateDirect3D::FunctionName
 */
const char *megamol::core::view::CallUpdateDirect3D::FunctionName(
        unsigned int idx) {
    if (idx < CallUpdateDirect3D::FunctionCount()) {
        return CallUpdateDirect3D::FUNCTIONS[idx];
    } else {
        return NULL;
    }
}


/*
 * megamol::core::view::CallUpdateDirect3D::CallUpdateDirect3D
 */
megamol::core::view::CallUpdateDirect3D::CallUpdateDirect3D(void) : Base(), 
        device(NULL) {
}


/*
 * megamol::core::view::CallUpdateDirect3D::~CallUpdateDirect3D
 */
megamol::core::view::CallUpdateDirect3D::~CallUpdateDirect3D(void) {
#ifdef MEGAMOLCORE_WITH_DIRECT3D11
    SAFE_RELEASE(this->device);
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
}


/*
 * megamol::core::view::CallUpdateDirect3D::SetDevice
 */
void megamol::core::view::CallUpdateDirect3D::SetDevice(ID3D11Device *device) {
    ASSERT(device != NULL);

#ifdef MEGAMOLCORE_WITH_DIRECT3D11
    /* Increment reference counter first (in case of setting same objects)! */
    device->AddRef();
    SAFE_RELEASE(this->device);
    this->device = device;
#endif /* MEGAMOLCORE_WITH_DIRECT3D11 */
}


/*
 * megamol::core::view::CallUpdateDirect3D::operator =
 */
megamol::core::view::CallUpdateDirect3D&
megamol::core::view::CallUpdateDirect3D::operator =(
        const CallUpdateDirect3D& rhs) {
    Base::operator =(rhs);
    if (this != &rhs) {
        this->SetDevice(rhs.device);
    }
    return *this;
}


/*
 * megamol::core::view::CallUpdateDirect3D::FUNCTIONS
 */
const char *megamol::core::view::CallUpdateDirect3D::FUNCTIONS[] = {
    "UpdateDevice"
};
