/*
 * D3D9AdapterInformation.cpp
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/D3D9AdapterInformation.h"

#include "vislib/d3dverify.h"
#include "vislib/SystemException.h"


/*
 * vislib::graphics::d3d::D3D9AdapterInformation::GetAdapterInformation
 */
void vislib::graphics::d3d::D3D9AdapterInformation::GetAdapterInformation(
            vislib::PtrArray<D3D9AdapterInformation>& outAdapterInformation,
            IDirect3D9 *d3d) {
    VLSTACKTRACE("D3D9AdapterInformation::GetAdapterInformation",
        __FILE__, __LINE__);
    USES_D3D_VERIFY;
    D3DCAPS9 d3dCaps;           // Capabilities of the hardware.

    ASSERT(d3d != NULL);

    /* Clear output. */
    outAdapterInformation.Clear();

    /* Collect the information. */
    for (UINT i = 0; i < d3d->GetAdapterCount(); i++) {
        ::ZeroMemory(&d3dCaps, sizeof(d3dCaps));
        D3D_VERIFY_THROW(d3d->GetDeviceCaps(i, D3DDEVTYPE_HAL, &d3dCaps));

        if (d3dCaps.AdapterOrdinalInGroup == 0) {
            outAdapterInformation.Add(new D3D9AdapterInformation(
                d3d, d3dCaps.AdapterOrdinal));
        }
    }
}


/*
 * vislib::graphics::d3d::D3D9AdapterInformation::D3D9AdapterInformation
 */
vislib::graphics::d3d::D3D9AdapterInformation::D3D9AdapterInformation(
        IDirect3D9 *d3d, const UINT adapterOrdinal) {
    VLSTACKTRACE("D3D9AdapterInformation::D3D9AdapterInformation",
        __FILE__, __LINE__);

    HRESULT hr = D3D_OK;            // Direct3D API results.
    Output output;                  // Receives current output infos.
    HMONITOR hMon = NULL;           // Monitor pseudo handle.
    UINT masterAdapterOrdinal = 0;  // ID of the master adapter.

    ASSERT(d3d != NULL);

    /* 
     * Get the ID of the master adapter. In order to get all outputs/swap 
     * chains, we must be sure that the ID the user passed in is the ID of
     * a master adpeter. If this is not the case, we correct it silently.
     */
    masterAdapterOrdinal = adapterOrdinal;
    ::ZeroMemory(&output.d3dCaps, sizeof(output.d3dCaps));
    if (D3D_FAILED(hr = d3d->GetDeviceCaps(adapterOrdinal, D3DDEVTYPE_HAL,
            &output.d3dCaps))) {
        this->~D3D9AdapterInformation();
        throw D3DException(hr, __FILE__, __LINE__);
    }
    if (output.d3dCaps.AdapterOrdinalInGroup != 0) {
        masterAdapterOrdinal = output.d3dCaps.MasterAdapterOrdinal;
    }

    /* 
     * Enumerate all adapters that belong to the adapter group we identified 
     * above.
     */
    for (UINT i = 0; i < d3d->GetAdapterCount(); i++) {
        ::ZeroMemory(&output, sizeof(output));
        output.monInfo.cbSize = sizeof(output.monInfo);

        if (D3D_FAILED(hr = d3d->GetDeviceCaps(i, D3DDEVTYPE_HAL, 
                &output.d3dCaps))) {
            this->~D3D9AdapterInformation();
            throw D3DException(hr, __FILE__, __LINE__);
        }

        if ((output.d3dCaps.AdapterOrdinal == masterAdapterOrdinal)
                || (output.d3dCaps.MasterAdapterOrdinal 
                == masterAdapterOrdinal)) {
            hMon = d3d->GetAdapterMonitor(output.d3dCaps.AdapterOrdinal);
            if (!::GetMonitorInfo(hMon, &output.monInfo)) {
                this->~D3D9AdapterInformation();
                throw vislib::sys::SystemException(__FILE__, __LINE__);
            }

            this->infos.Add(output);
        }
    }

    this->infos.Trim();
}


/*
 * vislib::graphics::d3d::D3D9AdapterInformation::~D3D9AdapterInformation
 */
vislib::graphics::d3d::D3D9AdapterInformation::~D3D9AdapterInformation(void) {
    VLSTACKTRACE("D3D9AdapterInformation::D3D9AdapterInformation",
        __FILE__, __LINE__);
    this->infos.Clear(true);
}


/*
 * vislib::graphics::d3d::D3D9AdapterInformation::GetDirect3DCapabilites
 */
const D3DCAPS9& 
vislib::graphics::d3d::D3D9AdapterInformation::GetDirect3DCapabilites(
        const SIZE_T outputIdx) {
    VLSTACKTRACE("D3D9AdapterInformation::GetDirect3DCapabilites", __FILE__, 
        __LINE__);

    /* Range check. */
    if (outputIdx > this->infos.Count()) {
        throw OutOfRangeException(outputIdx, 0, 
            static_cast<int>(this->infos.Count()), __FILE__, __LINE__);
    }

    return this->infos[outputIdx].d3dCaps;
}


/*
 * vislib::graphics::d3d::D3D9AdapterInformation::GetOutputCount
 */
SIZE_T vislib::graphics::d3d::D3D9AdapterInformation::GetOutputCount(
        void) const {
    VLSTACKTRACE("D3D9AdapterInformation::GetOutputCount", __FILE__, __LINE__);
    return this->infos.Count();
}


/*
 * vislib::graphics::d3d::D3D9AdapterInformation::operator =
 */
vislib::graphics::d3d::D3D9AdapterInformation& 
vislib::graphics::d3d::D3D9AdapterInformation::operator =(
        const D3D9AdapterInformation& rhs) {
    VLSTACKTRACE("D3D9AdapterInformation::operator =", __FILE__, __LINE__);

    if (this != &rhs) {
        this->infos = rhs.infos;
    }

    return *this;
}


/*
 * vislib::graphics::d3d::D3D9AdapterInformation::getMonitorInfo
 */
const MONITORINFOEXW&
vislib::graphics::d3d::D3D9AdapterInformation::getMonitorInfo(
        const SIZE_T outputIdx) const {
    VLSTACKTRACE("D3D9AdapterInformation::getMonitorInfo", __FILE__, __LINE__);

    for (SIZE_T i = 0; i < this->infos.Count(); i++) {
        if (this->infos[i].d3dCaps.AdapterOrdinalInGroup == outputIdx) {
            return (this->infos[i].monInfo);
        }
    }
    /* Not found. */

    throw OutOfRangeException(outputIdx, 0, 
        static_cast<int>(this->infos.Count()), __FILE__, __LINE__);
}
