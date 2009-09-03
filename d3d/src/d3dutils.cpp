/*
 * d3dutils.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/d3dutils.h"

#include "vislib/D3DException.h"
#include "vislib/IllegalParamException.h"
#include "vislib/StackTrace.h"


/**
 * Answer whether the resolution of the display mode 'displayMode' is larger 
 * than the given resolution of 'width' x 'height' with respect to 'criterion'.
 * If both are equal, false is returned.
 *
 * @param displayMode The display mode to test.
 * @param width       The width of the reference resolution.
 * @param height      The height of the reference resolution.
 * @param criterion   The comparison criterion.
 *
 * @return true if 'displayMode' has a larger resolution, false otherwise
 *         (also if equal).
 */
static bool IsDisplayModeLargerThan(const D3DDISPLAYMODE& displayMode, 
        const UINT width, const UINT height,
        const vislib::graphics::d3d::FullscreenSizeCriterion criterion) {
    using namespace vislib::graphics::d3d;

    switch (criterion) {
        case CRITERION_AREA:
            return ((displayMode.Width * displayMode.Height) 
                > (width * height));

        case CRITERION_WIDTH:
            return (displayMode.Width > width);

        case CRITERION_HEIGHT:
            return (displayMode.Height > height);

        default:
            throw vislib::IllegalParamException("criterion", __FILE__, 
                __LINE__);
    }
}


/*
 * vislib::graphics::d3d::GetBackbufferSize
 */
void vislib::graphics::d3d::GetBackbufferSize(UINT& outWidth, UINT& outHeight,
        IDirect3DDevice9 *device) {
    VLSTACKTRACE("GetBackbufferSize", __FILE__, __LINE__);
    using vislib::graphics::d3d::D3DException;

    HRESULT hr = D3D_OK;
    LPDIRECT3DSURFACE9 backBuffer = NULL;
    D3DSURFACE_DESC backBufferDesc;

    if (device == NULL) {
        throw IllegalParamException("device", __FILE__, __LINE__);
    }

    if (FAILED(hr = device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, 
            &backBuffer))) {
        throw D3DException(hr, __FILE__, __LINE__);
    }

    if (FAILED(hr = backBuffer->GetDesc(&backBufferDesc))) {
        backBuffer->Release();
        throw D3DException(hr, __FILE__, __LINE__);
    }
    backBuffer->Release();

    outWidth = backBufferDesc.Width;
    outHeight = backBufferDesc.Height;
}


/*
 * vislib::graphics::d3d::GetMaximumFullscreenResolution
 */
void vislib::graphics::d3d::GetMaximumFullscreenResolution(
        UINT& outWidth,
        UINT& outHeight, 
        UINT& outRefreshRate,
        IDirect3D9 *d3d, 
        const UINT adapterID, 
        const D3DFORMAT& format,
        const FullscreenSizeCriterion criterion) {
    VLSTACKTRACE("GetMaximumFullscreenResolution", __FILE__, __LINE__);
    using vislib::graphics::d3d::D3DException;

    HRESULT hr = D3D_OK;
    UINT cntAdapterModes = 0;
    D3DDISPLAYMODE displayMode;

    /* Sanity checks. */
    if (d3d == NULL) {
        throw IllegalParamException("d3d", __FILE__, __LINE__);
    }

    outWidth = outHeight = outRefreshRate = 0;
    cntAdapterModes = d3d->GetAdapterModeCount(adapterID, format); 
    for (UINT i = 0; i < cntAdapterModes; i++) {
        if (FAILED(hr = d3d->EnumAdapterModes(adapterID, format, i, 
                &displayMode))) {
            throw D3DException(hr, __FILE__, __LINE__);
        }
        ASSERT(displayMode.Format == format);

        if (::IsDisplayModeLargerThan(displayMode, outWidth, outHeight, 
                criterion)) {
            outWidth = displayMode.Width;
            outHeight = displayMode.Height;
            outRefreshRate = displayMode.RefreshRate;
        }
    }
}


/*
 * vislib::graphics::d3d::GetMaximumSharedFullscreenResolution
 */
bool vislib::graphics::d3d::GetMaximumSharedFullscreenResolution(
        UINT& outWidth, 
        UINT& outHeight, 
        UINT& outRefreshRate,
        IDirect3D9 *d3d, 
        const D3DFORMAT& format,
        const FullscreenSizeCriterion criterion) {
    VLSTACKTRACE("GetMaximumSharedFullscreenResolution", __FILE__, __LINE__);
    using vislib::graphics::d3d::D3DException;

    HRESULT hr = D3D_OK;                // API return values.
    bool retval = false;                // Remember whether match was found.
    UINT cntAdapters = 0;               // The number of adapters found.
    UINT cntAdapterModesRef = 0;        // # of modes of adapter 0.
    UINT cntAdapterModesOther = 0;      // # of modes of each other adapter.
    D3DDISPLAYMODE displayModeRef;      // Display mode of adapter 0.
    D3DDISPLAYMODE displayModeOther;    // Display mode of each other adapter.

    /* Sanity checks. */
    if (d3d == NULL) {
        throw IllegalParamException("d3d", __FILE__, __LINE__);
    }

    cntAdapters = d3d->GetAdapterCount();
    outWidth = outHeight = outRefreshRate = 0;

    /* Get adapter 0 as reference. */
    cntAdapterModesRef = d3d->GetAdapterModeCount(0, format); 
    for (UINT i = 0; i < cntAdapterModesRef; i++) {
        if (FAILED(hr = d3d->EnumAdapterModes(0, format, i, &displayModeRef))) {
            throw D3DException(hr, __FILE__, __LINE__);
        }
        ASSERT(displayModeRef.Format == format);

        if (::IsDisplayModeLargerThan(displayModeRef, outWidth, outHeight,
                criterion)) {
            retval = true;

            /* Check whether all other adapters support this mode. */
            for (UINT j = 1; j < cntAdapters; j++) {
                cntAdapterModesOther = d3d->GetAdapterModeCount(j, format);
                for (UINT k = 0; k < cntAdapterModesOther; k++) {
                    if (FAILED(hr = d3d->EnumAdapterModes(j, format, k, 
                            &displayModeOther))) {
                        throw D3DException(hr, __FILE__, __LINE__);
                    }
                    ASSERT(displayModeOther.Format == format);                
                
                    if ((displayModeRef.Width != displayModeOther.Width)
                            || (displayModeRef.Height 
                            != displayModeOther.Height)
                            || (displayModeRef.RefreshRate 
                            != displayModeOther.RefreshRate)) {
                        retval = false;
                        break;
                    }
                } /* end for (UINT k = 0; k < cntAdapterModesOther; k++) */

                /* If one adapter failed, we do not need to check the others. */
                if (!retval) {
                    break;
                }
            } /* end for (UINT j = 1; j < cntAdapters; j++) */

            /* This one is good, so remember it as possible solution. */
            if (retval) {
                outWidth = displayModeRef.Width;
                outHeight = displayModeRef.Height;
                outRefreshRate = displayModeRef.RefreshRate;
            }
        } /* end if (::IsDisplayModeLargerThan(displayModeRef, outWidth, ... */
    } /* end for (UINT i = 0; i < cntAdapterModesRef; i++) */

    return retval;
}


/*
 * vislib::graphics::d3d::IsFullscreenResolutionSupported
 */
bool vislib::graphics::d3d::IsFullscreenResolutionSupported(
        IDirect3D9 *d3d,
        const UINT adapterID, 
        const D3DFORMAT& format,
        const UINT width,
        const UINT height,
        UINT *outRefreshRate) {
    VLSTACKTRACE("GetMaximumFullscreenResolution", __FILE__, __LINE__);
    using vislib::graphics::d3d::D3DException;

    HRESULT hr = D3D_OK;
    UINT cntAdapterModes = 0;
    D3DDISPLAYMODE displayMode;

    /* Sanity checks. */
    if (d3d == NULL) {
        throw IllegalParamException("d3d", __FILE__, __LINE__);
    }

    cntAdapterModes = d3d->GetAdapterModeCount(adapterID, format); 
    for (UINT i = 0; i < cntAdapterModes; i++) {
        if (FAILED(hr = d3d->EnumAdapterModes(adapterID, format, i, 
                &displayMode))) {
            throw D3DException(hr, __FILE__, __LINE__);
        }
        ASSERT(displayMode.Format == format);

        if ((displayMode.Width == width) && (displayMode.Height == height)) {
            if (outRefreshRate != NULL) {
                *outRefreshRate = displayMode.RefreshRate;
            }
            return true;
        }
    }
    /* Not found. */

    return false;
}
