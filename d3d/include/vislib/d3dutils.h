/*
 * d3dutils.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#ifndef VISLIB_D3DUTILS_H_INCLUDED
#define VISLIB_D3DUTILS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include <d3d9.h>


#ifndef SAFE_RELEASE
#define SAFE_RELEASE(obj) if (obj != NULL) { obj->Release(); obj = NULL; }
#endif /* !_SAFE_RELEASE */


namespace vislib {
namespace graphics {
namespace d3d {

    /** This enumeration is used to identify different Direct3D API versions. */
    typedef enum ApiVersion_t {
        D3DVERSION_UNKNOWN = 0,
        D3DVERSION_9,
        D3DVERSION_10,
        D3DVERSION_11
    } ApiVersion;

    /** 
     * This enumeration is used to judge wether a screen resolution should be
     * considered larger than another one.
     */
    typedef enum FullscreenSizeCriterion_t {
        CRITERION_AREA = 1, //< Maximise the screen area.
        CRITERION_WIDTH,    //< Maximise the screen width.
        CRITERION_HEIGHT    //< Maximise the screen height.
    } FullscreenSizeCriterion;

    /**
     * Answer the dimensions of the backbuffer of the specified device.
     *
     * @param outWidth
     * @param outHeight
     * @param device
     *
     * @throws
     */
    void GetBackbufferSize(UINT& outWidth, UINT& outHeight, 
        IDirect3DDevice9 *device);

    /**
     * Answer the maximum resolution the specified adapter supports.
     *
     * @param outWidth
     * @param outHeight
     * @param outRefreshRate
     * @param d3d
     * @param adapterID
     * @param format
     * @param criterion
     *
     * @throws
     * @throws
     */
    void GetMaximumFullscreenResolution(UINT& outWidth, 
        UINT& outHeight, 
        UINT& outRefreshRate,
        IDirect3D9 *d3d, 
        const UINT adapterID, 
        const D3DFORMAT& format,
        const FullscreenSizeCriterion criterion = CRITERION_AREA);

    /**
     * Answer the maximum resolution that is supported by all adapters of the
     * specified device.
     *
     * @param outWidth
     * @param outHeight
     * @param outRefreshRate
     * @param d3d
     * @param format
     * @param criterion
     *
     * @return true if a mode was found that is supported by all adapters, 
     *         false otherwise.
     *
     * @throws
     * @throws
     */
    bool GetMaximumSharedFullscreenResolution(UINT& outWidth, 
        UINT& outHeight, 
        UINT& outRefreshRate,
        IDirect3D9 *d3d, 
        const D3DFORMAT& format,
        const FullscreenSizeCriterion criterion = CRITERION_AREA);

    /**
     * Answer whether the specified fullscreen resolution is supported by the
     * specified adapter. If requested, return the refresh rate for that mode.
     * 
     * @param d3d
     * @param adapterID
     * @param format
     * @param width
     * @param height
     * @param outRefreshRate
     *
     * @return
     *
     * @throws
     * @throws
     */
    bool IsFullscreenResolutionSupported(IDirect3D9 *d3d, 
        const UINT adapterID, 
        const D3DFORMAT& format,
        const UINT width,
        const UINT height,
        UINT *outRefreshRate = NULL);

} /* end namespace d3d */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_D3DUTILS_H_INCLUDED */
