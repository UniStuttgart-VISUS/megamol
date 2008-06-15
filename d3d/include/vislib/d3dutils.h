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
        D3DVERSION_10
    } ApiVersion;

} /* end namespace d3d */
} /* end namespace graphics */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_D3DUTILS_H_INCLUDED */
