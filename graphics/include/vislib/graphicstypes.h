/*
 * graphicstypes.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_GRAPHICSTYPES_H_INCLUDED
#define VISLIB_GRAPHICSTYPES_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


namespace vislib {
namespace graphics {

    /** This type is used for values in scene space. */
    typedef double SceneSpaceType;

    /** 
     * This type is used for values in image space.
     * Implementation note: using float instead of unsigned int to be able 
     * to place elements with subpixel precision
     */
    typedef float ImageSpaceType;
    
} /* end namespace graphics */
} /* end namespace vislib */


#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERA_H_INCLUDED */
