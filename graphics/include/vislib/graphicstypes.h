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


namespace vislib {
namespace graphics {

    /** This type is used for values in scene space. */
    typedef float SceneSpaceType;

    /** 
     * This type is used for values in image space.
     * Implementation note: using float instead of unsigned int to be able 
     * to place elements with subpixel precision
     */
    typedef float ImageSpaceType;

    /** This type is used for values in cursor space. */
    typedef float CursorSpaceType;
    
} /* end namespace graphics */
} /* end namespace vislib */


#endif /* VISLIB_CAMERA_H_INCLUDED */
