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


#include "vislib/Dimension.h"
#include "vislib/Point.h"
#include "vislib/Rectangle.h"
#include "vislib/Vector.h"


namespace vislib {
namespace graphics {

    /** 
     * This type is used for values in image space.
     * Implementation note: using float instead of unsigned int to be able 
     * to place elements with subpixel precision
     */
    typedef float ImageSpaceType;

    /** A type for specifying image space dimensions. */
    typedef math::Dimension<ImageSpaceType, 2> ImageSpaceDimension;

    /** A type for specifying image space rectangle. */
    typedef math::Rectangle<ImageSpaceType> ImageSpaceRectangle;

    /** This type is used for values in scene space. */
    typedef float SceneSpaceType;

    /** A 2D scene space point. */
    typedef math::Point<SceneSpaceType, 2> SceneSpacePoint2D;

    /** A 3D scene space point. */
    typedef math::Point<SceneSpaceType, 3> SceneSpacePoint3D;

    /** A 2D scene space vector. */
    typedef math::Vector<SceneSpaceType, 2> SceneSpaceVector2D;

    /** A 3D scene space vector. */
    typedef math::Vector<SceneSpaceType, 3> SceneSpaceVector3D;
    
} /* end namespace graphics */
} /* end namespace vislib */


#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERA_H_INCLUDED */
