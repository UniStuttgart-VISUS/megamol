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
#include "vislib/ShallowDimension.h"
#include "vislib/ShallowPoint.h"
#include "vislib/ShallowRectangle.h"
#include "vislib/ShallowVector.h"


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

    /** A shallow type for specifying image space dimensions. */
    typedef math::ShallowDimension<ImageSpaceType, 2> 
        ShallowImageSpaceDimension2D;

    /** A type for specifying image space rectangle. */
    typedef math::Rectangle<ImageSpaceType> ImageSpaceRectangle;

    /** A shallow type for specifying image space rectangle. */
    typedef math::ShallowRectangle<ImageSpaceType> ShallowImageSpaceRectangle;

    /** This type is used for values in scene space. */
    typedef float SceneSpaceType;

    /** A 2D scene space point. */
    typedef math::Point<SceneSpaceType, 2> SceneSpacePoint2D;

    /** A shallow 2D scene space point. */
    typedef math::ShallowPoint<SceneSpaceType, 2> ShallowSceneSpacePoint2D;

    /** A 3D scene space point. */
    typedef math::Point<SceneSpaceType, 3> SceneSpacePoint3D;

    /** A shallow 3D scene space point. */
    typedef math::ShallowPoint<SceneSpaceType, 3> ShallowSceneSpacePoint3D;

    /** A 2D scene space vector. */
    typedef math::Vector<SceneSpaceType, 2> SceneSpaceVector2D;

    /** A shallow 2D scene space vector. */
    typedef math::ShallowVector<SceneSpaceType, 2> ShallowSceneSpaceVector2D;

    /** A 3D scene space point. */
    typedef math::Point<SceneSpaceType, 3> SceneSpacePoint3D;

    /** A shallow 3D scene space point. */
    typedef math::ShallowPoint<SceneSpaceType, 3> ShallowSceneSpacePoint3D;

    /** A 3D scene space vector. */
    typedef math::Vector<SceneSpaceType, 3> SceneSpaceVector3D;

    /** A shallow 3D scene space vector. */
    typedef math::ShallowVector<SceneSpaceType, 3> ShallowSceneSpaceVector3D;

} /* end namespace graphics */
} /* end namespace vislib */


#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERA_H_INCLUDED */
