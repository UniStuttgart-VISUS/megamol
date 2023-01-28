/*
 * graphicstypes.h
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_GRAPHICSTYPES_H_INCLUDED
#define VISLIB_GRAPHICSTYPES_H_INCLUDED
#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/math/Cuboid.h"
#include "vislib/math/Dimension.h"
#include "vislib/math/Point.h"
#include "vislib/math/Rectangle.h"
#include "vislib/math/RectangularPyramidalFrustum.h"
#include "vislib/math/ShallowDimension.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/math/ShallowRectangle.h"
#include "vislib/math/ShallowVector.h"
#include "vislib/math/Vector.h"
#include "vislib/math/ViewFrustum.h"


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
typedef math::ShallowDimension<ImageSpaceType, 2> ShallowImageSpaceDimension2D;

/** A type for specifying image space rectangle. */
typedef math::Rectangle<ImageSpaceType> ImageSpaceRectangle;

/** A shallow type for specifying image space rectangle. */
typedef math::ShallowRectangle<ImageSpaceType> ShallowImageSpaceRectangle;

/** This type is used for values in scene space. */
typedef float SceneSpaceType;

/** A 3D scene space cuboid. */
typedef math::Cuboid<SceneSpaceType> SceneSpaceCuboid;

/** A scene space frustum. */
typedef math::RectangularPyramidalFrustum<SceneSpaceType> SceneSpaceFrustum;

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

/** A 3D scene space vector. */
typedef math::Vector<SceneSpaceType, 3> SceneSpaceVector3D;

/** A shallow 3D scene space vector. */
typedef math::ShallowVector<SceneSpaceType, 3> ShallowSceneSpaceVector3D;

/** A scene space view frustum. */
typedef math::ViewFrustum<SceneSpaceType> SceneSpaceViewFrustum;

} /* end namespace graphics */
} /* end namespace vislib */


#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CAMERA_H_INCLUDED */
