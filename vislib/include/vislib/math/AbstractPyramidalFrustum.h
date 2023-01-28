/*
 * AbstractPyramidalFrustum.h
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTPYRAMIDALFRUSTUM_H_INCLUDED
#define VISLIB_ABSTRACTPYRAMIDALFRUSTUM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Array.h"
#include "vislib/Pair.h"
#include "vislib/assert.h"
#include "vislib/types.h"

#include "vislib/math/Point.h"
#include "vislib/math/mathfunctions.h"

#include "vislib/MissingImplementationException.h"

namespace vislib::math {


/**
 * This is the superclass for pyramidal frustums in the VISlib. It provides
 * some common functionality.
 */
template<class T>
class AbstractPyramidalFrustum {

public:
    /**
     * Index of the left bottom point (as seen from the apex of the pyramid)
     * in the array returned by GetBottomBasePoints() or GetTopBasePoints().
     */
    static const UINT_PTR IDX_LEFT_BOTTOM_POINT;

    /**
     * Index of the left top point (as seen from the apex of the pyramid)
     * in the array returned by GetBottomBasePoints() or GetTopBasePoints().
     */
    static const UINT_PTR IDX_LEFT_TOP_POINT;

    /**
     * Index of the right bottom point (as seen from the apex of the
     * pyramid) in the array returned by GetBottomBasePoints() or
     * GetTopBasePoints().
     */
    static const UINT_PTR IDX_RIGHT_BOTTOM_POINT;

    /**
     * Index of the right top point (as seen from the apex of the pyramid)
     * in the array returned by GetBottomBasePoints() or GetTopBasePoints().
     */
    static const UINT_PTR IDX_RIGHT_TOP_POINT;

    /** Dtor. */
    virtual ~AbstractPyramidalFrustum();

    /**
     * Answer the points that form the bottom base of the frustum.
     *
     * The points must be located at IDX_LEFT_BOTTOM_POINT, etc. with
     * viewing direction from the apex of the pyramid.
     *
     * @param outPoints An array that will receive the points. All existing
     *                  content will be replaced.
     */
    virtual void GetBottomBasePoints(vislib::Array<Point<T, 3>>& outPoints) const = 0;

    /**
     * Answer the eight border points of the frustum. The points are ordered
     * as returned by GetTopBasePoints() followed by GetBottomBasePoints(),
     * ie the left/bottom/front point is at IDX_LEFT_BOTTOM_POINT and the
     * left/bottom/back point is at (4 + IDX_LEFT_BOTTOM_POINT).
     *
     * @tparam I An output iterator for Point<T, 3>.
     *
     * @param oit An output iterator which can recieve at least eight
     *            elements of type Point<T, 3>.
     */
    template<class I>
    inline void GetPoints(I oit) const {
        Array<Point<T, 3>> points(4);
        this->GetTopBasePoints(points);
        *oit++ = points[0];
        *oit++ = points[1];
        *oit++ = points[2];
        *oit++ = points[3];
        points.Clear();
        this->GetBottomBasePoints(points);
        *oit++ = points[0];
        *oit++ = points[1];
        *oit++ = points[2];
        *oit++ = points[3];
    }

    /**
     * Answer the line segments between the points forming the frustum.
     *
     * @tparam I An output iterator for Pair<Point<T, 3>, Point<T, 3>>.
     *
     * @param oit An output iterator which can recieve at least twelve
     *            pairs of points.
     */
    template<class I>
    void GetLineSegments(I oit) const;

    /**
     * Answer the points that form the top base of the frustum.
     *
     * The points must be located at IDX_LEFT_BOTTOM_POINT, etc. with
     * viewing direction from the apex of the pyramid.
     *
     * @param outPoints An array that will receive the points. All existing
     *                  content will be replaced.
     */
    virtual void GetTopBasePoints(vislib::Array<Point<T, 3>>& outPoints) const = 0;

protected:
    /**
     * Disallow instances.
     */
    inline AbstractPyramidalFrustum() {}

    /**
     * Performs a sanity check whether the bottom and top base points can
     * form a pyramidal frustum.
     *
     * Subclasses can use this method as necessary to check user input.
     * Subclasses may also overwrite this method to check additional
     * constraints.
     *
     * @param bottomBasePoints
     * @param topBasePoints
     *
     * @return
     */
    virtual bool checkSanity(
        const vislib::Array<Point<T, 3>>& bottomBasePoints, const vislib::Array<Point<T, 3>>& topBasePoints) const;
};


/*
 * vislib::math::AbstractPyramidalFrustum<T>::IDX_LEFT_BOTTOM_POINT
 */
template<class T>
const UINT_PTR AbstractPyramidalFrustum<T>::IDX_LEFT_BOTTOM_POINT = 0;


/*
 * vislib::math::AbstractPyramidalFrustum<T>::IDX_LEFT_TOP_POINT
 */
template<class T>
const UINT_PTR AbstractPyramidalFrustum<T>::IDX_LEFT_TOP_POINT = 3;


/*
 * vislib::math::AbstractPyramidalFrustum<T>::IDX_RIGHT_BOTTOM_POINT
 */
template<class T>
const UINT_PTR AbstractPyramidalFrustum<T>::IDX_RIGHT_BOTTOM_POINT = 1;


/*
 * vislib::math::AbstractPyramidalFrustum<T>::IDX_RIGHT_TOP_POINT
 */
template<class T>
const UINT_PTR AbstractPyramidalFrustum<T>::IDX_RIGHT_TOP_POINT = 2;


/*
 * vislib::math::AbstractPyramidalFrustum<T>::~AbstractPyramidalFrustum
 */
template<class T>
AbstractPyramidalFrustum<T>::~AbstractPyramidalFrustum() {}


/*
 * AbstractPyramidalFrustum<T>::GetLineSegments
 */
template<class T>
template<class I>
void AbstractPyramidalFrustum<T>::GetLineSegments(I oit) const {
    typedef Pair<Point<T, 3>, Point<T, 3>> PairType;
    Array<Point<T, 3>> bottom(4);
    Array<Point<T, 3>> top(4);

    this->GetBottomBasePoints(bottom);
    this->GetTopBasePoints(top);

    *oit++ = PairType(top[IDX_LEFT_BOTTOM_POINT], top[IDX_LEFT_TOP_POINT]);
    *oit++ = PairType(top[IDX_LEFT_TOP_POINT], top[IDX_RIGHT_TOP_POINT]);
    *oit++ = PairType(top[IDX_RIGHT_TOP_POINT], top[IDX_RIGHT_BOTTOM_POINT]);
    *oit++ = PairType(top[IDX_RIGHT_BOTTOM_POINT], top[IDX_LEFT_BOTTOM_POINT]);

    *oit++ = PairType(bottom[IDX_LEFT_BOTTOM_POINT], bottom[IDX_LEFT_TOP_POINT]);
    *oit++ = PairType(bottom[IDX_LEFT_TOP_POINT], bottom[IDX_RIGHT_TOP_POINT]);
    *oit++ = PairType(bottom[IDX_RIGHT_TOP_POINT], bottom[IDX_RIGHT_BOTTOM_POINT]);
    *oit++ = PairType(bottom[IDX_RIGHT_BOTTOM_POINT], bottom[IDX_LEFT_BOTTOM_POINT]);

    *oit++ = PairType(bottom[IDX_LEFT_BOTTOM_POINT], top[IDX_LEFT_BOTTOM_POINT]);
    *oit++ = PairType(bottom[IDX_RIGHT_BOTTOM_POINT], top[IDX_RIGHT_BOTTOM_POINT]);
    *oit++ = PairType(bottom[IDX_LEFT_TOP_POINT], top[IDX_LEFT_TOP_POINT]);
    *oit++ = PairType(bottom[IDX_RIGHT_TOP_POINT], top[IDX_RIGHT_TOP_POINT]);
}


/*
 * vislib::math::AbstractPyramidalFrustum<T>::checkSanity
 */
template<class T>
bool AbstractPyramidalFrustum<T>::checkSanity(
    const vislib::Array<Point<T, 3>>& bottomBasePoints, const vislib::Array<Point<T, 3>>& topBasePoints) const {
    /*
     * Bottom and top base must be at least triangles and have the same
     * shape.
     */
    if ((bottomBasePoints.Count() < 3) || (topBasePoints.Count() < 3)) {
        return false;
    }
    if (bottomBasePoints.Count() != topBasePoints.Count()) {
        return false;
    }

    // TODO: Check plane
    // TODO: Check parallel
    // TODO: Check base size
    // TODO: Check apex
    throw MissingImplementationException("checkSanity", __FILE__, __LINE__);
    return true;
}

} // namespace vislib::math

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTPYRAMIDALFRUSTUM_H_INCLUDED */
