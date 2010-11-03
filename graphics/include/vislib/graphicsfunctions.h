/*
 * graphicsfunctions.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_GRAPHICSFUNCTIONS_H_INCLUDED
#define VISLIB_GRAPHICSFUNCTIONS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/Array.h"
#include "vislib/mathfunctions.h"
#include "vislib/Point.h"
#include "vislib/Vector.h"


namespace vislib {
namespace graphics {


    /**
     * Sorts the vertices of a flat convex polygon
     *
     * @param polygon the vertices of a flat convex polygon
     */
    template<class T> void FlatPolygonSort(Array<math::Point<T, 3> >& polygon) {
        typedef math::Point<T, 3> Point;
        typedef math::Vector<T, 3> Vector;
        // sorts the points for a flat, concave, non-self-intersecting polygon
        if (polygon.Count() < 3) return; // no need for sort

        Point center(polygon[0]);
        for (SIZE_T i = 1; i < polygon.Count(); i++) {
            center.Set(center.X() + polygon[i].X(),
                center.Y() + polygon[i].Y(),
                center.Z() + polygon[i].Z());
        }
        center.Set(center.X() / static_cast<T>(polygon.Count()),
            center.Y() / static_cast<T>(polygon.Count()),
            center.Z() / static_cast<T>(polygon.Count()));

        Vector normal;
        Vector x(polygon[0] - center);
        x.Normalise();
        normal.SetNull();
        SIZE_T i = 0;
        while (normal.IsNull()) {
            i++;
            if (i >= polygon.Count()) return; // all points in the same place?
            normal = x.Cross(polygon[i] - center);
        }
        normal.Normalise();
        Vector y = x.Cross(normal);

        Array<Pair<SIZE_T, double> > as;
        as.SetCount(polygon.Count());
        for (SIZE_T i = 0; i < polygon.Count(); i++) {
            Vector v(polygon[i] - center);
            as[i].First() = i;
            as[i].Second() = ::atan2(static_cast<double>(v.Dot(y)),
                static_cast<double>(v.Dot(x)));
        }

        as.Sort(vislib::math::ComparePairsSecond);

        Array<Point> tmp(polygon);
        polygon.Clear();
        for (SIZE_T i = 0; i < tmp.Count(); i++) {
            polygon.Add(tmp[as[i].First()]);
        }
    }


} /* end namespace graphics */
} /* end namespace vislib */


#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_GRAPHICSFUNCTIONS_H_INCLUDED */

