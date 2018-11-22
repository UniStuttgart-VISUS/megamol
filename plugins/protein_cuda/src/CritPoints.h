/*
 * CritPoints.h
 *
 * Copyright (C) 2012 by University of Stuttgart (VISUS).
 * All rights reserved.
 *
 * $Id$
 */

#ifndef MMPROTEINCUDAPLUGIN_CRITPOINTS_H
#define MMPROTEINCUDAPLUGIN_CRITPOINTS_H

#include "protein_calls/CrystalStructureDataCall.h"
#include "UniGrid3D.h"

#include "vislib/math/Vector.h"
#include "vislib/Array.h"

#include "helper_math.h"

namespace megamol {
namespace protein_cuda {

/**
 * Computes critical points in a 3D uniform vector field.
 */
class CritPoints {

public:

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "CritPoints";
    }

    /**
     * Recursively computes the location of critical points of the vector field
     * by calculating the topological degree.
     *
     * @param uniGrid  The 3D uniform grid containing the vector field
     * @param minCoord The lower corner of the current cell
     * @param maxCoord The upper corner of the current cell
     *
     * @return The array containing the critical points
     */
    static vislib::Array<float> GetCritPoints(UniGrid3D<float3> &uniGrid,
            vislib::math::Vector<float, 3> minCoord,
            vislib::math::Vector<float, 3> maxCoord);

    /**
     * TODO
     */
    static vislib::Array<float> GetCritPointsGreene(
            UniGrid3D<float3> &uniGrid,
                    vislib::math::Vector<float, 3> minGridCoord,
                    vislib::math::Vector<float, 3> maxGridCoord,
                    float cellCize);

protected:

    /**
     * Computes the topological degree of a cell. If the degree is != 0 the cell
     * contains a critical point.
     *
     * @param uniGrid The 3D uniform grid containing the vector field
     * @param minCoord The lower corner of the current cell
     * @param maxCoord The upper corner of the current cell
     *
     * @return The degree of the cell
     */
    static int calcDegreeOfCell(UniGrid3D<float3> &uniGrid,
            vislib::math::Vector<float, 3> minCoord,
            vislib::math::Vector<float, 3> maxCoord);

    /**
     * Computes the solid angle covered by the vectors of a given triangle.
     *
     * @param uniGrid The 3D uniform grid containing the vector field
     * @param points  The points of the triangle
     *
     * @returm The solid angle
     */
    static float calcSolidAngleOfTriangle(
            UniGrid3D<float3> &uniGrid,
            vislib::Array<vislib::math::Vector<float, 3> > points);

    static float calcSolidAngleOfTriangleAlt(
            UniGrid3D<float3> &uniGrid,
            vislib::Array<vislib::math::Vector<float, 3> > points);

private:

    /**
     * Sample the uniform grid by nearest neighbour manner.
     *
     * @param uniGrid The 3D uniform grid
     * @param The position in world coordinates
     *
     * @return The sampled vector
     */
    static vislib::math::Vector<float, 3> sampleUniGridNearestNeighbour(
            UniGrid3D<float3> &uniGrid,
            vislib::math::Vector<float, 3> pos);

};

} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif /* MMPROTEINCUDAPLUGIN_CRITPOINTS_H */
