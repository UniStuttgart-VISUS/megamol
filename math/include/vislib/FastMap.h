/*
 * FastMap.h
 *
 * Copyright (C) 2007 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef VISLIB_FASTMAP_H_INCLUDED
#define VISLIB_FASTMAP_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include <stdlib.h>
#include <time.h>

#include "vislib/Array.h"
#include "vislib/Vector.h"

/** Number of iterations for choosing the maximally distant points. The
    original paper suggests 5. */
#define MAX_PIVOT_ITERATIONS 5

namespace vislib {
namespace math {


    /**
     * A simple dimensionality reduction Algorithm.
     * C. Faloutsos, K. Lin,
     * "FastMap: A Fast Algorithm for Indexing, Data-Mining and Visualization
     * of Tradditional and Multimedia Datasets,"
     * in Proceedings of 1995 ACM SIGMOD, SIGMOD RECORD (June 1995), vol.24,
     * no.2, p 163-174.
     * http://citeseer.ist.psu.edu/faloutsos95fastmap.htm
     *
     * Background: Find maximally distant pivot points in high-D space,
     * project all points onto that line (first coordinate). After that,
     * measure distance on a hyperplane perpendicular to the pivot line,
     * select additional pivots, and project again. Repeat until desired
     * number of dimensions is reached.
     *
     * TI needs to support a float TI::Distance(TI &other) method, results
     * will be returned as a vislib::Array<vislib::math::Vector<TO, D>>.
     *
     * One of the strengths of FastMap is the ability to just project
     * additional points in O(1) without having to recompute pivots.
     *
     * The main weakness of FastMap is that only distances to the pivots are
     * used for the projection. The choice of pivots considers all
     * inter-object distances though.
     * Thus sparse distance matrices almost necessarily yield unsatisfactory
     * results.
     */
    template<class TI, class TO, unsigned int D>
    class FastMap {

        /** Output Element Type */
        typedef Vector<TO, D> OutElement;

        /** Result Type (reduced dimensionality) */
        typedef Array<Vector<TO, D> > ResultType;

    public:
        /**
         * Compute reduced-dimensionality positions for inData.
         *
         * @param inputData The high-dimensional dataset.
         * @param[out] out  A pre-allocated
         * vislib::Array<vislib::math::Vector<TO, D>>
         *                  for the reduced coordinates.
         */
        FastMap(Array<TI> &inputData, ResultType &out) :
            inData(inputData), outData(out) {
            this->compute();
        }

        /** dtor. */
        ~FastMap() {
        }

        /**
         * Reduce an additional item based on the already chosen pivots. out
         * must again have this index already allocated.
         *
         * @param index The index of the new element in the inputData array.
         */
        void ComputeSingle(SIZE_T index) {
            OutElement out;
            unsigned int currDim;
            float xi;

            for (currDim = 0; currDim < D; currDim++) {
                if (calcDistanceSquared(currDim, a[currDim], b[currDim]) == 0) {
                    outData[index][currDim] = 0;
                } else {
                    xi = calcDistanceSquared(currDim, a[currDim], index);
                    xi += dabSq[currDim];
                    xi -= calcDistanceSquared(currDim, b[currDim], index);
                    xi /= 2 * dab[currDim];
                    outData[index][currDim] = xi;
                }
            }
        }


    private:

        /**
         * Calculate the distance in high-D (dim = 0) or on the corresponding
         * hyperplane
         * (dim > 0) between the two points inputData[x] and inputData[y].
         *
         * @param dim The dimension for which to calculate the distance.
         * @param x   Index of one object.
         * @param y   Index of the other object.
         */
        float calcDistanceSquared(unsigned int dim, SIZE_T x, SIZE_T y);

        /**
         * Heuristically choose two maximally distant pivot points using the
         * distance calculated according to dim. a, b, dab, and dasSq are set
         * accordingly.
         *
         * This takes O(N) (exactly N * MAX_PIVOT_ITERATIONS).
         *
         * @param dim The (target) dimension for which to choose pivots.
         */
        void chooseDistant(unsigned int dim);

        /**
         * Executes the mapping from inData into outData. It calls
         * chooseDistant once per destination dimension.
         */
        void compute(void);

        /** Reference to source data */
        Array<TI> &inData;

        /** Reference to output data */
        ResultType &outData;

        /** The pivot indices per (output) dimension */
        SIZE_T a[D], b[D];

        /** The pivot distance per (output) dimension */
        float dab[D];

        /** Square of the pivot distance per (output) dimension */
        float dabSq[D];

    };

    /*
     * vislib::math::FastMap<TI, TO, D>::calcDistanceSquared
     */
    template<class TI, class TO, unsigned int D>
    float FastMap<TI, TO, D>::calcDistanceSquared(
        unsigned int dim, SIZE_T x, SIZE_T y) {
        float d;
        if (dim == 0) {
            d = inData[x].Distance(inData[y]);
            return d * d;
        }
        d = (outData[x][dim - 1] - outData[y][dim - 1]);
        d *= d;
        return calcDistanceSquared(dim - 1, x, y) - d;
    }

    /*
     * vislib::math::FastMap<TI, TO, D>::chooseDistant
     */
    template<class TI, class TO, unsigned int D>
    void FastMap<TI, TO, D>::chooseDistant(unsigned int dim) {
        SIZE_T a = 0;
        SIZE_T b;
        float dist, tmpDist;

        // This should be done outside to give more control to the application
        // srand(static_cast<unsigned>(time(NULL)));

        b = SIZE_T ((rand() / static_cast<double>(RAND_MAX)) * inData.Count());
        for (unsigned int j = 0; j <= MAX_PIVOT_ITERATIONS; j++) {
            dist = 0.0f;
            for (SIZE_T i = 0; i < inData.Count(); i++) {
                tmpDist = calcDistanceSquared(dim, i, b);
                if (tmpDist > dist) {
                    dist = tmpDist;
                    a = i;
                }
            }
            SIZE_T i = b;
            b = a;
            a = i;
        }
        this->a[dim] = a;
        this->b[dim] = b;
        this->dabSq[dim] = dist;
        this->dab[dim] = sqrt(dist);
    }

    /*
     * vislib::math::FastMap<TI, TO, D>::compute
     */
    template<class TI, class TO, unsigned int D>
    void FastMap<TI, TO, D>::compute(void) {
        unsigned int currDim;
        float xi;

        for (currDim = 0; currDim < D; currDim++) {
            chooseDistant(currDim);

            //if (calcDistanceSquared(currDim, a[currDim], b[currDim]) == 0) {
            if (vislib::math::IsEqual(calcDistanceSquared(currDim, a[currDim], b[currDim]), 0.0f)) {
                for (SIZE_T i = 0; i < inData.Count(); i++) {
                    outData[i][currDim] = 0;
                }
            } else {
                for (SIZE_T i = 0; i < inData.Count(); i++) {
                    xi = calcDistanceSquared(currDim, a[currDim], i);
                    xi += dabSq[currDim];
                    xi -= calcDistanceSquared(currDim, b[currDim], i);
                    xi /= 2 * dab[currDim];
                    outData[i][currDim] = xi;
                }
            }
        }
    }

} /* end namespace math */
} /* end namespace vislib */



#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_FASTMAP_H_INCLUDED */
