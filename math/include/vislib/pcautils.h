/*
 * pcautils.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_PCAUTILS_H_INCLUDED
#define VISLIB_PCAUTILS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/Array.h"
#include "vislib/assert.h"
#include "vislib/AbstractVector.h"
#include "vislib/AbstractMatrix.h"
#include "vislib/mathfunctions.h"
#include "vislib/Pair.h"
#include "vislib/Vector.h"


namespace vislib {
namespace math {

    /**
     * Calculate the covariance matrix from a list of relative coordinates
     *
     * T is the type for the matrix type (and the vector type)
     * D is the dimensionality for the matrix type (and the vector type)
     * L is the matrix type layout
     * S is the matrix storage type
     * L2 is the array lock class
     * C is the array constructor class
     * T2 is the array type class (the vector type)
     *
     * @param outMatrix The variable to receive the calculated covariance
     *                  matrix
     * @param relCoords The relative coordinates
     */
    template<class T, unsigned int D, MatrixLayout L, class S, class L2,
        class C, class T2>
    inline void CalcCovarianceMatrix(AbstractMatrix<T, D, L, S>& outMatrix,
            const Array<T2, L2, C>& relCoords) {
        CalcCovarianceMatrix(outMatrix, relCoords.PeekElements(),
            relCoords.Count());
    }

    /**
     * Calculate the covariance matrix from a list of relative coordinates
     *
     * T is the type for the matrix type and the vector type
     * D is the dimensionality for the matrix type and the vector type
     * L is the matrix type layout
     * S1 is the matrix storage type
     * S2 is the vector storage type
     *
     * @param outMatrix The variable to receive the calculated covariance
     *                  matrix
     * @param relCoords Pointer to the relative coordinates. Must not be NULL
     * @param relCoordsCnt The number of relative coordinates 'relCoords'
     *                     points to
     */
    template<class T, unsigned int D, MatrixLayout L, class S1, class S2>
    void CalcCovarianceMatrix(AbstractMatrix<T, D, L, S1>& outMatrix,
            const AbstractVector<T, D, S2> *relCoords, SIZE_T relCoordsCnt) {
        ASSERT(relCoords != NULL);

        for (unsigned int x = 0; x < D; x++) {
            for (unsigned int y = 0; y < D; y++) {
                outMatrix(x, y) = static_cast<T>(0);
            }
        }

        if (relCoordsCnt > 0) {
            for (SIZE_T i = 0; i < relCoordsCnt; i++) {
                for (unsigned int x = 0; x < D; x++) {
                    for (unsigned int y = 0; y < D; y++) {
                        outMatrix(x, y) += (relCoords[i][x] * relCoords[i][y]);
                    }
                }
            }

            outMatrix /= static_cast<T>(relCoordsCnt);

        }

    }

    /**
     * sorts a list of eigenvectors according to their eigenvalues ascending.
     *
     * @param evec The array of eigenvectors
     * @param eval The array of eigenvalues
     * @param cnt The number of eigenvectors/eigenvalues
     */
    template<class T, unsigned int D, class S>
    void SortEigenvectors(AbstractVector<T, D, S> *evec, T *eval, SIZE_T cnt) {
        if (cnt < 2) return; // lol

        if (cnt < 5) {
            if (eval[0] > eval[1]) {
                vislib::math::Swap(eval[0], eval[1]);
                vislib::math::Swap(evec[0], evec[1]);
            }
            if (cnt > 2) {
                if (eval[0] > eval[2]) {
                    vislib::math::Swap(eval[0], eval[2]);
                    vislib::math::Swap(evec[0], evec[2]);
                }
                if (eval[1] > eval[2]) {
                    vislib::math::Swap(eval[1], eval[2]);
                    vislib::math::Swap(evec[1], evec[2]);
                }
                if (cnt == 4) {
                    if (eval[0] > eval[3]) {
                        vislib::math::Swap(eval[0], eval[3]);
                        vislib::math::Swap(evec[0], evec[3]);
                    }
                    if (eval[1] > eval[3]) {
                        vislib::math::Swap(eval[1], eval[3]);
                        vislib::math::Swap(evec[1], evec[3]);
                    }
                    if (eval[2] > eval[3]) {
                        vislib::math::Swap(eval[2], eval[3]);
                        vislib::math::Swap(evec[2], evec[3]);
                    }
                }
            }
            return; // simple (and common)
        }

        // more than four dimensions. I so don't care
        vislib::Array<vislib::Pair<T, Vector<T, D> > > tb;
        tb.SetCount(cnt);
        for (SIZE_T i = 0; i < cnt; i++) {
            tb[i].SetFirst(eval[i]);
            tb[i].SetSecond(evec[i]);
        }
        tb.Sort(&ComparePairsFirst<T, Vector<T, D> >);
        for (SIZE_T i = 0; i < cnt; i++) {
            eval[i] = tb[i].First();
            evec[i] = tb[i].Second();
        }

    }


} /* end namespace math */
} /* end namespace vislib */


#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_PCAUTILS_H_INCLUDED */
