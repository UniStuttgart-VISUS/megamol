//
// VecField3f.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//

#ifndef MMPROTEINCUDAPLUGIN_VECFIELD3D_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_VECFIELD3D_H_INCLUDED

#include "CUDAFieldTopology.cuh"
#include <vector_types.h>

#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "helper_functions.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include <cuda_gl_interop.h>

//#include "vislib_vector_typedefs.h"
#include "vislib/math/Matrix.h"
#include "vislib/math/Vector.h"
typedef vislib::math::Matrix<float, 3, vislib::math::COLUMN_MAJOR> Mat3f;
typedef vislib::math::Vector<float, 3> Vec3f;
typedef vislib::math::Vector<unsigned int, 3> Vec3u;
#include "vislib/Array.h"

namespace megamol {
namespace protein_cuda {

class VecField3f {

public:

    class CritPoint {

    public:

        /// Possible types of critical points
        enum Type {UNKNOWN=0, SOURCE, SINK, REPELLING_SADDLE,
            ATTRACTING_SADDLE}; // TODO extend

        /** CTor */
        CritPoint() : pos(0.0f, 0.0f, 0.0f), cellId(0, 0, 0), t(UNKNOWN) {
        }

        /** CTor */
        CritPoint(Vec3f pos,
                Vec3u cellId,
                Type t) : pos(pos), cellId(cellId), t(t) {
        }

        /** DTor */
        ~CritPoint() {
        }

        /**
         * Answer the null points position in world space.
         *
         * @return The position of the critical point
         */
        const Vec3f GetPos() const {
            return this->pos;
        }

        /**
         * Answer the cell id of the cell containing the critical point.
         *
         * @return The cell id of the according cell
         */
        const Vec3u GetCellId() const {
            return this->cellId;
        }

        /**
         * Answer the type of the critical point.
         *
         * @return The type of the critical point
         */
        Type GetType() const {
            return this->t;
        }

        /**
         * Test for equality.
         *
         * @param rhs The right hand side parameter
         * @return True if the two operands are equal
         */
        bool operator==(const CritPoint& rhs) const {
            return (this->pos == rhs.pos)
                && (this->cellId == rhs.cellId)
                && (this->t == rhs.t);
        }

    private:

        /// The exact position of the critical point
        Vec3f pos;

        /// The cell containing the critical point
        Vec3u cellId;

        /// The type of the critical point
        Type t;
    };

    /** CTor */
    VecField3f() : dimX(0), dimY(0), dimZ(0), spacingX(0.0f), spacingY(0.0f),
        spacingZ(0.0f), orgX(0.0f), orgY(0.0f), orgZ(0.0f), data(NULL) {
    }

    /** DTor */
    ~VecField3f() {
        delete[] this->data;
    }

    /**
     * Answers the vector at a given grid position.
     *
     * @param posX, posY, poZ The position in the grid.
     * @return The vector at the given position.
     * @throws OutOfRangeException, if the given position exceeds the dimensions
     *         of the grid.
     */
    Vec3f GetAt(unsigned int posX, unsigned int posY,
            unsigned int posZ);

    /**
     * Answers the vector at a given grid position. The vector is obtained
     * using trilinear interpolation.
     *
     * @param posX, posY, posZ The position in the grid.
     * @param normalize If true, the interpoaltion is done based on the
     *                  normalized grid.
     * @return The vector at the given position.
     */
    Vec3f GetAtTrilin(float posX, float posY, float posZ, bool normalize=false);

    /**
     * Answers the vector at a given grid position. The vector is obtained
     * using trilinear interpolation.
     *
     * @param pos The position in the grid.
     * @param normalize If true, the interpoaltion is done based on the
     *                  normalized grid.
     * @return The vector at the given position.
     */
    Vec3f GetAtTrilin(Vec3f pos, bool normalize=false) {
        return this->GetAtTrilin(pos.X(), pos.Y(), pos.Z(), normalize);
    }


    /**
     * Answers the cell id of the cell a given position is located in.
     *
     * @param pos The position in the grid.
     * @return The cell id
     */
    Vec3u GetCellId(Vec3f pos) {
        float cx,cy,cz;
        cx = (pos.X() - this->orgX)/this->spacingX;
        cy = (pos.Y() - this->orgY)/this->spacingY;
        cz = (pos.Z() - this->orgZ)/this->spacingZ;

        Vec3u cellId;
        cellId[0] = static_cast<unsigned int>(cx);
        cellId[1] = static_cast<unsigned int>(cy);
        cellId[2] = static_cast<unsigned int>(cz);

        return cellId;
    }

    /**
     * Answer the number of critical points found.
     *
     * @return The number of critical points
     */
    unsigned int GetCritPointCount() {
        return static_cast<unsigned int>(this->critPoints.Count());
    }

    /**
     * Get a reference to the critical point at the index 'idx'.
     *
     * @param idx The index of the critical point
     */
    const CritPoint& GetCritPoint(unsigned int idx) const {
        return this->critPoints[idx];
    }

    /**
     * Answer the dimensions of the grid
     *
     * @return The dimensions of the grid
     */
    const Vec3u GetDim() const {
        return Vec3u(this->dimX, this->dimY,
                this->dimZ);
    }

    /**
     * Answer the spacing of the grid
     *
     * @return The spacing of the grid
     */
    const Vec3f GetSpacing() const {
        return Vec3f(this->spacingX, this->spacingY,
                this->spacingZ);
    }

    /**
     * Answer the world space origin of the grid
     *
     * @return The origin of the grid
     */
    const Vec3f GetOrg() const {
        return Vec3f(this->orgX,
                this->orgY,
                this->orgZ);
    }

    /**
     * Get the jacobian at a given grid position. Partial derivatives are
     * obtained using central differences.
     *
     * @param x,y,z The grid position
     * @param normalize If true, the vector field is normalized before computing
     *                  the jacobian.
     * @return The jacobian
     */
    Mat3f GetJacobianAt(unsigned int x, unsigned int y, unsigned int z,
            bool normalize=false);

    /**
     * Check whether a given position lies inside a certain cell.
     *
     * @param cellId The id of the cell.
     * @param pos The position.
     * @return True, if the position is inside the cell, false otherwise.
     */
    bool IsPosInCell(Vec3u cellId,
            Vec3f pos);

    /**
     * Check whether a given position is inside the grid boundaries.
     *
     * @return True, if the position is inside the grid boundaries, false
     *         otherwise.
     */
    bool IsValidGridpos(Vec3f pos);

    /**
     * Returns a pointer to the data containing the vector data. Might be null.
     *
     * @return A pointer to the vector data.
     */
    const float *PeekBuff() {
        return this->data;
    }

    /**
     * Search null points and classify them according to their Eigenvalues.
     *
     * @param maxBisections The maximum number of bisections
     * @param maxItNewton   The maximum number of iterations when using the
     *                      Newton-Raphson iteration
     * @param stepNewton    The step size for the Newton iteration
     * @param epsNewton     Magnitude at which the vector field is considered
     *                      to be vanishing
     */
    void SearchCritPoints(unsigned int maxBisections, unsigned int maxItNewton,
            float stepNewton, float epsNewton);

    /**
     * Search null points and classify them according to their Eigenvalues. The
     * bisection is done using CUDA.
     *
     * @param maxItNewton   The maximum number of iterations when using the
     *                      Newton-Raphson iteration
     * @param stepNewton    The step size for the Newton iteration
     * @param epsNewton     Magnitude at which the vector field is considered
     *                      to be vanishing
     */
    void SearchCritPointsCUDA(unsigned int maxItNewton, float stepNewton,
            float epsNewton);

    /**
     * Sets the dimensions/ the extent of the grid and make a deep copy of the
     * data.
     *
     * @param dx,dy,dz         The dimensions of the grid
     * @param sx, sy, sz       The spacing of the grid
     * @param orgx, orgy, orgz The origin of the grid (in WS coordinates)
     */
    void SetData(const float *data, unsigned int dX, unsigned int dY,
            unsigned int dZ, float sx, float sy, float sz, float orgX,
            float orgY, float orgZ);

private:

    /**
     * Classify a null point according to its Eigenvalues.
     *
     * @param cellId The cell containing the null point
     * @param pos    The position of the null point
     * @return The type of the critical point
     */
    CritPoint::Type classifyCritPoint(Vec3u cellId, Vec3f pos);

    /**
     * Check whether the field is vanishing inside a given cell. The cell is
     * recursively bisected.
     *
     * @param currDepth The current number of bisections.
     * @param maxDepth  The maximum number of bisections.
     * @param n         The corner values of the current (sub)cell.
     */
    bool isFieldVanishingInCellBisectionRec(
            unsigned int currDepth, unsigned int maxDepth,
            vislib::Array<Vec3f > n);

    /**
     * Approximates the point where the field is vanishing in a given cell
     * beginning at a given starting position. The approximation is done using
     * the Newton-Raphson iteration.
     *
     * @param maxIt    The maximum number of iterations
     * @param startPos The position where the iteration starts
     * @param cellId   The id of the cell
     * @param step     The step size
     * @param eps      The magnitude at which the vector field is considered
     *                 to be vanishing
     * @return The approximated position of the null point.
     */
    Vec3f searchNullPointNewton(
            unsigned int maxIt,
            Vec3f startPos,
            Vec3u cellId,
            float step, float eps);

    /// Dimensions of the grid
    unsigned int dimX, dimY, dimZ;

    /// Spacing of the grid
    float spacingX, spacingY, spacingZ;

    /// origin of the grid in world space coordinates
    float orgX, orgY, orgZ;

    /// Array containing the data
    float *data;

    /// Array containing critical points
    vislib::Array<CritPoint> critPoints;
};

} // end namespace protein_cuda
} // end namespace megamol

#endif // MMPROTEINCUDAPLUGIN_VECFIELD3D_H_INCLUDED
