/*
 * LIC.h
 *
 * Copyright (C) 2012 by University of Stuttgart (VISUS).
 * All rights reserved.
 *
 * $Id$
 */

#ifndef MMPROTEINCUDAPLUGIN_LIC_H
#define MMPROTEINCUDAPLUGIN_LIC_H

#include "vislib/math/Vector.h"
#include "helper_cuda.h"
#include "helper_math.h"

#include "UniGrid3D.h"

namespace megamol {
namespace protein_cuda {

/**
 * Class offering computation of LIC (Line Integral Convolution) based on a
 * 3D uniform grid.
 */
class LIC {

public:

    /**
     * Computes LIC for plane parallel to x-y-plane. The computation is based
     * on 3D data, but if flagged, all vectors in the uniform grid are projected
     * onto the x-y-plane in order to achieve better results.
     *
     * @param[in]  grid          The 3D uniform grid containing the vector field
     * @param[in]  randBuff      Buffer containing random values
     * @param[in]  streamLen     The length of the streamlines
     * @param[in]  minStreamCnt  Minimal number of streamlines crossing one voxel
     * @param[in]  vecScale      The scale factor for the vectors
     * @param[in]  licDim        The dimension of the LIC textures
     * @param[in]  projectVec2D  Flag whether vectors should be projected onto
     *                           the x-y-plane.
     * @param[out] licBuffX      The buffer holding the LIC value
     * @param[out] licBuffXAlpha The buffer holding the LIC alpha value
     *
     * @return 'True' on success, 'false' otherwise
     */
    static bool CalcLicX(UniGrid3D<float3> &grid,
            UniGrid3D<float> &randBuff,
            unsigned int streamLen,
            float minStreamCnt,
            float vecScale,
            vislib::math::Vector<unsigned int, 3> licDim,
            bool projectVec2D,
            UniGrid3D<float> &licBuffX,
            UniGrid3D<float> &licBuffXAlpha);

    /**
     * Computes LIC for plane parallel to x-z-plane. The computation is based
     * on 3D data, but if flagged all vectors in the uniform grid are projected
     * onto the x-z-plane in order to achieve better results.
     *
     * @param[in]  grid          The 3D uniform grid containing the vector field
     * @param[in]  randBuff      Buffer containing random values
     * @param[in]  streamLen     The length of the streamlines
     * @param[in]  minStreamCnt  Minimal number of streamlines crossing one voxel
     * @param[in]  vecScale      The scale factor for the vectors
     * @param[in]  licDim        The dimension of the LIC textures
     * @param[in]  projectVec2D  Flag whether vectors should be projected onto
     *                           the x-z-plane.
     * @param[out] licBuffY      The buffer holding the LIC value
     * @param[out] licBuffYAlpha The buffer holding the LIC alpha value
     *
     * @return 'True' on success, 'false' otherwise
     */
    static bool CalcLicY(UniGrid3D<float3> &grid,
            UniGrid3D<float> &randBuff,
            unsigned int streamLen,
            float minStreamCnt,
            float vecScale,
            vislib::math::Vector<unsigned int, 3> licDim,
            bool projectVec2D,
            UniGrid3D<float> &licBuffY,
            UniGrid3D<float> &licBuffYAlpha);

    /**
     * Computes LIC for plane parallel to y-z-plane. The computation is based
     * on 3D data, but if flagged all vectors in the uniform grid are projected
     * onto the y-z-plane in order to achieve better results.
     *
     * @param[in]  grid          The 3D uniform grid containing the vector field
     * @param[in]  randBuff      Buffer containing random values
     * @param[in]  streamLen     The length of the streamlines
     * @param[in]  minStreamCnt  Minimal number of streamlines crossing one voxel
     * @param[in]  vecScale      The scale factor for the vectors
     * @param[in]  licDim        The dimension of the LIC textures
     * @param[in]  projectVec2D  Flag whether vectors should be projected onto
     *                           the y-z-plane.
     * @param[out] licBuffZ      The buffer holding the LIC value
     * @param[out] licBuffZAlpha The buffer holding the LIC alpha value
     *
     * @return 'True' on success, 'false' otherwise
     */
    static bool CalcLicZ(UniGrid3D<float3> &grid,
            UniGrid3D<float> &randBuff,
            unsigned int streamLen,
            float minStreamCnt,
            float vecScale,
            vislib::math::Vector<unsigned int, 3> licDim,
            bool projectVec2D,
            UniGrid3D<float> &licBuffZ,
            UniGrid3D<float> &licBuffZAlpha);

protected:

    /**
     * Sample the uniform grid containing the vector field. Special treatment
     * necessary because the dimensions of the grid and the LIC buffer could be
     * different.
     *
     * @param[in] pos     The position in the lic buffer
     * @param[in] grid    The uniform grid containing the vector field
     * @param[in] licDim  The dimensions of the LIC textures
     *
     * @return The according value in the vector field
     */
    static vislib::math::Vector<float, 3> sampleUniGrid(
            vislib::math::Vector<float, 3> pos,
            UniGrid3D<float3> &grid,
            vislib::math::Vector<float, 3> licDim);

    /**
     * Sample the random buffer based on LIC buffer position while wrapping
     * coordinates.
     *
     * @param[in] randBuff The random buffer
     * @param[in] pos      The sampling position
     *
     * @return The sampled random value
     */
    static float sampleRandBuffWrap(UniGrid3D<float> &randBuff,
            vislib::math::Vector<float, 3> pos);

private:

    /**
     * Helper function which clamps a vector according to a given value.
     *
     * @param vec The vector to be clamped
     * @param min The lower boundary
     * @param max The upper boundary
     */
    static void clampVec(vislib::math::Vector<float, 3> &vec,
            vislib::math::Vector<float, 3> min,
            vislib::math::Vector<float, 3> max);
};

} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif /* MMPROTEINCUDAPLUGIN_LIC_H */
