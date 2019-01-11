/*
 * LIC.cpp
 *
 * Copyright (C) 2012 by University of Stuttgart (VISUS).
 * All rights reserved.
 *
 * $Id$
 */

#include "stdafx.h"
#include "LIC.h"
#include "UniGrid3D.h"

#include "helper_math.h"

#include "vislib/sys/Log.h"
#include "vislib/Array.h"

using namespace megamol;

/*
 * protein_cuda::LIC::CalcLicX
 */
bool protein_cuda::LIC::CalcLicX(UniGrid3D<float3> &grid,
        UniGrid3D<float> &randBuff,
        unsigned int streamLen,
        float minStreamCnt,
        float vecScale,
        vislib::math::Vector<unsigned int, 3> licDim,
        bool projectVec2D,
        UniGrid3D<float> &licBuffX,
        UniGrid3D<float> &licBuffXAlpha) {

    using namespace vislib;
    using namespace vislib::sys;
    using namespace vislib::math;

    //printf("(Re)calculating LIC texture for x-plane.\n"); // DEBUG

    time_t t = clock();

    // Calc tex coords for planes // TODO handle case in which gridStep is not 1.0f
	int gridCoordX = static_cast<int>(licBuffX.GetGridOrg().X() -
            grid.GetGridOrg().X()/(grid.GetGridDim().X()*grid.GetGridStepSize())*grid.GetGridDim().X());
    //printf("Grid x coord in unigrid space %i\n", gridCoordX); // DEBUG
	gridCoordX *= static_cast<int>((float)(licDim.X()) / (float)(grid.GetGridDim().X()));
    //printf("Grid x coord in lic tex space %i\n", gridCoordX); // DEBUG

    //float minStreamLines = 3.0f;
    int length = static_cast<int> (streamLen);
    vislib::math::Vector<float, 3> minC(0.0f, 0.0f, 0.0f);
    float invVal = 1.0f/(2.0f*static_cast<float>(length)+1.0f);

    // Init with zero
#pragma omp parallel for
    for(int y = 0; y < static_cast<int>(licBuffX.GetGridDim().Y()); y++) {
#pragma omp parallel for
        for(int z = 0; z < static_cast<int>(licBuffX.GetGridDim().Z()); z++) {
            licBuffX.SetAt(0, y, z, 0.0f);
            licBuffXAlpha.SetAt(0, y, z, 0.0f);
        }
    }

    // Compute LIC values

    for(int cnt = 0; cnt < minStreamCnt; cnt ++) {
        for(int y = 0; y < static_cast<int>(licBuffX.GetGridDim().Y()); y++) {
            for(int z = 0; z < static_cast<int>(licBuffX.GetGridDim().Z()); z++) {

                if(licBuffXAlpha.GetAt(0, y, z) < minStreamCnt) {

                    float licCol;
                    Vector<float, 3> vPos, vDir;
                    Array<Vector<float, 3> > vPosForward, vPosBackward;
                    Array<float> fRandTexForward, fRandTexBackward;
                    vPosForward.SetCount(length);
                    vPosBackward.SetCount(length);
                    fRandTexForward.SetCount(length);
                    fRandTexBackward.SetCount(length);

                    // Compute convolution for this voxel
					vPos.Set(static_cast<float>(gridCoordX), static_cast<float>(y), static_cast<float>(z));
                    licCol = LIC::sampleRandBuffWrap(randBuff, vPos);
                    vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                    if(projectVec2D) vDir.SetX(0.0f);

                    // Go 'forward'
                    for(int p = 1; p <= length; p++) {
                        vPos = vPos + vDir*vecScale;
                        LIC::clampVec(vPos, minC, licDim);
                        float randVal = LIC::sampleRandBuffWrap(randBuff, vPos);
                        licCol += randVal;
                        vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                        if(projectVec2D) vDir.SetX(0.0f);
                        vPosForward[p-1] = Vector<float, 3>(vPos);
                        fRandTexForward[p-1] = randVal;
                    }

					vPos.Set(static_cast<float>(gridCoordX), static_cast<float>(y), static_cast<float>(z));
                    vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                    if(projectVec2D) vDir.SetX(0.0f);

                    // Go 'backwards'
                    for(int p = 1; p <= length; p++) {
                        vPos = vPos - vDir*vecScale;
                        LIC::clampVec(vPos, minC, licDim);
                        float randVal = LIC::sampleRandBuffWrap(randBuff, vPos);
                        licCol += randVal;
                        vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                        if(projectVec2D) vDir.SetX(0.0f);
                        vPosBackward[p-1] = Vector<float, 3>(vPos);
                        fRandTexBackward[p-1] = randVal;
                    }

                    // Add result to this voxel and increment alpha

                    licBuffX.SetAt(0, y, z, licBuffX.GetAt(0, y, z) + licCol/(2.0f*static_cast<float>(length)+1.0f));
                    licBuffXAlpha.SetAt(0, y, z, licBuffXAlpha.GetAt(0, y, z) + 1.0f);

                    float forwardVal, backwardVal;
                    forwardVal = licBuffX.GetAt(0, y, z);
                    backwardVal = licBuffX.GetAt(0, y, z);

                    // Follow the stream line forward

                    vPos = vPosForward[length-1];
                    vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                    if(projectVec2D) vDir.SetX(0.0f);

                    for(int str = 1; str <= length; str++) {

                        vPos = vPos + vDir*vecScale;
                        LIC::clampVec(vPos, minC, licDim);

                        forwardVal -= fRandTexBackward[length-str]*invVal;
                        forwardVal += LIC::sampleRandBuffWrap(randBuff, vPos)*invVal;

                        // Compute convolution by adding/subtracting one value
                        licBuffX.SetAt(
                                0,
                                static_cast<unsigned int>(vPosForward[str-1].Y()),
                                static_cast<unsigned int>(vPosForward[str-1].Z()),
                                licBuffX.GetAt(
                                        0,
                                        static_cast<unsigned int>(vPosForward[str-1].Y()),
                                        static_cast<unsigned int>(vPosForward[str-1].Z())) + forwardVal);

                        licBuffXAlpha.SetAt(
                                0,
                                static_cast<unsigned int>(vPosForward[str-1].Y()),
                                static_cast<unsigned int>(vPosForward[str-1].Z()),
                                licBuffXAlpha.GetAt(
                                        0,
                                        static_cast<unsigned int>(vPosForward[str-1].Y()),
                                        static_cast<unsigned int>(vPosForward[str-1].Z())) + 1.0f);

                        vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                        if(projectVec2D) vDir.SetX(0.0f);
                    }

                    // Follow the stream line backwards

                    vPos = vPosBackward[length-1];
                    vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                    if(projectVec2D) vDir.SetX(0.0f);

                    for(int str = 1; str <= length; str++) {

                        vPos = vPos - vDir*vecScale;
                        LIC::clampVec(vPos, minC, licDim);

                        backwardVal -= fRandTexForward[length-str]*invVal;
                        backwardVal += LIC::sampleRandBuffWrap(randBuff, vPos)*invVal;

                        // Compute convolution by adding/subtracting one value
                        licBuffX.SetAt(
                                0,
                                static_cast<unsigned int>(vPosBackward[str-1].Y()),
                                static_cast<unsigned int>(vPosBackward[str-1].Z()),
                                licBuffX.GetAt(
                                        0,
                                        static_cast<unsigned int>(vPosBackward[str-1].Y()),
                                        static_cast<unsigned int>(vPosBackward[str-1].Z())) + backwardVal);

                        licBuffXAlpha.SetAt(
                                0,
                                static_cast<unsigned int>(vPosBackward[str-1].Y()),
                                static_cast<unsigned int>(vPosBackward[str-1].Z()),
                                licBuffXAlpha.GetAt(
                                        0,
                                        static_cast<unsigned int>(vPosBackward[str-1].Y()),
                                        static_cast<unsigned int>(vPosBackward[str-1].Z())) + 1.0f);

                        vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                        if(projectVec2D) vDir.SetX(0.0f);

                    }
                }
            }
        }
    }

    // Normalize LIC values according to alpha
#pragma omp parallel for
    for(int y = 0; y < static_cast<int>(licBuffX.GetGridDim().Y()); y++) {
#pragma omp parallel for
        for(int z = 0; z < static_cast<int>(licBuffX.GetGridDim().Z()); z++) {
            licBuffX.SetAt(0, y, z, licBuffX.GetAt(0, y, z)/licBuffXAlpha.GetAt(0, y, z));
        }
    }

    /*float min=licBuffX.GetAt(0, 0, 0), max=licBuffX.GetAt(0, 0, 0);
    for(int y = 0; y < static_cast<int>(licBuffX.GetGridDim().Y()); y++) {
        for(int z = 0; z < static_cast<int>(licBuffX.GetGridDim().Z()); z++) {
            if(max < licBuffX.GetAt(0, y, z)) {
                max = licBuffX.GetAt(0, y, z);
            }
            if(min > licBuffX.GetAt(0, y, z)) {
                min = licBuffX.GetAt(0, y, z);
            }
        }
    }
    printf("min %f, max %f\n", min, max);*/ // DEBUG

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Time for computing LIC texture (x-Plane) : %f",
            (double(clock()-t)/double(CLOCKS_PER_SEC) )); // DEBUG

    return true;
}


/*
 * protein_cuda::LIC::CalcLicY
 */
bool protein_cuda::LIC::CalcLicY(UniGrid3D<float3> &grid,
        UniGrid3D<float> &randBuff,
        unsigned int streamLen,
        float minStreamCnt,
        float vecScale,
        vislib::math::Vector<unsigned int, 3> licDim,
        bool projectVec2D,
        UniGrid3D<float> &licBuffY,
        UniGrid3D<float> &licBuffYAlpha) {

    using namespace vislib;
    using namespace vislib::sys;
    using namespace vislib::math;

    //printf("(Re)calculating LIC texture for y-plane.\n"); // DEBUG

    time_t t = clock();

    // Calc tex coords for planes // TODO handle case in which gridStep is not 1.0f
    int gridCoordY = static_cast<int>(licBuffY.GetGridOrg().Y() -
            grid.GetGridOrg().Y()/(grid.GetGridDim().Y()*grid.GetGridStepSize())*grid.GetGridDim().Y());
    gridCoordY *= static_cast<int>(((float)(licDim.Y())/(float)(grid.GetGridDim().Y())));

    //float minStreamLines = 3.0f;
    int length = static_cast<int> (streamLen);
    vislib::math::Vector<float, 3> minC(0.0f, 0.0f, 0.0f);
    float invVal = 1.0f/(2.0f*static_cast<float>(length)+1.0f);

    // Init with zero
#pragma omp parallel for
    for(int x = 0; x < static_cast<int>(licBuffY.GetGridDim().X()); x++) {
#pragma omp parallel for
        for(int z = 0; z < static_cast<int>(licBuffY.GetGridDim().Z()); z++) {
            licBuffY.SetAt(x, 0, z, 0.0f);
            licBuffYAlpha.SetAt(x, 0, z, 0.0f);
        }
    }

    // Compute LIC values

    for(int cnt = 0; cnt < minStreamCnt; cnt ++) {
        for(int x = 0; x < static_cast<int>(licBuffY.GetGridDim().X()); x++) {
            for(int z = 0; z < static_cast<int>(licBuffY.GetGridDim().Z()); z++) {

                if(licBuffYAlpha.GetAt(x, 0, z) < minStreamCnt) {

                    float licCol;
                    Vector<float, 3> vPos, vDir;
                    Array<Vector<float, 3> > vPosForward, vPosBackward;
                    Array<float> fRandTexForward, fRandTexBackward;
                    vPosForward.SetCount(length);
                    vPosBackward.SetCount(length);
                    fRandTexForward.SetCount(length);
                    fRandTexBackward.SetCount(length);

                    // Compute convolution for this voxel
					vPos.Set(static_cast<float>(x), static_cast<float>(gridCoordY), static_cast<float>(z));
                    licCol = LIC::sampleRandBuffWrap(randBuff, vPos);
                    vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                    if(projectVec2D) vDir.SetY(0.0f);

                    // Go 'forward'
                    for(int p = 1; p <= length; p++) {
                        vPos = vPos + vDir*vecScale;
                        LIC::clampVec(vPos, minC, licDim);
                        float randVal = LIC::sampleRandBuffWrap(randBuff, vPos);
                        licCol += randVal;
                        vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                        if(projectVec2D) vDir.SetY(0.0f);
                        vPosForward[p-1] = Vector<float, 3>(vPos);
                        fRandTexForward[p-1] = randVal;
                    }

					vPos.Set(static_cast<float>(x), static_cast<float>(gridCoordY), static_cast<float>(z));
                    vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                    if(projectVec2D) vDir.SetY(0.0f);

                    // Go 'backwards'
                    for(int p = 1; p <= length; p++) {
                        vPos = vPos - vDir*vecScale;
                        LIC::clampVec(vPos, minC, licDim);
                        float randVal = LIC::sampleRandBuffWrap(randBuff, vPos);
                        licCol += randVal;
                        vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                        if(projectVec2D) vDir.SetY(0.0f);
                        vPosBackward[p-1] = Vector<float, 3>(vPos);
                        fRandTexBackward[p-1] = randVal;
                    }

                    // Add result to this voxel and increment alpha

                    licBuffY.SetAt(x, 0, z, licBuffY.GetAt(x, 0, z) + licCol/(2.0f*static_cast<float>(length)+1.0f));
                    licBuffYAlpha.SetAt(x, 0, z, licBuffYAlpha.GetAt(x, 0, z) + 1.0f);

                    float forwardVal, backwardVal;
                    forwardVal = licBuffY.GetAt(x, 0, z);
                    backwardVal = licBuffY.GetAt(x, 0, z);

                    // Follow the stream line forward

                    vPos = vPosForward[length-1];
                    vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                    if(projectVec2D) vDir.SetY(0.0f);

                    for(int str = 1; str <= length; str++) {

                        vPos = vPos + vDir*vecScale;
                        LIC::clampVec(vPos, minC, licDim);

                        forwardVal -= fRandTexBackward[length-str]*invVal;
                        forwardVal += LIC::sampleRandBuffWrap(randBuff, vPos)*invVal;

                        // Compute convolution by adding/subtracting one value
                        licBuffY.SetAt(
                                static_cast<unsigned int>(vPosForward[str-1].X()),
                                0,
                                static_cast<unsigned int>(vPosForward[str-1].Z()),
                                licBuffY.GetAt(
                                        static_cast<unsigned int>(vPosForward[str-1].X()),
                                        0,
                                        static_cast<unsigned int>(vPosForward[str-1].Z())) + forwardVal);

                        licBuffYAlpha.SetAt(
                                static_cast<unsigned int>(vPosForward[str-1].X()),
                                0,
                                static_cast<unsigned int>(vPosForward[str-1].Z()),
                                licBuffYAlpha.GetAt(
                                        static_cast<unsigned int>(vPosForward[str-1].X()),
                                        0,
                                        static_cast<unsigned int>(vPosForward[str-1].Z())) + 1.0f);

                        vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                        if(projectVec2D) vDir.SetY(0.0f);
                    }


                    // Follow the stream line backwards


                    vPos = vPosBackward[length-1];
                    vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                    if(projectVec2D) vDir.SetY(0.0f);

                    for(int str = 1; str <= length; str++) {

                        vPos = vPos - vDir*vecScale;
                        LIC::clampVec(vPos, minC, licDim);

                        backwardVal -= fRandTexForward[length-str]*invVal;
                        backwardVal += LIC::sampleRandBuffWrap(randBuff, vPos)*invVal;


                        // Compute convolution by adding/subtracting one value
                        licBuffY.SetAt(
                                static_cast<unsigned int>(vPosBackward[str-1].X()),
                                0,
                                static_cast<unsigned int>(vPosBackward[str-1].Z()),
                                licBuffY.GetAt(
                                        static_cast<unsigned int>(vPosBackward[str-1].X()),
                                        0,
                                        static_cast<unsigned int>(vPosBackward[str-1].Z())) + backwardVal);

                        licBuffYAlpha.SetAt(
                                static_cast<unsigned int>(vPosBackward[str-1].X()),
                                0,
                                static_cast<unsigned int>(vPosBackward[str-1].Z()),
                                licBuffYAlpha.GetAt(
                                        static_cast<unsigned int>(vPosBackward[str-1].X()),
                                        0,
                                        static_cast<unsigned int>(vPosBackward[str-1].Z())) + 1.0f);

                        vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                        if(projectVec2D) vDir.SetY(0.0f);

                    }
                }
            }
        }
    }

    // Normalize LIC values according to alpha
#pragma omp parallel for
    for(int x = 0; x < static_cast<int>(licBuffY.GetGridDim().X()); x++) {
#pragma omp parallel for
        for(int z = 0; z < static_cast<int>(licBuffY.GetGridDim().Z()); z++) {
            licBuffY.SetAt(x, 0, z, licBuffY.GetAt(x, 0, z)/licBuffYAlpha.GetAt(x, 0, z));
        }
    }

    /*float min=licBuffY.GetAt(0, 0, 0), max=licBuffY.GetAt(0, 0, 0);
    for(int x = 0; x < static_cast<int>(licBuffY.GetGridDim().X()); x++) {
        for(int z = 0; z < static_cast<int>(licBuffY.GetGridDim().Z()); z++) {
            if(max < licBuffY.GetAt(x, 0, z)) {
                max = licBuffY.GetAt(x, 0, z);
            }
            if(min > licBuffY.GetAt(x, 0, z)) {
                min = licBuffY.GetAt(x, 0, z);
            }
        }
    }
    printf("min %f, max %f\n", min, max);*/ // DEBUG

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Time for computing LIC texture (y-Plane) : %f",
            (double(clock()-t)/double(CLOCKS_PER_SEC) )); // DEBUG

    return true;
}


/*
 * protein_cuda::LIC::CalcLicZ
 */
bool protein_cuda::LIC::CalcLicZ(UniGrid3D<float3> &grid,
        UniGrid3D<float> &randBuff,
        unsigned int streamLen,
        float minStreamCnt,
        float vecScale,
        vislib::math::Vector<unsigned int, 3> licDim,
        bool projectVec2D,
        UniGrid3D<float> &licBuffZ,
        UniGrid3D<float> &licBuffZAlpha) {

    using namespace vislib;
    using namespace vislib::sys;
    using namespace vislib::math;

    //printf("(Re)calculating LIC texture for z-plane.\n"); // DEBUG

    time_t t = clock();

    // Calc tex coords for planes // TODO handle case in which gridStep is not 1.0f
    int gridCoordZ = static_cast<int>(licBuffZ.GetGridOrg().Z() -
            grid.GetGridOrg().Z()/(grid.GetGridDim().Z()*grid.GetGridStepSize())*grid.GetGridDim().Z());
    gridCoordZ *= static_cast<int>((float)(licDim.Z())/(float)(grid.GetGridDim().Z()));

    //float minStreamLines = 3.0f;
    int length = static_cast<int> (streamLen);
    vislib::math::Vector<float, 3> minC(0.0f, 0.0f, 0.0f);
    float invVal = 1.0f/(2.0f*static_cast<float>(length)+1.0f);

    // Init with zero
#pragma omp parallel for
    for(int x = 0; x < static_cast<int>(licBuffZ.GetGridDim().X()); x++) {
#pragma omp parallel for
        for(int y = 0; y < static_cast<int>(licBuffZ.GetGridDim().Y()); y++) {
            licBuffZ.SetAt(x, y, 0, 0.0f);
            licBuffZAlpha.SetAt(x, y, 0, 0.0f);
        }
    }

    // Compute LIC values

    for(int cnt = 0; cnt < minStreamCnt; cnt ++) {
        for(int x = 0; x < static_cast<int>(licBuffZ.GetGridDim().X()); x++) {
            for(int y = 0; y < static_cast<int>(licBuffZ.GetGridDim().Y()); y++) {

                if(licBuffZAlpha.GetAt(x, y, 0) < minStreamCnt) {

                    float licCol;
                    Vector<float, 3> vPos, vDir;
                    Array<Vector<float, 3> > vPosForward, vPosBackward;
                    Array<float> fRandTexForward, fRandTexBackward;
                    vPosForward.SetCount(length);
                    vPosBackward.SetCount(length);
                    fRandTexForward.SetCount(length);
                    fRandTexBackward.SetCount(length);

                    // Compute convolution for this voxel
					vPos.Set(static_cast<float>(x), static_cast<float>(y), static_cast<float>(gridCoordZ));
                    licCol = LIC::sampleRandBuffWrap(randBuff, vPos);
                    vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                    if(projectVec2D) vDir.SetZ(0.0f);

                    // Go 'forward'
                    for(int p = 1; p <= length; p++) {
                        vPos = vPos + vDir*vecScale;
                        LIC::clampVec(vPos, minC, licDim);
                        float randVal = LIC::sampleRandBuffWrap(randBuff, vPos);
                        licCol += randVal;
                        vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                        if(projectVec2D) vDir.SetZ(0.0f);
                        vPosForward[p-1] = Vector<float, 3>(vPos);
                        fRandTexForward[p-1] = randVal;
                    }

					vPos.Set(static_cast<float>(x), static_cast<float>(y), static_cast<float>(gridCoordZ));
                    vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                    if(projectVec2D) vDir.SetZ(0.0f);

                    // Go 'backwards'
                    for(int p = 1; p <= length; p++) {
                        vPos = vPos - vDir*vecScale;
                        LIC::clampVec(vPos, minC, licDim);
                        float randVal = LIC::sampleRandBuffWrap(randBuff, vPos);
                        licCol += randVal;
                        vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                        if(projectVec2D) vDir.SetZ(0.0f);
                        vPosBackward[p-1] = Vector<float, 3>(vPos);
                        fRandTexBackward[p-1] = randVal;
                    }

                    // Add result to this voxel and increment alpha

                    licBuffZ.SetAt(x, y, 0, licBuffZ.GetAt(x, y, 0) + licCol/(2.0f*static_cast<float>(length)+1.0f));
                    licBuffZAlpha.SetAt(x, y, 0, licBuffZAlpha.GetAt(x, y, 0) + 1.0f);

                    float forwardVal, backwardVal;
                    forwardVal = licBuffZ.GetAt(x, y, 0);
                    backwardVal = licBuffZ.GetAt(x, y, 0);

                    // Follow the stream line forward

                    vPos = vPosForward[length-1];
                    vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                    if(projectVec2D) vDir.SetZ(0.0f);

                    for(int str = 1; str <= length; str++) {

                        vPos = vPos + vDir*vecScale;
                        LIC::clampVec(vPos, minC, licDim);

                        forwardVal -= fRandTexBackward[length-str]*invVal;
                        forwardVal += LIC::sampleRandBuffWrap(randBuff, vPos)*invVal;

                        // Compute convolution by adding/subtracting one value
                        licBuffZ.SetAt(
                                static_cast<unsigned int>(vPosForward[str-1].X()),
                                static_cast<unsigned int>(vPosForward[str-1].Y()),
                                0,
                                licBuffZ.GetAt(
                                        static_cast<unsigned int>(vPosForward[str-1].X()),
                                        static_cast<unsigned int>(vPosForward[str-1].Y()),
                                        0) + forwardVal);

                        licBuffZAlpha.SetAt(
                                static_cast<unsigned int>(vPosForward[str-1].X()),
                                static_cast<unsigned int>(vPosForward[str-1].Y()),
                                0,
                                licBuffZAlpha.GetAt(
                                        static_cast<unsigned int>(vPosForward[str-1].X()),
                                        static_cast<unsigned int>(vPosForward[str-1].Y()),
                                        0) + 1.0f);

                        vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                        if(projectVec2D) vDir.SetZ(0.0f);
                    }


                    // Follow the stream line backwards


                    vPos = vPosBackward[length-1];
                    vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                    if(projectVec2D) vDir.SetZ(0.0f);

                    for(int str = 1; str <= length; str++) {

                        vPos = vPos - vDir*vecScale;
                        LIC::clampVec(vPos, minC, licDim);

                        backwardVal -= fRandTexForward[length-str]*invVal;
                        backwardVal += LIC::sampleRandBuffWrap(randBuff, vPos)*invVal;


                        // Compute convolution by adding/subtracting one value
                        licBuffZ.SetAt(
                                static_cast<unsigned int>(vPosBackward[str-1].X()),
                                static_cast<unsigned int>(vPosBackward[str-1].Y()),
                                0,
                                licBuffZ.GetAt(
                                        static_cast<unsigned int>(vPosBackward[str-1].X()),
                                        static_cast<unsigned int>(vPosBackward[str-1].Y()),
                                        0) + backwardVal);

                        licBuffZAlpha.SetAt(
                                static_cast<unsigned int>(vPosBackward[str-1].X()),
                                static_cast<unsigned int>(vPosBackward[str-1].Y()),
                                0,
                                licBuffZAlpha.GetAt(
                                        static_cast<unsigned int>(vPosBackward[str-1].X()),
                                        static_cast<unsigned int>(vPosBackward[str-1].Y()),
                                        0) + 1.0f);

                        vDir = LIC::sampleUniGrid(vPos, grid, licDim);
                        if(projectVec2D) vDir.SetZ(0.0f);

                    }
                }
            }
        }
    }

    // Normalize LIC values according to alpha
#pragma omp parallel for
    for(int x = 0; x < static_cast<int>(licBuffZ.GetGridDim().X()); x++) {
#pragma omp parallel for
        for(int y = 0; y < static_cast<int>(licBuffZ.GetGridDim().Y()); y++) {
            licBuffZ.SetAt(x, y, 0, licBuffZ.GetAt(x, y, 0)/licBuffZAlpha.GetAt(x, y, 0));
        }
    }

    /*float min=licBuffZ.GetAt(0, 0, 0), max=licBuffZ.GetAt(0, 0, 0);
    for(int x = 0; x < static_cast<int>(licBuffZ.GetGridDim().X()); x++) {
        for(int y = 0; y < static_cast<int>(licBuffZ.GetGridDim().Y()); y++) {
            if(max < licBuffZ.GetAt(x, y, 0)) {
                max = licBuffZ.GetAt(x, y, 0);
            }
            if(min > licBuffZ.GetAt(x, y, 0)) {
                min = licBuffZ.GetAt(x, y, 0);
            }
        }
    }
    printf("min %f, max %f\n", min, max);*/ // DEBUG

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Time for computing LIC texture (z-Plane) : %f",
            (double(clock()-t)/double(CLOCKS_PER_SEC) )); // DEBUG

    return true;
}


/*
 * protein_cuda::LIC::sampleUniGrid
 */
vislib::math::Vector<float, 3> protein_cuda::LIC::sampleUniGrid(
        vislib::math::Vector<float, 3> pos,
        UniGrid3D<float3> &grid,
        vislib::math::Vector<float, 3> licDim) {

    // Nearest neighbour sampling
    vislib::math::Vector<unsigned int, 3> p(
            (unsigned int)(pos.X()*((float)grid.GetGridDim().X()/(float)licDim.X())),
            (unsigned int)(pos.Y()*((float)grid.GetGridDim().Y()/(float)licDim.Y())),
            (unsigned int)(pos.Z()*((float)grid.GetGridDim().Z()/(float)licDim.Z())));

    //printf("==== unigrid dimensions: %u %u %u\n", grid.GetGridDim().X(), grid.GetGridDim().Y(), grid.GetGridDim().Z()); // DEBUG
    //printf("==== sampling at pos: %u %u %u\n", p.X(), p.Y(), p.Z()); // DEBUG

    float3 res = grid.GetAt(p.X(), p.Y(), p.Z());
    return vislib::math::Vector<float, 3>(res.x, res.y, res.z);
}


/*
 * protein_cuda::LIC::sampleRandBuffWrap
 */
float protein_cuda::LIC::sampleRandBuffWrap(UniGrid3D<float> &randBuff,
        vislib::math::Vector<float, 3> pos) {

    vislib::math::Vector<unsigned int, 3> p(
            static_cast<unsigned int>(pos.X())%randBuff.GetGridDim().X(),
            static_cast<unsigned int>(pos.Y())%randBuff.GetGridDim().Y(),
            static_cast<unsigned int>(pos.Z())%randBuff.GetGridDim().Z());

    return randBuff.GetAt(p.X(), p.Y(), p.Z());
}


/*
 * protein_cuda::LIC::clampVec
 */
void protein_cuda::LIC::clampVec(vislib::math::Vector<float, 3> &vec,
        vislib::math::Vector<float, 3> min, vislib::math::Vector<float, 3> max) {
    //printf("===== vec %f %f %f\n", vec.X(), vec.Y(), vec.Z()); // DEBUG
    //printf("===== min %f %f %f\n", min.X(), min.Y(), min.Z()); // DEBUG
    //printf("===== max %f %f %f\n", max.X(), max.Y(), max.Z()); // DEBUG
    ASSERT(max.X() >= min.X());
    ASSERT(max.Y() >= min.Y());
    ASSERT(max.Z() >= min.Z());
    if(vec.X() < min.X()) vec.SetX(min.X());
    if(vec.Y() < min.Y()) vec.SetY(min.Y());
    if(vec.Z() < min.Z()) vec.SetZ(min.Z());
    if(vec.X() >= max.X()) vec.SetX(max.X()-1.0f); // TODO?
    if(vec.Y() >= max.Y()) vec.SetY(max.Y()-1.0f);
    if(vec.Z() >= max.Z()) vec.SetZ(max.Z()-1.0f);
}
