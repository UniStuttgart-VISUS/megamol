/*
 * SphereRenderer.cpp
 *
 * Copyright (C) 2009 - 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "MolecularAOShader.h"
#include "protein_calls/MolecularDataCall.h"
#include <omp.h>
#include <algorithm>

using namespace megamol::protein_cuda;

/*
 * MolecularAOShader::SphereRenderer
 */
MolecularAOShader::MolecularAOShader(void) 
    : volSizeX(8), volSizeY(8), volSizeZ(8), genFac(1.0f)
{
    // Do nothing.
}

/*
 * MolecularAOShader::~MolecularAOShader
 */
MolecularAOShader::~MolecularAOShader(void)
{
    // Do nothing.
}

/*
 * MolecularAOShader::getVolumeSizeX
 */
int MolecularAOShader::getVolumeSizeX() const
{
    return this->volSizeX;
}

/*
 * MolecularAOShader::getVolumeSizeY
 */
int MolecularAOShader::getVolumeSizeY() const
{
    return this->volSizeY;
}

/*
 * MolecularAOShader::getVolumeSizeZ
 */
int MolecularAOShader::getVolumeSizeZ() const
{
    return this->volSizeZ;
}

/*
 * MolecularAOShader::setVolumeSize
 */
void MolecularAOShader::setVolumeSize(int volSizeX, int volSizeY, int volSizeZ)
{
    this->volSizeX = std::max(4, volSizeX);
    this->volSizeY = std::max(4, volSizeY);
    this->volSizeZ = std::max(4, volSizeZ);
}

/*
  * MolecularAOShader::setGenerationFactor
 */
void MolecularAOShader::setGenerationFactor(float genFac)
{
    this->genFac = std::max(0.0f, genFac);
}

/*
 * MolecularAOShader::createVolume
 */
float* MolecularAOShader::createVolume(class megamol::protein_calls::MolecularDataCall& mol)
{
    int sx = this->volSizeX - 2;
    int sy = this->volSizeY - 2;
    int sz = this->volSizeZ - 2;

    // Allocate empty volume.
    float **vol = new float*[omp_get_max_threads()];
    int init, i, j;
#pragma omp parallel for
    for( init = 0; init < omp_get_max_threads(); init++ ) {
        vol[init] = new float[sx * sy * sz];
        ::memset(vol[init], 0, sizeof(float) * sx * sy * sz);
    }

    float minOSx = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Left();
    float minOSy = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Bottom();
    float minOSz = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Back();
    float rangeOSx = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Width();
    float rangeOSy = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Height();
    float rangeOSz = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Depth();

    float voxelVol = (rangeOSx / static_cast<float>(sx))
        * (rangeOSy / static_cast<float>(sy))
        * (rangeOSz / static_cast<float>(sz));

    // Compute AO Factors for ech atom.
#pragma omp parallel for
	for (i = 0; i < (int)mol.AtomCount(); i++) {
        int x = static_cast<int>(((mol.AtomPositions()[i*3+0] - minOSx) / rangeOSx) * static_cast<float>(sx));
        if (x < 0) x = 0; else if (x >= sx) x = sx - 1;
        int y = static_cast<int>(((mol.AtomPositions()[i*3+1] - minOSy) / rangeOSy) * static_cast<float>(sy));
        if (y < 0) y = 0; else if (y >= sy) y = sy - 1;
        int z = static_cast<int>(((mol.AtomPositions()[i*3+2] - minOSz) / rangeOSz) * static_cast<float>(sz));
        if (z < 0) z = 0; else if (z >= sz) z = sz - 1;
        float rad = mol.AtomTypes()[mol.AtomTypeIndices()[i]].Radius();
        float spVol = 4.0f / 3.0f * static_cast<float>(M_PI) * rad * rad * rad;

        vol[omp_get_thread_num()][x + (y + z * sy) * sx] += (spVol / voxelVol) * this->genFac;
    }

    // Aggregate AO Factors.
#pragma omp parallel for
    for (j = 0; j < sx * sy * sz; j++ ) {
        for ( unsigned int i = 1; i < static_cast<unsigned int>(omp_get_max_threads()); i++ ) {
            vol[0][j] += vol[i][j];
        }
    }

    float* result = vol[0];

    // Cleanup (exept for result)
#pragma omp parallel for
	for (init = 1; init < omp_get_max_threads(); init++) {
        delete[] vol[init];
    }
    delete[] vol;

    return result;
}

/*
 * MolecularAOShader::createVolumeDebug
 */
float* MolecularAOShader::createVolumeDebug(class megamol::protein_calls::MolecularDataCall& mol)
{
    int sx = this->volSizeX - 2;
    int sy = this->volSizeY - 2;
    int sz = this->volSizeZ - 2;

    // Allocate empty volume.
    float *vol;
    int i;
    vol = new float[sx * sy * sz];
    ::memset(vol, 0, sizeof(float) * sx * sy * sz);

    float minOSx = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Left();
    float minOSy = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Bottom();
    float minOSz = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Back();
    float rangeOSx = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Width();
    float rangeOSy = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Height();
    float rangeOSz = mol.AccessBoundingBoxes().ObjectSpaceClipBox().Depth();

    float voxelVol = (rangeOSx / static_cast<float>(sx))
        * (rangeOSy / static_cast<float>(sy))
        * (rangeOSz / static_cast<float>(sz));

    // Compute AO Factors for ech atom.
	for (i = 0; i < (int)mol.AtomCount(); i++) {
        int x = static_cast<int>(((mol.AtomPositions()[i*3+0] - minOSx) / rangeOSx) * static_cast<float>(sx));
        if (x < 0) x = 0; else if (x >= sx) x = sx - 1;
        int y = static_cast<int>(((mol.AtomPositions()[i*3+1] - minOSy) / rangeOSy) * static_cast<float>(sy));
        if (y < 0) y = 0; else if (y >= sy) y = sy - 1;
        int z = static_cast<int>(((mol.AtomPositions()[i*3+2] - minOSz) / rangeOSz) * static_cast<float>(sz));
        if (z < 0) z = 0; else if (z >= sz) z = sz - 1;
        float rad = mol.AtomTypes()[mol.AtomTypeIndices()[i]].Radius();
        float spVol = 4.0f / 3.0f * static_cast<float>(M_PI) * rad * rad * rad;

        vol[x + (y + z * sy) * sx] += (spVol / voxelVol) * this->genFac;
    }

    return vol;
}
