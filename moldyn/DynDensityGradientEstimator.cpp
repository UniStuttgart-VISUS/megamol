/*
 * DynDensityGradientEstimator.cpp
 *
 *  Created on: May 22, 2014
 *      Author: scharnkn@visus.uni-stuttgart.de
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include "DynDensityGradientEstimator.h"
#include "moldyn/MultiParticleDataCall.h"
#include "moldyn/DirectionalParticleDataCall.h"
#include <omp.h>
#include "param/IntParam.h"

using namespace megamol;
using namespace megamol::core;

/*
 * moldyn::DynDensityGradientEstimator::DynDensityGradientEstimator
 */
moldyn::DynDensityGradientEstimator::DynDensityGradientEstimator(void)
        : Module(), getPartDataSlot("getPartData", "..."),
             putDirDataSlot("putDirData", "..."),
             dens(NULL),
             xResSlot("sizex", "The size of the volume in numbers of voxels"),
             yResSlot("sizey", "The size of the volume in numbers of voxels"),
             zResSlot("sizez", "The size of the volume in numbers of voxels") {

    this->getPartDataSlot.SetCompatibleCall<
            moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getPartDataSlot);

    this->putDirDataSlot.SetCallback("DirectionalParticleDataCall", "GetData",
            &DynDensityGradientEstimator::getData);
    this->putDirDataSlot.SetCallback("DirectionalParticleDataCall", "GetExtent",
            &DynDensityGradientEstimator::getExtent);
    this->MakeSlotAvailable(&this->putDirDataSlot);

    this->xResSlot << new core::param::IntParam(16);
    this->MakeSlotAvailable(&this->xResSlot);

    this->yResSlot << new core::param::IntParam(16);
    this->MakeSlotAvailable(&this->yResSlot);

    this->zResSlot << new core::param::IntParam(16);
    this->MakeSlotAvailable(&this->zResSlot);

}

/*
 * moldyn::DynDensityGradientEstimator::~DynDensityGradientEstimator
 */
moldyn::DynDensityGradientEstimator::~DynDensityGradientEstimator(void) {
    this->Release();
}

/*
 * moldyn::DynDensityGradientEstimator::create
 */
bool moldyn::DynDensityGradientEstimator::create(void) {
    return true;
}

/*
 * moldyn::DynDensityGradientEstimator::release
 */
void moldyn::DynDensityGradientEstimator::release(void) {
    if (this->dens) delete[] this->dens;
}

/*
 * moldyn::DynDensityGradientEstimator::getExtent
 */
bool moldyn::DynDensityGradientEstimator::getExtent(Call& call) {

    // Get a pointer to the incoming data call.
    moldyn::DirectionalParticleDataCall *callIn = dynamic_cast<moldyn::DirectionalParticleDataCall*>(&call);
    if(callIn == NULL) return false;

    // Get a pointer to the outgoing data call.
    moldyn::MultiParticleDataCall *callOut = this->getPartDataSlot.CallAs<moldyn::MultiParticleDataCall>();
    if(callOut == NULL) return false;

    // Get extend.
    if(!(*callOut)(1)) return false;

    // Set extend.
    callIn->AccessBoundingBoxes().Clear();
    callIn->SetExtent(callOut->FrameCount(), callOut->AccessBoundingBoxes());

    return true;
}

/*
 * moldyn::DynDensityGradientEstimator::getData
 */
bool moldyn::DynDensityGradientEstimator::getData(Call& call) {

    // Get a pointer to the incoming data call.
    moldyn::DirectionalParticleDataCall *callIn =
            dynamic_cast<moldyn::DirectionalParticleDataCall*>(&call);
    if (callIn == NULL) return false;

    // Get a pointer to the outgoing data call.
    moldyn::MultiParticleDataCall *callOut =
            this->getPartDataSlot.CallAs<moldyn::MultiParticleDataCall>();
    if (callOut == NULL) {
        return false;
    }

    callOut->SetFrameID(callIn->FrameID());

    // Get data + extent.
    if (!(*callOut)(0)) {
        return false;
    }
//    if (!(*callOut)(1)) {
//        return false;
//    }

    size_t gridResX = this->xResSlot.Param<param::IntParam>()->Value();
    size_t gridResY = this->yResSlot.Param<param::IntParam>()->Value();
    size_t gridResZ = this->zResSlot.Param<param::IntParam>()->Value();

    // Create particle list for output
    float orgX = callOut->AccessBoundingBoxes().ObjectSpaceBBox().Left();
    float orgY = callOut->AccessBoundingBoxes().ObjectSpaceBBox().Bottom();
    float orgZ = callOut->AccessBoundingBoxes().ObjectSpaceBBox().Back();
    float stepX = callOut->AccessBoundingBoxes().ObjectSpaceBBox().Width()/static_cast<float>(gridResX-1);
    float stepY = callOut->AccessBoundingBoxes().ObjectSpaceBBox().Height()/static_cast<float>(gridResY-1);
    float stepZ = callOut->AccessBoundingBoxes().ObjectSpaceBBox().Depth()/static_cast<float>(gridResZ-1);

    this->gridPos.SetCount(4*gridResX*gridResY*gridResZ);
    for (size_t x = 0; x < gridResX; ++x) {
        for (size_t y = 0; y < gridResY; ++y) {
            for (size_t z = 0; z < gridResZ; ++z) {
                this->gridPos[4*(gridResX*(gridResY*z+y)+x)+0] = orgX + x*stepX;
                this->gridPos[4*(gridResX*(gridResY*z+y)+x)+1] = orgY + y*stepY;
                this->gridPos[4*(gridResX*(gridResY*z+y)+x)+2] = orgZ + z*stepZ;
                this->gridPos[4*(gridResX*(gridResY*z+y)+x)+3] = 4.0; // TODO Param?
            }
        }
    }

    // Sample particles to a density grid
    if (!this->createVolumeCPU(*callOut)) {
        return false;
    }

    // Compute vector field
    this->dir.SetCount(4*gridResX*gridResY*gridResZ);
    for (size_t x = 0; x < gridResX; ++x) {
        for (size_t y = 0; y < gridResY; ++y) {
            for (size_t z = 0; z < gridResZ; ++z) {
                this->dir[3*(gridResX*(gridResY*z+y)+x)+0] = this->dens[gridResX*(gridResY*z+y)+x]; // Dummy data
                this->dir[3*(gridResX*(gridResY*z+y)+x)+1] = this->dens[gridResX*(gridResY*z+y)+x];
                this->dir[3*(gridResX*(gridResY*z+y)+x)+2] = this->dens[gridResX*(gridResY*z+y)+x];
            }
        }
    }

    // Put data to incoming call

    callIn->SetParticleListCount(1);
    callIn->AccessParticles(0).SetCount(this->gridPos.Count()/4);
    callIn->AccessParticles(0).SetVertexData(
                    MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR,
                    this->gridPos.PeekElements());
    callIn->AccessParticles(0).SetDirData(
                    DirectionalParticleDataCall::Particles::DIRDATA_FLOAT_XYZ,
                    this->dir.PeekElements());
    callIn->AccessParticles(0).SetGlobalColour(255, 0, 0);
    callIn->AccessParticles(0).SetColourData(MultiParticleDataCall::Particles::COLDATA_NONE, NULL, 0);

    return true;
}


/*
 *  moldyn::DynDensityGradientEstimator::createVolumeCPU
 */
bool moldyn::DynDensityGradientEstimator::createVolumeCPU(
        class megamol::core::moldyn::MultiParticleDataCall& c2) {

    int sx = this->xResSlot.Param<param::IntParam>()->Value();
    int sy = this->yResSlot.Param<param::IntParam>()->Value();
    int sz = this->zResSlot.Param<param::IntParam>()->Value();

//    size_t gridResX = this->xResSlot.Param<param::IntParam>()->Value();
//    size_t gridResY = this->yResSlot.Param<param::IntParam>()->Value();
//    size_t gridResZ = this->zResSlot.Param<param::IntParam>()->Value();

    float **vol = new float*[omp_get_max_threads()];
    int init, j;
#pragma omp parallel for
    for( init = 0; init < omp_get_max_threads(); init++ ) {
        vol[init] = new float[sx * sy * sz];
        ::memset(vol[init], 0, sizeof(float) * sx * sy * sz);
    }

    float minOSx = c2.AccessBoundingBoxes().ObjectSpaceBBox().Left();
    float minOSy = c2.AccessBoundingBoxes().ObjectSpaceBBox().Bottom();
    float minOSz = c2.AccessBoundingBoxes().ObjectSpaceBBox().Back();
    float rangeOSx = c2.AccessBoundingBoxes().ObjectSpaceBBox().Width();
    float rangeOSy = c2.AccessBoundingBoxes().ObjectSpaceBBox().Height();
    float rangeOSz = c2.AccessBoundingBoxes().ObjectSpaceBBox().Depth();

    //float volGenFac = this->aoGenFacSlot.Param<megamol::core::param::FloatParam>()->Value();
    float volGenFac = 1.0f;
//    float voxelVol = (rangeOSx / static_cast<float>(sx))
//        * (rangeOSy / static_cast<float>(sy))
//        * (rangeOSz / static_cast<float>(sz));
    float voxelVol = (rangeOSx / static_cast<float>(sx-1))
        * (rangeOSy / static_cast<float>(sy-1))
        * (rangeOSz / static_cast<float>(sz-1));

    for (unsigned int i = 0; i < c2.GetParticleListCount(); i++) {
        megamol::core::moldyn::MultiParticleDataCall::Particles &parts = c2.AccessParticles(i);
        const float *pos = static_cast<const float*>(parts.GetVertexData());
        unsigned int posStride = parts.GetVertexDataStride();
        float globRad = parts.GetGlobalRadius();
        bool useGlobRad = (parts.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ);
        if (parts.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE) {
            continue;
        }
        if (useGlobRad) {
            if (posStride < 12) posStride = 12;
        } else {
            if (posStride < 16) posStride = 16;
        }
        float globSpVol = 4.0f / 3.0f * static_cast<float>(M_PI) * globRad * globRad * globRad;

#pragma omp parallel for
        for (j = 0; j < parts.GetCount(); j++ ) {
            const float *ppos = reinterpret_cast<const float*>(reinterpret_cast<const char*>(pos) + posStride * j);
            int x = static_cast<int>(((ppos[0] - minOSx) / rangeOSx) * static_cast<float>(sx-1));
            if (x < 0) x = 0; else if (x >= sx) x = sx - 1;
            int y = static_cast<int>(((ppos[1] - minOSy) / rangeOSy) * static_cast<float>(sy-1));
            if (y < 0) y = 0; else if (y >= sy) y = sy - 1;
            int z = static_cast<int>(((ppos[2] - minOSz) / rangeOSz) * static_cast<float>(sz-1));
            if (z < 0) z = 0; else if (z >= sz) z = sz - 1;
            float spVol = globSpVol;
            if (!useGlobRad) {
                float rad = ppos[3];
                spVol = 4.0f / 3.0f * static_cast<float>(M_PI) * rad * rad * rad;
            }
            vol[omp_get_thread_num()][x + (y + z * sy) * sx] += (spVol / voxelVol) * volGenFac;
        }
    }

#pragma omp parallel for
    for (j = 0; j < sx * sy * sz; j++ ) {
        for ( unsigned int i = 1; i < omp_get_max_threads(); i++) {
            vol[0][j] += vol[i][j];
        }
    }

//    ::glEnable(GL_TEXTURE_3D);
//    ::glBindTexture(GL_TEXTURE_3D, this->volTex);
//    ::glTexSubImage3D(GL_TEXTURE_3D, 0, 1, 1, 1, sx, sy, sz, GL_LUMINANCE, GL_FLOAT, vol[0]);
//    ::glBindTexture(GL_TEXTURE_3D, 0);
//    ::glDisable(GL_TEXTURE_3D);
    if (this->dens) delete[] this->dens;
    this->dens = new float[sx*sy*sz];
    memcpy(this->dens, vol[0], sx*sy*sz*sizeof(float));

//    for (size_t z = 0; z < GRIDRES; ++z) {
//        printf("Z = %u\n", z);
//        for (size_t y = 0; y < GRIDRES; ++y) {
//            for (size_t x = 0; x < GRIDRES; ++x) {
//                printf("%f ", this->dens[GRIDRES*(GRIDRES*z+y)+x]);
//            }
//        }
//    }

    // Cleanup
#pragma omp parallel for
    for( init = 0; init < omp_get_max_threads(); init++ ) {
        delete[] vol[init];
    }
    delete[] vol;

    return true;
}
