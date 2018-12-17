/*
 *	ParticlesToMeshConverter.cpp
 *
 *	Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 *	All rights reserved
 */

#include "stdafx.h"

//#define _USE_MATH_DEFINES 1

#include "ParticlesToMeshConverter.h"
#include "mmcore/CoreInstance.h"
#include <gl/GLU.h>
#include "geometry_calls/CallTriMeshData.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/ButtonParam.h"

#include "vislib/StringTokeniser.h"

#include <channel_descriptor.h>
#include <driver_functions.h>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::moldyn;
using namespace megamol::protein_cuda;
using namespace megamol::protein_calls;
using namespace megamol::core::misc;

VISLIB_FORCEINLINE void copyCol_FLOAT_I(const void* source, float* dest) {
    float i = *reinterpret_cast<const float*>(source);
    float* d = reinterpret_cast<float*>(dest);
    *d = i;
    *(d + 1) = i;
    *(d + 2) = i;
    *(d + 3) = i;
}

VISLIB_FORCEINLINE void copyCol_FLOAT_RGB(const void* source, float* dest) {
    const float* s = reinterpret_cast<const float*>(source);
    float* d = reinterpret_cast<float*>(dest);
    *d = *s;
    *(d + 1) = *(s + 1);
    *(d + 2) = *(s + 2);
    *(d + 3) = 1.0f;
}

VISLIB_FORCEINLINE void copyCol_FLOAT_RGBA(const void* source, float* dest) {
    const float* s = reinterpret_cast<const float*>(source);
    float* d = reinterpret_cast<float*>(dest);
    *d = *s;
    *(d + 1) = *(s + 1);
    *(d + 2) = *(s + 2);
    *(d + 3) = *(s + 3);
}

VISLIB_FORCEINLINE void copyCol_UINT8_RGB(const void* source, float* dest) {
    const UINT8* s = reinterpret_cast<const UINT8*>(source);
    float* d = reinterpret_cast<float*>(dest);
    *d = static_cast<float>(*s) / 255.0f;
    *(d + 1) = static_cast<float>(*(s + 1)) / 255.0f;
    *(d + 2) = static_cast<float>(*(s + 2)) / 255.0f;
    *(d + 3) = 1.0f;
}

VISLIB_FORCEINLINE void copyCol_UINT8_RGBA(const void* source, float* dest) {
    const UINT8* s = reinterpret_cast<const UINT8*>(source);
    float* d = reinterpret_cast<float*>(dest);
    *d = static_cast<float>(*s) / 255.0f;
    *(d + 1) = static_cast<float>(*(s + 1)) / 255.0f;
    *(d + 2) = static_cast<float>(*(s + 2)) / 255.0f;
    *(d + 3) = static_cast<float>(*(s + 3)) / 255.0f;
}

typedef void(*copyColFunc)(const void*, float*);

/*
 *	ParticlesToMeshConverter::ParticlesToMeshConverter
 */
ParticlesToMeshConverter::ParticlesToMeshConverter(void) : Module(),
    particleDataSlot("getData", "Connects the converter with the particle data storage"),
    meshDataSlot("dataOut", "Connects the converter with the mesh consumer"),
    qualityParam("quicksurf::quality", "Quality"),
    radscaleParam("quicksurf::radscale", "Radius scale"),
    gridspacingParam("quicksurf::gridspacing", "Grid spacing"),
    isovalParam("quicksurf::isoval", "Isovalue"),
    scalingFactor("quicksurf::scalingFactor", "Scaling factor for the density values and particle radii"),
    concFactorParam("quicksurf::concentrationFactor", "Scaling factor for particle radii based on their concentration"),
    maxRadius("quicksurf::maxRadius", "The maximal particle influence radius the quicksurf algorithm uses"),
    setCUDAGLDevice(true),
    firstTransfer(true),
    recomputeVolume(true),
    particlesSize(0), lastMDCHash(0), lastMPDCHash(0), lastVDCHash(0), lastFrameID(0)
{
    this->particleDataSlot.SetCompatibleCall<MultiParticleDataCallDescription>();
    this->particleDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->particleDataSlot.SetCompatibleCall<VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->particleDataSlot);

    this->qualityParam.SetParameter(new param::IntParam(1, 0, 4));
    this->MakeSlotAvailable(&this->qualityParam);

    this->radscaleParam.SetParameter(new param::FloatParam(1.0f, 0.0f));
    this->MakeSlotAvailable(&this->radscaleParam);

    //this->gridspacingParam.SetParameter(new param::FloatParam(0.01953125f, 0.0f));
    this->gridspacingParam.SetParameter(new param::FloatParam(0.2f, 0.0f));
    this->MakeSlotAvailable(&this->gridspacingParam);

    this->isovalParam.SetParameter(new param::FloatParam(1.0f, 0.0f));
    this->MakeSlotAvailable(&this->isovalParam);

    this->scalingFactor.SetParameter(new param::FloatParam(1.0f, 0.0f));
    this->MakeSlotAvailable(&this->scalingFactor);

    this->concFactorParam.SetParameter(new param::FloatParam(0.5f, 0.0f));
    this->MakeSlotAvailable(&this->concFactorParam);

    this->maxRadius.SetParameter(new param::FloatParam(0.5, 0.0f));
    this->MakeSlotAvailable(&this->maxRadius);

    lastViewport.Set(0, 0);

    volumeExtent = make_cudaExtent(0, 0, 0);

    cudaqsurf = nullptr;
    volumeArray = nullptr;
    particles = nullptr;
    texHandle = 0;
    mcVertices_d = nullptr;
    mcNormals_d = nullptr;
    mcColors_d = nullptr;

    curTime = 0;
}

/*
 *	QuickSurfRaycaster::~QuickSurfRaycaster
 */
ParticlesToMeshConverter::~ParticlesToMeshConverter(void) {
    if (cudaqsurf) {
        CUDAQuickSurfAlternative *cqs = (CUDAQuickSurfAlternative *)cudaqsurf;
        delete cqs;
        cqs = nullptr;
        cudaqsurf = nullptr;
    }

    if (mcVertices_d != NULL) {
        cudaFree(mcVertices_d);
        mcVertices_d = NULL;
    }

    if (mcNormals_d != NULL) {
        cudaFree(mcNormals_d);
        mcNormals_d = NULL;
    }

    if (mcColors_d != NULL) {
        cudaFree(mcColors_d);
        mcColors_d = NULL;
    }
    
    this->Release();
}

/*
 *	QuickSurfRaycaster::release
 */
void ParticlesToMeshConverter::release(void) {

}

bool ParticlesToMeshConverter::calcVolume(float3 bbMin, float3 bbMax, float* positions, int quality, float radscale, float gridspacing,
    float isoval, float minConcentration, float maxConcentration, bool useCol, int timestep) {

    float x = bbMax.x - bbMin.x;
    float y = bbMax.y - bbMin.y;
    float z = bbMax.z - bbMin.z;

    int numVoxels[3];
    numVoxels[0] = (int)ceil(x / gridspacing);
    numVoxels[1] = (int)ceil(y / gridspacing);
    numVoxels[2] = (int)ceil(z / gridspacing);

    x = (numVoxels[0] - 1) * gridspacing;
    y = (numVoxels[1] - 1) * gridspacing;
    z = (numVoxels[2] - 1) * gridspacing;

    //printf("vox %i %i %i \n", numVoxels[0], numVoxels[1], numVoxels[2]);

    volumeExtent = make_cudaExtent(numVoxels[0], numVoxels[1], numVoxels[2]);
    volumeExtentSmall = make_cudaExtent(numVoxels[0], numVoxels[1], numVoxels[2]);

    float gausslim = 2.0f;
    switch (quality) { // TODO adjust values
    case 3: gausslim = 4.0f; break;
    case 2: gausslim = 3.0f; break;
    case 1: gausslim = 2.5f; break;
    case 0:
    default: gausslim = 2.0f; break;
    }

    float origin[3] = { bbMin.x, bbMin.y, bbMin.z };

    if (cudaqsurf == NULL) {
        cudaqsurf = new CUDAQuickSurfAlternative();
    }

    CUDAQuickSurfAlternative *cqs = (CUDAQuickSurfAlternative*)cudaqsurf;

    int result = -1;
    result = cqs->calc_map((long)particleCnt, positions, colorTable.data(), 1, 
        origin, numVoxels, maxConcentration, radscale, gridspacing, 
        isoval, gausslim, false, timestep, 20);

    checkCudaErrors(cudaDeviceSynchronize());

    volumeExtent = make_cudaExtent(cqs->getMapSizeX(), cqs->getMapSizeY(), cqs->getMapSizeZ());

    // make the initial plane more beautiful
    if (volumeExtent.depth > volumeExtentSmall.depth)
        volumeExtentSmall.depth = volumeExtentSmall.depth + 1;

    return (result == 0);
}

/*
 *	QuickSurfRaycaster::create
 */
bool ParticlesToMeshConverter::create(void) {
    return true; // initOpenGL();
}

/*
 *	QuickSurfRaycaster::convertToMesh
 */
void ParticlesToMeshConverter::makeMesh(float * volumeData, cudaExtent volSize, float3 bbMin, float3 bbMax, float isoValue, float concMin, float concMax) {

	uint3 extents = make_uint3(static_cast<unsigned int>(volSize.width), static_cast<unsigned int>(volSize.height), static_cast<unsigned int>(volSize.depth));
    unsigned int chunkmaxverts = 3 * extents.x * extents.y * extents.z;

    float * data = volumeData;

#define DOWNSAMPLE
#ifdef DOWNSAMPLE
    extents = make_uint3(extents.x / 2, extents.y / 2, extents.z / 2);
    chunkmaxverts = 12 * extents.x * extents.y * extents.z; // TODO was: 3* (which seemed weird)

    data = new float[extents.x * extents.y * extents.z];

    float newMin = FLT_MAX;
    float newMax = FLT_MIN;

	for (int index = 0; index < static_cast<int>(chunkmaxverts / 3); index++) {
        int i = (index % (extents.x * extents.y)) % extents.x;
        int j = (index % (extents.x * extents.y)) / extents.x;
        int k = index / (extents.x * extents.y);

        float val = 0.0f;
        val += volumeData[(2*k)   * extents.x * extents.y + (2*j)   * extents.x + (2*i)];
        val += volumeData[(2*k)   * extents.x * extents.y + (2*j)   * extents.x + (2*i+1)];
        val += volumeData[(2*k)   * extents.x * extents.y + (2*j+1) * extents.x + (2*i)];
        val += volumeData[(2*k)   * extents.x * extents.y + (2*j+1) * extents.x + (2*i+1)];
        val += volumeData[(2*k+1) * extents.x * extents.y + (2*j)   * extents.x + (2*i)];
        val += volumeData[(2*k+1) * extents.x * extents.y + (2*j)   * extents.x + (2*i+1)];
        val += volumeData[(2*k+1) * extents.x * extents.y + (2*j+1) * extents.x + (2*i)];
        val += volumeData[(2*k+1) * extents.x * extents.y + (2*j+1) * extents.x + (2*i+1)];
        data[k * extents.x * extents.y + j * extents.x + i] = val / 8.0f;

        if (val / 8.0f < newMin) newMin = val / 8.0f;
        if (val / 8.0f > newMax) newMax = val / 8.0f;
    }
#else
    float newMin = concMin;
    float newMax = concMax;
#endif

#define NORMALIZE
#ifdef NORMALIZE
	for (int i = 0; i < static_cast<int>(chunkmaxverts / 3); i++) {
        data[i] = (data[i] - newMin) / (newMax - newMin);
    }
#endif

    if (cudaMarching == NULL) {
        cudaMarching = new CUDAMarchingCubes();
        ((CUDAMarchingCubes*)cudaMarching)->Initialize(extents);
    }

    CUDAMarchingCubes * cmc = (CUDAMarchingCubes*)cudaMarching;
    uint3 oldExtents = cmc->GetMaxGridSize();
    
    if (extents.x > oldExtents.x || extents.y > oldExtents.y || extents.z > oldExtents.z) {
        cmc->Initialize(extents);
    }

    float3 bbSize = make_float3(bbMax.x - bbMin.x, bbMax.y - bbMin.y, bbMax.z - bbMin.z);
    float3 bbNewMin = make_float3(0.0f, 0.0f, 0.0f);

    //std::cout << chunkmaxverts * sizeof(float3) << std::endl;
    printf("min %f ; max %f\n", newMin, newMax);

    float myIsoVal = isoValue;
    //myIsoVal = 0.2f;

    // TODO is the linear interpolation necessary? (use only isoValue instead)
    //cmc->SetIsovalue((1.0f - myIsoVal) * newMin + myIsoVal * newMax);
    cmc->SetIsovalue(myIsoVal);
    if (!cmc->SetVolumeData(data, NULL, extents, bbMin, bbSize, false)) {
        printf("SetVolumeData failed!\n");
    }

    if (mcVertices_d != NULL) {
        cudaFree(mcVertices_d);
        mcVertices_d = NULL;
    }

    if (mcNormals_d != NULL) {
        cudaFree(mcNormals_d);
        mcNormals_d = NULL;
    }

    if (mcColors_d != NULL) {
        cudaFree(mcColors_d);
        mcColors_d = NULL;
    }

    checkCudaErrors(cudaMalloc((void**)&mcVertices_d, chunkmaxverts * sizeof(float3)));
    checkCudaErrors(cudaMalloc((void**)&mcNormals_d, chunkmaxverts * sizeof(float3)));
    checkCudaErrors(cudaMalloc((void**)&mcColors_d, chunkmaxverts * sizeof(float3)));

    cmc->computeIsosurface(mcVertices_d, mcNormals_d, mcColors_d, chunkmaxverts);
    checkCudaErrors(cudaDeviceSynchronize());

    int64_t count = cmc->GetVertexCount();

    float *mcVertices_h = new float[count * 3 * sizeof(float)];
    float *mcNormals_h = new float[count * 3 * sizeof(float)];

    checkCudaErrors(cudaMemcpy(mcVertices_h, mcVertices_d, count * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(mcNormals_h, mcNormals_d, count * 3 * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Found " << count / 3 << " triangles" << std::endl;

    std::ofstream file("mesh.obj");
    for (int i = 0; i < count; i++) { // vertices
        file << "v " << std::to_string(mcVertices_h[i * 3 + 0]) << " " << std::to_string(mcVertices_h[i * 3 + 1]) << " " << std::to_string(mcVertices_h[i * 3 + 2]) << std::endl;
    }
    file << std::endl;
    for (int i = 0; i < count; i++) { // normals
        file << "vn " << std::to_string(mcNormals_h[i * 3 + 0]) << " " << std::to_string(mcNormals_h[i * 3 + 1]) << " " << std::to_string(mcNormals_h[i * 3 + 2]) << std::endl;
    }
    file << std::endl;
    for (int i = 0; i < count / 3; i++) { // triangles
        std::string i1 = std::to_string(i * 3 + 1);
        std::string i2 = std::to_string(i * 3 + 2);
        std::string i3 = std::to_string(i * 3 + 3);

        file << "f " << i1 << "//" << i1 << " " << i2 << "//" << i2 << " " << i3 << "//" << i3 << std::endl;
    }

    file.close();

    delete[] mcVertices_h;
    mcVertices_h = nullptr;
    delete[] mcNormals_h;
    mcNormals_h = nullptr;

#ifdef DOWNSAMPLE
    delete[] data;
    data = nullptr;
#endif
}

/*
 *	ParticlesToMeshConverter::GetExtents
 */
bool ParticlesToMeshConverter::GetExtents(Call& call) {
    view::AbstractCallRender3D *cr3d = dynamic_cast<view::AbstractCallRender3D *>(&call);
    if (cr3d == NULL) return false;

    MultiParticleDataCall *mpdc = this->particleDataSlot.CallAs<MultiParticleDataCall>();
    MolecularDataCall *mdc = this->particleDataSlot.CallAs<MolecularDataCall>();
    VolumetricDataCall *vdc = this->particleDataSlot.CallAs<VolumetricDataCall>();

    if (mpdc == NULL && mdc == NULL && vdc == NULL) return false;

    // MultiParticleDataCall in use
    if (mpdc != NULL) {
        if (!(*mpdc)(1)) return false;

        float scale;
        if (!vislib::math::IsEqual(mpdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
            scale = 2.0f / mpdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        }
        else {
            scale = 1.0f;
        }
        cr3d->AccessBoundingBoxes() = mpdc->AccessBoundingBoxes();
        cr3d->AccessBoundingBoxes().MakeScaledWorld(scale);
        cr3d->SetTimeFramesCount(mpdc->FrameCount());
    } // MolecularDataCall in use
    else if (mdc != NULL) {
        if (!(*mdc)(1)) return false;

        float scale;
        if (!vislib::math::IsEqual(mdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
            scale = 2.0f / mdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        }
        else {
            scale = 1.0f;
        }
        cr3d->AccessBoundingBoxes() = mdc->AccessBoundingBoxes();
        cr3d->AccessBoundingBoxes().MakeScaledWorld(scale);
        cr3d->SetTimeFramesCount(mdc->FrameCount());
    }
    else if (vdc != NULL) {
        if (!(*vdc)(vdc->IDX_GET_EXTENTS)) return false;

        float scale;
        if (!vislib::math::IsEqual(vdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
            scale = 2.0f / vdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        }
        else {
            scale = 1.0f;
        }

        cr3d->AccessBoundingBoxes() = vdc->AccessBoundingBoxes();
        cr3d->AccessBoundingBoxes().MakeScaledWorld(scale);
        cr3d->SetTimeFramesCount(vdc->FrameCount());
    }

    return true;
}


/*
 *	QuickSurfRaycaster::GetData
 */
bool ParticlesToMeshConverter::GetData(Call& call) {
    geocalls::CallTriMeshData *cmesh = dynamic_cast<geocalls::CallTriMeshData * > (&call);
    if (cmesh == NULL) return false;

    if (concFactorParam.IsDirty() || gridspacingParam.IsDirty() 
        || isovalParam.IsDirty() || qualityParam.IsDirty() 
        || radscaleParam.IsDirty() || scalingFactor.IsDirty()) {

        concFactorParam.ResetDirty();
        gridspacingParam.ResetDirty();
        isovalParam.ResetDirty(); 
        qualityParam.ResetDirty();
        radscaleParam.ResetDirty();
        scalingFactor.ResetDirty();

        recomputeVolume = true;
    }

    if (cmesh->FrameID() != lastFrameID) {
        recomputeVolume = true;
    }

    MultiParticleDataCall * mpdc = particleDataSlot.CallAs<MultiParticleDataCall>();
    MolecularDataCall * mdc = particleDataSlot.CallAs<MolecularDataCall>();
    VolumetricDataCall * vdc = particleDataSlot.CallAs<VolumetricDataCall>();

    float3 bbMin;
    float3 bbMax;

    float3 clipBoxMin;
    float3 clipBoxMax;

    bool onlyVolumetric = false;

    if (mpdc == NULL && mdc == NULL && vdc == NULL) return false;

    if (mpdc != NULL && mpdc->DataHash() != lastMPDCHash) {

        mpdc->SetFrameID(cmesh->FrameID());
        if (!(*mpdc)(1)) return false;
        if (!(*mpdc)(0)) return false;

        auto bb = mpdc->GetBoundingBoxes().ObjectSpaceBBox();
        bbMin = make_float3(bb.Left(), bb.Bottom(), bb.Back());
        bbMax = make_float3(bb.Right(), bb.Top(), bb.Front());
        bb = mpdc->GetBoundingBoxes().ClipBox();
        clipBoxMin = make_float3(bb.Left(), bb.Bottom(), bb.Back());
        clipBoxMax = make_float3(bb.Right(), bb.Top(), bb.Front());

        if (recomputeVolume) {

            numParticles = 0;
            for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
                numParticles += mpdc->AccessParticles(i).GetCount();
            }

            if (numParticles == 0) {
                return true;
            }

            if (this->particlesSize < this->numParticles * 4) {
                if (this->particles) {
                    delete[] this->particles;
                    this->particles = nullptr;
                }
                this->particles = new float[this->numParticles * 4];
                this->particlesSize = this->numParticles * 4;
            }
            memset(this->particles, 0, this->numParticles * 4 * sizeof(float));

            particleCnt = 0;
            this->colorTable.clear();
            this->colorTable.resize(numParticles * 4, 0.0f);

            //printf("bbMin %f %f %f\n", bbMin.x, bbMin.y, bbMin.z);
            //exit(-1);

            particleCnt = 0;
            copyColFunc copier = nullptr;

            for (unsigned int i = 0; i < mpdc->GetParticleListCount(); i++) {
                MultiParticleDataCall::Particles &parts = mpdc->AccessParticles(i);
                const float *pos = static_cast<const float*>(parts.GetVertexData());
                const float *colorPos = static_cast<const float*>(parts.GetColourData());
                unsigned int posStride = parts.GetVertexDataStride();
                unsigned int colStride = parts.GetColourDataStride();
                float globalRadius = parts.GetGlobalRadius();
                bool useGlobRad = (parts.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ);
                int numColors = 0;

                switch (parts.GetColourDataType()) {
                case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: numColors = 1; copier = copyCol_FLOAT_I; break;
                case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB: numColors = 3; copier = copyCol_FLOAT_RGB; break;
                case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA: numColors = 4; copier = copyCol_FLOAT_RGBA; break;
                case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB: numColors = 0; copier = copyCol_UINT8_RGB; break; // TODO
                case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA: numColors = 0; copier = copyCol_UINT8_RGBA; break; // TODO
                }

                // if the vertices have no type, take the next list
                if (parts.GetVertexDataType() == megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_NONE) {
                    continue;
                }
                if (useGlobRad) { // TODO is this correct?
                    if (posStride < 12) posStride = 12;
                }
                else {
                    if (posStride < 16) posStride = 16;
                }

                for (UINT64 j = 0; j < parts.GetCount(); j++, pos = reinterpret_cast<const float*>(reinterpret_cast<const char*>(pos)+posStride),
                    colorPos = reinterpret_cast<const float*>(reinterpret_cast<const char*>(colorPos)+colStride)) {
                        particles[particleCnt * 4 + 0] = pos[0] - bbMin.x;
                        particles[particleCnt * 4 + 1] = pos[1] - bbMin.y;
                        particles[particleCnt * 4 + 2] = pos[2] - bbMin.z;
                        if (useGlobRad) {
                            particles[particleCnt * 4 + 3] = globalRadius;
                        }
                        else {
                            particles[particleCnt * 4 + 3] = pos[3];
                        }

                        /*---------------------------------choose-one---------------------------------------------------------*/

                        copier(colorPos, &this->colorTable[particleCnt * 4]);
//#define ALWAYS4COLORS
//#ifndef ALWAYS4COLORS
//						// 1. copy all available values into the color, the rest gets filled up with the last available value
//						for (int k = 0; k < numColors; k++) {
//							for (int l = 0; l < 3 - k; l++) {
//								this->colorTable[particleCnt * 4 + k + l] = colorPos[k];
//							}
//						}
//#else
//						for (int k = 0; k < 4; k++) {
//							this->colorTable[particleCnt * 4 + k] = colorPos[k];
//						}
//#endif
//
//						// normalization of the values happens later
//						this->colorTable[particleCnt * 4 + 3] = colorPos[numColors - 1];
//
//						// 2. fill r,g & b with the last available color value (should be density)
//						/*for (int k = 0; k < 3; k++) {
//							this->colorTable[particleCnt * 4 + k] = colorPos[numColors - 1];
//							}*/
//
//						/*---------------------------------------------------------------------------------------------------*/

                        particleCnt++;
                }
            }

            if (particleCnt == 0) {
                return true;
            }

        }

        //glPushMatrix();
        //float scale = 1.0f;
        //if (!vislib::math::IsEqual(mpdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
        //	scale = 2.0f / mpdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        //}
        //glScalef(scale, scale, scale);

        mpdc->Unlock();

        onlyVolumetric = false;
        lastMPDCHash = mpdc->DataHash();

    } else if (mdc != NULL && mdc->DataHash() != lastMDCHash) {
        // TODO
        printf("MolecularDataCall currently not supported\n");
        mdc->Unlock();
        onlyVolumetric = false;
        //lastMDCHash = mdc->DataHash();
        return false;
    } else if (vdc != NULL && vdc->DataHash() != lastVDCHash) {

        vdc->SetFrameID(cmesh->FrameID());
        if (!(*vdc)(vdc->IDX_GET_EXTENTS)) return false;
        if (!(*vdc)(vdc->IDX_GET_METADATA)) return false;
        if (!(*vdc)(vdc->IDX_GET_DATA)) return false;
        
        //glPushMatrix();
        //float scale = 1.0f;
        //if (!vislib::math::IsEqual(vdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
        //	scale = 2.0f / vdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        //}
        //glScalef(scale, scale, scale);
        onlyVolumetric = true;
    }

    float factor = scalingFactor.Param<param::FloatParam>()->Value();

    float density = 0.5f;
    float brightness = 1.0f;
    float transferOffset = 0.0f;
    float transferScale = 1.0f;

    /*printf("min: %f %f %f \n", clipBoxMin.x, clipBoxMin.y, clipBoxMin.z);
    printf("max: %f %f %f \n\n", clipBoxMax.x, clipBoxMax.y, clipBoxMax.z);*/

    //dim3 blockSize = dim3(8, 8);
    //dim3 gridSize = dim3(iDivUp(viewport.GetWidth(), blockSize.x), iDivUp(viewport.GetHeight(), blockSize.y));

    if (recomputeVolume && !onlyVolumetric) {
        if (cudaqsurf == NULL) {
            cudaqsurf = new CUDAQuickSurfAlternative();
        }

        bool suc = this->calcVolume(bbMin, bbMax, particles,
            this->qualityParam.Param<param::IntParam>()->Value(),
            this->radscaleParam.Param<param::FloatParam>()->Value(),
            this->gridspacingParam.Param<param::FloatParam>()->Value(),
            1.0f, // necessary to switch off velocity scaling
            0.0f, this->maxRadius.Param<param::FloatParam>()->Value(), true,
            cmesh->FrameID());
    
        //if (!suc) return false;
    }

    if (recomputeVolume && !onlyVolumetric) {
        CUDAQuickSurfAlternative * cqs = (CUDAQuickSurfAlternative*)cudaqsurf;
        map = cqs->getMap();
    }

    // TODO: now use map (device) OR vdc->GetData() (host) as marching cubes stuff!

        //transferNewVolume(map, volumeExtent);

#if 0
    if (!onlyVolumetric) { // quicksurfed data
        transferNewVolume(map, volumeExtentSmall);
    }
    else { // pure volume data
        auto xDir = vdc->GetResolution(0);
        auto yDir = vdc->GetResolution(1);
        auto zDir = vdc->GetResolution(2);
        volumeExtentSmall = make_cudaExtent(xDir, yDir, zDir);
        volumeExtent = volumeExtentSmall;

        auto bb = vdc->GetBoundingBoxes().ObjectSpaceBBox();
        bbMin = make_float3(bb.Left(), bb.Bottom(), bb.Back());
        bbMax = make_float3(bb.Right(), bb.Top(), bb.Front());

        if (vdc->GetComponents() > 1 || vdc->GetComponents() < 1) {
            return false;
        }

        if (vdc->GetScalarType() != VolumetricDataCall::ScalarType::FLOATING_POINT) {
            return false;
        }

        auto volPtr = vdc->GetData();
        auto voxelNumber = volumeExtentSmall.width * volumeExtentSmall.height * volumeExtentSmall.depth;
        float * fPtr = reinterpret_cast<float*>(volPtr);

        if (volPtr != NULL){
            transferVolumeDirect(volPtr, volumeExtentSmall, concMin, concMax);
            checkCudaErrors(cudaDeviceSynchronize());
        }
        else {
            printf("Volume data was NULL");
            return false;
        }

            printf("Converting the isosurface of %f to a mesh\n", this->convertedIsoValueParam.Param<param::FloatParam>()->Value());
            makeMesh(static_cast<float*>(volPtr), volumeExtentSmall, bbMin, bbMax, this->convertedIsoValueParam.Param<param::FloatParam>()->Value(), concMin, concMax);
            this->triggerConvertButtonParam.ResetDirty();
    }
#endif
    checkCudaErrors(cudaDeviceSynchronize());

    if (onlyVolumetric)
        vdc->Unlock();

    recomputeVolume = false;

    lastVDCHash = vdc->DataHash();
    return true;
}