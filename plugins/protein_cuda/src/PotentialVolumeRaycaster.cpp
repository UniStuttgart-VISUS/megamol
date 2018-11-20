//
// PotentialVolumeRaycaster.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 16, 2013
//     Author: scharnkn
//
// TODO
// + z test is buggy
// + texture coordinates on view aligned slice are incorrect
// + Set max frame to min fram


#include "stdafx.h"
#include "PotentialVolumeRaycaster.h"

#include "protein_calls/VTIDataCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "CUDAQuickSurf.h"
#include "slicing.h"
#include "cuda_error_check.h"
#include "ogl_error_check.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/view/CallRender3D.h"

#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/math/Matrix.h"

#include "vislib/graphics/gl/IncludeAllGL.h"
#include <GL/glu.h>

#include <cmath>
#include <algorithm>


using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein_calls;
using namespace megamol::protein_cuda;


/*
 * megamol::protein_cuda::PotentialVolumeRaycaster::PotentialVolumeRaycaster
 */
PotentialVolumeRaycaster::PotentialVolumeRaycaster(void):
                Renderer3DModuleDS(),
                /* Caller slots */
                potentialDataCallerSlot("getVolData", "Connects the rendering with data storage"),
                particleDataCallerSlot("getParticleData", "Connects the rendering with data storage"),
                rendererCallerSlot("protren", "Connects the renderer with another render module"),
                /* Parameters for volume rendering */
                colorMinPotentialSclSlot("colorMinPotentialScl", "Minimum potential value for the color map"),
                colorMaxPotentialSclSlot("colorMaxPotentialScl", "Maximum potential value for the color map"),
                colorMinPotentialSlot("colorMinPotential", "Color for the minimum potential value"),
                colorZeroPotentialSlot("colorZeroPotential", "Color for zero potential"),
                colorMaxPotentialSlot("colorMaxPotential", "Color for the maximum potential value"),
                volIsoValSlot("volIsoVal", "Iso value for volume rendering"),
                volDeltaSlot("volDelta", "Delta for ray casting"),
                volAlphaSclSlot("volAlphaScl", "Alpha scale factor for iso surface"),
                volClipZSlot("volClipZ", "Parameter for the distance of the volume clipping plane"),
                volMaxItSlot("volMaxIt", "Parameter for the maximum number of steps when raycasting"),
                gradOffsSlot("gradOffs", "Parameter for the gradient offset"),
                /* Parameters for the 'quicksurf' class */
                qsParticleRadSlot("qsParticleRad", "Assumed radius of density grid data"),
                qsGaussLimSlot("qsGaussLim", "The cutoff radius for the gaussian kernel"),
                qsGridSpacingSlot("qsGridSpacing", "Spacing for the density grid"),
                qsSclVanDerWaalsSlot("qsSclVanDerWaals", "Toggle scaling of the density radius by van der Waals radius"),
                /* Parameters for slice rendering */
                sliceRMSlot("texslice::render", "The rendering mode for the slices"),
                xPlaneSlot("texslice::xPlanePos", "Change the position of the x-Plane"),
                yPlaneSlot("texslice::yPlanePos", "Change the position of the y-Plane"),
                zPlaneSlot("texslice::zPlanePos", "Change the position of the z-Plane"),
                toggleXPlaneSlot("texslice::showXPlane", "Change the position of the x-Plane"),
                toggleYPlaneSlot("texslice::showYPlane", "Change the position of the y-Plane"),
                toggleZPlaneSlot("texslice::showZPlane", "Change the position of the z-Plane"),
                sliceColorMinPotentialSclSlot("texslice::colorMinPotentialScl", "Minimum potential value for the color map"),
                sliceColorMidPotentialSclSlot("texslice::colorMidPotentialScl", "Minimum potential value for the color map"),
                sliceColorMaxPotentialSclSlot("texslice::colorMaxPotentialScl", "Maximum potential value for the color map"),
                sliceColorMinPotentialSlot("texslice::colorMinPotential", "Color for the minimum potential value"),
                sliceColorZeroPotentialSlot("texslice::colorZeroPotential", "Color for zero potential"),
                sliceColorMaxPotentialSlot("texslice::colorMaxPotential", "Color for the maximum potential value"),
                sliceMinValSlot("texslice::minTex", "Minimum texture value"),
                sliceMaxValSlot("texslice::maxTex", "Maximum texture value"),
                /* Raycasting */
                fboDim(-1, -1), volumeTex(0),
                potentialTex(0), volume(NULL),
                /* Volume generation */
                cudaqsurf(NULL),
                /* The data */
                datahashParticles(0), datahashPotential(0),
                /* Boolean flags */
                computeVolume(true), initPotentialTex(true), frameOld(-1) {


    // Data caller slota for the potential maps
    this->potentialDataCallerSlot.SetCompatibleCall<VTIDataCallDescription>();
    this->MakeSlotAvailable(&this->potentialDataCallerSlot);
    // Data caller slot for the particles
    this->particleDataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->particleDataCallerSlot);

    // Renderer caller slot
    this->rendererCallerSlot.SetCompatibleCall<view::CallRender3DDescription>();
    this->MakeSlotAvailable(&this->rendererCallerSlot);


    /* Parameters for the GPU volume rendering */

    // Parameter for minimum potential value for the color map
    this->colorMinPotentialScl = -1.0f;
    this->colorMinPotentialSclSlot.SetParameter(new core::param::FloatParam(this->colorMinPotentialScl));
    this->MakeSlotAvailable(&this->colorMinPotentialSclSlot);

    // Parameter for maximum potential value for the color map
    this->colorMaxPotentialScl = 1.0f;
    this->colorMaxPotentialSclSlot.SetParameter(new core::param::FloatParam(this->colorMaxPotentialScl));
    this->MakeSlotAvailable(&this->colorMaxPotentialSclSlot);

    // Parameter for minimum potential value for the color map
    this->colorMinPotential.Set(0.75f, 0.01f, 0.15f);
    this->colorMinPotentialSlot.SetParameter(new core::param::StringParam(
            utility::ColourParser::ToString(this->colorMinPotential.X(),
                    this->colorMinPotential.Y(), this->colorMinPotential.Z())));
    this->MakeSlotAvailable(&this->colorMinPotentialSlot);

    // Parameter for zero potential for the color map
    this->colorZeroPotential.Set(1.0f, 1.0f, 1.0f);
    this->colorZeroPotentialSlot.SetParameter(new core::param::StringParam(
            utility::ColourParser::ToString(this->colorZeroPotential.X(),
                    this->colorZeroPotential.Y(), this->colorZeroPotential.Z())));
    this->MakeSlotAvailable(&this->colorZeroPotentialSlot);

    // Parameter for maximum potential value for the color map
    this->colorMaxPotential.Set(0.23f, 0.29f, 0.75f);
    this->colorMaxPotentialSlot.SetParameter(new core::param::StringParam(
            utility::ColourParser::ToString(this->colorMaxPotential.X(),
                    this->colorMaxPotential.Y(), this->colorMaxPotential.Z())));
    this->MakeSlotAvailable(&this->colorMaxPotentialSlot);

    // Parameter for iso value for volume rendering
    this->volIsoVal = 0.5f;
    this->volIsoValSlot.SetParameter(new core::param::FloatParam(this->volIsoVal, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->volIsoValSlot);

    // Parameters for delta for volume rendering
    this->volDelta = 0.5f;
    this->volDeltaSlot.SetParameter(new core::param::FloatParam(this->volDelta, 0.0f));
    this->MakeSlotAvailable(&this->volDeltaSlot);

    // Parameter for alpha scale factor for the isosurface
    this->volAlphaScl = 0.5f;
    this->volAlphaSclSlot.SetParameter(new core::param::FloatParam(this->volAlphaScl, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->volAlphaSclSlot);

    // Parameter for the distance of the volume clipping plane
    this->volClipZ = 0.0f;
    this->volClipZSlot.SetParameter(new core::param::FloatParam(this->volClipZ, 0.0f, 100.0f));
    this->MakeSlotAvailable(&this->volClipZSlot);

    // Parameter for the maximum number of iterations when raycasting
    this->volMaxIt = 10000;
    this->volMaxItSlot.SetParameter(new core::param::IntParam(static_cast<int>(this->volMaxIt), 0));
    this->MakeSlotAvailable(&this->volMaxItSlot);

    // Parameter for the gradient offset
    this->gradOffs = 0.01f;
    this->gradOffsSlot.SetParameter(new core::param::FloatParam(this->gradOffs, 0.01f));
    this->MakeSlotAvailable(&this->gradOffsSlot);


    /* Parameters for the 'quicksurf' class */

    // Parameter for assumed radius of density grid data
    this->qsParticleRad = 1.0f;
    this->qsParticleRadSlot.SetParameter(new core::param::FloatParam(this->qsParticleRad, 0.0f));
    this->MakeSlotAvailable(&this->qsParticleRadSlot);

    // Parameter for the cutoff radius for the gaussian kernel
    this->qsGaussLim = 1.0f;
    this->qsGaussLimSlot.SetParameter(new core::param::FloatParam(this->qsGaussLim, 0.0f));
    this->MakeSlotAvailable(&this->qsGaussLimSlot);

    // Parameter for assumed radius of density grid data
    this->qsGridSpacing = 1.0f;
    this->qsGridSpacingSlot.SetParameter(new core::param::FloatParam(this->qsGridSpacing, 0.1f));
    this->MakeSlotAvailable(&this->qsGridSpacingSlot);

    // Parameter to toggle scaling by van der Waals radius
    this->qsSclVanDerWaals = true;
    this->qsSclVanDerWaalsSlot.SetParameter(new core::param::BoolParam(this->qsSclVanDerWaals));
    this->MakeSlotAvailable(&this->qsSclVanDerWaalsSlot);


    /* Parameters for slice rendering */

    // Render modes for slices
    this->sliceRM = 0;
    param::EnumParam *srm = new core::param::EnumParam(this->sliceRM);
    srm->SetTypePair(0, "Potential");
    srm->SetTypePair(1, "Density");
    this->sliceRMSlot << srm;
    this->MakeSlotAvailable(&this->sliceRMSlot);

    // X-plane position
    this->xPlane = 0.0f;
    this->xPlaneSlot.SetParameter(new core::param::FloatParam(this->xPlane, -120.0f, 120.0f));
    this->MakeSlotAvailable(&this->xPlaneSlot);

    // X-plane visibility
    this->showXPlane = false;
    this->toggleXPlaneSlot.SetParameter(new core::param::BoolParam(this->showXPlane));
    this->MakeSlotAvailable(&this->toggleXPlaneSlot);

    // Y-plane position
    this->yPlane = 0.0f;
    this->yPlaneSlot.SetParameter(new core::param::FloatParam(this->yPlane, -120.0f, 120.0f));
    this->MakeSlotAvailable(&this->yPlaneSlot);

    // Y-plane visibility
    this->showYPlane = false;
    this->toggleYPlaneSlot.SetParameter(new core::param::BoolParam(this->showYPlane));
    this->MakeSlotAvailable(&this->toggleYPlaneSlot);

    // Z-plane position
    this->zPlane = 0.0f;
    this->zPlaneSlot.SetParameter(new core::param::FloatParam(this->zPlane, -120.0f, 120.0f));
    this->MakeSlotAvailable(&this->zPlaneSlot);

    // Z-plane visibility
    this->showZPlane = false;
    this->toggleZPlaneSlot.SetParameter(new core::param::BoolParam(this->showZPlane));
    this->MakeSlotAvailable(&this->toggleZPlaneSlot);

    // Parameter for minimum potential value for the color map
    this->sliceColorMinPotentialScl = -1.0f;
    this->sliceColorMinPotentialSclSlot.SetParameter(new core::param::FloatParam(this->sliceColorMinPotentialScl));
    this->MakeSlotAvailable(&this->sliceColorMinPotentialSclSlot);

    // Parameter for maximum potential value for the color map
    this->sliceColorMidPotentialScl = 0.0f;
    this->sliceColorMidPotentialSclSlot.SetParameter(new core::param::FloatParam(this->sliceColorMidPotentialScl));
    this->MakeSlotAvailable(&this->sliceColorMidPotentialSclSlot);

    // Parameter for maximum potential value for the color map
    this->sliceColorMaxPotentialScl = 1.0f;
    this->sliceColorMaxPotentialSclSlot.SetParameter(new core::param::FloatParam(this->sliceColorMaxPotentialScl));
    this->MakeSlotAvailable(&this->sliceColorMaxPotentialSclSlot);

    // Parameter for minimum potential value for the color map
    this->sliceColorMinPotential.Set(0.75f, 0.01f, 0.15f);
    this->sliceColorMinPotentialSlot.SetParameter(new core::param::StringParam(
            utility::ColourParser::ToString(this->sliceColorMinPotential.X(),
                    this->sliceColorMinPotential.Y(), this->sliceColorMinPotential.Z())));
    this->MakeSlotAvailable(&this->sliceColorMinPotentialSlot);

    // Parameter for zero potential for the color map
    this->sliceColorZeroPotential.Set(1.0f, 1.0f, 1.0f);
    this->sliceColorZeroPotentialSlot.SetParameter(new core::param::StringParam(
            utility::ColourParser::ToString(this->sliceColorZeroPotential.X(),
                    this->sliceColorZeroPotential.Y(), this->sliceColorZeroPotential.Z())));
    this->MakeSlotAvailable(&this->sliceColorZeroPotentialSlot);

    // Parameter for maximum potential value for the color map
    this->sliceColorMaxPotential.Set(0.23f, 0.29f, 0.75f);
    this->sliceColorMaxPotentialSlot.SetParameter(new core::param::StringParam(
            utility::ColourParser::ToString(this->sliceColorMaxPotential.X(),
                    this->sliceColorMaxPotential.Y(), this->sliceColorMaxPotential.Z())));
    this->MakeSlotAvailable(&this->sliceColorMaxPotentialSlot);

    // Minimum texture value
    this->sliceMinVal = -1.0f;
    this->sliceMinValSlot.SetParameter(new core::param::FloatParam(this->sliceMinVal));
    this->MakeSlotAvailable(&this->sliceMinValSlot);

    // Maximum texture value
    this->sliceMaxVal = 1.0f;
    this->sliceMaxValSlot.SetParameter(new core::param::FloatParam(this->sliceMaxVal));
    this->MakeSlotAvailable(&this->sliceMaxValSlot);


    /* Initialize grid parameters */

    this->gridDensMap.minC[0] = -1.0f;
    this->gridDensMap.minC[1] = -1.0f;
    this->gridDensMap.minC[2] = -1.0f;
    this->gridDensMap.maxC[0] = 1.0f;
    this->gridDensMap.maxC[1] = 1.0f;
    this->gridDensMap.maxC[2] = 1.0f;
    this->gridDensMap.delta[0] = 1.0f;
    this->gridDensMap.delta[1] = 1.0f;
    this->gridDensMap.delta[2] = 1.0f;
    this->gridDensMap.size[0] = 2;
    this->gridDensMap.size[1] = 2;
    this->gridDensMap.size[2] = 2;

    this->gridPotentialMap.minC[0] = -1.0f;
    this->gridPotentialMap.minC[1] = -1.0f;
    this->gridPotentialMap.minC[2] = -1.0f;
    this->gridPotentialMap.maxC[0] = 1.0f;
    this->gridPotentialMap.maxC[1] = 1.0f;
    this->gridPotentialMap.maxC[2] = 1.0f;
    this->gridPotentialMap.delta[0] = 1.0f;
    this->gridPotentialMap.delta[1] = 1.0f;
    this->gridPotentialMap.delta[2] = 1.0f;
    this->gridPotentialMap.size[0] = 2;
    this->gridPotentialMap.size[1] = 2;
    this->gridPotentialMap.size[2] = 2;
}


/*
 * megamol::protein_cuda::PotentialVolumeRaycaster::~PotentialVolumeRaycaster
 */
PotentialVolumeRaycaster::~PotentialVolumeRaycaster(void) {
    this->Release();
}


/*
 * megamol::protein_cuda::PotentialVolumeRaycaster::release
 */
void PotentialVolumeRaycaster::release(void) {

    if(glIsTexture(this->volumeTex)) glDeleteTextures(1, &this->volumeTex);
    if(glIsTexture(this->potentialTex)) glDeleteTextures(1, &this->potentialTex);

    this->rcShader.Release();
    this->rcShaderRay.Release();

    if(this->cudaqsurf != NULL) {
        CUDAQuickSurf *cqs = (CUDAQuickSurf *)this->cudaqsurf;
        delete cqs;
    }

    this->freeBuffers();
    cudaDeviceReset();
}


/*
 * megamol::protein_cuda::PotentialVolumeRaycaster::computeVolumeTex
 */
bool PotentialVolumeRaycaster::computeDensityMap(
        const MolecularDataCall *mol) {

    using namespace vislib::sys;
    using namespace vislib::math;

    //Vec3f gridXAxis, gridYAxis, gridZAxis;

    int volSizeOld = gridDensMap.size[0]*gridDensMap.size[1]*gridDensMap.size[2];

    // Init uniform grid
    gridDensMap.minC[0] = this->bbox.ObjectSpaceBBox().Left();
    gridDensMap.minC[1] = this->bbox.ObjectSpaceBBox().Bottom();
    gridDensMap.minC[2] = this->bbox.ObjectSpaceBBox().Back();
    gridDensMap.maxC[0] = this->bbox.ObjectSpaceBBox().Right();
    gridDensMap.maxC[1] = this->bbox.ObjectSpaceBBox().Top();
    gridDensMap.maxC[2] = this->bbox.ObjectSpaceBBox().Front();

//    printf("raycaster bbox density %f %f %f %f %f %f\n",
//            this->bbox.ObjectSpaceBBox().Left(), this->bbox.ObjectSpaceBBox().Right(),
//            this->bbox.ObjectSpaceBBox().Bottom(), this->bbox.ObjectSpaceBBox().Top(),
//            this->bbox.ObjectSpaceBBox().Back(),this->bbox.ObjectSpaceBBox().Front());

    gridXAxis[0] = gridDensMap.maxC[0] - gridDensMap.minC[0];
    gridYAxis[1] = gridDensMap.maxC[1] - gridDensMap.minC[1];
    gridZAxis[2] = gridDensMap.maxC[2] - gridDensMap.minC[2];

    //printf("gridMaxCoord0: %f %f %f\n", gridMaxCoord[0], gridMaxCoord[1], gridMaxCoord[2]);
    //printf("grids: %f %f %f\n", gridXAxis[0], gridYAxis[1], gridZAxis[2]);
    //printf("spacing %f\n", this->qsGridSpacing);
    //printf("rad %f\n", this->qsParticleRad);
    gridDensMap.size[0] = (int) ceil(gridXAxis[0] / this->qsGridSpacing);
    gridDensMap.size[1] = (int) ceil(gridYAxis[1] / this->qsGridSpacing);
    gridDensMap.size[2] = (int) ceil(gridZAxis[2] / this->qsGridSpacing);
    gridXAxis[0] = (gridDensMap.size[0]-1) * this->qsGridSpacing;
    gridYAxis[1] = (gridDensMap.size[1]-1) * this->qsGridSpacing;
    gridZAxis[2] = (gridDensMap.size[2]-1) * this->qsGridSpacing;
    gridDensMap.maxC[0] = gridDensMap.minC[0] + gridXAxis[0];
    gridDensMap.maxC[1] = gridDensMap.minC[1] + gridYAxis[1];
    gridDensMap.maxC[2] = gridDensMap.minC[2] + gridZAxis[2];


    float *gridDataPos = new float[mol->AtomCount()*4]; // TODO Do not allocate every in frame

	if (volSizeOld < (int)(gridDensMap.size[0] * gridDensMap.size[1] * gridDensMap.size[2])) { // TODO Do not allocate every time
        if(this->volume != NULL) {
            delete[] this->volume;
        }
        this->volume = new float[gridDensMap.size[0]*gridDensMap.size[1]*gridDensMap.size[2]];
    }

    // Gather atom positions for the density map
    uint particleCnt = 0;
    float maxRad = 0.0f;
    for (uint c = 0; c < mol->ChainCount(); ++c) { // Loop through chains
        // Only use non-solvent atoms for density map
        if (mol->Chains()[c].Type() != MolecularDataCall::Chain::SOLVENT) {

            MolecularDataCall::Chain chainTmp = mol->Chains()[c];
            for (uint m = 0; m < chainTmp.MoleculeCount(); ++m) { // Loop through molecules

                MolecularDataCall::Molecule molTmp = mol->Molecules()[chainTmp.FirstMoleculeIndex()+m];
                for (uint res = 0; res < molTmp.ResidueCount(); ++res) { // Loop through residues

                    const MolecularDataCall::Residue *resTmp = mol->Residues()[molTmp.FirstResidueIndex()+res];
                    for (uint at = 0; at < resTmp->AtomCount(); ++at) { // Loop through atoms
                        uint atomIdx = resTmp->FirstAtomIndex() + at;

                        gridDataPos[4*particleCnt+0] = mol->AtomPositions()[3*atomIdx+0] - gridDensMap.minC[0];
                        gridDataPos[4*particleCnt+1] = mol->AtomPositions()[3*atomIdx+1] - gridDensMap.minC[1];
                        gridDataPos[4*particleCnt+2] = mol->AtomPositions()[3*atomIdx+2] - gridDensMap.minC[2];
                        if(this->qsSclVanDerWaals) {
                            gridDataPos[4*particleCnt+3] = mol->AtomTypes()[mol->AtomTypeIndices()[atomIdx]].Radius();
                        }
                        else {
                            gridDataPos[4*particleCnt+3] = 1.0f;
                        }
                        if(gridDataPos[4*particleCnt+3] > maxRad) {
                            maxRad = gridDataPos[4*particleCnt+3];
                        }
                        particleCnt++;
                    }
                }
            }
        }
    }

    // Compute uniform grid
    CUDAQuickSurf *cqs = (CUDAQuickSurf *) this->cudaqsurf;
    int rc = cqs->calc_map(
            particleCnt,
            &gridDataPos[0],
            NULL,                 // Pointer to 'color' array
            false,                // Do not use 'color' array
            (float*)&gridDensMap.minC,
            (int*)&gridDensMap.size,
            maxRad,
            this->qsParticleRad, // Radius scaling
            this->qsGridSpacing,
            this->volIsoVal,
            this->qsGaussLim);

    // DEBUG print parameters
    /*printf("QS PARAM Atom count %u\n", mol->AtomCount());
    printf("QS PARAM GridPos %f %f %f\n", gridDataPos[0], gridDataPos[1], gridDataPos[2]);
    printf("QS PARAM GridOrg %f %f %f\n", gridOrg.X(), gridOrg.Y(), gridOrg.Z());
    printf("QS PARAM GridDim %u %u %u\n", gridDim.X(), gridDim.Y(), gridDim.Z());
    printf("QS PARAM MaxRad %f\n", maxRad);
    printf("QS PARAM Radius Scale %f\n", this->qsParticleRad);
    printf("QS PARAM Grid Spacing %f\n", this->qsGridSpacing);
    printf("QS PARAM iso val %f\n", this->volIsoVal);
    printf("QS PARAM gauss limit %f\n", this->qsGaussLim);*/

    if(rc != 0) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Quicksurf class returned val != 0\n", this->ClassName());
        return false;
    }

    // Copy data from device to host
    CudaSafeCall(cudaMemcpy(this->volume, cqs->getMap(),
            gridDensMap.size[0]*gridDensMap.size[1]*gridDensMap.size[2]*sizeof(float),
            cudaMemcpyDeviceToHost));
    if(cudaGetLastError() != cudaSuccess) {
        return false;
    }

//    for(int i = 0; i < gridDim[0]*gridDim[1]*gridDim[2]; i++) {
//        printf("volume (raycaster) %i: %f\n", i, this->volume0[i]);
//    }

//    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
//            "%s: time for computing volume #0 %f",
//            this->ClassName(),
//            (double(clock()-t)/double(CLOCKS_PER_SEC))); // DEBUG

    delete[] gridDataPos;


    //  Setup texture

    glEnable(GL_TEXTURE_3D);

    if(!glIsTexture(this->volumeTex)) {
        glGenTextures(1, &this->volumeTex);
    }
    glBindTexture(GL_TEXTURE_3D, this->volumeTex);
    glTexImage3DEXT(GL_TEXTURE_3D,
            0,
            GL_ALPHA,
            gridDensMap.size[0],
            gridDensMap.size[1],
            gridDensMap.size[2],
            0,
            GL_ALPHA,
            GL_FLOAT,
            this->volume);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);

    CheckForGLError();

    return true;
}


/*
 * megamol::protein_cuda::PotentialVolumeRaycaster::create
 */
bool PotentialVolumeRaycaster::create() {
    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    // Create quicksurf objects
    if(!this->cudaqsurf) {
        this->cudaqsurf = new CUDAQuickSurf();
    }

    // Init extensions
    if(! ogl_IsVersionGEQ(2,0) || !areExtsAvailable("\
            GL_EXT_texture3D \
            GL_EXT_framebuffer_object \
            GL_ARB_multitexture \
            GL_ARB_draw_buffers \
            GL_ARB_vertex_buffer_object")) {
        return false;
    }
    if(!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        return false;
    }

    // Load shader sources
    ShaderSource vertSrc, fragSrc, geomSrc;

    core::CoreInstance *ci = this->GetCoreInstance();
    if(!ci) return false;

    // Load raycasting vertex shader
    if(!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("electrostatics::raycasting::vertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for the raycasting shader",
                this->ClassName());
        return false;
    }
    // Load raycasting fragment shader
    if(!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("electrostatics::raycasting::fragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for the raycasting shader", this->ClassName());
        return false;
    }
    try {
        if(!this->rcShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch(vislib::Exception &e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to create the raycasting shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    // Load raycasting vertex shader
    if(!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("electrostatics::raycasting::vertexRay", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for the raycasting shader",
                this->ClassName());
        return false;
    }
    // Load raycasting fragment shader
    if(!this->GetCoreInstance()->ShaderSourceFactory().MakeShaderSource("electrostatics::raycasting::fragmentRay", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to load vertex shader source for the raycasting shader", this->ClassName());
        return false;
    }
    try {
        if(!this->rcShaderRay.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count())) {
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__ );
        }
    }
    catch(vislib::Exception &e) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "%s: Unable to create the raycasting shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }


    // Load slice shader
    if(!ci->ShaderSourceFactory().MakeShaderSource("electrostatics::slice::vertex", vertSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load vertex shader source: slice shader", this->ClassName());
        return false;
    }
    if(!ci->ShaderSourceFactory().MakeShaderSource("electrostatics::slice::fragment", fragSrc)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to load fragment shader source:  slice shader", this->ClassName());
        return false;
    }
    try {
        if(!this->sliceShader.Create(vertSrc.Code(), vertSrc.Count(), fragSrc.Code(), fragSrc.Count()))
            throw vislib::Exception("Generic creation failure", __FILE__, __LINE__);
    }
    catch(vislib::Exception &e){
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to create shader: %s\n", this->ClassName(), e.GetMsgA());
        return false;
    }

    return true;
}


/*
 * megamol::protein_cuda::PotentialVolumeRaycaster::createFbos
 */
bool PotentialVolumeRaycaster::createFbos(UINT width, UINT height) {

    using namespace vislib::sys;
    using namespace vislib::graphics::gl;

    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
               "%s: (re)creating raycasting fbo.", this->ClassName());

    glEnable(GL_TEXTURE_2D);

    FramebufferObject::ColourAttachParams cap0[3];
    cap0[0].internalFormat = GL_RGBA32F;
    cap0[0].format = GL_RGBA;
    cap0[0].type = GL_FLOAT;
    cap0[1].internalFormat = GL_RGBA32F;
    cap0[1].format = GL_RGBA;
    cap0[1].type = GL_FLOAT;
    cap0[2].internalFormat = GL_RGBA32F;
    cap0[2].format = GL_RGBA;
    cap0[2].type = GL_FLOAT;

    FramebufferObject::DepthAttachParams dap0;
    dap0.format = GL_DEPTH_COMPONENT24;
    dap0.state = FramebufferObject::ATTACHMENT_DISABLED;

    FramebufferObject::StencilAttachParams sap0;
    sap0.format = GL_STENCIL_INDEX;
    sap0.state = FramebufferObject::ATTACHMENT_DISABLED;

    if(!this->rcFbo.Create(width, height, 3, cap0, dap0, sap0)) return false;

    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
               "%s: (re)creating source fbo.", this->ClassName());

    glEnable(GL_TEXTURE_2D);

    FramebufferObject::ColourAttachParams cap1[3];
    cap1[0].internalFormat = GL_RGBA32F;
    cap1[0].format = GL_RGBA;
    cap1[0].type = GL_FLOAT;

    FramebufferObject::DepthAttachParams dap1;
    dap1.format = GL_DEPTH_COMPONENT32;
    dap1.state = FramebufferObject::ATTACHMENT_TEXTURE;

    FramebufferObject::StencilAttachParams sap1;
    sap1.format = GL_STENCIL_INDEX;
    sap1.state = FramebufferObject::ATTACHMENT_DISABLED;

    return this->srcFbo.Create(width, height, 1, cap1, dap1, sap1);
}


/*
 * megamol::protein_cuda::PotentialVolumeRaycaster::GetExtents
 */
bool PotentialVolumeRaycaster::GetExtents(core::Call& call) {

//    printf("PotentialVolumeRaycaster::GetExtents\n");

    core::view::CallRender3D *cr3d = dynamic_cast<core::view::CallRender3D *>(&call);
    if (cr3d == NULL) return false;

    // Get pointer to potential map data call
	protein_calls::VTIDataCall *cmd =
			this->potentialDataCallerSlot.CallAs<protein_calls::VTIDataCall>();
    if (cmd == NULL) return false;
    if (!(*cmd)(VTIDataCall::CallForGetExtent)) return false;

    // Get a pointer to particle data call
    MolecularDataCall *mol = this->particleDataCallerSlot.CallAs<MolecularDataCall>();
    if (mol == NULL) return false;
    if (!(*mol)(MolecularDataCall::CallForGetExtent)) return false;

    // Get a pointer to the outgoing render call
    view::CallRender3D *ren = this->rendererCallerSlot.CallAs<view::CallRender3D>();
    if(ren == NULL) return false;
    if(!(*ren)(1)) return false;

    this->bboxParticles = mol->AccessBoundingBoxes();
    this->bboxPotential = cmd->AccessBoundingBoxes();

    core::BoundingBoxes bbox_external = ren->AccessBoundingBoxes();

    // Calc union of all bounding boxes
    vislib::math::Cuboid<float> bboxTmp;

    bboxTmp = cmd->AccessBoundingBoxes().ObjectSpaceBBox();
    bboxTmp.Union(ren->AccessBoundingBoxes().ObjectSpaceBBox());
    bboxTmp.Union(mol->AccessBoundingBoxes().ObjectSpaceBBox());
    bboxTmp.Union(bbox_external.ObjectSpaceBBox());
    this->bbox.SetObjectSpaceBBox(bboxTmp);

    bboxTmp = cmd->AccessBoundingBoxes().ObjectSpaceClipBox();
    bboxTmp.Union(ren->AccessBoundingBoxes().ObjectSpaceClipBox());
    bboxTmp.Union(mol->AccessBoundingBoxes().ObjectSpaceClipBox());
    bboxTmp.Union(bbox_external.ObjectSpaceClipBox());
    this->bbox.SetObjectSpaceClipBox(bboxTmp);

    bboxTmp = cmd->AccessBoundingBoxes().WorldSpaceBBox();
    bboxTmp.Union(ren->AccessBoundingBoxes().WorldSpaceBBox());
    bboxTmp.Union(mol->AccessBoundingBoxes().WorldSpaceBBox());
    bboxTmp.Union(bbox_external.WorldSpaceBBox());
    this->bbox.SetWorldSpaceBBox(bboxTmp);

    bboxTmp = cmd->AccessBoundingBoxes().WorldSpaceClipBox();
    bboxTmp.Union(ren->AccessBoundingBoxes().WorldSpaceClipBox());
    bboxTmp.Union(mol->AccessBoundingBoxes().WorldSpaceClipBox());
    bboxTmp.Union(bbox_external.WorldSpaceClipBox());
    this->bbox.SetWorldSpaceClipBox(bboxTmp);

    float scale;
    if(!vislib::math::IsEqual(this->bbox.ObjectSpaceBBox().LongestEdge(), 0.0f) ) {
        scale = 2.0f / this->bbox.ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }

//    // DEBUG
//    printf("Raycaster scale %f       , org %f %f %f, maxCoord %f %f %f\n",
//            scale,
//            this->bbox.ObjectSpaceBBox().GetOrigin().X(),
//            this->bbox.ObjectSpaceBBox().GetOrigin().Y(),
//            this->bbox.ObjectSpaceBBox().GetOrigin().Z(),
//            this->bbox.ObjectSpaceBBox().GetRightTopFront().X(),
//            this->bbox.ObjectSpaceBBox().GetRightTopFront().Y(),
//            this->bbox.ObjectSpaceBBox().GetRightTopFront().Z());

    this->bbox.MakeScaledWorld(scale);
    cr3d->AccessBoundingBoxes() = this->bbox;
    cr3d->SetTimeFramesCount(std::min(cmd->FrameCount(), mol->FrameCount()));

    return true;
}


/*
 * megamol::protein_cuda::PotentialVolumeRaycaster::freeBuffers
 */
void PotentialVolumeRaycaster::freeBuffers() {
    if(this->volume != NULL) { delete[] this->volume; this->volume = NULL; }
}


/*
 * megamol::protein_cuda::PotentialVolumeRaycaster::initPotential
 */
bool PotentialVolumeRaycaster::initPotential(VTIDataCall *cmd) {
    using namespace vislib::sys;

    // Setup grid parameters
    gridPotentialMap.minC[0] = cmd->AccessBoundingBoxes().ObjectSpaceBBox().GetLeft();
    gridPotentialMap.minC[1] = cmd->AccessBoundingBoxes().ObjectSpaceBBox().GetBottom();
    gridPotentialMap.minC[2] = cmd->AccessBoundingBoxes().ObjectSpaceBBox().GetBack();
    gridPotentialMap.maxC[0] = cmd->AccessBoundingBoxes().ObjectSpaceBBox().GetRight();
    gridPotentialMap.maxC[1] = cmd->AccessBoundingBoxes().ObjectSpaceBBox().GetTop();
    gridPotentialMap.maxC[2] = cmd->AccessBoundingBoxes().ObjectSpaceBBox().GetFront();
    gridPotentialMap.size[0] = cmd->GetGridsize().X();
    gridPotentialMap.size[1] = cmd->GetGridsize().X();
    gridPotentialMap.size[2] = cmd->GetGridsize().X();
    gridPotentialMap.delta[0] = cmd->GetSpacing().X();
    gridPotentialMap.delta[1] = cmd->GetSpacing().Y();
    gridPotentialMap.delta[2] = cmd->GetSpacing().Z();

    //  Setup textures
    glEnable(GL_TEXTURE_3D);
    if(!glIsTexture(this->potentialTex)) {
        glGenTextures(1, &this->potentialTex);
    }
    glBindTexture(GL_TEXTURE_3D, this->potentialTex);
    glTexImage3DEXT(GL_TEXTURE_3D,
            0,
            GL_RGBA32F,
            cmd->GetGridsize().X(),
            cmd->GetGridsize().Y(),
            cmd->GetGridsize().Z(),
            0,
            GL_ALPHA,
            GL_FLOAT,
            (float*)(cmd->GetPointDataByIdx(0, 0)));
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
//    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
//    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);

    glDisable(GL_TEXTURE_3D);

    CheckForGLError();

    return true;
}


/*
 * megamol::protein_cuda::PotentialVolumeRaycaster::Render
 */
bool PotentialVolumeRaycaster::Render(core::Call& call) {

    using namespace vislib::sys;
    using namespace vislib::math;

    // Get render call
    core::view::AbstractCallRender3D *cr3d =
            dynamic_cast<core::view::AbstractCallRender3D *>(&call);
    if (cr3d == NULL) {
        return false;
    }

    float calltime = cr3d->Time();
    uint frameIdx = static_cast<int>(calltime);

    // Get a pointer to the outgoing render call
    view::CallRender3D *ren = this->rendererCallerSlot.CallAs<view::CallRender3D>();
    if (ren == NULL) {
        return false;
    }
    // Set calltime
    ren->SetTime(calltime);
    ren->SetTimeFramesCount(frameIdx);

    // Get the potential data
	protein_calls::VTIDataCall *cmd =
		this->potentialDataCallerSlot.CallAs<protein_calls::VTIDataCall>();
    if (cmd == NULL) {
        return false;
    }
    cmd->SetCalltime(calltime);       // Set calltime
    cmd->SetFrameID(frameIdx, true);  // Set 'force' flag
    if (!(*cmd)(VTIDataCall::CallForGetData)) {
        return false;
    }

    // Get the particle data
    MolecularDataCall *mol = this->particleDataCallerSlot.CallAs<MolecularDataCall>();
    if (mol == NULL) {
        return false;
    }
    mol->SetCalltime(calltime);      // Set calltime
    mol->SetFrameID(frameIdx, true); // Set 'force' flag
    if (!(*mol)(MolecularDataCall::CallForGetData)) {
        return false;
    }


    // Update parameters if necessary
    if(!this->updateParams()) {
        Log::DefaultLog.WriteMsg(
                Log::LEVEL_ERROR, "%s: Unable to update parameters",
                this->ClassName());
        return false;
    }

    // (Re-)compute volume texture if necessary
    if ((this->computeVolume)
            || (mol->DataHash() != this->datahashParticles)
            || (this->frameOld != frameIdx)) {
        this->datahashParticles = mol->DataHash();
        if(!this->computeDensityMap(mol)) return false;
        this->computeVolume = false;
    }

    // (Re-)compute potential texture if necessary
    if ((this->initPotentialTex)
            || (cmd->DataHash() != this->datahashPotential)
            || (this->frameOld != frameIdx)) {
        this->datahashPotential = cmd->DataHash();
        if (!this->initPotential(cmd)) {
            return false;
        }
        this->initPotentialTex = false;
    }

    // Get camera information
    this->cameraInfo =  dynamic_cast<core::view::CallRender3D*>(&call)->GetCameraParameters();
    ren->SetCameraParameters(this->cameraInfo);

    // Set call time
    ren->SetTime(cr3d->Time());

    // Get current viewport and recreate fbo if necessary
    float curVP[4];
    glGetFloatv(GL_VIEWPORT, curVP);
    if((curVP[2] != this->fboDim.X()) || (curVP[3] != fboDim.Y())) {
        this->fboDim.SetX(static_cast<int>(curVP[2]));
        this->fboDim.SetY(static_cast<int>(curVP[3]));
		if (!this->createFbos(static_cast<GLsizei>(curVP[2]), static_cast<GLsizei>(curVP[3]))) return false;
    }


    /* Offscreen rendering of scene objects */

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    //glLoadIdentity();

    // Render scene objects to source fbo
    this->srcFbo.Enable(0);

    //glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Apply scaling based on combined bounding box
    float scaleCombined;
    if(!vislib::math::IsEqual(this->bbox.ObjectSpaceBBox().LongestEdge(), 0.0f)) {
        scaleCombined = 2.0f/this->bbox.ObjectSpaceBBox().LongestEdge();
    } else {
        scaleCombined = 1.0f;
    }
    glScalef(scaleCombined, scaleCombined, scaleCombined);
    //printf("Scale by %f (scaleCombined)\n", scaleCombined); // DEBUG

    // Call additional renderer
    glPushMatrix();
    // Revert scaling done by external renderer
    float scaleRevert;
    if(!vislib::math::IsEqual(ren->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
        scaleRevert = 2.0f/ren->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scaleRevert = 1.0f;
    }
    scaleRevert = 1.0f/scaleRevert;
    glScalef(scaleRevert, scaleRevert, scaleRevert);
    //printf("Scale by %f (scaleRevert)\n", scaleRevert); // DEBUG
    (*ren)(0);
    glPopMatrix();

    CheckForGLError();

//    // DEBUG
//    glDisable(GL_CULL_FACE);
//    glBegin(GL_QUADS);
//        glColor3f(0.0, 1.0, 0.0);
//        glVertex3f(10.0, 10.0, 11.0);
//        glVertex3f(60.0, 10.0, 11.0);
//        glVertex3f(60.0, 60.0, 11.0);
//        glVertex3f(10.0, 60.0, 11.0);
//    glEnd();

    // Render slices
    if (!this->renderSlices(this->volumeTex, this->potentialTex,
            this->gridPotentialMap, this->gridDensMap)) {
        return false;
    }

    this->srcFbo.Disable();


    /* Volume rendering */

    // Render back of the cube to fbo
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
    glDisable(GL_LIGHTING);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_TEXTURE_3D);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    this->rcFbo.EnableMultiple(3, GL_COLOR_ATTACHMENT0_EXT,
            GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT2_EXT);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    this->rcShaderRay.Enable();
    this->RenderVolCube(); // Render back of the cube and store depth values in fbo texture
    this->rcShaderRay.Disable();

    this->rcFbo.Disable();

    // Render the front of the cube
    glCullFace(GL_BACK);
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);

    this->rcShader.Enable();
    glUniform1iARB(this->rcShader.ParameterLocation("densityTex"), 0);
    glUniform1iARB(this->rcShader.ParameterLocation("srcColorBuff"), 1);
    glUniform1iARB(this->rcShader.ParameterLocation("srcDepthBuff"), 2);
    glUniform1iARB(this->rcShader.ParameterLocation("tcBuff0"), 3);
    glUniform1iARB(this->rcShader.ParameterLocation("posESBuff"), 4);
    glUniform1iARB(this->rcShader.ParameterLocation("tcBuff1"), 5);
    glUniform1iARB(this->rcShader.ParameterLocation("potential"), 6);
    glUniform1fARB(this->rcShader.ParameterLocation("delta"), this->volDelta*0.01f);
    glUniform1fARB(this->rcShader.ParameterLocation("isoVal"), this->volIsoVal);
    glUniform1fARB(this->rcShader.ParameterLocation("alphaScl"), this->volAlphaScl);
	glUniform1iARB(this->rcShader.ParameterLocation("maxIt"), static_cast<int>(this->volMaxIt));
    glUniform1fARB(this->rcShader.ParameterLocation("gradOffset"), this->gradOffs/10.0f);
    glUniform1fARB(this->rcShader.ParameterLocation("colorMinVal"), this->colorMinPotentialScl);
    glUniform1fARB(this->rcShader.ParameterLocation("colorMaxVal"), this->colorMaxPotentialScl);
    glUniform3fvARB(this->rcShader.ParameterLocation("colorMin"), 1, this->colorMinPotential.PeekComponents());
    glUniform3fvARB(this->rcShader.ParameterLocation("colorZero"), 1, this->colorZeroPotential.PeekComponents());
    glUniform3fvARB(this->rcShader.ParameterLocation("colorMax"), 1, this->colorMaxPotential.PeekComponents());
	glUniform4fARB(this->rcShader.ParameterLocation("viewportDim"), static_cast<float>(fboDim.X()), static_cast<float>(fboDim.Y()),
            this->cameraInfo->NearClip(), this->cameraInfo->FarClip());
    glUniform3fvARB(this->rcShader.ParameterLocation("potTexOrg"), 1, (float*)&gridPotentialMap.minC);
    glUniform3fARB(this->rcShader.ParameterLocation("potTexSize"),
            gridPotentialMap.delta[0]*(gridPotentialMap.size[0]-1),
            gridPotentialMap.delta[1]*(gridPotentialMap.size[1]-1),
            gridPotentialMap.delta[2]*(gridPotentialMap.size[2]-1));
    glUniform3fvARB(this->rcShader.ParameterLocation("volTexOrg"), 1, (float*)&gridDensMap.minC);
    glUniform3fARB(this->rcShader.ParameterLocation("volTexSize"),
            gridDensMap.delta[0]*(gridDensMap.size[0]-1),
            gridDensMap.delta[1]*(gridDensMap.size[1]-1),
            gridDensMap.delta[2]*(gridDensMap.size[2]-1));

    glActiveTextureARB(GL_TEXTURE1_ARB);
    this->srcFbo.BindColourTexture(0);

    glActiveTextureARB(GL_TEXTURE2_ARB);
    this->srcFbo.BindDepthTexture();

    glActiveTextureARB(GL_TEXTURE3_ARB);
    this->rcFbo.BindColourTexture(0);

    glActiveTextureARB(GL_TEXTURE4_ARB);
    this->rcFbo.BindColourTexture(1);

    glActiveTextureARB(GL_TEXTURE5_ARB);
    this->rcFbo.BindColourTexture(2);

    glActiveTextureARB(GL_TEXTURE6_ARB);
    glBindTexture(GL_TEXTURE_3D, this->potentialTex);

    glActiveTextureARB(GL_TEXTURE0_ARB);
    glBindTexture(GL_TEXTURE_3D, this->volumeTex);


    glEnable(GL_TEXTURE_2D);
    glEnable(GL_TEXTURE_3D);

    this->RenderVolCube();

    glDisable(GL_CULL_FACE);
    Vec3f bboxExtents(this->gridXAxis.X(), this->gridYAxis.Y(), this->gridZAxis.Z());
    Vec3d view;

    GLfloat m[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, m);
    Matrix<GLfloat, 4, COLUMN_MAJOR> modelMatrix(&m[0]);
    modelMatrix.Invert();
    view.SetX(modelMatrix.GetAt(0, 2));
    view.SetY(modelMatrix.GetAt(1, 2));
    view.SetZ(modelMatrix.GetAt(2, 2));

    glPushMatrix();
    glTranslatef(this->bbox.ObjectSpaceBBox().GetOrigin().X(),
            this->bbox.ObjectSpaceBBox().GetOrigin().Y(),
            this->bbox.ObjectSpaceBBox().GetOrigin().Z());
    this->viewSlicing.setupSingleSlice(view.PeekComponents(), bboxExtents.PeekComponents());
    // Render single view aligned slice
    glColor3f(1.0f, 0.0, 0.0);
    glBegin(GL_TRIANGLE_FAN);
	this->viewSlicing.drawSingleSlice(static_cast<float>(VS_EPS));
    //this->viewSlicing.drawSingleSlice(VS_EPS - this->volClipZ);
    glEnd();
    glEnable(GL_CULL_FACE);
    glPopMatrix();

    glDisable(GL_TEXTURE_3D);
    glDisable(GL_TEXTURE_2D);

    this->rcShader.Disable();

    glPopMatrix();

    CheckForGLError();

    this->frameOld = frameIdx;

    mol->Unlock();
    cmd->Unlock();

    return true;
}


/*
 * PotentialVolumeRaycaster::renderSlices
 */
bool PotentialVolumeRaycaster::renderSlices(GLuint densityTex,
        GLuint potentialTex, gridParams potGrid, gridParams densGrid) {

    Vec3f gridMinCoord, gridMaxCoord;
    gridMinCoord[0] = this->bbox.ObjectSpaceBBox().Left();
    gridMinCoord[1] = this->bbox.ObjectSpaceBBox().Bottom();
    gridMinCoord[2] = this->bbox.ObjectSpaceBBox().Back();
    gridMaxCoord[0] = this->bbox.ObjectSpaceBBox().Right();
    gridMaxCoord[1] = this->bbox.ObjectSpaceBBox().Top();
    gridMaxCoord[2] = this->bbox.ObjectSpaceBBox().Front();

    // Texture coordinates for potential map
    float minXPotTC = (this->bbox.ObjectSpaceBBox().Left() - potGrid.minC[0])/(potGrid.maxC[0]-potGrid.minC[0]);
    float minYPotTC = (this->bbox.ObjectSpaceBBox().Bottom() - potGrid.minC[1])/(potGrid.maxC[1]-potGrid.minC[1]);
    float minZPotTC = (this->bbox.ObjectSpaceBBox().Back() - potGrid.minC[2])/(potGrid.maxC[2]-potGrid.minC[2]);
    float maxXPotTC = (this->bbox.ObjectSpaceBBox().Right() - potGrid.minC[0])/(potGrid.maxC[0]-potGrid.minC[0]);
    float maxYPotTC = (this->bbox.ObjectSpaceBBox().Top() - potGrid.minC[1])/(potGrid.maxC[1]-potGrid.minC[1]);
    float maxZPotTC = (this->bbox.ObjectSpaceBBox().Front() - potGrid.minC[2])/(potGrid.maxC[2]-potGrid.minC[2]);
    float planeXPotTC = (this->xPlane-potGrid.minC[0])/(potGrid.maxC[0]- potGrid.minC[0]);
    float planeYPotTC = (this->yPlane-potGrid.minC[1])/(potGrid.maxC[1]- potGrid.minC[1]);
    float planeZPotTC = (this->zPlane-potGrid.minC[2])/(potGrid.maxC[2]- potGrid.minC[2]);

    // Texture coordinates for the density grid
    float minXDensTC = (this->bbox.ObjectSpaceBBox().Left() - densGrid.minC[0])/(densGrid.maxC[0]-densGrid.minC[0]);
    float minYDensTC = (this->bbox.ObjectSpaceBBox().Bottom() - densGrid.minC[1])/(densGrid.maxC[1]-densGrid.minC[1]);
    float minZDensTC = (this->bbox.ObjectSpaceBBox().Back() - densGrid.minC[2])/(densGrid.maxC[2]-densGrid.minC[2]);
    float maxXDensTC = (this->bbox.ObjectSpaceBBox().Right() - densGrid.minC[0])/(densGrid.maxC[0]-densGrid.minC[0]);
    float maxYDensTC = (this->bbox.ObjectSpaceBBox().Top() - densGrid.minC[1])/(densGrid.maxC[1]-densGrid.minC[1]);
    float maxZDensTC = (this->bbox.ObjectSpaceBBox().Front() - densGrid.minC[2])/(densGrid.maxC[2]-densGrid.minC[2]);
    float planeXDensTC = (this->xPlane-densGrid.minC[0])/(densGrid.maxC[0]-densGrid.minC[0]);
    float planeYDensTC = (this->yPlane-densGrid.minC[1])/(densGrid.maxC[1]-densGrid.minC[1]);
    float planeZDensTC = (this->zPlane-densGrid.minC[2])/(densGrid.maxC[2]-densGrid.minC[2]);

    this->sliceShader.Enable();
    glUniform1iARB(this->sliceShader.ParameterLocation("potentialTex"), 0);
    glUniform1iARB(this->sliceShader.ParameterLocation("densityMap"), 1);
    glUniform1iARB(this->sliceShader.ParameterLocation("renderMode"), this->sliceRM);
    glUniform1fARB(this->sliceShader.ParameterLocation("colorMinVal"), this->sliceMinVal);
    glUniform1fARB(this->sliceShader.ParameterLocation("colorMaxVal"), this->sliceMaxVal);
    glUniform3fvARB(this->sliceShader.ParameterLocation("colorMin"), 1, this->sliceColorMinPotential.PeekComponents());
    glUniform3fvARB(this->sliceShader.ParameterLocation("colorZero"), 1, this->sliceColorZeroPotential.PeekComponents());
    glUniform3fvARB(this->sliceShader.ParameterLocation("colorMax"), 1, this->sliceColorMaxPotential.PeekComponents());
    glUniform1fARB(this->sliceShader.ParameterLocation("minPotential"), this->sliceColorMinPotentialScl);
    glUniform1fARB(this->sliceShader.ParameterLocation("midPotential"), this->sliceColorMidPotentialScl);
    glUniform1fARB(this->sliceShader.ParameterLocation("maxPotential"), this->sliceColorMaxPotentialScl);
    glUniform1fARB(this->sliceShader.ParameterLocation("isoval"), this->volIsoVal);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    glActiveTextureARB(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_3D, densityTex);

    glActiveTextureARB(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, potentialTex);

    // X-plane
    if (this->showXPlane) {
        glBegin(GL_QUADS);
        glMultiTexCoord3fARB(GL_TEXTURE0, planeXPotTC, maxYPotTC, minZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, planeXDensTC, maxYDensTC, minZDensTC);
        glVertex3f(this->xPlane, gridMaxCoord[1], gridMinCoord[2]);
        glMultiTexCoord3fARB(GL_TEXTURE0, planeXPotTC, minYPotTC, minZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, planeXDensTC, minYDensTC, minZDensTC);
        glVertex3f(this->xPlane, gridMinCoord[1], gridMinCoord[2]);
        glMultiTexCoord3fARB(GL_TEXTURE0, planeXPotTC, minYPotTC, maxZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, planeXDensTC, minYDensTC, maxZDensTC);
        glVertex3f(this->xPlane, gridMinCoord[1], gridMaxCoord[2]);
        glMultiTexCoord3fARB(GL_TEXTURE0, planeXPotTC, maxYPotTC, maxZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, planeXDensTC, maxYDensTC, maxZDensTC);
        glVertex3f(this->xPlane, gridMaxCoord[1], gridMaxCoord[2]);
        glEnd();
    }

    // Y-plane
    if (this->showYPlane) {
        glBegin(GL_QUADS);
        glMultiTexCoord3fARB(GL_TEXTURE0, minXPotTC, planeYPotTC, maxZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, minXDensTC, planeYDensTC, maxZDensTC);
        glVertex3f(gridMinCoord[0], this->yPlane, gridMaxCoord[2]);
        glMultiTexCoord3fARB(GL_TEXTURE0, minXPotTC, planeYPotTC, minZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, minXDensTC, planeYDensTC, minZDensTC);
        glVertex3f(gridMinCoord[0], this->yPlane, gridMinCoord[2]);
        glMultiTexCoord3fARB(GL_TEXTURE0, maxXPotTC, planeYPotTC, minZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, maxXDensTC, planeYDensTC, minZDensTC);
        glVertex3f(gridMaxCoord[0], this->yPlane, gridMinCoord[2]);
        glMultiTexCoord3fARB(GL_TEXTURE0, maxXPotTC, planeYPotTC, maxZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, maxXDensTC, planeYDensTC, maxZDensTC);
        glVertex3f(gridMaxCoord[0], this->yPlane, gridMaxCoord[2]);
        glEnd();
    }

    // Z-plane
    if (this->showZPlane) {
        glBegin(GL_QUADS);
        glMultiTexCoord3fARB(GL_TEXTURE0, maxXPotTC, minYPotTC, planeZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, maxXDensTC, minYDensTC, planeZDensTC);
        glVertex3f(gridMaxCoord[0], gridMinCoord[1], this->zPlane);
        glMultiTexCoord3fARB(GL_TEXTURE0, maxXPotTC, maxYPotTC, planeZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, maxXDensTC, maxYDensTC, planeZDensTC);
        glVertex3f(gridMaxCoord[0], gridMaxCoord[1], this->zPlane);
        glMultiTexCoord3fARB(GL_TEXTURE0, minXPotTC, maxYPotTC, planeZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, minXDensTC, maxYDensTC, planeZDensTC);
        glVertex3f(gridMinCoord[0], gridMaxCoord[1], this->zPlane);
        glMultiTexCoord3fARB(GL_TEXTURE0, minXPotTC, minYPotTC, planeZPotTC);
        glMultiTexCoord3fARB(GL_TEXTURE1, minXDensTC, minYDensTC, planeZDensTC);
        glVertex3f(gridMinCoord[0], gridMinCoord[1], this->zPlane);
        glEnd();
    }

    glEnable(GL_CULL_FACE);

    this->sliceShader.Disable();

    return CheckForGLError();
}


/*
 * megamol::protein_cuda::PotentialVolumeRaycaster::RenderVolCube
 */
bool PotentialVolumeRaycaster::RenderVolCube() {

    Vec3f gridMinCoord, gridMaxCoord;
    gridParams potGrid, densGrid;

    gridMinCoord[0] = this->bbox.ObjectSpaceBBox().Left();
    gridMinCoord[1] = this->bbox.ObjectSpaceBBox().Bottom();
    gridMinCoord[2] = this->bbox.ObjectSpaceBBox().Back();
    gridMaxCoord[0] = this->bbox.ObjectSpaceBBox().Right();
    gridMaxCoord[1] = this->bbox.ObjectSpaceBBox().Top();
    gridMaxCoord[2] = this->bbox.ObjectSpaceBBox().Front();

    // Texture coordinates for potential map
    potGrid = this->gridPotentialMap;
    float minXPotTC = (this->bbox.ObjectSpaceBBox().Left() - potGrid.minC[0])/(potGrid.maxC[0]-potGrid.minC[0]);
    float minYPotTC = (this->bbox.ObjectSpaceBBox().Bottom() - potGrid.minC[1])/(potGrid.maxC[1]-potGrid.minC[1]);
    float minZPotTC = (this->bbox.ObjectSpaceBBox().Back() - potGrid.minC[2])/(potGrid.maxC[2]-potGrid.minC[2]);
    float maxXPotTC = (this->bbox.ObjectSpaceBBox().Right() - potGrid.minC[0])/(potGrid.maxC[0]-potGrid.minC[0]);
    float maxYPotTC = (this->bbox.ObjectSpaceBBox().Top() - potGrid.minC[1])/(potGrid.maxC[1]-potGrid.minC[1]);
    float maxZPotTC = (this->bbox.ObjectSpaceBBox().Front() - potGrid.minC[2])/(potGrid.maxC[2]-potGrid.minC[2]);

    // Texture coordinates for the density grid
    densGrid = this->gridDensMap;
    float minXDensTC = (this->bbox.ObjectSpaceBBox().Left() - densGrid.minC[0])/(densGrid.maxC[0]-densGrid.minC[0]);
    float minYDensTC = (this->bbox.ObjectSpaceBBox().Bottom() - densGrid.minC[1])/(densGrid.maxC[1]-densGrid.minC[1]);
    float minZDensTC = (this->bbox.ObjectSpaceBBox().Back() - densGrid.minC[2])/(densGrid.maxC[2]-densGrid.minC[2]);
    float maxXDensTC = (this->bbox.ObjectSpaceBBox().Right() - densGrid.minC[0])/(densGrid.maxC[0]-densGrid.minC[0]);
    float maxYDensTC = (this->bbox.ObjectSpaceBBox().Top() - densGrid.minC[1])/(densGrid.maxC[1]-densGrid.minC[1]);
    float maxZDensTC = (this->bbox.ObjectSpaceBBox().Front() - densGrid.minC[2])/(densGrid.maxC[2]-densGrid.minC[2]);

    glBegin(GL_QUADS);

    // Front

    glColor3f(maxXDensTC, minYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, maxXDensTC, minYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, maxXPotTC, minYPotTC, maxZPotTC);
    glVertex3f(gridMaxCoord[0], gridMinCoord[1], gridMaxCoord[2]);

    glColor3f(maxXDensTC, maxYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, maxXDensTC, maxYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, maxXPotTC, maxYPotTC, maxZPotTC);
    glVertex3f(gridMaxCoord[0], gridMaxCoord[1], gridMaxCoord[2]);

    glColor3f(minXDensTC, maxYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, minXDensTC, maxYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, minXPotTC, maxYPotTC, maxZPotTC);
    glVertex3f(gridMinCoord[0], gridMaxCoord[1], gridMaxCoord[2]);

    glColor3f(minXDensTC, minYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, minXDensTC, minYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, minXPotTC, minYPotTC, maxZPotTC);
    glVertex3f(gridMinCoord[0], gridMinCoord[1], gridMaxCoord[2]);

    // Back

    glColor3f(minXDensTC, maxYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, minXDensTC, maxYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, minXPotTC, maxYPotTC, minZPotTC);
    glVertex3f(gridMinCoord[0], gridMaxCoord[1], gridMinCoord[2]);

    glColor3f(maxXDensTC, maxYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, maxXDensTC, maxYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, maxXPotTC, maxYPotTC, minZPotTC);
    glVertex3f(gridMaxCoord[0], gridMaxCoord[1], gridMinCoord[2]);

    glColor3f(maxXDensTC, minYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, maxXDensTC, minYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, maxXPotTC, minYPotTC, minZPotTC);
    glVertex3f(gridMaxCoord[0], gridMinCoord[1], gridMinCoord[2]);

    glColor3f(minXDensTC, minYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, minXDensTC, minYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, minXPotTC, minYPotTC, minZPotTC);
    glVertex3f(gridMinCoord[0], gridMinCoord[1], gridMinCoord[2]);

    // Left

    glColor3f(minXDensTC, minYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, minXDensTC, minYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, minXPotTC, minYPotTC, maxZPotTC);
    glVertex3f(gridMinCoord[0], gridMinCoord[1], gridMaxCoord[2]);

    glColor3f(minXDensTC, maxYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, minXDensTC, maxYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, minXPotTC, maxYPotTC, maxZPotTC);
    glVertex3f(gridMinCoord[0], gridMaxCoord[1], gridMaxCoord[2]);

    glColor3f(minXDensTC, maxYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, minXDensTC, maxYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, minXPotTC, maxYPotTC, minZPotTC);
    glVertex3f(gridMinCoord[0], gridMaxCoord[1], gridMinCoord[2]);

    glColor3f(minXDensTC, minYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, minXDensTC, minYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, minXPotTC, minYPotTC, minZPotTC);
    glVertex3f(gridMinCoord[0], gridMinCoord[1], gridMinCoord[2]);

    // Right

    glColor3f(maxXDensTC, maxYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, maxXDensTC, maxYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, maxXPotTC, maxYPotTC, minZPotTC);
    glVertex3f(gridMaxCoord[0], gridMaxCoord[1], gridMinCoord[2]);

    glColor3f(maxXDensTC, maxYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, maxXDensTC, maxYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, maxXPotTC, maxYPotTC, maxZPotTC);
    glVertex3f(gridMaxCoord[0], gridMaxCoord[1], gridMaxCoord[2]);

    glColor3f(maxXDensTC, minYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, maxXDensTC, minYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, maxXPotTC, minYPotTC, maxZPotTC);
    glVertex3f(gridMaxCoord[0], gridMinCoord[1], gridMaxCoord[2]);

    glColor3f(maxXDensTC, minYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, maxXDensTC, minYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, maxXPotTC, minYPotTC, minZPotTC);
    glVertex3f(gridMaxCoord[0], gridMinCoord[1], gridMinCoord[2]);

    // Top

    glColor3f(minXDensTC, maxYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, minXDensTC, maxYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, minXPotTC, maxYPotTC, maxZPotTC);
    glVertex3f(gridMinCoord[0], gridMaxCoord[1], gridMaxCoord[2]);

    glColor3f(maxXDensTC, maxYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, maxXDensTC, maxYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, maxXPotTC, maxYPotTC, maxZPotTC);
    glVertex3f(gridMaxCoord[0], gridMaxCoord[1], gridMaxCoord[2]);

    glColor3f(maxXDensTC, maxYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, maxXDensTC, maxYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, maxXPotTC, maxYPotTC, minZPotTC);
    glVertex3f(gridMaxCoord[0], gridMaxCoord[1], gridMinCoord[2]);

    glColor3f(minXDensTC, maxYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, minXDensTC, maxYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, minXPotTC, maxYPotTC, minZPotTC);
    glVertex3f(gridMinCoord[0], gridMaxCoord[1], gridMinCoord[2]);

    // Bottom

    glColor3f(maxXDensTC, minYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, maxXDensTC, minYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, maxXPotTC, minYPotTC, minZPotTC);
    glVertex3f(gridMaxCoord[0], gridMinCoord[1], gridMinCoord[2]);

    glColor3f(maxXDensTC, minYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, maxXDensTC, minYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, maxXPotTC, minYPotTC, maxZPotTC);
    glVertex3f(gridMaxCoord[0], gridMinCoord[1], gridMaxCoord[2]);

    glColor3f(minXDensTC, minYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, minXDensTC, minYDensTC, maxZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, minXPotTC, minYPotTC, maxZPotTC);
    glVertex3f(gridMinCoord[0], gridMinCoord[1], gridMaxCoord[2]);

    glColor3f(minXDensTC, minYDensTC,minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE0, minXDensTC, minYDensTC, minZDensTC);
    glMultiTexCoord3fARB(GL_TEXTURE1, minXPotTC, minYPotTC, minZPotTC);
    glVertex3f(gridMinCoord[0], gridMinCoord[1], gridMinCoord[2]);

    glEnd();

    CheckForGLError();

    return true;
}


/*
 * megamol::protein_cuda::PotentialVolumeRaycaster::PotentialVolumeRaycaster
 */
bool PotentialVolumeRaycaster::updateParams() {

    /* Parameters for the GPU volume rendering */

    // Parameter for minimum potential value for the color map
    if (this->colorMinPotentialSclSlot.IsDirty()) {
        this->colorMinPotentialScl = this->colorMinPotentialSclSlot.Param<core::param::FloatParam>()->Value();
        this->colorMinPotentialSclSlot.ResetDirty();
    }

    // Parameter for maximum potential value for the color map
    if (this->colorMaxPotentialSclSlot.IsDirty()) {
        this->colorMaxPotentialScl = this->colorMaxPotentialSclSlot.Param<core::param::FloatParam>()->Value();
        this->colorMaxPotentialSclSlot.ResetDirty();
    }

    // Parameter for minimum potential color
    if (this->colorMinPotentialSlot.IsDirty()) {
        float r,g,b;
        utility::ColourParser::FromString(this->colorMinPotentialSlot.Param<core::param::StringParam>()->Value(), r, g, b);
        this->colorMinPotential.Set(r, g, b);
        this->colorMinPotentialSlot.ResetDirty();
    }

    // Parameter for zero potential color
    if (this->colorZeroPotentialSlot.IsDirty()) {
        float r,g,b;
        utility::ColourParser::FromString(this->colorZeroPotentialSlot.Param<core::param::StringParam>()->Value(), r, g, b);
        this->colorZeroPotential.Set(r, g, b);
        this->colorZeroPotentialSlot.ResetDirty();
    }

    // Parameter for maximum potential color
    if (this->colorMaxPotentialSlot.IsDirty()) {
        float r,g,b;
        utility::ColourParser::FromString(this->colorMaxPotentialSlot.Param<core::param::StringParam>()->Value(), r, g, b);
        this->colorMaxPotential.Set(r, g, b);
        this->colorMaxPotentialSlot.ResetDirty();
    }

    // The isovalue for the isosurface
    if (this->volIsoValSlot.IsDirty()) {
        this->volIsoVal = this->volIsoValSlot.Param<core::param::FloatParam>()->Value();
        this->volIsoValSlot.ResetDirty();
        this->computeVolume = true;
    }

    // The delta value for the ray marching
    if (this->volDeltaSlot.IsDirty()) {
        this->volDelta = this->volDeltaSlot.Param<core::param::FloatParam>()->Value();
        this->volDeltaSlot.ResetDirty();
    }

    // The alpha scale factor for the iso surface
    if (this->volAlphaSclSlot.IsDirty()) {
        this->volAlphaScl = this->volAlphaSclSlot.Param<core::param::FloatParam>()->Value();
        this->volAlphaSclSlot.ResetDirty();
    }

    // Parameter for the distance of the volume clipping plane
    if (this->volClipZSlot.IsDirty()) {
        this->volClipZ = this->volClipZSlot.Param<core::param::FloatParam>()->Value();
        this->volClipZSlot.ResetDirty();
    }

    // Parameter for the maximum number of raycasting steps
    if (this->volMaxItSlot.IsDirty()) {
        this->volMaxIt = static_cast<float>(this->volMaxItSlot.Param<core::param::IntParam>()->Value());
        this->volMaxItSlot.ResetDirty();
    }

    // Parameter for the gradient offset
    if (this->gradOffsSlot.IsDirty()) {
        this->gradOffs = this->gradOffsSlot.Param<core::param::FloatParam>()->Value();
        this->gradOffsSlot.ResetDirty();
    }


    /* Parameters for the 'quicksurf' class */

    // Parameter for assumed radius of density grid data
    if (this->qsParticleRadSlot.IsDirty()) {
        this->qsParticleRad = this->qsParticleRadSlot.Param<core::param::FloatParam>()->Value();
        this->qsParticleRadSlot.ResetDirty();
        this->computeVolume = true;
    }

    // Parameter for the cutoff radius for the gaussian kernel
    if (this->qsGaussLimSlot.IsDirty()) {
        this->qsGaussLim = this->qsGaussLimSlot.Param<core::param::FloatParam>()->Value();
        this->qsGaussLimSlot.ResetDirty();
        this->computeVolume = true;
    }

    // Parameter for assumed radius of density grid data
    if (this->qsGridSpacingSlot.IsDirty()) {
        this->qsGridSpacing = this->qsGridSpacingSlot.Param<core::param::FloatParam>()->Value();
        this->qsGridSpacingSlot.ResetDirty();
        this->computeVolume = true;
    }

    // Parameter to toggle scaling by van der Waals radius
    if (this->qsSclVanDerWaalsSlot.IsDirty()) {
        this->qsSclVanDerWaals = this->qsSclVanDerWaalsSlot.Param<core::param::BoolParam>()->Value();
        this->qsSclVanDerWaalsSlot.ResetDirty();
        this->computeVolume = true;
    }


    /* Parameters for slice rendering */

    // Render mode for slices
    if (this->sliceRMSlot.IsDirty()) {
        this->sliceRM= this->sliceRMSlot.Param<core::param::EnumParam>()->Value();
        this->sliceRMSlot.ResetDirty();
    }

    // X-plane position
    if (this->xPlaneSlot.IsDirty()) {
        this->xPlane = this->xPlaneSlot.Param<core::param::FloatParam>()->Value();
        this->xPlaneSlot.ResetDirty();
    }

    // X-plane visibility
    if (this->toggleXPlaneSlot.IsDirty()) {
        this->showXPlane = this->toggleXPlaneSlot.Param<core::param::BoolParam>()->Value();
        this->toggleXPlaneSlot.ResetDirty();
    }

    // Y-plane position
    if (this->yPlaneSlot.IsDirty()) {
        this->yPlane = this->yPlaneSlot.Param<core::param::FloatParam>()->Value();
        this->yPlaneSlot.ResetDirty();
    }

    // Y-plane visibility
    if (this->toggleYPlaneSlot.IsDirty()) {
        this->showYPlane = this->toggleYPlaneSlot.Param<core::param::BoolParam>()->Value();
        this->toggleYPlaneSlot.ResetDirty();
    }

    // Z-plane position
    if (this->zPlaneSlot.IsDirty()) {
        this->zPlane = this->zPlaneSlot.Param<core::param::FloatParam>()->Value();
        this->zPlaneSlot.ResetDirty();
    }

    // Z-plane visibility
    if (this->toggleZPlaneSlot.IsDirty()) {
        this->showZPlane = this->toggleZPlaneSlot.Param<core::param::BoolParam>()->Value();
        this->toggleZPlaneSlot.ResetDirty();
    }

    // Parameter for minimum potential value for the color map
    if (this->sliceColorMinPotentialSclSlot.IsDirty()) {
        this->sliceColorMinPotentialScl = this->sliceColorMinPotentialSclSlot.Param<core::param::FloatParam>()->Value();
        this->sliceColorMinPotentialSclSlot.ResetDirty();
    }

    // Parameter for mid potential value for the color map
    if (this->sliceColorMidPotentialSclSlot.IsDirty()) {
        this->sliceColorMidPotentialScl = this->sliceColorMidPotentialSclSlot.Param<core::param::FloatParam>()->Value();
        this->sliceColorMidPotentialSclSlot.ResetDirty();
    }

    // Parameter for maximum potential value for the color map
    if (this->sliceColorMaxPotentialSclSlot.IsDirty()) {
        this->sliceColorMaxPotentialScl = this->sliceColorMaxPotentialSclSlot.Param<core::param::FloatParam>()->Value();
        this->sliceColorMaxPotentialSclSlot.ResetDirty();
    }

    // Parameter for minimum potential color
    if (this->sliceColorMinPotentialSlot.IsDirty()) {
        float r,g,b;
        utility::ColourParser::FromString(this->sliceColorMinPotentialSlot.Param<core::param::StringParam>()->Value(), r, g, b);
        this->sliceColorMinPotential.Set(r, g, b);
        this->sliceColorMinPotentialSlot.ResetDirty();
    }

    // Parameter for zero potential color
    if (this->sliceColorZeroPotentialSlot.IsDirty()) {
        float r,g,b;
        utility::ColourParser::FromString(this->sliceColorZeroPotentialSlot.Param<core::param::StringParam>()->Value(), r, g, b);
        this->sliceColorZeroPotential.Set(r, g, b);
        this->sliceColorZeroPotentialSlot.ResetDirty();
    }

    // Parameter for maximum potential color
    if (this->sliceColorMaxPotentialSlot.IsDirty()) {
        float r,g,b;
        utility::ColourParser::FromString(this->sliceColorMaxPotentialSlot.Param<core::param::StringParam>()->Value(), r, g, b);
        this->sliceColorMaxPotential.Set(r, g, b);
        this->sliceColorMaxPotentialSlot.ResetDirty();
    }

    // Minimum texture value
    if (this->sliceMinValSlot.IsDirty()) {
        this->sliceMinVal = this->sliceMinValSlot.Param<core::param::FloatParam>()->Value();
        this->sliceMinValSlot.ResetDirty();
    }

    // Maximum texture value
    if (this->sliceMaxValSlot.IsDirty()) {
        this->sliceMaxVal = this->sliceMaxValSlot.Param<core::param::FloatParam>()->Value();
        this->sliceMaxValSlot.ResetDirty();
    }


    return true;
}
