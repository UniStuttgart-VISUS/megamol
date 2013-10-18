//
// PotentialCalculator.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 19, 2013
//     Author: scharnkn
//

/** NOT WORKING ATM !!**/

#include "stdafx.h"
#include "PotentialCalculator.h"
#include "MolecularDataCall.h"
#include "VTIDataCall.h"
#include "VTKImageData.h"
//#include "vislib_vector_typedefs.h"
#include "param/IntParam.h"
#include "param/FloatParam.h"
#include "param/EnumParam.h"
#include "vislib/Log.h"

#if (defined(WITH_CUDA) && (WITH_CUDA))
#include "CUDAQuickSurf.h"
#include "cuda_error_check.h"
#include "PotentialCalculator.cuh"
#endif

// TODO
// + CUDAQuicksurf segfaults on release because vertex buffer object extension
//   is not initialized

using namespace megamol;
using namespace megamol::protein;

typedef unsigned int uint;


/*
 * PotentialCalculator::PotentialCalculator
 */
PotentialCalculator::PotentialCalculator() : Module(),
        dataCallerSlot("getdata", "Connects the module with the data source"),
        dataCalleeSlot("dataout", "Connects the module with a calling data module"),
        computationalMethodSlot("computationalMethod", "The algorithm used to compute the potential map"),
        bboxTypeSlot("bboxType", "The periodic bounding box of the particle data"),
        chargesGridSpacingSlot("chargesGridSpacing", "Grid resolution for the charge distribution"),
        potentialGridSpacingSlot("potentialGridSpacing", "Grid resolution for the potential map"),
#if (defined(WITH_CUDA) && (WITH_CUDA))
        cudaqsurf(NULL),
#endif
        maxParticleRad(0.0f), minPotential(0.0f),
        maxPotential(0.0f), jobDone(false) {

    // Make data caller slot available
    this->dataCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable (&this->dataCallerSlot);

    // Make data callee slot available
    this->dataCalleeSlot.SetCallback(
            VTIDataCall::ClassName(),
            VTIDataCall::FunctionName(VTIDataCall::CallForGetData),
            &PotentialCalculator::getData);
    this->dataCalleeSlot.SetCallback(
            VTIDataCall::ClassName(),
            VTIDataCall::FunctionName(VTIDataCall::CallForGetExtent),
            &PotentialCalculator::getExtent);
    this->MakeSlotAvailable(&this->dataCalleeSlot);

    // Parameter to choose the computational method for the potential map
    this->computationalMethod = GPU_POISSON_SOLVER;
    core::param::EnumParam *cm = new core::param::EnumParam(int(this->computationalMethod));
    cm->SetTypePair(DIRECT_COULOMB_SUMMATION_NO_PERIODIC_BOUNDARIES, "DCS (no periodic boundaries)");
    cm->SetTypePair(DIRECT_COULOMB_SUMMATION, "DCS");
    cm->SetTypePair(EWALD_SUMMATION, "Ewald sum");
    //cm->SetTypePair(PARTICLE_MESH_EWALD, "Particle Mesh Ewald");  // TODO Not supported atm
    //cm->SetTypePair(CONTINUUM_SOLVATION_POISSON_BOLTZMAN, "Continuum Solvation"); // TODO Not supported atm
    cm->SetTypePair(GPU_POISSON_SOLVER, "GPU Poisson Solver (by Georg Rempfer)");
    this->computationalMethodSlot << cm;
    this->MakeSlotAvailable(&this->computationalMethodSlot);

    // Parameter to choose the computational method for the potential map
    this->bboxType = BBOX_CUBIC;
    core::param::EnumParam *bb = new core::param::EnumParam(int(this->bboxType));
    bb->SetTypePair(BBOX_CUBIC, "Cubic");
    //bb->SetTypePair(BBOX_TRUNCATED_OCTAHEDRON, "Truncated Octahedron"); // TODO Not supported atm
    this->bboxTypeSlot << bb;
    this->MakeSlotAvailable(&this->bboxTypeSlot);

    // Parameter for the grid resolution of the charge distribution
    this->chargesGridSpacing = 2.0f;
    this->chargesGridSpacingSlot << new core::param::FloatParam(this->chargesGridSpacing, 0.0f);
    this->MakeSlotAvailable(&this->chargesGridSpacingSlot);

    // Parameter for the grid resolution of the potential map
    this->potentialGridSpacing = 2.0f;
    this->potentialGridSpacingSlot << new core::param::FloatParam(this->potentialGridSpacing, 0.0f);
    this->MakeSlotAvailable(&this->potentialGridSpacingSlot);
}


/*
 * PotentialCalculator::~PotentialCalculator
 */
PotentialCalculator::~PotentialCalculator() {
     this->Release();
 }


/*
 * PotentialCalculator::create
 */
bool PotentialCalculator::create(void) {
#if (defined(WITH_CUDA) && (WITH_CUDA))
    // Create quicksurf object
    if (!this->cudaqsurf) {
        this->cudaqsurf = new CUDAQuickSurf();
    }
#endif
    return true;
}


/*
 * PotentialCalculator::getData
 */
bool PotentialCalculator::getData(core::Call& call) {
    using namespace vislib::math;

    // Get the incoming data call
    VTIDataCall *dcIn = dynamic_cast<VTIDataCall*>(&call);
    if (dcIn == NULL) {
        return false;
    }

    // Call for particle data extent
    MolecularDataCall *dcOut = this->dataCallerSlot.CallAs<MolecularDataCall>();
    if (dcOut == NULL) {
        return false;
    }
    if (!(*dcOut)(MolecularDataCall::CallForGetData)) {
        return false;
    }

    this->updateParams(); // Update parameter slots


    /* Compute charge distribution */

    if (!this->computeChargeDistribution(dcOut)) {
        return false;
    }


    /* Compute potential map based on particles and charge distribution */

    if (!this->computePotentialMap(dcOut)) {
        return false;
    }


    /* Set piece data pointer for potential in incoming data call */

// TODO Fix this?
//    dcIn->PeekPiece(0).SetExtent(Cuboid<uint>(0, 0, 0,
//            this->potentialGrid.size[0]-1,
//            this->potentialGrid.size[1]-1,
//            this->potentialGrid.size[2]-1));
//    dcIn->PeekPiece(0).SetScalarPointData((void*)(this->potential.Peek()),
//            this->minPotential, this->maxPotential,
//            VTKImageData::DataArray::VTI_FLOAT, "potential");

    return true;
}


/*
 * PotentialCalculator::getExtent
 */
bool PotentialCalculator::getExtent(core::Call& call) {
    using namespace vislib::math;

    // Get the incoming data call
    VTIDataCall *dcIn = dynamic_cast<VTIDataCall*>(&call);
    if (dcIn == NULL) {
        return false;
    }

    // Call for particle data extent
    MolecularDataCall *dcOut = this->dataCallerSlot.CallAs<MolecularDataCall>();
    if (dcOut == NULL) {
        return false;
    }
    if (!(*dcOut)(MolecularDataCall::CallForGetExtent)) {
        return false;
    }

    // TODO Make sure this is the actual bounding box from the simulation
    // (atm padded version from pdb loader) !!
    this->particleBBox.Set(
            dcOut->AccessBoundingBoxes().ObjectSpaceBBox().Left(),
            dcOut->AccessBoundingBoxes().ObjectSpaceBBox().Bottom(),
            dcOut->AccessBoundingBoxes().ObjectSpaceBBox().Back(),
            dcOut->AccessBoundingBoxes().ObjectSpaceBBox().Right(),
            dcOut->AccessBoundingBoxes().ObjectSpaceBBox().Top(),
            dcOut->AccessBoundingBoxes().ObjectSpaceBBox().Front());

    // Init grid parameters for the charge distribution
    this->chargesGrid.delta[0] = this->chargesGridSpacingSlot.Param<core::param::FloatParam>()->Value();
    this->chargesGrid.delta[1] = this->chargesGridSpacingSlot.Param<core::param::FloatParam>()->Value();
    this->chargesGrid.delta[2] = this->chargesGridSpacingSlot.Param<core::param::FloatParam>()->Value();
    this->initGridParams(this->chargesGrid, dcOut);

    // Init grid dimensions for potential map
    this->potentialGrid.delta[0] = this->potentialGridSpacingSlot.Param<core::param::FloatParam>()->Value();
    this->potentialGrid.delta[1] = this->potentialGridSpacingSlot.Param<core::param::FloatParam>()->Value();
    this->potentialGrid.delta[2] = this->potentialGridSpacingSlot.Param<core::param::FloatParam>()->Value();
    this->initGridParams(this->potentialGrid, dcOut);

    // Set bbox of the data set
    this->potentialBBox.Set(this->potentialGrid.minC[0], this->potentialGrid.minC[1],
            this->potentialGrid.minC[2], this->potentialGrid.maxC[0],
            this->potentialGrid.maxC[1], this->potentialGrid.maxC[2]);

    // Set extent in the incoming call
    // TODO Fix this
//    dcIn->SetFrameCount(dcOut->FrameCount());
//    dcIn->AccessBoundingBoxes().Clear();
//    dcIn->AccessBoundingBoxes().SetObjectSpaceBBox(this->potentialBBox);
//    dcIn->AccessBoundingBoxes().SetObjectSpaceClipBox(this->potentialBBox);
//    dcIn->SetWholeExtent(Cuboid<uint>(0, 0, 0, this->potentialGrid.size[0]-1,
//            this->potentialGrid.size[1]-1, this->potentialGrid.size[2]-1));
//    dcIn->SetOrigin(Vec3f(this->potentialGrid.minC));
//    dcIn->SetSpacing(Vec3f(this->potentialGrid.delta));
//    dcIn->SetNumberOfPieces(1);

    return true;
}


/*
 * PotentialCalculator::initGridParams
 */
void PotentialCalculator::initGridParams(gridParams &grid, MolecularDataCall *dcOut) {

    float padding;
    float gridXAxisLen, gridYAxisLen, gridZAxisLen;

    // Determine maximum particle radius (needed for correct padding)
    this->maxParticleRad = 0.0f;
    for (int cnt = 0; cnt < static_cast<int>(dcOut->AtomCount()); ++cnt) {
        if (dcOut->AtomTypes()[dcOut->AtomTypeIndices()[cnt]].Radius() > this->maxParticleRad) {
            this->maxParticleRad = dcOut->AtomTypes()[dcOut->AtomTypeIndices()[cnt]].Radius();
        }
    }

    // Compute safety padding for the grid
    padding = this->maxParticleRad + grid.delta[0]*2;

    // Init grid parameters
    grid.minC[0] = dcOut->AccessBoundingBoxes().ObjectSpaceBBox().GetLeft()   - padding;
    grid.minC[1] = dcOut->AccessBoundingBoxes().ObjectSpaceBBox().GetBottom() - padding;
    grid.minC[2] = dcOut->AccessBoundingBoxes().ObjectSpaceBBox().GetBack()   - padding;
    grid.maxC[0] = dcOut->AccessBoundingBoxes().ObjectSpaceBBox().GetRight() + padding;
    grid.maxC[1] = dcOut->AccessBoundingBoxes().ObjectSpaceBBox().GetTop()   + padding;
    grid.maxC[2] = dcOut->AccessBoundingBoxes().ObjectSpaceBBox().GetFront() + padding;
    gridXAxisLen = grid.maxC[0] - grid.minC[0];
    gridYAxisLen = grid.maxC[1] - grid.minC[1];
    gridZAxisLen = grid.maxC[2] - grid.minC[2];
    grid.size[0] = (int) ceil(gridXAxisLen / grid.delta[0]);
    grid.size[1] = (int) ceil(gridYAxisLen / grid.delta[1]);
    grid.size[2] = (int) ceil(gridZAxisLen / grid.delta[2]);

//    // FFT needs the grid to be a power of two
//    if (this->computationalMethod == GPU_POISSON_SOLVER) {
//        // Expand each grid side to be a power of two
//        for (int i = 0; i < 3; ++i) {
//            uint power = 1;
//            while(power < grid.size[i]) {
//                power*=2;
//            }
//            grid.size[i] = power;
//        }
//    }

    gridXAxisLen = (grid.size[0]-1) * grid.delta[0];
    gridYAxisLen = (grid.size[1]-1) * grid.delta[1];
    gridZAxisLen = (grid.size[2]-1) * grid.delta[2];
    grid.maxC[0] = grid.minC[0] + gridXAxisLen;
    grid.maxC[1] = grid.minC[1] + gridYAxisLen;
    grid.maxC[2] = grid.minC[2] + gridZAxisLen;

//    // DEBUG
//    printf("Grid params: minC %f %f %f, maxCoord %f %f %f, size %u %u %u\n",
//            grid.minC[0], grid.minC[1], grid.minC[2], grid.maxC[0],
//            grid.maxC[1], grid.maxC[2], grid.size[0], grid.size[1],
//            grid.size[2]);
}


/*
 * PotentialCalculator::computeChargeDistribution
 */
bool PotentialCalculator::computeChargeDistribution(const MolecularDataCall *mol) {
    using namespace vislib::sys;
    using namespace vislib::math;

#if (defined(WITH_CUDA) && (WITH_CUDA)) // GPU version

    unsigned int volSize;

    // (Re-)allocate memory for intermediate particle data
    this->particlePos.Validate(mol->AtomCount()*4);
    this->particleCharges.Validate(mol->AtomCount()*4);

    volSize = this->chargesGrid.size[0]*this->chargesGrid.size[1]*
            this->chargesGrid.size[2];

    // (Re-)allocate volume memory if necessary
    this->charges.Validate(volSize);
    this->chargesBuff.Validate(volSize*3);

    // Set particle positions
#pragma omp parallel for
    for (int cnt = 0; cnt < static_cast<int>(mol->AtomCount()); ++cnt) {
        this->particlePos.Peek()[4*cnt+0] = mol->AtomPositions()[3*cnt+0] - this->chargesGrid.minC[0];
        this->particlePos.Peek()[4*cnt+1] = mol->AtomPositions()[3*cnt+1] - this->chargesGrid.minC[1];
        this->particlePos.Peek()[4*cnt+2] = mol->AtomPositions()[3*cnt+2] - this->chargesGrid.minC[2];
        this->particlePos.Peek()[4*cnt+3] = mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius();
        this->particleCharges.Peek()[4*cnt+0] = mol->AtomOccupancies()[cnt];
        this->particleCharges.Peek()[4*cnt+1] = 0.0f; // Not used atm
        this->particleCharges.Peek()[4*cnt+2] = 0.0f; // Not used atm
        this->particleCharges.Peek()[4*cnt+3] = 0.0f; // Not used atm
    }

//    for (int cnt = 0; cnt < static_cast<int>(mol->AtomCount()); ++cnt) {
//        this->particlePos.Peek()[4*cnt+3] = mol->AtomTypes()[mol->AtomTypeIndices()[cnt]].Radius();
//        printf("Atom %i occupancy %f\n", cnt, mol->AtomOccupancies()[cnt]);
//    }

    // Compute uniform grid
    CUDAQuickSurf *cqs = (CUDAQuickSurf *)this->cudaqsurf;
    int rc = cqs->calc_map(
            mol->AtomCount(),
            &this->particlePos.Peek()[0],     // Pointer to 'particle positions
            &this->particleCharges.Peek()[0], // Pointer to 'color' array
            true,                             // Do not use 'color' array
            (float*)&this->chargesGrid.minC[0],
            (int*)&this->chargesGrid.size[0],
            this->maxParticleRad,
            5.0f,   // Radius scale
            this->chargesGrid.delta[0],
            0.5f,   // Isovalue
            20.0f); // Cut off for Gaussian

    if (rc != 0) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Quicksurf class returned val != 0\n", this->ClassName());
        return false;
    }

    CudaSafeCall(cudaMemcpy((void *)this->chargesBuff.Peek(),
            (const void *)cqs->getColorMap(),
            sizeof(float)*this->chargesBuff.GetCount(),
            cudaMemcpyDeviceToHost));

    // Set charges
#pragma omp parallel for
    for (int cnt = 0; cnt < static_cast<int>(volSize); ++cnt) {
        this->charges.Peek()[cnt] = this->chargesBuff.Peek()[3*cnt];
    }

//    for (int i = 0; i < this->charges.GetCount(); ++i)
//        printf("Charges %f\n", this->charges.Peek()[i]);

#else // CPU version
    // TODO
#endif // (defined(WITH_CUDA) && (WITH_CUDA))

    return true;
}


/*
 * PotentialCalculator::computePotentialMap
 */
bool PotentialCalculator::computePotentialMap(const MolecularDataCall *mol) {
    using namespace vislib::sys;

    size_t volSize = this->potentialGrid.size[0]*
            this->potentialGrid.size[1]*
            this->potentialGrid.size[2];

    this->potential.Validate(volSize);

    time_t t = clock(); // DEBUG

    switch (this->computationalMethod) {
    case DIRECT_COULOMB_SUMMATION_NO_PERIODIC_BOUNDARIES:
        if (!this->computePotentialMapDCS(mol, 100.0f)) {
            return false;
        }
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "Time for computing potential map (Direct Coulomb Summation, CPU): %fs",
                (double(clock()-t)/double(CLOCKS_PER_SEC) )); // DEBUG
        break;
    case DIRECT_COULOMB_SUMMATION:
        if (!this->computePotentialMapDCS(mol, 100.0f, true)) {
            return false;
        }
        break; // TODO Parameter for cutoff radius
    case PARTICLE_MESH_EWALD: // TODO
    case CONTINUUM_SOLVATION_POISSON_BOLTZMAN: break; // TODO
    case GPU_POISSON_SOLVER:
#ifdef WITH_CUDA
        if (!CudaSafeCall(this->potential_D.Validate(volSize))) {
            return false;
        }
        CudaSafeCall(SolvePoissonEq(this->potentialGridSpacing,
                make_uint3(this->potentialGrid.size[0],
                        this->potentialGrid.size[1],
                        this->potentialGrid.size[2]),
                        this->charges.Peek(),
                        this->potential_D.Peek(),
                        this->potential.Peek()));
#endif // WITH_CUDA
        break;
    case EWALD_SUMMATION:
        if (!this->computePotentialMapEwaldSum(mol, 0.25f)) {
            return false;
        }
        break;
    }

//    // DEBUG Use output charges
//    Vec3f center(this->potentialGrid.maxC[0] - this->potentialGrid.minC[0],
//            this->potentialGrid.maxC[1] - this->potentialGrid.minC[1],
//            this->potentialGrid.maxC[2] - this->potentialGrid.minC[2]);
//    for (uint i = 0; i < volSize; ++i) {
//        this->potential.Peek()[i] = this->charges.Peek()[i];
//    }

    // Get min/max potential
    this->minPotential = this->potential.Peek()[0];
    this->maxPotential = this->potential.Peek()[0];
    for (uint i = 0; i < volSize; ++i) {
        if (this->potential.Peek()[i] < this->minPotential) {
            this->minPotential = this->potential.Peek()[i];
        }
        if (this->potential.Peek()[i] > this->maxPotential) {
            this->maxPotential = this->potential.Peek()[i];
        }
    }

    return true;
}


/*
 * PotentialCalculator::computePotentialMapDCS
 */
bool PotentialCalculator::computePotentialMapDCS(const MolecularDataCall *mol,
        float sphericalCutOff, bool usePeriodicImages) {

    // Compute electrostatic potential for all grid points using Direct
    // Coulomb Summation
#if (defined(WITH_CUDA) && (WITH_CUDA)) // GPU

    // Init atom data
    this->atomData.Validate(mol->AtomCount()*4);
#pragma omp parallel for
    for (int at = 0; at < static_cast<int>(mol->AtomCount()); ++at) {
        this->atomData.Peek()[at*4+0] = mol->AtomPositions()[3*at+0];
        this->atomData.Peek()[at*4+1] = mol->AtomPositions()[3*at+1];
        this->atomData.Peek()[at*4+2] = mol->AtomPositions()[3*at+2];
        this->atomData.Peek()[at*4+3] = mol->AtomOccupancies()[at];
    }

    if (!CudaSafeCall(DirectCoulombSummation(
            this->atomData.Peek(),
            mol->AtomCount(),
            this->potential.Peek(),
            make_uint3(this->potentialGrid.size[0],
                       this->potentialGrid.size[1],
                       this->potentialGrid.size[2]),
                       this->potentialGridSpacing))) {
        return false;
    }

#else // CPU

    // Init lattice positions
     vislib::Array<float> latticePos;
     uint volSize = this->potentialGrid.size[0]*this->potentialGrid.size[1]*
             this->potentialGrid.size[2];
     latticePos.SetCount(volSize*3);

 #pragma omp parallel for
     for (int pos = 0; pos < static_cast<int>(volSize); ++pos) {
         Vec3i ipos(pos % this->potentialGrid.size[0],
                 (pos / this->potentialGrid.size[0]) % this->potentialGrid.size[1],
                 (pos / this->potentialGrid.size[0]) / this->potentialGrid.size[1]);
         latticePos[3*pos+0] = this->potentialGrid.minC[0] + ipos[0]*this->potentialGrid.delta[0];
         latticePos[3*pos+1] = this->potentialGrid.minC[1] + ipos[1]*this->potentialGrid.delta[1];
         latticePos[3*pos+2] = this->potentialGrid.minC[2] + ipos[2]*this->potentialGrid.delta[2];
     }

     // Calc potential map
 #pragma omp parallel for

     // Loop through all lattice positions
     for (int pos = 0; pos < static_cast<int>(volSize); ++pos) {

         this->potential.Peek()[pos] = 0.0f;
         Vec3f lattPos(&latticePos[3*pos]);

         // Loop through all particles
         for (int at = 0; at < static_cast<int>(mol->AtomCount()); ++at) {
             Vec3f atomPos(&mol->AtomPositions()[3*at]);
             float minDist = (lattPos-atomPos).Length();

             // If periodic boundary conditions are to be used find the minimum
             // distance image of the respective particle
             if (usePeriodicImages) {
                 vislib::Array<Vec3f> imgArr;
                 // Get list of periodic images
                 this->getPeriodicImages(atomPos, imgArr);
                 for (uint i = 0; i < imgArr.Count(); ++i) {
                     if ((lattPos-imgArr[i]).Length() < minDist) {
                         atomPos = imgArr[i];
                         minDist = (lattPos-imgArr[i]).Length();
                     }
                 }
             }

             // Add potential contribution
             this->potential.Peek()[pos] +=
                     // Charge is saved in occupancy column
                     (mol->AtomOccupancies()[at]) / minDist;
         }
     }

#endif

    return true;
}


/**
 * TODO
 */
bool PotentialCalculator::computePotentialMapEwaldSum(const MolecularDataCall *mol,
        float beta) {

    // Volume of one unit cell of the periodic lattice
    size_t v = this->potentialGrid.size[0]*this->potentialGrid.size[1]*
            this->potentialGrid.size[2];

    // For now use only order of N terms, this leeds to sufficient accuracy
    // when computing the direct term with sufficiently large beta
    // TODO Minimum image convention?
    const int maxImg = 0;

    /* Compute direct term */

    // Loop through lattice positions
#pragma omp parallel for
    for (int x = 0; x < static_cast<int>(this->potentialGrid.size[0]); ++x) {
        for (int y = 0; y < static_cast<int>(this->potentialGrid.size[1]); ++y) {
            for (int z = 0; z < static_cast<int>(this->potentialGrid.size[2]); ++z) {

                uint posIdx = this->potentialGrid.size[0]*
                        (this->potentialGrid.size[1]*z+y)+x;

                this->potential.Peek()[posIdx] = 0.0f;
                Vec3f latticePos(this->potentialGrid.minC[0] + x*this->potentialGrid.delta[0],
                        this->potentialGrid.minC[1] + y*this->potentialGrid.delta[1],
                        this->potentialGrid.minC[2] + z*this->potentialGrid.delta[2]);

                printf("Grid pos %f %f %f\n",
                        latticePos.X(),
                        latticePos.Y(),
                        latticePos.Z());

                // Loop through all translation vectors
                for (uint nx = -maxImg; nx <= maxImg; ++nx) {
                    for (uint ny = -maxImg; ny <= maxImg; ++ny) {
                        for (uint nz = -maxImg; nz <= maxImg; ++nz) {
                            Vec3f n(static_cast<float>(nx), static_cast<float>(ny), static_cast<float>(nz));

                            /*// Loop through all particles
                            for (uint p = 0; p < mol->AtomCount(); ++p) {
                                // Compute distance
                                Vec3f atomPos(&mol->AtomPositions()[p]);
                                float dist = (latticePos-atomPos).Length();
                                this->potential.Peek()[posIdx] += mol->AtomOccupancies()[p]*erfc(beta*dist)/dist;
                            }*/ // TODO erfc not available under windows?
                            this->potential.Peek()[posIdx] = 1.0f;
                        }
                    }
                }
            }
        }
    }

    return true;
}

float PotentialCalculator::computeChargeWeightedStructureFactor(
        uint maxWaveLength,
        MolecularDataCall *mol) {

//    float S = 0.0f;
//    for (uint k = 0; k < maxWaveLength; ++k) {
//        for (uint p = 0; p < mol->AtomCount(); ++p) {
//            S +=
//        }
//    }
//    return S;
            return 0.0f;
}


/*
 * PotentialCalculator::getPeriodicImages
 */
void PotentialCalculator::getPeriodicImages(Vec3f atomPos,
        vislib::Array<Vec3f> &imgArr) {

    if (this->bboxType == BBOX_CUBIC) {
        imgArr.SetCount(6*3); // 6 images per atom

        // x+1, y, z
        imgArr.Add(Vec3f(atomPos.X() + this->particleBBox.Width(), atomPos.Y(),
                atomPos.Z()));
        // x-1, y, z
        imgArr.Add(Vec3f(atomPos.X() - this->particleBBox.Width(), atomPos.Y(),
                atomPos.Z()));
        // x, y+1, z
        imgArr.Add(Vec3f(atomPos.X(), atomPos.Y() + this->particleBBox.Height(),
                atomPos.Z()));
        // x, y-1, z
        imgArr.Add(Vec3f(atomPos.X(), atomPos.Y() - this->particleBBox.Height(),
                atomPos.Z()));
        // x, y, z+1
        imgArr.Add(Vec3f(atomPos.X(), atomPos.Y(),
                atomPos.Z() + this->particleBBox.Depth()));
        // x, y, z-1
        imgArr.Add(Vec3f(atomPos.X(), atomPos.Y(),
                atomPos.Z() - this->particleBBox.Depth()));
    } else if (this->bboxType == BBOX_TRUNCATED_OCTAHEDRON) {
        // TODO
    }

}


/*
 * PotentialCalculator::release
 */
void PotentialCalculator::release(void) {
    this->charges.Release();
    this->potential.Release();
    this->particlePos.Release();
    this->particleCharges.Release();
    this->chargesBuff.Release();
#if (defined(WITH_CUDA) && (WITH_CUDA))
    if (this->cudaqsurf != NULL) {
        CUDAQuickSurf *cqs = (CUDAQuickSurf *)this->cudaqsurf;
        delete cqs;
    }
    cudaDeviceReset();
#endif
}


/*
 * PotentialCalculator::updateParams()
 */
void PotentialCalculator::updateParams() {

    // Parameter to choose the computational method for the potential map
    if (this->computationalMethodSlot.IsDirty()) {
        this->computationalMethodSlot.ResetDirty();
        this->computationalMethod = static_cast<ComputationalMethod>
            (this->computationalMethodSlot.Param<core::param::EnumParam>()->Value());
    }

    // Parameter for the grid resolution of the charge distribution
    if (this->chargesGridSpacingSlot.IsDirty()) {
        this->chargesGridSpacing = this->chargesGridSpacingSlot.Param<core::param::FloatParam>()->Value();
        this->chargesGridSpacingSlot.ResetDirty();
    }

    // Parameter for the grid resolution of the potential map
    if (this->potentialGridSpacingSlot.IsDirty()) {
        this->potentialGridSpacing = this->potentialGridSpacingSlot.Param<core::param::FloatParam>()->Value();
        this->potentialGridSpacingSlot.ResetDirty();
    }
}
