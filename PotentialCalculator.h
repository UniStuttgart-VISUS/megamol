//
// PotentialCalculator.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 19, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINPLUGIN_POTENTIALCALCULATOR_H_INCLUDED
#define MMPROTEINPLUGIN_POTENTIALCALCULATOR_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif // (defined(_MSC_VER) && (_MSC_VER > 1000))

#include "Module.h"
#include "CallerSlot.h"
#include "CalleeSlot.h"
#include "param/ParamSlot.h"
#include "gridParams.h"
#include "MolecularDataCall.h"
#include "HostArr.h"
#include "CudaDevArr.h"
#include "vislib_vector_typedefs.h"

typedef unsigned int uint;

namespace megamol {
namespace protein {

class PotentialCalculator : public core::Module {
public:

    /// Enum representing the different algorithms used to compute the electro-
    /// static potential
    enum ComputationalMethod {DIRECT_COULOMB_SUMMATION_NO_PERIODIC_BOUNDARIES=0,
        DIRECT_COULOMB_SUMMATION, PARTICLE_MESH_EWALD,
        CONTINUUM_SOLVATION_POISSON_BOLTZMAN, GPU_POISSON_SOLVER,
        EWALD_SUMMATION
    };

    /// Enum for different types of periodic bounding boxes
    enum BBoxType {BBOX_CUBIC=0, BBOX_TRUNCATED_OCTAHEDRON};

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "PotentialCalculator";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Module calculating the electrostatic potential of molecular \
                data using different algorithms.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /**
     * Disallow usage in quickstarts
     *
     * @return false
     */
    static bool SupportQuickstart(void) {
        return false;
    }

    /**
     * Ctor
     */
    PotentialCalculator();

    /**
     * Dtor
     */
    virtual ~PotentialCalculator();

protected:

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Call callback to get the data
     *
     * @param call The calling call
     * @return True on success
     */
    bool getData(core::Call& call);

    /**
     * Call callback to get the extent of the data
     *
     * @param call The calling call
     * @return True on success
     */
    bool getExtent(core::Call& call);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

private:

    /**
     * TODO
     */
    bool computeChargeDistribution(const MolecularDataCall *mol);

    /**
     * TODO
     */
    bool computePotentialMap(const MolecularDataCall *mol);

    /**
     * TODO
     */
    bool computePotentialMapDCS(const MolecularDataCall *mol,
            float sphericalCutOff,
            bool usePeriodicImages=false);

    /**
     * TODO
     */
    bool computePotentialMapEwaldSum(const MolecularDataCall *mol,
            float beta);

    /**
     * TODO
     */
    float computeChargeWeightedStructureFactor(uint maxWaveLength,
            MolecularDataCall *mol);

    /**
     * TODO
     */
    void getPeriodicImages(Vec3f atomPos,
            vislib::Array<Vec3f> &imgArr);

    /**
     * TODO
     */
    void initGridParams(gridParams &grid, MolecularDataCall *dcOut);

    /**
     * TODO
     */
    void updateParams();

    core::CallerSlot dataCallerSlot;  ///> Data caller slot
    core::CalleeSlot dataCalleeSlot;  ///> Data callee slot

    /// Parameter for the potential calculation method
    core::param::ParamSlot computationalMethodSlot;
    ComputationalMethod computationalMethod;

    /// Parameter for the bounding box type
    core::param::ParamSlot bboxTypeSlot;
    BBoxType bboxType;

    /// Parameter for the grid resolution for the charge distribution
    core::param::ParamSlot chargesGridSpacingSlot;
    float chargesGridSpacing;

    /// Parameter for the grid resolution for the potential map
    core::param::ParamSlot potentialGridSpacingSlot;
    float potentialGridSpacing;

#if (defined(WITH_CUDA) && (WITH_CUDA))
    void *cudaqsurf;             ///> Pointer to CUDAQuickSurf objects
#endif
    gridParams chargesGrid;         ///> Grid params for the charge distribution
    HostArr<float> charges;         ///> The charge distribution
    HostArr<float> particlePos;     ///> Array containing the grid positions
    HostArr<float> particleCharges; ///> Array containing the charges
    HostArr<float> chargesBuff;     ///> Array containing intermediate data
    float maxParticleRad;           ///> Maximum particle radius
    HostArr<float> atomData;        ///> Array containing particle data (xyzq)

    /// The (cubic) bounding box of the particle data
    vislib::math::Cuboid<float> particleBBox;

    gridParams potentialGrid;       ///> Grid params for the potential map
    HostArr<float> potential;       ///> The potential map (host memory)
#ifdef WITH_CUDA
    CudaDevArr<float> potential_D;  ///> The potential map (device memory)
#endif // WITH_CUDA
    float minPotential;             ///> Minimum potential value
    float maxPotential;             ///> Maximum potential value

    /// The bounding box of the generated potential map
    vislib::math::Cuboid<float> potentialBBox;

    bool jobDone;       ///> Flag whether the job is done
};

} // end namespace protein
} // end namespace megamol

#endif // MMPROTEINPLUGIN_POTENTIALCALCULATOR_H_INCLUDED
