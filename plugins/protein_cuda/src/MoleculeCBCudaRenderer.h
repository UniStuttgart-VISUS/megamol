/*
 * MoleculeCBCudaRenderer.h
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_MOLSURFREN_CONTOURBUILDUP_CUDA_H_INCLUDED
#define MEGAMOL_MOLSURFREN_CONTOURBUILDUP_CUDA_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#include "glowl/BufferObject.hpp"
#include "glowl/GLSLProgram.hpp"
#include "glowl/Texture2D.hpp"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore_gl/view/Renderer3DModuleGL.h"
#include "protein_calls/MolecularDataCall.h"
#include "vislib_gl/graphics/gl/GLSLShader.h"
#include <memory>
#include <vector>

#include "cuda_runtime_api.h"
#include "particles_kernel.cuh"
#include "vector_functions.h"

namespace megamol {
namespace protein_cuda {

/**
 * Molecular Surface Renderer class.
 * Computes and renders the solvent excluded (Connolly) surface
 * using the Contour-Buildup Algorithm by Totrov & Abagyan.
 */
class MoleculeCBCudaRenderer : public megamol::core_gl::view::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "MoleculeCBCudaRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Offers molecular surface renderings.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** ctor */
    MoleculeCBCudaRenderer(void);

    /** dtor */
    virtual ~MoleculeCBCudaRenderer(void);

    /**********************************************************************
     * 'get'-functions
     **********************************************************************/

    /** Get probe radius */
    const float GetProbeRadius() const {
        return probeRadius;
    };

    /**********************************************************************
     * 'set'-functions
     **********************************************************************/

    /** Set probe radius */
    void SetProbeRadius(const float rad) {
        probeRadius = rad;
    };

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * Implementation of 'release'.
     */
    virtual void release(void);

    /**
     * Initialize CUDA
     */
    bool initCuda(megamol::protein_calls::MolecularDataCall* mol, uint gridDim, core_gl::view::CallRender3DGL* cr3d);

    /**
     * Write atom positions and radii to an array for processing in CUDA
     */
    void writeAtomPositions(const megamol::protein_calls::MolecularDataCall* mol);

    /**
     * Write atom positions and radii to a VBO for processing in CUDA
     */
    void writeAtomPositionsVBO(megamol::protein_calls::MolecularDataCall* mol);

private:
    // This function returns the best GPU (with maximum GFLOPS)
    VISLIB_FORCEINLINE int cudaUtilGetMaxGflopsDeviceId() const {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);

        cudaDeviceProp device_properties;
        int max_gflops_device = 0;
        int max_gflops = 0;

        int current_device = 0;
        cudaGetDeviceProperties(&device_properties, current_device);
        max_gflops = device_properties.multiProcessorCount * device_properties.clockRate;
        ++current_device;

        while (current_device < device_count) {
            cudaGetDeviceProperties(&device_properties, current_device);
            int gflops = device_properties.multiProcessorCount * device_properties.clockRate;
            if (gflops > max_gflops) {
                max_gflops = gflops;
                max_gflops_device = current_device;
            }
            ++current_device;
        }

        return max_gflops_device;
    }

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(megamol::core_gl::view::CallRender3DGL& call);

    /**
     * Open GL Render call.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool Render(megamol::core_gl::view::CallRender3DGL& call);

    /**
     * CUDA version of contour buildup algorithm
     *
     * TODO
     *
     */
    void ContourBuildupCuda(megamol::protein_calls::MolecularDataCall* mol);

    /**
     * Update all parameter slots.
     *
     * @param mol   Pointer to the data call.
     */
    void UpdateParameters(const megamol::protein_calls::MolecularDataCall* mol);

    /**
     * Deinitialises this renderer. This is only called if there was a
     * successful call to "initialise" before.
     */
    virtual void deinitialise(void);

    /**********************************************************************
     * variables
     **********************************************************************/

    // caller slot
    megamol::core::CallerSlot molDataCallerSlot;

    // parameter slots
    megamol::core::param::ParamSlot probeRadiusParam;
    megamol::core::param::ParamSlot opacityParam;
    megamol::core::param::ParamSlot stepsParam;

    // camera information
    core::view::Camera cameraInfo;

    std::shared_ptr<glowl::GLSLProgram> sphereShader_;
    std::shared_ptr<glowl::GLSLProgram> torusShader_;
    std::shared_ptr<glowl::GLSLProgram> sphericalTriangleShader_;

    GLuint sphereVAO_;
    GLuint torusVAO_;
    GLuint sphericalTriangleVAO_;

    // the bounding box of the protein
    vislib::math::Cuboid<float> bBox;

    // radius of the probe atom
    float probeRadius;

    // max number of neighbors per atom
    const unsigned int atomNeighborCount;

    // params
    bool cudaInitalized;
    uint numAtoms;
    SimParams params;
    uint3 gridSize;
    uint numGridCells;

    // CPU data
    std::vector<glm::vec4> hPos_;

    // GPU data
    float* m_dPos;
    float* m_dSortedPos;
    float* m_dSortedProbePos;
    uint* m_dNeighborCount;
    uint* m_dNeighbors;
    float* m_dSmallCircles;
    uint* m_dSmallCircleVisible;
    uint* m_dSmallCircleVisibleScan;
    float* m_dArcs;
    uint* m_dArcIdxK;
    uint* m_dArcCount;
    uint* m_dArcCountScan;

    // grid data for sorting method
    uint* m_dGridParticleHash;  // grid hash value for each particle
    uint* m_dGridParticleIndex; // particle index for each particle
    uint* m_dGridProbeHash;     // grid hash value for each probe
    uint* m_dGridProbeIndex;    // particle index for each probe
    uint* m_dCellStart;         // index of start of each cell in sorted list
    uint* m_dCellEnd;           // index of end of cell
    uint gridSortBits;

    enum class Buffers : GLuint {
        PROBE_POS = 0,
        SPHERE_TRIA_VEC_1 = 1,
        SPHERE_TRIA_VEC_2 = 2,
        SPHERE_TRIA_VEC_3 = 3,
        TORUS_POS = 4,
        TORUS_VS = 5,
        TORUS_AXIS = 6,
        SING_TEX = 7,
        TEX_COORD = 8,
        ATOM_POS = 9,
        BUFF_COUNT = 10
    };
    std::array<std::unique_ptr<glowl::BufferObject>, static_cast<int>(Buffers::BUFF_COUNT)> buffers_;

    // singularity texture
    std::unique_ptr<glowl::Texture2D> singTex_;

    // maximum number of probe neighbors
    uint probeNeighborCount;
    unsigned int texHeight;
    unsigned int texWidth;
    unsigned int width;
    unsigned int height;

    bool setCUDAGLDevice;
};

} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif /* MEGAMOL_MOLSURFACERENDERERCONTOURBUILDUP_CUDA_H_INCLUDED */
