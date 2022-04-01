#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore_gl/view/CallRender3DGL.h"
#include "mmcore_gl/view/Renderer3DModuleGL.h"
#include "protein_calls/MolecularDataCall.h"
#include "protein_calls/ProteinColor.h"
#include "protein_gl/DeferredRenderingProvider.h"

#include "quicksurf/CUDAQuickSurf.h"

namespace megamol::protein_cuda {
class QuickSurf : public megamol::core_gl::view::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "QuickSurf";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Offers rendering of gaussian density surfaces as well as the calculation of the rendered volume data";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /** Ctor. */
    QuickSurf(void);

    /** Dtor. */
    virtual ~QuickSurf(void);

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

private:
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
     * The Open GL Render callback.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool Render(megamol::core_gl::view::CallRender3DGL& call);

    /**
     * Calculates whether some of the parameters are dirty and we have to recalculate the surface/volume
     *
     * @param mdc The already filled input data call
     * @return True, if at least one parameter has been changed, false otherwise.
     */
    bool HandleParameters(protein_calls::MolecularDataCall& mdc);

    /**
     * Calculates the QuickSurf surface for given molecular data
     *
     * @param mdc The call containing the molecular data
     * @param quality The quality identifier. 0 means minimum quality, 3 maximum quality.
     * @param radscale Scaling factor for the radial influence of each particle
     * @param gridspacing Spacing factor for the
     * @param isoval The isovalue of the calculated surface
     * @param sortTriangles If true, the resulting triangles will be sorted according to their view space depth
     *
     * @return True on success, false otherwise
     */
    bool calculateSurface(protein_calls::MolecularDataCall& mdc, int quality, float radscale, float gridspacing,
        float isoval, bool sortTriangles);

    /**
     *
     */
    bool calculateVolume();

    /** The data input slot */
    core::CallerSlot dataInSlot_;

    /** Input slot for the light information */
    core::CallerSlot lightInSlot_;

    /** Parameter slot for the quicksurf quality */
    core::param::ParamSlot qs_qualityParam_;

    /** Parameter slot for the quicksurf radius scaling */
    core::param::ParamSlot qs_radScaleParam_;

    /** Parameter slot for the quicksurf grid spacing */
    core::param::ParamSlot qs_gridSpacingParam_;

    /** Parameter slot for the isovalue of the quicksurf surface */
    core::param::ParamSlot qs_isovalueParam_;

    /** Parameter slot for the file path to the file containing the color table */
    core::param::ParamSlot colorTableFileParam_;

    /** Parameter slot for the first coloring mode */
    core::param::ParamSlot coloringMode0Param_;

    /** Parameter slot for the second coloring mode */
    core::param::ParamSlot coloringMode1Param_;

    /** Parameter slot for the weighting factor between the coloring modes */
    core::param::ParamSlot coloringModeWeightParam_;

    /** Parameter slot for the color corresponding to the minimum value */
    core::param::ParamSlot minGradColorParam_;

    /** Parameter slot for the color corresponding to the mid value */
    core::param::ParamSlot midGradColorParam_;

    /** Parameter slot for the color corresponding to the maximum value */
    core::param::ParamSlot maxGradColorParam_;

    /** Submodule for the deferred rendering */
    protein_gl::DeferredRenderingProvider deferredProvider_;

    /** Pointer to the quicksurf provider */
    std::unique_ptr<CUDAQuickSurf> cudaqsurf_;

    /** Voxel size of the quicksurf grid */
    glm::ivec3 numVoxels_;

    /** The real x-axis of the quicksurf grid */
    glm::vec3 xAxis_;

    /** The real y-axis of the quicksurf grid */
    glm::vec3 yAxis_;

    /** The real z-axis of the quicksurf grid */
    glm::vec3 zAxis_;

    /** The origin of the quicksurf grid */
    glm::vec3 gridOrigin_;

    /** Vector containing all necessary vertex buffers */
    std::vector<QuickSurfGraphicBuffer> vertBuffers_;

    /** Name of the used vertex array object */
    std::vector<GLuint> vaoHandles_;

    /** Pointer to the shader program for the surface mesh */
    std::unique_ptr<glowl::GLSLProgram> meshShader_;

    /** Flag determining whether we have read a color table from file */
    bool tableFromFile_;

    /** Color table for the atoms */
    std::vector<glm::vec3> atomColorTable_;

    /** Color table read from file */
    std::vector<glm::vec3> fileColorTable_;

    /** Rainbow color table */
    std::vector<glm::vec3> rainbowColorTable_;

    /** The currently used first coloring mode */
    protein_calls::ProteinColor::ColoringMode currentColoringMode0_;

    /** The currently used second coloring mode */
    protein_calls::ProteinColor::ColoringMode currentColoringMode1_;

    /** Flag determining whether we have to recompute the color tables */
    bool refreshColors_;

    /** Last data hash of the source module that was calculated */
    size_t lastHash_;

    /** Last frame ID of the source module that was calculated */
    size_t lastFrame_;
};
} // namespace megamol::protein_cuda
