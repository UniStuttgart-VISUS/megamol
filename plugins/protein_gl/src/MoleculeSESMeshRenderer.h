/*
 * MoleculeSESMeshRenderer.h
 *
 * Copyright (C) 2009-2021 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#pragma once

#include <algorithm>
#include <list>
#include <set>
#include <tuple>
#include <vector>

#include <glowl/glowl.h>

#include "geometry_calls_gl/CallTriMeshDataGL.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Camera.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"
#include "protein/Icosphere.h"
#include "protein/ReducedSurface.h"
#include "protein_calls/BindingSiteCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "protein_calls/ProteinColor.h"
#include "protein_gl/DeferredRenderingProvider.h"
#include "vislib/Array.h"
#include "vislib/String.h"
#include "vislib/math/Quaternion.h"
#include "vislib_gl/graphics/gl/SimpleFont.h"

namespace megamol {
namespace protein_gl {

/**
 * Molecular Surface Renderer class.
 * Computes and renders the solvent excluded (Connolly) surface.
 */
class MoleculeSESMeshRenderer : public megamol::mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "MoleculeSESMeshRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Offers solvent excluded surface mesh rendering.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        // return true;
        return true;
    }

    /** ctor */
    MoleculeSESMeshRenderer(void);

    /** dtor */
    virtual ~MoleculeSESMeshRenderer(void);

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

    /** The data update hash */
    SIZE_T datahash;

private:
    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getTriangleDataCallback(core::Call& caller);

    /**
     * Finds nearest Vertice to the given one on an vector
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    std::vector<unsigned int> findNearestVertice(const std::vector<std::vector<unsigned int>>& edgelord,
        unsigned int& referenceIndex0, unsigned int& referenceIndex1, const std::vector<float>& vertex, int index);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getExtentCallback(core::Call& caller);

    /**
     * Update all parameter slots.
     *
     * @param mol   Pointer to the data call.
     */
    void UpdateParameters(
        const megamol::protein_calls::MolecularDataCall* mol, const protein_calls::BindingSiteCall* bs = 0);

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(mmstd_gl::CallRender3DGL& call);

    /**
     * Open GL Render call.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool Render(mmstd_gl::CallRender3DGL& call);

    void RenderAtoms(const megamol::protein_calls::MolecularDataCall* mol);
    void RenderReducedSurface(protein::ReducedSurface* rs);

    /**
     * Deinitialises this renderer. This is only called if there was a
     * successful call to "initialise" before.
     */
    virtual void deinitialise(void);

    /**********************************************************************
     * variables
     **********************************************************************/

    /** MolecularDataCall caller slot */
    megamol::core::CallerSlot molDataCallerSlot;
    /** BindingSiteCall caller slot */
    megamol::core::CallerSlot bsDataCallerSlot;
    /** Light data caller slot */
    megamol::core::CallerSlot getLightsSlot;

    /** The slot for requesting data */
    core::CalleeSlot getTriangleDataSlot;

    /** camera information */
    // vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;
    core::view::Camera camera;

    /** framebuffer information */
    std::shared_ptr<glowl::FramebufferObject> fbo_;

    // camera information
    // vislib::SmartPtr<vislib::graphics::CameraParameters> MoleculeSESMeshRenderercameraInfo;
    core::view::Camera MoleculeSESMeshRenderercameraInfo;

    /** parameter slot for coloring mode */
    megamol::core::param::ParamSlot coloringModeParam0;
    /** parameter slot for coloring mode */
    megamol::core::param::ParamSlot coloringModeParam1;
    /** parameter slot for coloring mode weighting*/
    megamol::core::param::ParamSlot cmWeightParam;
    /** parameter slot for min color of gradient color mode */
    megamol::core::param::ParamSlot minGradColorParam;
    /** parameter slot for mid color of gradient color mode */
    megamol::core::param::ParamSlot midGradColorParam;
    /** parameter slot for max color of gradient color mode */
    megamol::core::param::ParamSlot maxGradColorParam;
    /** parameter slot for color table filename */
    megamol::core::param::ParamSlot colorTableFileParam;
    /** parameter slot for adjusting the probe radius */
    megamol::core::param::ParamSlot probeRadiusSlot;

    /** the reduced surface(s) */
    std::vector<protein::ReducedSurface*> reducedSurface;

    std::shared_ptr<glowl::GLSLProgram> sphereShader_;

    ////////////

    // the bounding box of the protein
    vislib::math::Cuboid<float> bBox;

    // epsilon value for float-comparison
    float epsilon;

    // radius of the probe atom
    float probeRadius;

    std::vector<glm::vec3> atomColorTable;
    unsigned int currentArray;

    /** 'true' if the data for the current render mode is computed, 'false' otherwise */
    bool preComputationDone;

    uint32_t atomCount_;

    /** The current coloring mode */
    protein_calls::ProteinColor::ColoringMode currentColoringMode0;
    protein_calls::ProteinColor::ColoringMode currentColoringMode1;

    /** vertex and color array for raycasting the spheres */
    std::vector<vislib::Array<float>> sphereVertexArray;
    std::vector<vislib::Array<float>> sphereColors;

    // width and height of view
    unsigned int width;
    unsigned int height;

    /** The color lookup table (for chains, amino acids,...) */
    std::vector<glm::vec3> colorLookupTable;
    std::vector<glm::vec3> fileLookupTable;
    /** The color lookup table which stores the rainbow colors */
    std::vector<glm::vec3> rainbowColors;

    // flag for SES computation (false = one SES per molecule)
    bool computeSesPerMolecule;
    glm::mat4 view_;
    glm::mat4 proj_;
    glm::mat4 invview_;
    glm::mat4 transview_;
    glm::mat4 invproj_;
    glm::mat4 invtransview_;
    glm::mat4 mvp_;
    glm::mat4 mvpinverse_;
    glm::mat4 mvptranspose_;

    std::unique_ptr<glowl::BufferObject> sphereVertexBuffer_;
    std::unique_ptr<glowl::BufferObject> sphereColorBuffer_;

    std::unique_ptr<glowl::BufferObject> pointLightBuffer_;
    std::unique_ptr<glowl::BufferObject> directionalLightBuffer_;

    GLuint vertexArraySphere_;

    std::vector<geocalls_gl::CallTriMeshDataGL::Mesh*> triaMesh;

    DeferredRenderingProvider deferredProvider_;
    static std::vector<std::vector<unsigned int>> getMultipleVertices(Icosphere* pIcosphere);
};

} // namespace protein_gl
} /* end namespace megamol */
