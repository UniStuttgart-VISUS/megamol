//
// ComparativeFieldTopologyRenderer.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Feb 07, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_FIELDTOPOLOGYRENDERER_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_FIELDTOPOLOGYRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/graphics/gl/GLSLGeometryShader.h"
#include "mmcore/CallerSlot.h"
#include "protein_calls/VTIDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Renderer3DModule.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/CameraParameters.h"
#include "vislib/math/Matrix.h"
#include "VecField3f.h"
#include "Streamline.h"

namespace megamol {
namespace protein_cuda {

class ComparativeFieldTopologyRenderer : public megamol::core::view::Renderer3DModule {

public:

    /** CTor */
    ComparativeFieldTopologyRenderer(void);

    /** DTor */
    virtual ~ComparativeFieldTopologyRenderer(void);

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "ComparativeFieldTopologyRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Renders topology information of vector fields.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

protected:

    enum StreamlineShading {UNIFORM=0, POTENTIAL};

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
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(core::Call& call);

    /**
     * Open GL Render call.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool Render(core::Call& call);

private:

    /**
     * (Re)calculate the electric field using the provided potential map.
     * The field direction is defined by -grad(potential). Partial derivatives
     * are obtained using finite differences.
     *
     * @param cmd        The data call
     * @param efield     Vector field object for the electrostatic field
     * @param egradfield Vector field object for the gradient of the electric
     *                   field strength
     */
    void calcElectrostaticField(protein_calls::VTIDataCall *cmd,
            VecField3f &efield, VecField3f &egradfield);

    /**
     * Render critical points using sphere glyph ray casting.
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool renderCritPointsSpheres();

    /**
     * Render arrow glyphs representing the electric field.
     *
     * @param cmd The data call.
     */
    bool renderFieldArrows(const protein_calls::VTIDataCall *cmd);

    /**
     * Renders a streamline bundle around the manually set seed point.
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool renderStreamlineBundleManual();

    /**
     * Renders streamlines in regions of interest, i.e. areas where the
     * two electrostatic fields differ.
     *
     * @param cmd0 The data call for the first field
     * @param cmd1 The data call for the second field
     * @return 'True' on success, 'false' otherwise
     */
    bool renderStreamlinesRoi(
            const protein_calls::VTIDataCall *cmd0,
            const protein_calls::VTIDataCall *cmd1);

    /**
     * Search for neighbours in a certain distance for all sources/sinks in
     * the electric field. Critical points without neighbours represent a change
     * in polarization and, therefore, a region of interest.
     */
    void findNeighbours();

    /**
     * Update all parameters and set boolean flags accordingly.
     */
    void updateParams();

    /** TODO */
    inline float GetPotentialAt(protein_calls::VTIDataCall *dc, unsigned int x, unsigned int y,
            unsigned int z) {
        return dc->GetPointDataByIdx(0, 0)[
                 dc->GetGridsize().X()*(dc->GetGridsize().Y()*z+y)+x];
    }


    /* Data caller slots */

    /// Data caller slots
    megamol::core::CallerSlot dataCallerSlot0, dataCallerSlot1;


    /* Parameters related to fog */

    /// Parameter for fog starting point
    core::param::ParamSlot fogZSlot;
    float fogZ;


    /* Parameters for arrow glyph rendering */

    /// Parameter arrow radius
    core::param::ParamSlot arrowRadSclSlot;
    float arrowRadScl;

    /// Parameter for arrow scale factor
    core::param::ParamSlot arrowLenSclSlot;
    float arrowLenScl;

    /// Parameter for max x filter
    core::param::ParamSlot arrowFilterXMaxSlot;
    float arrowFilterXMax;

    /// Parameter for max y filter
    core::param::ParamSlot arrowFilterYMaxSlot;
    float arrowFilterYMax;

    /// Parameter for max z filter
    core::param::ParamSlot arrowFilterZMaxSlot;
    float arrowFilterZMax;

    /// Parameter for min x filter
    core::param::ParamSlot arrowFilterXMinSlot;
    float arrowFilterXMin;

    /// Parameter for min y filter
    core::param::ParamSlot arrowFilterYMinSlot;
    float arrowFilterYMin;

    /// Parameter for min z filter
    core::param::ParamSlot arrowFilterZMinSlot;
    float arrowFilterZMin;


    /* Parameters for critical point analysis */

    /// Parameter forsphere radius
    core::param::ParamSlot critPointsSphereSclSlot;
    float critPointsSphereScl;

    /// Parameter to determine the maximum number of bisections
    core::param::ParamSlot critPointsMaxBisectionsSlot;
    unsigned int critPointsMaxBisections;

    /// Parameter for maximum number of Newton iterations
    core::param::ParamSlot critPointsNewtonMaxStepsSlot;
    unsigned int critPointsNewtonMaxSteps;

    /// Parameter for the stepsize of the Newton iteration
    core::param::ParamSlot critPointsNewtonStepsizeSlot;
    float critPointsNewtonStepsize;

    /// Parameter for the epsilon for the Newton iteration
    core::param::ParamSlot critPointsNewtonEpsSlot;
    float critPointsNewtonEps;

    /// param slot to determine whether all critpints are to be shown
    core::param::ParamSlot critPointsShowAllSlot;
    bool critPointsShowAll;


    /* Parameters for streamlines */

    /// Parameter for streamline maximum steps
    core::param::ParamSlot streamlineMaxStepsSlot;
    unsigned int streamlineMaxSteps;

    /// Parameter to determine streamline shading
    core::param::ParamSlot streamlineShadingSlot;
    StreamlineShading streamlineShading;

    /// Parameter to set the radius of the streamline bundle
    core::param::ParamSlot streamBundleRadSlot;
    float streamBundleRad;

    /// Parameter to set the resolution of the streamline bundle
    core::param::ParamSlot streamBundleResSlot;
    unsigned int streamBundleRes;

    /// Parameter to set the step size of the streamline bundle
    core::param::ParamSlot streamBundlePhiSlot;
    float streamBundlePhi;

    /// Parameter to set the epsilon for stream line terminations
    core::param::ParamSlot streamlineEpsSlot;
    float streamlineEps;

    /// Parameter to set the stepsize for streamline integration
    core::param::ParamSlot streamlineStepsizeSlot;
    float streamlineStepsize;

    /// Parameter to toggle rendering of streamlines
    core::param::ParamSlot toggleStreamlinesSlot;
    bool toggleStreamlines;

    /// Parameter to show all streamlines
    core::param::ParamSlot streamlinesShowAllSlot;
    bool streamlinesShowAll;


    /* Parameters for manually set streamline seed point */

    /// Parameter for x coord of streamline seed point
    core::param::ParamSlot streamBundleSeedXSlot;
    float streamBundleSeedX;

    /// Parameter for y coord of streamline seed point
    core::param::ParamSlot streamBundleSeedYSlot;
    float streamBundleSeedY;

    /// Parameter for z coord of streamline seed point
    core::param::ParamSlot streamBundleSeedZSlot;
    float streamBundleSeedZ;

    /// Parameter for streamline maximum steps
    core::param::ParamSlot streamlineMaxStepsManualSlot;
    unsigned int streamlineMaxStepsManual;

    /// Parameter to determine streamline shading
    core::param::ParamSlot streamlineShadingManualSlot;
    StreamlineShading streamlineShadingManual;

    /// Parameter to set the radius of the streamline bundle
    core::param::ParamSlot streamBundleRadManualSlot;
    float streamBundleRadManual;

    /// Parameter to set the resolution of the streamline bundle
    core::param::ParamSlot streamBundleResManualSlot;
    unsigned int streamBundleResManual;

    /// Parameter to set the step size of the streamline bundle
    core::param::ParamSlot streamBundlePhiManualSlot;
    float streamBundlePhiManual;

    /// Parameter to set the epsilon for stream line terminations
    core::param::ParamSlot streamlineEpsManualSlot;
    float streamlineEpsManual;

    /// Parameter to set the stepsize for streamline integration
    core::param::ParamSlot streamlineStepsizeManualSlot;
    float streamlineStepsizeManual;

    /// Parameter to toggle rendering of streamlines based on manual seed
    core::param::ParamSlot toggleStreamlinesManualSlot;
    bool toggleStreamlinesManual;


    /* Parameters for finding regions of interest */

    /// Param slot for maximum euclidean distance between critpoints
    core::param::ParamSlot roiMaxDistSlot;
    float roiMaxDist;


    /* Parameters for debugging purposes */

    /// Param for minimum potential on texture slice
    core::param::ParamSlot texMinValSlot;
    float texMinVal;

    /// Param for maximum potential on texture slice
    core::param::ParamSlot texMaxValSlot;
    float texMaxVal;

    /// Param for z position of the texture
    core::param::ParamSlot texPosZSlot;
    float texPosZ;


    /* Electrostatics related data */

    /// The electric fields
    VecField3f efield0, efield1;

    /// The gradients of the electric field strength
    VecField3f egradfield0, egradfield1;

    /// Texture containing the potential map
    GLuint potentialTex0, potentialTex1;


    /* GLSL shader objects */

    /// Shader for rendering arrows
    vislib::graphics::gl::GLSLGeometryShader arrowShader;

    /// Shader for rendering spheres
    vislib::graphics::gl::GLSLGeometryShader sphereShader;

    /// Shader for illuminated streamlines
    vislib::graphics::gl::GLSLShader streamlineShader;

    /// Shader for testure slices
    vislib::graphics::gl::GLSLShader sliceShader;


    /* Arrow glyph data */

    /// Array conatining arrow glyph orientations
    vislib::Array<float> arrowData;

    /// Array containing arrow positions
    vislib::Array<float> arrowDataPos;

    /// Array containing arrow glyph colors
    vislib::Array<float> arrowCol;

    /// Array containing indices of visible arrows
    vislib::Array<int> arrowVisIdx;

    /// Array containing grid positions on which the arrows are defined
    vislib::Array <vislib::math::Vector<float, 3> > gridPos;


    /* Streamline related data */

    /// Arrays containing indices of ending streamlines for every cell
    vislib::Array<vislib::Array<unsigned int > > cellEndpoints0, cellEndpoints1;

    /// Arrays containing indices of ending streamlines for every cell (backward)
    vislib::Array<vislib::Array<unsigned int > > cellEndpointsBackward0, cellEndpointsBackwards1;

    /// Array containing streamline objects
    vislib::Array<Streamline*> streamlines0, streamlines1;

    /// Array containing streamline objects based on the manually set seedpoint
    vislib::Array<Streamline*> streamlinesManualSeed0, streamlinesManualSeed1;

    /// Arrays containing starting points for streamlines
    vislib::Array<float> seedPosStart0, seedPosStart1;

    /// Arrays containing ending cells for all seed points
    vislib::Array<unsigned int> seedCellEnd0, seedCellEnd1;

    /// Arrays containing ending cells for all seed points (backward)
    vislib::Array<unsigned int> seedCellEndBackward0, seedCellEndBackward1;


    /* Critical point related data */

    /// List with pointers to partner critpoints in different data sets
    vislib::Array<vislib::Array< const VecField3f::CritPoint* > > neighbours0, neighbours1;


    /* Boolean flags */

    /// Triggers recalculation of the elctrostatic field
    bool recalcEfield;

    /// Triggers recalculation of grid positions
    //bool recalcGridpos;

    /// Triggers recalculation of streamlines
    bool recalcStreamlines;

    /// Triggers recalculation of critical points
    bool recalcCritPoints;

    /// Triggers recalculation of streamlines (manual seed point)
    bool recalcStreamlinesManualSeed;

    /// Triggers neigbbour search for sinks/sources
    bool recalcNeighbours;

    /// Triggers recalculation of arrow data
    bool recalcArrowData;


    /* Misc */

    /// Hold information about the current viewport
    float viewportStuff[4];

    /// The current modelview matrix
    GLfloat modelMatrix[16];

    /// The current projection matrix
    GLfloat projMatrix [16];

    /// The light position
    GLfloat lightPos[4];

    /// The data sets bounding box
    vislib::math::Cuboid<float> bbox;

    /// Hash value of the current data set
    SIZE_T dataHash;

    /// Camera information
    vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;

};

} // namespace protein_cuda
} // namespace megamol

#endif // MMPROTEINCUDAPLUGIN_ComparativeFieldTopologyRenderer_H_INCLUDED
