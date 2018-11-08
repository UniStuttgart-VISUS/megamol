//
// SecPlaneRenderer.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Sep 13, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINPLUGIN_SECPLANERENDERER_H_INCLUDED
#define MMPROTEINPLUGIN_SECPLANERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/param/ParamSlot.h"
#include "vislib/graphics/gl/GLSLShader.h"

namespace megamol {
namespace protein {

/*
 * Class that renders sectional planes of 3D textures.
 */
class SecPlaneRenderer : public core::view::Renderer3DModule {
public:

    enum ShadingMode {DENSITY=0, POTENTIAL, LIC, ISOLINES};

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char *ClassName(void) {
        return "SecPlaneRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char *Description(void) {
        return "Offers molecule renderings.";
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
    SecPlaneRenderer(void);

    /** Dtor. */
    virtual ~SecPlaneRenderer(void);

protected:

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(megamol::core::Call& call);

    /**
     * Initializes the random noise texture used for LIC.
     *
     * @return 'True' on success, 'false' otherwise
     */
    bool initLIC();

    /**
     * Implementation of 'release'.
     */
    virtual void release(void);

    /**
     * The Open GL Render callback.
     *
     * @param call The calling call.
     * @return The return value of the function.
     */
    virtual bool Render(megamol::core::Call& call);

private:

    /// Data caller slot
    core::CallerSlot textureSlot;

    /// The data bounding box
    core::BoundingBoxes bbox;

    /// The OpenGL texture handle
    GLuint tex;


    /* Parameter slots */

    /// Parameter slot for shading mode
    core::param::ParamSlot shadingSlot;

    /// Parameter slot for the minimum texture value (used for shading)
    core::param::ParamSlot shadingMinTexSlot;

    /// Parameter slot for the minimum texture value (used for shading)
    core::param::ParamSlot shadingMaxTexSlot;

    /// Parameter slot for LIC contrast
    core::param::ParamSlot licContrastSlot;

    /// Parameter slot for LIC licBrightness
    core::param::ParamSlot licBrightnessSlot;

    /// Parameter slot for LIC stepsize scale factor
    core::param::ParamSlot licDirSclSlot;

    /// Parameter slor for LIC random noise texture coordinates scale
    core::param::ParamSlot licTCSclSlot;

    /// Parameter for isovalue
    core::param::ParamSlot isoValueSlot;

    /// Parameter for isoline threshold
    core::param::ParamSlot isoThreshSlot;

    /// Parameter for isoline distribution
    core::param::ParamSlot isoDistributionSlot;

    /// Parameter slot for x-Plane
    core::param::ParamSlot xPlaneSlot;

    /// Parameter slot for y-Plane
    core::param::ParamSlot yPlaneSlot;

    /// Parameter slot for z-Plane
    core::param::ParamSlot zPlaneSlot;

    /// Parameter slot to toggle x-Plane
    core::param::ParamSlot toggleXPlaneSlot;

    /// Parameter slot to toggle y-Plane
    core::param::ParamSlot toggleYPlaneSlot;

    /// Parameter slot to toggle z-Plane
    core::param::ParamSlot toggleZPlaneSlot;


    /* Rendering */

    /// Shader for slice rendering
    vislib::graphics::gl::GLSLShader sliceShader;

    /// Random noise texture for LIC
    GLuint randNoiseTex;

};


} // namespace protein
} // namespace megamol

#endif // MMPROTEINPLUGIN_SECPLANERENDERER_H_INCLUDED
