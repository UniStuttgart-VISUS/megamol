/*
 * SphereRenderer.h
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS). 
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_SPHERERENDERER_H_INCLUDED
#define MMPROTEINPLUGIN_SPHERERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "SphereDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/graphics/gl/GLSLShader.h"

namespace megamol {
namespace protein {

    /*
     * Simple Molecular Renderer class
     */

    class SphereRenderer : public megamol::core::view::Renderer3DModule
    {
    public:
        /** The coloring modes */
        enum ColoringMode {
            COLOR_TYPE    = 0,
            COLOR_CHARGE  = 1
        };

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) 
        {
            /// braunms: Changed name to prevent name ambiguity with core::moldyn::SphereRenderer in Configurartor
            return "MolecularSphereRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) 
        {
            return "Offers sphere renderings.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) 
        {
            return true;
        }

        /** Ctor. */
        SphereRenderer(void);

        /** Dtor. */
        virtual ~SphereRenderer(void);

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

       /**********************************************************************
        * 'render'-functions
        **********************************************************************/

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetExtents( megamol::core::Call& call);

        /**
         * The Open GL Render callback.
         *
         * @param call The calling call.
         * @return The return value of the function.
         */
        virtual bool Render( megamol::core::Call& call);

        /**
         * Compute the color array.
         *
         * @param sphere    The data call.
         * @param mode      The coloring mode.
         */
        void ComputeColors( const SphereDataCall *sphere, ColoringMode mode);

        /**********************************************************************
         * variables
         **********************************************************************/

        // caller slot
        megamol::core::CallerSlot sphereDataCallerSlot;

        // camera information
        vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;

        // parameter slots
        megamol::core::param::ParamSlot coloringModeParam;
        megamol::core::param::ParamSlot minValueParam;
        megamol::core::param::ParamSlot maxValueParam;

        // shader for the spheres (raycasting view)
        vislib::graphics::gl::GLSLShader sphereShader;
        // shader for the cylinders (raycasting view)
        vislib::graphics::gl::GLSLShader cylinderShader;

        // attribute locations for GLSL-Shader
        GLint attribLocInParams;
        GLint attribLocQuatC;
        GLint attribLocColor1;
        GLint attribLocColor2;

        // the color array
        vislib::Array<float> colors;
    };


} /* end namespace protein */
} /* end namespace megamol */

#endif // MMPROTEINPLUGIN_SPHERERENDERER_H_INCLUDED
