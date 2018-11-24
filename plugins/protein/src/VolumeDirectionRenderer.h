/*
 * VolumeDirectionRenderer.h
 *
 * Copyright (C) 2013 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_VOLDIRRENDERER_H_INCLUDED
#define MMPROTEINPLUGIN_VOLDIRRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "protein_calls/VTIDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer3DModuleDS.h"
#include "mmcore/view/AbstractCallRender3D.h"

#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"


namespace megamol {
namespace protein {

    /*
     * Simple Molecular Renderer class
     */

    class VolumeDirectionRenderer : public megamol::core::view::Renderer3DModuleDS
    {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "VolumeDirectionRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Offers arrow rendering for volumetric data.";
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
        VolumeDirectionRenderer(void);

        /** Dtor. */
        virtual ~VolumeDirectionRenderer(void);

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
         * Update all parameter slots.
         *
         * @param mol   Pointer to the data call.
         */
		void UpdateParameters(const protein_calls::VTIDataCall *vti = 0);


        /**********************************************************************
         * variables
         **********************************************************************/
        
        /** VTIData caller slot */
        megamol::core::CallerSlot vtiDataCallerSlot;

        /** parameter slots */
        unsigned int arrowCount;
        /** arrow lenght scale parameter slots */
        core::param::ParamSlot lengthScaleParam;
        /** lenght filter param slot */
        core::param::ParamSlot lengthFilterParam;
        /** lenght filter param slot */
        core::param::ParamSlot minDensityFilterParam;

        /** camera information */
        vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;

        /** the arrow shader */
        vislib::graphics::gl::GLSLShader arrowShader;

        /** trigger recomputation of arrow arrays */
        bool triggerArrowComputation;

        /** data arrays for rendering */
        vislib::Array<float> vertexArray;
        vislib::Array<float> colorArray;
        vislib::Array<float> dirArray;
        float minC, maxC;
        
        /** The call for Transfer function */
        megamol::core::CallerSlot getTFSlot;
        
        /** A simple black-to-white transfer function texture as fallback */
        unsigned int greyTF;
        
        int datahash;
    };


} /* end namespace protein */
} /* end namespace megamol */

#endif // MMPROTEINPLUGIN_VOLDIRRENDERER_H_INCLUDED
