/*
 * SolventRenderer.h
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MEGAMOLCORE_SOLVENTRENDERER_H_INCLUDED
#define MEGAMOLCORE_SOLVENTRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CallProteinData.h"
#include "CallFrame.h"
#include "param/ParamSlot.h"
#include "CallerSlot.h"
#include "view/Renderer3DModule.h"
#include "view/CallRender3D.h"
#include "vislib/GLSLShader.h"
#include "vislib/String.h"
#include <vector>

namespace megamol {
namespace protein {

	/*
     * Solvent Renderer class
     */

	class SolventRenderer : public megamol::core::view::Renderer3DModule
	{
	public:
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) 
		{
            return "SolventRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) 
		{
            return "Offers solvent renderings.";
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
        SolventRenderer(void);

        /** Dtor. */
        virtual ~SolventRenderer(void);

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
         * Refresh all dirty parameter values.
         */
        void ParameterRefresh();

	   /**********************************************************************
		* 'render'-functions
	    **********************************************************************/

        /**
         * The get capabilities callback. The module should set the members
         * of 'call' to tell the caller its capabilities.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetCapabilities( megamol::core::Call& call);

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

		/**********************************************************************
		 * variables
		 **********************************************************************/

        // caller slot
		megamol::core::CallerSlot protDataCallerSlot;

		// camera information
		vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;

        megamol::core::param::ParamSlot radiusScaleParam;
        megamol::core::param::ParamSlot elementFilterParam;
        megamol::core::param::ParamSlot minChargeParam;
        megamol::core::param::ParamSlot maxChargeParam;
        megamol::core::param::ParamSlot distanceParam;

		// shader for the spheres (raycasting view)
		vislib::graphics::gl::GLSLShader sphereShader;
		// shader for the cylinders (raycasting view)
		vislib::graphics::gl::GLSLShader cylinderShader;

		// attribute locations for GLSL-Shader
		GLint attribLocInParams;
		GLint attribLocQuatC;
		GLint attribLocColor1;
		GLint attribLocColor2;

        // radius scale
        float radiusScale;
        // element filter string
        vislib::StringA elementFilter;
        // minimum occupancy
        float minCharge;
        // mmaximum occupancy
        float maxCharge;
        // distance
        float distance;

        // array for atom positions
        float *atomPos;
        // size of the atom position array
        unsigned int atomPosSize;
        // array for atom colors
        unsigned char *atomColor;
        // size of the atom color array
        unsigned int atomColorSize;

	};


} /* end namespace protein */
} /* end namespace megamol */

#endif // MEGAMOLCORE_SOLVENTRENDERER_H_INCLUDED
