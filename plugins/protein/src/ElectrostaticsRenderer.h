/*
 * ElectrostaticsRenderer.h
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS). 
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_ELECTRORENDERER_H_INCLUDED
#define MMPROTEINPLUGIN_ELECTRORENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "ParticleDataCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/graphics/gl/GLSLShader.h"

namespace megamol {
namespace protein {

	/*
     * Electrostatics Renderer class
     */

	class ElectrostaticsRenderer : public megamol::core::view::Renderer3DModule {
	public:
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "ElectrostaticsRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Offers electrostatics renderings.";
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
        ElectrostaticsRenderer(void);

        /** Dtor. */
        virtual ~ElectrostaticsRenderer(void);

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
         * Compute the electrostatic field
         * 
         * @param particles The data call
         * @param stepWidth The designated length of a grid cell
         */
        void ComputeElectrostaticField( ParticleDataCall *particles, float stepWidth);

		/**********************************************************************
		 * variables
		 **********************************************************************/

        // caller slot
		megamol::core::CallerSlot dataCallerSlot;

        /** parameter slot for grid cell length */
        megamol::core::param::ParamSlot cellLenghtParam;

		// camera information
		vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;

		// shader for the spheres (raycasting view)
		vislib::graphics::gl::GLSLShader sphereShader;

		// attribute locations for GLSL-Shader
		GLint attribLocInParams;
		GLint attribLocQuatC;
		GLint attribLocColor1;
		GLint attribLocColor2;

        // The color array
        vislib::Array<float> colors;

        // The vector field size
        unsigned int fieldSize;

        // The vector field dimensions
        unsigned int fieldDim[3];

        // The vector field
        vislib::math::Vector<float, 3> *field;
	};


} /* end namespace protein */
} /* end namespace megamol */

#endif // MMPROTEINPLUGIN_ELECTRORENDERER_H_INCLUDED
