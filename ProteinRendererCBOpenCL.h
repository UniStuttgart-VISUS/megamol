/*
 * ProteinRendererCBOpenCL.h
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#if (defined(WITH_OPENCL) && (WITH_OPENCL))

#ifndef MEGAMOL_MOLSURFACERENDERERCONTOURBUILDUP_OCL_H_INCLUDED
#define MEGAMOL_MOLSURFACERENDERERCONTOURBUILDUP_OCL_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#include "CallProteinData.h"
#include "param/ParamSlot.h"
#include "CallerSlot.h"
#include "CallFrame.h"
#include "view/Renderer3DModule.h"
#include "view/CallRender3D.h"
#include "vislib/SimpleFont.h"
#include <vislib/GLSLShader.h>
#include <vislib/Quaternion.h>
#include <vector>
#include <set>
#include <algorithm>
#include <list>
#include "vislib/FpsCounter.h"

#include "CL/cl.h"
#include "CL/clext.h"
#include "CL/cl_gl.h"

namespace megamol {
namespace protein {

	/**
	 * Molecular Surface Renderer class.
	 * Computes and renders the solvent excluded (Connolly) surface 
	 * using the Contour-Buildup Algorithm by Totrov & Abagyan.
	 */
	class ProteinRendererCBOpenCL : public megamol::core::view::Renderer3DModule
	{
	public:

		/**
		 * Answer the name of this module.
		 *
		 * @return The name of this module.
		 */
		static const char *ClassName(void)
		{
			return "ProteinRendererCBOpenCL";
		}

		/**
		 * Answer a human readable description of this module.
		 *
		 * @return A human readable description of this module.
		 */
		static const char *Description(void) 
		{
			return "Offers protein surface renderings.";
		}

		/**
		 * Answers whether this module is available on the current system.
		 *
		 * @return 'true' if the module is available, 'false' otherwise.
		 */
		static bool IsAvailable(void) 
		{
			//return true;
			return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable();
		}
		
		/** ctor */
		ProteinRendererCBOpenCL(void);
		
		/** dtor */
		virtual ~ProteinRendererCBOpenCL(void);

	   /**********************************************************************
		 * 'get'-functions
		 **********************************************************************/
		
		/** Get probe radius */
		const float GetProbeRadius() const { return probeRadius; };

		/**********************************************************************
		 * 'set'-functions
		 **********************************************************************/

		/** Set probe radius */
		void SetProbeRadius( const float rad) { probeRadius = rad; };
		
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
		 * Write neighbor atoms of all atoms to a 
		 */
		void writeNeighborAtoms( CallProteinData *protein);
    private:

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
		 * Open GL Render call.
		 *
		 * @param call The calling call.
		 * @return The return value of the function.
		 */
		virtual bool Render( megamol::core::Call& call);

		/**
		 * Deinitialises this renderer. This is only called if there was a 
		 * successful call to "initialise" before.
		 */
		virtual void deinitialise(void);
		
		/**********************************************************************
		 * variables
		 **********************************************************************/
		
		// caller slot
		megamol::core::CallerSlot protDataCallerSlot;
		
		// camera information
		vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;

		// shader for the sphere raycasting
		vislib::graphics::gl::GLSLShader sphereShader;

		// the bounding box of the protein
		vislib::math::Cuboid<float> bBox;

		// radius of the probe atom
		float probeRadius;

		// OpenCL vars
		cl_context cxGPUContext;
		cl_device_id cdDevice;
		cl_command_queue cqCommandQueue;
		cl_program cpProgram;
		cl_int ciErrNum;
		vislib::StringA cSourceCL;
		cl_kernel spherecutKernel;

		// array for neighbors for each atom
		float* atomNeighbors;
		unsigned int atomNeighborsSize;
		const unsigned int atomNeighborCount;
	};

} /* end namespace protein */
} /* end namespace megamol */

#endif /* MEGAMOL_MOLSURFACERENDERERCONTOURBUILDUP_OCL_H_INCLUDED */

#endif /* (defined(WITH_OPENCL) && (WITH_OPENCL)) */
