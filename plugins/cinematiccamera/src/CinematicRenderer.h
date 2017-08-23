/*
* CinematicRenderer.h
*
* Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOLCORE_CinematicRenderer_H_INCLUDED
#define MEGAMOLCORE_CinematicRenderer_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CinematicCamera/CinematicCamera.h"

#include "mmcore/BoundingBoxes.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/Renderer3DModule.h"

#include "vislib/graphics/CameraParameters.h"
#include "vislib/graphics/CameraParamsStore.h"
#include "vislib/math/Cuboid.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/memutils.h"
#include "vislib/String.h"


namespace megamol {
	namespace cinematiccamera {
		


		/**
		* A renderer that passes the render call to another renderer
		*/
		
		class CinematicRenderer : public core::view::Renderer3DModule {
		public:

			/**
			* The names of the rendering modes
			*/
			enum RenderingMode {
				OVERVIEW	= 0,
				PREVIEW		= 1
			};

			/**
			* Gets the name of this module.
			*
			* @return The name of this module.
			*/
			static const char *ClassName(void) {
				return "CinematicRenderer";
			}

			/**
			* Gets a human readable description of the module.
			*
			* @return A human readable description of the module.
			*/
			static const char *Description(void) {
				return "Renderer that passes the render call to another renderer";
			}

			/**
			* Gets whether this module is available on the current system.
			*
			* @return 'true' if the module is available, 'false' otherwise.
			*/
			static bool IsAvailable(void) {
				return true;
			}

			/**
			* Disallow usage in quickstarts
			*
			* @return false
			*/
			static bool SupportQuickstart(void) {
				return false;
			}

			/** Ctor. */
			CinematicRenderer(void);

			/** Dtor. */
			virtual ~CinematicRenderer(void);

			/**
			* Implementation of 'Create'.
			*
			* @return 'true' on success, 'false' otherwise.
			*/
			virtual bool create(void);

			/**
			* Implementation of 'Release'.
			*/
			virtual void release(void);

		protected:

			/**
			* The get capabilities callback. The module should set the members
			* of 'call' to tell the caller its capabilities.
			*
			* @param call The calling call.
			*
			* @return The return value of the function.
			*/
			virtual bool GetCapabilities(core::Call& call);

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
			* The render callback.
			*
			* @param call The calling call.
			*
			* @return The return value of the function.
			*/
			virtual bool Render(core::Call& call);

			/**
			* Callback for mouse events (move, press, and release)
			*
			* @param[in] x The x coordinate of the mouse in screen space
			* @param[in] y The y coordinate of the mouse in screen space
			* @param[in] flags The mouse flags
			* @return 'true' on success
			*/
			virtual bool MouseEvent(float x, float y, core::view::MouseFlags flags);

		private:

            /**********************************************************************
            * callback stuff
            **********************************************************************/

			/** The renderer caller slot */
			core::CallerSlot slaveRendererSlot;

			/** The keyframe keeper caller slot */
			core::CallerSlot keyframeKeeperSlot;

            /**********************************************************************
            * parameters
            **********************************************************************/
			
            /** Amount of interpolation steps between keyframes */
            core::param::ParamSlot stepsParam;

            /** Param to load total time from slave renderer */
            core::param::ParamSlot loadTimeParam;

            /**********************************************************************
            * parameters
            **********************************************************************/


		};

	} /* end namespace cinematiccamera */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CinematicRenderer_H_INCLUDED */
