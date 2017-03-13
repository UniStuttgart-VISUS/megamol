/*
*CinematicView.h
*/


#ifndef MEGAMOL_CINEMATICCAMERA_CinematicView_H_INCLUDED
#define MEGAMOL_CINEMATICCAMERA_CinematicView_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/graphics/Camera.h"
#include "vislib/math/Point.h"
#include "vislib/Serialisable.h"
#include "mmcore/view/View3D.h"

namespace megamol {
	namespace cinematiccamera {

		class CinematicView : public core::view::View3D {

		public:

			enum SkyboxSides {
				SKYBOX_NONE = 0,
				SKYBOX_FRONT = 1,
				SKYBOX_BACK = 2,
				SKYBOX_LEFT = 4,
				SKYBOX_RIGHT = 8, 
				SKYBOX_UP = 16,
				SKYBOX_DOWN = 32
			};

			typedef core::view::View3D Base;

			/**
			* Answer the name of this module.
			*
			* @return The name of this module.
			*/
			static const char *ClassName(void) {
				return "CinematicView";
			}

			/**
			* Answer a human readable description of this module.
			*
			* @return A human readable description of this module.
			*/
			static const char *Description(void) {
				return "Screenshot View Module";
			}

			/**
			* Answers whether this module is available on the current system.
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
			CinematicView(void);

			/** Dtor. */
			virtual ~CinematicView(void);

			/**
			* Renders this AbstractView3D in the currently active OpenGL context.
			*/
			virtual void Render(const mmcRenderViewContext& context);

		private:

			/** The keyframe keeper caller slot */
			core::CallerSlot keyframeKeeperSlot;

			core::param::ParamSlot selectedKeyframeParam;
			core::param::ParamSlot selectedSkyboxSideParam;
		};

	} /* end namespace cinematiccamera */
} /* end namespace megamol */

#endif /* MEGAMOL_CINEMATICCAMERA_CinematicView_H_INCLUDED */