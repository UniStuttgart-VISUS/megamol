/*
 *Screenshotview.h
 */


#ifndef MEGAMOL_CINEMATICCAMERA_SCREENSHOTVIEW_H_INCLUDED
#define MEGAMOL_CINEMATICCAMERA_SCREENSHOTVIEW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/graphics/Camera.h"
#include "vislib/math/Point.h"
#include "vislib/Serialisable.h"
#include "mmcore/view/View3D.h"

namespace megamol {
	namespace cinematiccamera {

		class Screenshotview : public core::view::View3D {

		public:

			typedef core::view::View3D Base;

			/**
			* Answer the name of this module.
			*
			* @return The name of this module.
			*/
			static const char *ClassName(void) {
				return "Screenshotview";
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
			Screenshotview(void);

			/** Dtor. */
			virtual ~Screenshotview(void);

			/**
			* Renders this AbstractView3D in the currently active OpenGL context.
			*/
			virtual void Render(const mmcRenderViewContext& context);

		private:

			/** The keyframe keeper caller slot */
			core::CallerSlot keyframeKeeperSlot;

			vislib::SmartPtr<vislib::graphics::CameraParameters> paramsOverride;
		};

		} /* end namespace cinematiccamera */
	} /* end namespace megamol */

#endif /* MEGAMOL_CINEMATICCAMERA_SCREENSHOTVIEW_H_INCLUDED */