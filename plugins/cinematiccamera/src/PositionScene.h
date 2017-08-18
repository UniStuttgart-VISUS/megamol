/*
* PositionScene.h
*
* Copyright (C) 2011 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOLCORE_POSITIONSCENE_H_INCLUDED
#define MEGAMOLCORE_POSITIONSCENE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/View3D.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
namespace megamol {
	namespace cinematiccamera {


		class PositionScene : public megamol::core::view::View3D {

		public:

			/**
			* Answer the name of this module.
			*
			* @return The name of this module.
			*/
			static const char *ClassName(void) {
				return "PositionScene";
			}

			/**
			* Answer a human readable description of this module.
			*
			* @return A human readable description of this module.
			*/
			static const char *Description(void) {
				return "Position Scene Module";
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
			PositionScene(void);

			/** Dtor. */
			virtual ~PositionScene(void);

			/**
			* Resets the view. This normally sets the camera parameters to
			* default values.
			* This also resets the object center for the relative cursor
			*/
			virtual void ResetView(void);

		private:

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

			/** The call for data */
			core::CallerSlot getDataSlot;
			/** The call for data */
			core::CallerSlot getCinematicCameraSlot;


			virtual void injectCamera(vislib::graphics::Camera incectionCam);
			virtual bool injectCamera();

		};

	} /* end namespace protein */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_POSITIONSCENE_H_INCLUDED */
