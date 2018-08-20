/*
* aurora_borealis_renderer.h
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOL_AURORA_BOREALIS_RENDERER_H_INCLUDED
#define MEGAMOL_AURORA_BOREALIS_RENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "aurora_borealis/aurora_borealis.h"
#include "visualize.h"

#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol {

	namespace ab {

		/**
		* Renderer for aurora borealis
		*/
		class AuroraBorealisRenderer : public core::view::Renderer3DModule {
		public:

			/**
			* Answer the name of this module.
			*
			* @return The name of this module.
			*/
			static const char *ClassName(void) {
				return "AuroraBorealisRenderer";
			}

			/**
			* Answer a human readable description of this module.
			*
			* @return A human readable description of this module.
			*/
			static const char *Description(void) {
				return "Renderer for aurora borealis.";
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
			AuroraBorealisRenderer(void);

			/** Dtor. */
			virtual ~AuroraBorealisRenderer(void);

		protected:

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


		private:
			//void resize(size_t n, GLsizei width, GLsizei height);

			/** The call for data */
			core::CallerSlot getDataSlot;

			GLuint test_program;
			GLuint vao, vbo;

			GLuint mvpID, vPosID, vColID;
			mat4x4 mvp;

			// sim
			Visualize mVis;
		};

	}
}

#endif // MEGAMOL_AURORA_BOREALIS_RENDERER_H_INCLUDED