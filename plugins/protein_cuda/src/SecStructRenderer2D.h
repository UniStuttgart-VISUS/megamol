/*
 *	SecStructRenderer2D.h
 *
 *	Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 *	All rights reserved
 */

#ifndef MMPROTEINCUDAPLUGIN_SECSTRUCTRENDERER2D_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_SECSTRUCTRENDERER2D_H_INCLUDED

#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer2DModule.h"

#include "vislib/math/Matrix.h"
#include "vislib/math/Plane.h"
#include "vislib/graphics/gl/GLSLTesselationShader.h"

namespace megamol {
namespace protein_cuda {

	class SecStructRenderer2D : public core::view::Renderer2DModule {
	public:
		/**
		 *	Answer the name of this module.
		 *
		 *	@return The name of this module.
		 */
		static const char *ClassName(void) {
			return "SecStructRenderer2D";
		}

		/**
		 *	Answer a human readable description of this module.
		 *
		 *	@return A human readable description of this module.
		 */
		static const char *Description(void) {
			return "Offers renderings of protein secondary structures in a 2D domain.";
		}

		/**
		 *	Answers whether this module is available on the current system.
		 *
		 *	@return 'true' if the module is available, 'false' otherwise.
		 */
		static bool IsAvailable(void) {
			return true;
		}

		/** Ctor */
		SecStructRenderer2D(void);

		/** Dtor */
		virtual ~SecStructRenderer2D(void);

	protected:

		/**
		 *	Implementation of 'Create'.
		 *
		 *	@return 'true' on success, 'false' otherwise.
		 */
		virtual bool create(void);
		
		/**
		 *	Implementation of 'Release'.
		 */
		virtual void release(void);

		/**
         *	Callback for mouse events (move, press, and release)
         *	
         *	@param x The x coordinate of the mouse in world space
         *	@param y The y coordinate of the mouse in world space
         *	@param flags The mouse flags
         */
        virtual bool MouseEvent(float x, float y, megamol::core::view::MouseFlags flags);
	private:

		/** Struct for a single c alpha atom */
		struct CAlpha
		{
			/** Position */
			float pos[3];
			/** 
			 *	Secondary structure type
			 *	0 = unclassified
			 *	1 = beta sheet
			 *	2 = alpha helix
			 *	3 = turn
			 */
			int type;

			void print() {
				std::cout << "C " << pos[0] << " " << pos[1] << " " << pos[2] << " " << type << std::endl;
			}
		};

		/**
         *	The get extents callback. The module should set the members of
         *	'call' to tell the caller the extents of its data (bounding boxes
         *	and times).
         *	
         *	@param call The calling call.
         *	
         *	@return The return value of the function.
         */
        virtual bool GetExtents(megamol::core::view::CallRender2D& call);

		/**
		 *	The Open GL Render callback.
		 *	
		 *	@param call The calling call.
		 *	@return The return value of the function.
		 */
        virtual bool Render(megamol::core::view::CallRender2D& call);

		/**
		 *	Rotates a given 3D plane to the xy plane and returns the needed matrix for the operation.
		 *
		 *	@param plane The source plane.
		 *	@return The matrix that is necessary to rotate and translate the given plane to the xy plane.
		 */
		vislib::math::Matrix<float, 4, vislib::math::MatrixLayout::COLUMN_MAJOR> rotatePlaneToXY(const vislib::math::Plane<float> plane);

		/** slot for incoming render data */
		core::CallerSlot dataInSlot;

		/** slot for incoming plane data */
		core::CallerSlot planeInSlot;

		/** random coil width parameter */
		core::param::ParamSlot coilWidthParam;

		/** sheet or helix width parameter */
		core::param::ParamSlot structureWidthParam;

		/** shall the backbone be shown? */
		core::param::ParamSlot showBackboneParam;

		/** shall the direct atom connections be shown? */
		core::param::ParamSlot showDirectConnectionsParam;

		/** shall the atom positions be shown */
		core::param::ParamSlot showAtomPositionsParam;

		/** shall the hydrogen bonds be shown? */
		core::param::ParamSlot showHydrogenBondsParam;

		/** Shall the tubes be shown? */
		core::param::ParamSlot showTubesParam;

		/** the mouse position */
		vislib::math::Vector<float, 2> mousePos;

		/** the last hash for the molecular data */
		SIZE_T lastDataHash;

		/** the last hash for the plane data */
		SIZE_T lastPlaneHash;

		/** The current transformation matrix */
		vislib::math::Matrix<float, 4, vislib::math::MatrixLayout::COLUMN_MAJOR> transformationMatrix;

		/** The c alpha atoms as a vector */
		std::vector<CAlpha> cAlphas;

		/** The numbers of c alpha atoms for each different molecule */
		std::vector<unsigned int> molSizes;

		/** The indices in the main chain of each available c alpha atom */
		std::vector<unsigned int> cAlphaIndices;

		/** Handle for the c alpha ssbo */
		GLuint ssbo;

		/** Shader for the tesselated GL_LINES */
		vislib::graphics::gl::GLSLTesselationShader lineShader;

		/** Shader for the tesselated tubes */
		vislib::graphics::gl::GLSLTesselationShader tubeShader;

		/** The bounding rectangle for the data */
		vislib::math::Rectangle<float> bbRect;

		/** Map for the the c alpha positions to the indices used here */
		std::vector<unsigned int> cAlphaMap;
	};

} /* end namespace protein_cuda */
} /* end namespace megamol */

#endif // #ifndef MMPROTEINCUDAPLUGIN_SECSTRUCTRENDERER2D_H_INCLUDED