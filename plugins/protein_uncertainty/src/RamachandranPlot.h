/*
 * UncertaintyColor.h
 *
 * Author: Karsten Schatz
 * Copyright (C) 2017 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 *
 */
#ifndef MM_PROTEIN_UNCERTAINTY_PLUGIN_RAMACHANDRANPLOT_H_INCLUDED
#define MM_PROTEIN_UNCERTAINTY_PLUGIN_RAMACHANDRANPLOT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/Renderer2DModule.h"
#include "vislib/graphics/gl/OpenGLTexture2D.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/ShaderSource.h"

#ifdef USE_SIMPLER_FONT
#include "vislib/graphics/gl/SimpleFont.h"
#else //  USE_SIMPLER_FONT
#include "vislib/graphics/gl/OutlineFont.h"
#include "vislib/graphics/gl/Verdana.inc"
#endif //  USE_SIMPLER_FONT

#include "protein_calls/MolecularDataCall.h"
#include "RamachandranDataCall.h"

#include "mmcore/CallerSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol {
namespace protein_uncertainty {

	class RamachandranPlot : public megamol::core::view::Renderer2DModule {
	public:
		/**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "RamachandranPlot";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Offers rendering of ramachandran plots.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
			return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable() && ogl_IsVersionGEQ(1, 2);;
        }

		/** ctor */
		RamachandranPlot(void);

		/** dtor */
		virtual ~RamachandranPlot(void);

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
         * Callback for mouse events (move, press, and release)
         *
         * @param x The x coordinate of the mouse in world space
         * @param y The y coordinate of the mouse in world space
         * @param flags The mouse flags
         */
        virtual bool MouseEvent(float x, float y, core::view::MouseFlags flags);
	private:

		/**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetExtents(core::view::CallRender2D& call);

		/**
		 * Callback providing the data that this module is able to compute
		 */
		virtual bool GetDataCallback(megamol::core::Call& call);

        /**
        * The Open GL Render callback.
        *
        * @param call The calling call.
        * @return The return value of the function.
        */
        virtual bool Render(megamol::core::view::CallRender2D& call);

		/**
		 * Computes the dihedral angles of the amino acids of a given protein
		 *
		 * @param mol Pointer the incoming molecular data call.
		 */
		void computeDihedralAngles(protein_calls::MolecularDataCall * mol);

		/**
		 * Computes the polygon positions of each available amino acid.
		 */
		void computePolygonPositions(void);

		/**
		 * Computes the assignment probabilities for each amino acid
		 */
		void computeAssignmentProbabilities(void);

		/**
		 *	Tells whether a given point lies inside a given polygon
		 *
		 * @param polyVector A vector containing all points of the polygon ordererd clockwise or counterclockwise
		 * @param inputPos The position to test against the polygon
		 * @return True, if the point lies inside the polygon, false otherwise
		 */
		bool locateInPolygon(const std::vector<vislib::math::Vector<float, 2>>& polyVector, const vislib::math::Vector<float, 2> inputPos);

		/**
		 * Computes the dihedral angle between four given vectors
		 *
		 * @param v1 The first vector
		 * @param v2 The second vector
		 * @param v3 The third vector
		 * @param v4 The fourth vector
		 * @return The dihedral angle between the four vectors in degrees. Range[-180, 180].
		 */
		float dihedralAngle(const vislib::math::Vector<float, 3>& v1, const vislib::math::Vector<float, 3>& v2,
			const vislib::math::Vector<float, 3>& v3, const vislib::math::Vector<float, 3>& v4);

		/**
		 * Initializes the procheck array.
		 */
		void initProcheckArray(void);

		/** The call for the molecular data */
		core::CallerSlot molDataSlot;
		/** The call to write out the ramachandran data. */
		core::CalleeSlot plotDataSlot;

		/** Parameter slot for the drawn point size */
		core::param::ParamSlot pointSize;
		/** Parameter for the drawing of the own bounding box */
		core::param::ParamSlot ownBBParam;
		/** Parameter for the point color */
		core::param::ParamSlot pointColorParam;

		/** Vector containing all dihedral angles for all molecules. */
		std::vector<float> angles;
		std::vector<RamachandranDataCall::PointState> pointStates;
		std::vector<float> probabilities;

		std::vector<std::vector<RamachandranDataCall::ProcheckState>> procheckArray;

		std::vector<std::vector<vislib::math::Vector<float, 2>>> sureHelixPolygons;
		std::vector<std::vector<vislib::math::Vector<float, 2>>> sureSheetPolygons;
		std::vector<std::vector<vislib::math::Vector<float, 2>>> semiHelixPolygons;
		std::vector<std::vector<vislib::math::Vector<float, 2>>> semiSheetPolygons;

		int selectedAcid;

#ifdef USE_SIMPLER_FONT
		vislib::graphics::gl::SimpleFont theFont;
#else
		vislib::graphics::gl::OutlineFont theFont;
#endif
	};
}
}

#endif /* MM_PROTEIN_UNCERTAINTY_PLUGIN_RAMACHANDRANPLOT_H_INCLUDED */