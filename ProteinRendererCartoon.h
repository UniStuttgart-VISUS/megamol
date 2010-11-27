/*
 * ProteinRendererCartoon.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PROTEINRENDERERCARTOON_H_INCLUDED
#define MEGAMOLCORE_PROTEINRENDERERCARTOON_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CallProteinData.h"
#include "CallFrame.h"
#include "Color.h"
#include "param/ParamSlot.h"
#include "BSpline.h"
#include "CallerSlot.h"
#include "view/Renderer3DModule.h"
#include "view/CallRender3D.h"
#include "vislib/GLSLShader.h"
#include "vislib/GLSLGeometryShader.h"
#include "vislib/SimpleFont.h"
#include <vector>

namespace megamol {
namespace protein {

	/*
     * Protein Renderer class
	 *
	 * TODO:
	 * - add Parameter:
	 *    o number of segments per amino acids
	 *    o number of tube segments for CARTOON_CPU
	 * - add coloring mode:
	 *    o value
	 *    o rainbow / "chain"-bow(?)
	 * - add RenderMode CARTOON_GPU
     */

	class ProteinRendererCartoon : public megamol::core::view::Renderer3DModule
	{
	public:
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) 
		{
            return "ProteinRendererCartoon";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) 
		{
            return "Offers protein cartoon renderings.";
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
        ProteinRendererCartoon(void);

        /** Dtor. */
        virtual ~ProteinRendererCartoon(void);

		enum CartoonRenderMode
		{
			CARTOON        = 0,
			CARTOON_SIMPLE = 1,
			CARTOON_CPU    = 2,
			CARTOON_GPU    = 3
		};


	   /**********************************************************************
		* 'get'-functions
	    **********************************************************************/
		
		/** Get radius for cartoon rendering mode */
		inline float GetRadiusCartoon(void) const { return m_radiusCartoon; };

		/** Get number of spline segments per amino acid for cartoon rendering mode */
		inline unsigned int GetNumberOfSplineSegments(void) const { return m_numberOfSplineSeg; };

		/** Get number of tube segments per 390 degrees in CPU cartoon rendering mode */
		inline unsigned int GetNumberOfTubeSegments(void) const { return m_numberOfTubeSeg; };

		/** Get the color of a certain atom of the protein. */
        const float* GetProteinAtomColor( unsigned int idx) { return &this->m_protAtomColorTable[idx*3]; };

	   /**********************************************************************
		* 'set'-functions
	    **********************************************************************/

		/** Set current render mode */
		void SetRenderMode( CartoonRenderMode rm) { m_currentRenderMode = rm; RecomputeAll(); };

		/** Set current coloring mode */
		void SetColoringMode( Color::ColoringMode cm) { m_currentColoringMode = cm; RecomputeAll(); };

		/** Set radius for cartoon rendering mode */
		inline void SetRadiusCartoon( float rad ) { m_radiusCartoon = rad; RecomputeAll(); };

		/** Set number of spline segments per amino acid for cartoon rendering mode */
		inline void SetNumberOfSplineSegments( unsigned int numSeg ) { m_numberOfSplineSeg = numSeg; RecomputeAll(); };

		/** Set number of tube segments per 390 degrees in CPU cartoon rendering mode */
		inline void SetNumberOfTubeSegments( unsigned int numSeg ) { m_numberOfTubeSeg = numSeg; RecomputeAll(); };

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

        /**
         * Draw label for current loaded RMS frame.
         *
         * @param call Ths calling CallFrame.
         */
        void DrawLabel(unsigned int frameID);

        /**
         * The CallFrame callback.
         *
         * @param call The calling call.
         * @return The return value of the function.
         */
        bool ProcessFrameRequest( megamol::core::Call& call);

		/**
		 * Render protein in hybrid CARTOON mode using the Geometry Shader.
		 *
		 * @param prot The data interface.
		 */
		void RenderCartoonHybrid( const CallProteinData *prot);

		/**
		 * Render protein in CPU CARTOON mode using OpenGL primitives.
		 *
		 * @param prot The data interface.
		 */
		void RenderCartoonCPU( const CallProteinData *prot);

		 /** 
		 * Render protein in GPU CARTOON mode using OpenGL primitives.
		 *
		 * @param prot The data interface.
		 */
		void RenderCartoonGPU( const CallProteinData *prot);
		
		 /** 
		  * Recompute all values.
		  * This function has to be called after every change rendering attributes,
		  * e.g. coloring or render mode.
		  */
		 void RecomputeAll(void);


		/**********************************************************************
		 * variables
		 **********************************************************************/

        // caller slot
		megamol::core::CallerSlot m_protDataCallerSlot;
        // callee slot
        megamol::core::CalleeSlot m_callFrameCalleeSlot;
		// caller slot
		megamol::core::CallerSlot solventRendererCallerSlot;

        // 'true' if there is rms data to be rendered
        bool m_renderRMSData;

        // label with id of current loaded frame
		vislib::graphics::AbstractFont *m_frameLabel;

		// camera information
		vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;

        megamol::core::param::ParamSlot m_renderingModeParam;
        megamol::core::param::ParamSlot m_coloringModeParam;
        megamol::core::param::ParamSlot m_smoothCartoonColoringParam;

		// shader for per pixel lighting (polygonal view)
		vislib::graphics::gl::GLSLShader m_lightShader;
		// shader for tube generation (cartoon view)
		vislib::graphics::gl::GLSLGeometryShader m_cartoonShader;
		vislib::graphics::gl::GLSLGeometryShader m_tubeShader;
		vislib::graphics::gl::GLSLGeometryShader m_arrowShader;
		vislib::graphics::gl::GLSLGeometryShader m_helixShader;
		vislib::graphics::gl::GLSLGeometryShader m_tubeSimpleShader;
		vislib::graphics::gl::GLSLGeometryShader m_arrowSimpleShader;
		vislib::graphics::gl::GLSLGeometryShader m_helixSimpleShader;
		vislib::graphics::gl::GLSLGeometryShader m_tubeSplineShader;
		vislib::graphics::gl::GLSLGeometryShader m_arrowSplineShader;
		vislib::graphics::gl::GLSLGeometryShader m_helixSplineShader;

		vislib::graphics::gl::GLSLShader sphereShader;
        vislib::graphics::gl::GLSLShader cylinderShader;

		// current render mode
		CartoonRenderMode m_currentRenderMode;
		// current coloring mode
		Color::ColoringMode m_currentColoringMode;
		// smooth coloring of cartoon mode
		bool m_smoothCartoonColoringMode;

		// attribute locations for GLSL-Shader
		GLint m_attribLocInParams;
		GLint m_attribLocQuatC;
		GLint m_attribLocColor1;
		GLint m_attribLocColor2;

		// is the geometry shader (and OGL V2) supported?
		bool m_geomShaderSupported;

		// has the hybrid CARTOON render mode to be prepared?
		bool m_prepareCartoonHybrid;
		// has the CPU CARTOON render mode to be prepared?
		bool m_prepareCartoonCPU;

		// counters, vertex- and color-arrays for cartoon mode
		float *m_vertTube;
		float *m_normalTube;
		float *m_colorsParamsTube;
		unsigned int m_totalCountTube;
		float *m_vertArrow;
		float *m_normalArrow;
		float *m_colorsParamsArrow;
		unsigned int m_totalCountArrow;
		float *m_vertHelix;
		float *m_normalHelix;
		float *m_colorsParamsHelix;
		unsigned int m_totalCountHelix;

		// number of spline segments per amino acid
		unsigned int m_numberOfSplineSeg;
		// number of tube segments per 390 degrees (only used with cartoon GPU)
		unsigned int m_numberOfTubeSeg;
		// radius for secondary structure elements with CARTOON render modes
		float m_radiusCartoon;

		// color table for amino acids
		vislib::Array<vislib::math::Vector<float, 3> > m_aminoAcidColorTable;
		// color palette vector: stores the color for chains
		vislib::Array<vislib::math::Vector<float, 3> > rainbowColors;
		// color table for protein atoms
		//vislib::Array<unsigned char> m_protAtomColorTable;
        vislib::Array<float> m_protAtomColorTable;
		
		// the Id of the current frame (for dynamic data)
		unsigned int m_currentFrameId;

        unsigned int atomCount;
        
	    // cylinder shader attribute locations
	    GLuint attribLocInParams;
	    GLuint attribLocQuatC;
	    GLuint attribLocColor1;
	    GLuint attribLocColor2;
	};


} /* end namespace protein */
} /* end namespace megamol */

#endif // MEGAMOLCORE_PROTEINRENDERERCARTOON_H_INCLUDED
