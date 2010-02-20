/*
 * ProteinVolumeRenderer.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PROTEINVOLRENDERER_H_INCLUDED
#define MEGAMOLCORE_PROTEINVOLRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CallProteinData.h"
#include "CallFrame.h"
#include "param/ParamSlot.h"
#include "CallerSlot.h"
#include "view/Renderer3DModule.h"
#include "view/CallRender3D.h"
#include "vislib/GLSLShader.h"
#include "vislib/SimpleFont.h"
#include "vislib/FramebufferObject.h"
#include <vector>

#define CHECK_FOR_OGL_ERROR() do { GLenum err; err = glGetError();if (err != GL_NO_ERROR) { fprintf(stderr, "%s(%d) glError: %s\n", __FILE__, __LINE__, gluErrorString(err)); } } while(0)

namespace megamol {
namespace protein {

	/**
	 * Protein Renderer class
	 */
	class ProteinVolumeRenderer : public megamol::core::view::Renderer3DModule
	{
	public:
		/**
		 * Answer the name of this module.
		 *
		 * @return The name of this module.
		 */
		static const char *ClassName(void) 
		{
			return "ProteinVolumeRenderer";
		}

		/**
		 * Answer a human readable description of this module.
		 *
		 * @return A human readable description of this module.
		 */
		static const char *Description(void) 
		{
			return "Offers protein volume renderings.";
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
		ProteinVolumeRenderer(void);

		/** Dtor. */
		virtual ~ProteinVolumeRenderer(void);
		
		enum RenderMode
		{
			LINES             = 0,
			STICK_RAYCASTING  = 1,
			STICK_POLYGON     = 2,
			BALL_AND_STICK    = 3,
			SPACEFILLING      = 4,
			SAS               = 5,
			SPACEFILLING_CLIP = 6
		};

		enum ColoringMode
		{
			ELEMENT   = 0,
			AMINOACID = 1,
			STRUCTURE = 2,
			VALUE     = 3,
			CHAIN_ID  = 4,
			RAINBOW   = 5,
			CHARGE    = 6
		};

	   /**********************************************************************
		 * 'get'-functions
	    **********************************************************************/

		/** Get radius for stick rendering mode */
		inline float GetRadiusStick(void) const { return this->radiusStick; };

		/** Get the color of a certain atom of the protein. */
		const unsigned char * GetProteinAtomColor( unsigned int idx) { return &this->protAtomColorTable[idx*3]; };

		/** Get probe radius for Solvent Accessible Surface mode. */
		inline float GetProbeRadius() const { return this->probeRadius; };

	   /**********************************************************************
		 * 'set'-functions
	    **********************************************************************/

		/** Set current render mode */
		inline void SetRenderMode( RenderMode rm) { currentRenderMode = rm; RecomputeAll(); };

		/** Set current coloring mode */
		inline void SetColoringMode( ColoringMode cm) { currentColoringMode = cm; RecomputeAll(); };

		/** Set radius for stick rendering mode */
		inline void SetRadiusStick( const float rad ) { radiusStick = rad; RecomputeAll(); };
		
		/** Set probe radius for Solvent Accessible Surface mode. */
		inline void SetRadiusProbe( const float rad) { probeRadius = rad; RecomputeAll(); };

		/** Set if atoms are drawn as dots in LINES mode */
		inline void DrawAtomsAsDotsWithLine( bool drawDot ) { drawDotsWithLine = drawDot; RecomputeAll(); };

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
		* Render protein data in LINES mode.
		*
		* @param prot The data interface.
		*/
		void RenderLines( const CallProteinData *prot);
		
		/**
		* Render protein data in STICK_RAYCASTING mode.
		*
		* @param prot The data interface.
		*/
		void RenderStickRaycasting( const CallProteinData *prot);
		
		/**
		* Render protein data in BALL_AND_STICK mode using GPU raycasting.
		*
		* @param prot The data interface.
		*/
		void RenderBallAndStick( const CallProteinData *prot);
		
		/**
		* Render protein data in SPACEFILLING mode using GPU raycasting.
		*
		* @param prot The data interface.
		*/
		void RenderSpacefilling( const CallProteinData *prot);

		/**
		* Render protein data in SPACEFILLING mode with a clipping plane using GPU raycasting.
		*
		* @param cpDir The direction of the clipping plane
		* @param cpBase The base of the clipping plane
		* @param prot The data interface.
		*/
		void RenderClippedSpacefilling( const vislib::math::Vector<float, 3> cpDir,
			const vislib::math::Vector<float, 3> cpBase, const CallProteinData *prot);
		
		/* Render protein data in SAS mode (Solvent Accessible Surface) using GPU raycasting.
		*
		* @param prot The data interface.
		*/
		void RenderSolventAccessibleSurface( const CallProteinData *prot);
		
		/**
		* Render disulfide bonds using GL_LINES.
		*
		* @param prot The data interface.
		*/
		void RenderDisulfideBondsLine( const CallProteinData *prot);
	
		/** 
		* Recompute all values.
		* This function has to be called after every change rendering attributes,
		* e.g. coloring or render mode.
		*/
		void RecomputeAll(void);
	
		/** fill amino acid color table */
		void FillAminoAcidColorTable(void);
		
		/**
		 * Creates a rainbow color table with 'num' entries.
		 *
		 * @param num The number of color entries.
		 */
		void MakeRainbowColorTable( unsigned int num);
		
		/**
		 * Make color table for all atoms acoording to the current coloring mode.
		 * The color table is only computed if it is empty or if the recomputation 
		 * is forced by parameter.
		 *
		 * @param prot The data interface.
		 * @param forceRecompute Force recomputation of the color table.
		 */
		void MakeColorTable( const CallProteinData *prot, bool forceRecompute = false);

        /**
         * Create a volume containing all protein atoms.
		 *
		 * @param prot The data interface.
         */
        void UpdateVolumeTexture( const CallProteinData *protein);

        /**
         * Draw the volume.
		 *
		 * @param prot The data interface.
         */
        void RenderVolume( const CallProteinData *protein);

        /**
         * Write the parameters of the ray to the textures.
		 *
		 * @param prot The data interface.
         */
        void RayParamTextures( const CallProteinData *protein);

        /**
         * Draw the bounding box of the protein.
		 *
		 * @param prot The data interface.
         */
        void DrawBoundingBoxTranslated( const CallProteinData *protein);

        /**
         * Draw the bounding box of the protein around the origin.
		 *
		 * @param prot The data interface.
         */
        void DrawBoundingBox( const CallProteinData *protein);

		/**********************************************************************
		 * variables
		 **********************************************************************/
		
		// caller slot
		megamol::core::CallerSlot protDataCallerSlot;
		// callee slot
		megamol::core::CalleeSlot callFrameCalleeSlot;
		
		// 'true' if there is rms data to be rendered
		bool renderRMSData;
		
		// label with id of current loaded frame
		vislib::graphics::AbstractFont *frameLabel;
		
		// camera information
		vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;
        // scaling factor for the scene
        float scale;
        // translation of the scene
        vislib::math::Vector<float, 3> translation;
		
		megamol::core::param::ParamSlot renderingModeParam;
		megamol::core::param::ParamSlot coloringModeParam;
		megamol::core::param::ParamSlot drawBackboneParam;
		megamol::core::param::ParamSlot drawDisulfideBondsParam;
		megamol::core::param::ParamSlot stickRadiusParam;
		megamol::core::param::ParamSlot probeRadiusParam;
		// parameters for the volume rendering
		megamol::core::param::ParamSlot volIsoValueParam;
		megamol::core::param::ParamSlot volFilterRadiusParam;
		megamol::core::param::ParamSlot volDensityScaleParam;
		megamol::core::param::ParamSlot volIsoOpacityParam;

		// shader for the spheres (raycasting view)
		vislib::graphics::gl::GLSLShader sphereShader;
		// shader for the cylinders (raycasting view)
		vislib::graphics::gl::GLSLShader cylinderShader;
		// shader for the clipped spheres (raycasting view)
		vislib::graphics::gl::GLSLShader clippedSphereShader;
        // shader for volume texture generation
        vislib::graphics::gl::GLSLShader updateVolumeShader;
        // shader for volume rendering
        vislib::graphics::gl::GLSLShader volumeShader;
        vislib::graphics::gl::GLSLShader volRayStartShader;
        vislib::graphics::gl::GLSLShader volRayStartEyeShader;
        vislib::graphics::gl::GLSLShader volRayLengthShader;
		
		// current render mode
		RenderMode currentRenderMode;
		// current coloring mode
		ColoringMode currentColoringMode;
		
		// attribute locations for GLSL-Shader
		GLint attribLocInParams;
		GLint attribLocQuatC;
		GLint attribLocColor1;
		GLint attribLocColor2;
		
		// draw only the backbone atoms of the protein?
		bool drawBackbone;
		// draw the disulfide bonds?
		bool drawDisulfideBonds;
		
		// display list [LINES]
		GLuint proteinDisplayListLines;
		// display list [disulfide bonds]
		GLuint disulfideBondsDisplayList;
		// has the STICK_RAYCASTING render mode to be prepared?
		bool prepareStickRaycasting;
		// has the BALL_AND_STICK render mode to be prepared?
		bool prepareBallAndStick;
		// has the SPACEFILLING render mode to be prepared?
		bool prepareSpacefilling;
		// has the SAS render mode to be prepared?
		bool prepareSAS;

		// vertex array for spheres [STICK_RAYCASTING]
		vislib::Array<float> vertSphereStickRay;
		// vertex array for cylinders [STICK_RAYCASTING]
		vislib::Array<float> vertCylinderStickRay;
		// attribute array for quaterinons of the cylinders [STICK_RAYCASTING]
		vislib::Array<float> quatCylinderStickRay;
		// attribute array for inParameters of the cylinders (radius and length) [STICK_RAYCASTING]
		vislib::Array<float> inParaCylStickRaycasting;
		// color array for spheres [STICK_RAYCASTING]
		vislib::Array<unsigned char> colorSphereStickRay;
		// first color array for cylinder [STICK_RAYCASTING]
		vislib::Array<float> color1CylinderStickRay;
		// second color array for cylinder [STICK_RAYCASTING]
		vislib::Array<float> color2CylinderStickRay;
		
		// draw dots for atoms in LINE mode
		bool drawDotsWithLine;
		
		// radius for spheres and sticks with STICK_ render modes
		float radiusStick;
		
		// probe radius for SAS rendering
		float probeRadius;
		
		// color table for amino acids
		vislib::Array<vislib::math::Vector<unsigned char, 3> > aminoAcidColorTable;
		// color palette vector: stores the color for chains
		std::vector<vislib::math::Vector<float,3> > rainbowColors;
		// color table for protein atoms
		vislib::Array<unsigned char> protAtomColorTable;
		
		// the Id of the current frame (for dynamic data)
		unsigned int currentFrameId;

        // the number of protein atoms
        unsigned int atomCount;

        // FBO for rendering the protein
        vislib::graphics::gl::FramebufferObject proteinFBO;

        // volume texture
        GLuint volumeTex;
        unsigned int volumeSize;
        // FBO for volume generation
        GLuint volFBO;
        // volume parameters
        float volFilterRadius;
        float volDensityScale;
        float volScale[3];
        float volScaleInv[3];
        // width and height of view
        unsigned int width, height;
        // current width and height of textures used for ray casting
        unsigned int volRayTexWidth, volRayTexHeight;
        // volume ray casting textures
        GLuint volRayStartTex;
        GLuint volRayLengthTex;
        GLuint volRayDistTex;

        // render the volume as isosurface
        bool renderIsometric;
        // the iso value
        float isoValue;
		// the opacity of the isosurface
		float volIsoOpacity;
	};


} /* end namespace protein */
} /* end namespace megamol */

#endif // MEGAMOLCORE_PROTEINVOLRENDERER_H_INCLUDED