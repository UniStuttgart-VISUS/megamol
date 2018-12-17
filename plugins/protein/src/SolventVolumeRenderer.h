/*
 * SolventVolumeRenderer.h
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_SOLVENTVOLRENDERER_H_INCLUDED
#define MEGAMOLCORE_SOLVENTVOLRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "slicing.h"
#include "Color.h"
#include "protein_calls/MolecularDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/SimpleFont.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include <list>
#include "GridNeighbourFinder.h"

#define CHECK_FOR_OGL_ERROR() do { GLenum err; err = glGetError();if (err != GL_NO_ERROR) { fprintf(stderr, "%s(%d) glError: %s\n", __FILE__, __LINE__, gluErrorString(err)); } } while(0)

namespace megamol {
namespace protein {

	/**
	 * Protein Renderer class
	 */
	class SolventVolumeRenderer : public megamol::core::view::Renderer3DModule
	{
	public:
		/**
		 * Answer the name of this module.
		 *
		 * @return The name of this module.
		 */
		static const char *ClassName(void) {
			return "SolventVolumeRenderer";
		}

		/**
		 * Answer a human readable description of this module.
		 *
		 * @return A human readable description of this module.
		 */
		static const char *Description(void) {
			return "Offers protein/solvent volume renderings.";
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
		SolventVolumeRenderer(void);

		/** Dtor. */
		virtual ~SolventVolumeRenderer(void);

	   /**********************************************************************
		 * 'get'-functions
	    **********************************************************************/

		/** Get the color of a certain atom of the protein. */
        const float* GetAtomColor( unsigned int idx) { return &this->atomColorTable[idx*3]; };

        /**
         * Call callback to get the volume data
         *
         * @param c The calling call
         *
         * @return True on success
         */
        bool getVolumeData( core::Call& call);

	   /**********************************************************************
		 * 'set'-functions
	    **********************************************************************/


	protected:
		
        enum {
        	VOLCM_SolventConcentration,
        	VOLCM_HydrogenBonds,
        	VOlCM_HydrogenBondStats
        };

		/**
		 * Implementation of 'Create'.
		 *
		 * @return 'true' on success, 'false' otherwise.
		 */
		virtual bool create(void);

		/**
		 * @return whether shader loading/creation was sucessful
		 */
		bool loadShader(vislib::graphics::gl::GLSLShader& shader, const vislib::StringA& vert, const vislib::StringA& frag);

		/**
		 * Implementation of 'release'.
		 */
		virtual void release(void);

	private:

		/**********************************************************************
		 * 'render'-functions
		 **********************************************************************/

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
		void UpdateColorTable(megamol::protein_calls::MolecularDataCall *mol);
		void ColorAtom(float *atomColor, megamol::protein_calls::MolecularDataCall *mol, int polymerColorMode, int atomIdx, int residueIdx);

		/**
         * Volume rendering using molecular data.
		*/
		bool RenderMolecularData(megamol::core::view::CallRender3D *call, megamol::protein_calls::MolecularDataCall *mol);
		
		/**
         * Render the current mouse position on the clipping plane as a small sphere.
         *
         * @param call The render call
         * @param rad The sphere radius
         */
        void RenderMousePosition( megamol::core::view::CallRender3D *call, float rad);

        /**
         * Refresh all parameters.
		*/
		void ParameterRefresh(megamol::core::view::CallRender3D *call, megamol::protein_calls::MolecularDataCall *mol);
		
		/** 
         * Create a volume containing all molecule atoms.
		 *
		 * @param mol The data interface.
		*/
		void UpdateVolumeTexture(megamol::protein_calls::MolecularDataCall *mol);
		void CreateSpatialProbabilitiesTexture(megamol::protein_calls::MolecularDataCall *mol);

		/**
         * Draw the volume.
		 *
		 * @param boundingbox The bounding box.
		 */
        void RenderVolume( vislib::math::Cuboid<float> boundingbox);

		/*
		 * Render the molecular data in stick mode.
		 * Special case when using solvent rendering: only render solvent molecules near the isosurface between the solvent and the molecule.
		 */
		void RenderMolecules( /*const*/ megamol::protein_calls::MolecularDataCall *mol, const float *atomPos);

		/*
		 * Render hydrogen bounds
		 */
		void RenderHydrogenBounds(megamol::protein_calls::MolecularDataCall *mol, const float *atomPos);

		/**
         * Write the parameters of the ray to the textures.
		 *
		 * @param boundingbox The bounding box.
		 */
        void RayParamTextures( vislib::math::Cuboid<float> boundingbox);

        /**
         * Draw the bounding box of the protein.
		 *
		 * @paramboundingbox The bounding box.
         */
        void DrawBoundingBoxTranslated( vislib::math::Cuboid<float> boundingbox);

        /**
         * Draw the bounding box of the protein around the origin.
		 *
		 * @param boundingbox The bounding box.
         */
        void DrawBoundingBox( vislib::math::Cuboid<float> boundingbox);

        /**
         * Draw the clipped polygon for correct clip plane rendering.
         *
         * @param boundingbox The bounding box.
         */
        void drawClippedPolygon( vislib::math::Cuboid<float> boundingbox);

		/**
		 * Count visible solvent molecules (by type).
		 *
		 * @param mol The data interface.
		 */
		void FindVisibleSolventMolecules(megamol::protein_calls::MolecularDataCall *mol);

		/**********************************************************************
		 * variables
		 **********************************************************************/
		
		/** caller slot */
		megamol::core::CallerSlot protDataCallerSlot;
		/** caller slot */
		megamol::core::CallerSlot protRendererCallerSlot;
        /** The volume data callee slot */
        megamol::core::CalleeSlot dataOutSlot;
				
		// camera information
		vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;
        // scaling factor for the scene
        float scale;
        // translation of the scene
        vislib::math::Vector<float, 3> translation;
		
		megamol::core::param::ParamSlot coloringModeSolventParam;
		megamol::core::param::ParamSlot coloringModePolymerParam;
		megamol::core::param::ParamSlot coloringModeVolSurfParam;
		megamol::core::param::ParamSlot colorFilterRadiusParam;
		megamol::core::param::ParamSlot colorIntensityScaleParam;
		// parameters for the volume rendering
		megamol::core::param::ParamSlot volIsoValue1Param;
		//megamol::core::param::ParamSlot volIsoValue2Param;
		megamol::core::param::ParamSlot volFilterRadiusParam;
		megamol::core::param::ParamSlot volDensityScaleParam;
		megamol::core::param::ParamSlot volIsoOpacityParam;
        megamol::core::param::ParamSlot volClipPlaneFlagParam;
        megamol::core::param::ParamSlot volClipPlane0NormParam;
        megamol::core::param::ParamSlot volClipPlane0DistParam;
        megamol::core::param::ParamSlot volClipPlaneOpacityParam;
        /** parameter slot for positional interpolation */
        megamol::core::param::ParamSlot interpolParam;
        /** parameter slot for stick radius */
        megamol::core::param::ParamSlot stickRadiusParam;
        megamol::core::param::ParamSlot atomRadiusFactorParam;
        megamol::core::param::ParamSlot atomSpaceFillingParam;

        /** parameter slot for color table filename */
        megamol::core::param::ParamSlot colorTableFileParam;
        /** parameter slot for min color of gradient color mode */
        megamol::core::param::ParamSlot minGradColorParam;
        /** parameter slot for mid color of gradient color mode */
        megamol::core::param::ParamSlot midGradColorParam;
        /** parameter slot for max color of gradient color mode */
        megamol::core::param::ParamSlot maxGradColorParam;

	    /** ;-list of residue names which compose the solvent */
        //megamol::core::param::ParamSlot solventResidues;
		//vislib::Array<int> solventResidueTypeIds;

		/** threshold of visible solvent-molecules ... */
		megamol::core::param::ParamSlot solventMolThreshold;

		/** clear volume or accumulate stuff over time? ... */
		megamol::core::param::ParamSlot accumulateColors;
		megamol::core::param::ParamSlot accumulateVolume;
		megamol::core::param::ParamSlot accumulateFactor;

        // compute number of solvent molecules (by residue type) in hydration shell
		megamol::core::param::ParamSlot countSolMolParam;

		// shader for the spheres (raycasting view)
		vislib::graphics::gl::GLSLShader sphereSolventShader;
		// shader for the cylinders (raycasting view)
		vislib::graphics::gl::GLSLShader cylinderSolventShader;
		// shader for hygrogen bonds
		vislib::graphics::gl::GLSLShader hbondLineSolventShader;
		// shader for the clipped spheres (raycasting view)
		vislib::graphics::gl::GLSLShader clippedSphereShader;
        // shader for volume texture generation
        vislib::graphics::gl::GLSLShader updateVolumeShaderMoleculeVolume;
        vislib::graphics::gl::GLSLShader updateVolumeShaderSolventColor;
        vislib::graphics::gl::GLSLShader updateVolumeShaderHBondColor;
        // shader for volume rendering
        vislib::graphics::gl::GLSLShader volumeShader;
        vislib::graphics::gl::GLSLShader dualIsosurfaceShader;
        vislib::graphics::gl::GLSLShader volRayStartShader;
        vislib::graphics::gl::GLSLShader volRayStartEyeShader;
        vislib::graphics::gl::GLSLShader volRayLengthShader;
		// DEBUG
        vislib::graphics::gl::GLSLShader sphereShader;
        vislib::graphics::gl::GLSLShader visMolShader;
        vislib::graphics::gl::GLSLShader solTypeCountShader;
		
		// attribute locations for GLSL-Shader
		GLint attribLocInParams;
		GLint attribLocQuatC;
		GLint attribLocColor1;
		GLint attribLocColor2;
		
		// color table for amino acids
		vislib::Array<vislib::math::Vector<float, 3> > aminoAcidColorTable;
        /** The color lookup table (for chains, amino acids,...) */
        vislib::Array<vislib::math::Vector<float, 3> > colorLookupTable;
        /** The color lookup table which stores the rainbow colors */
        vislib::Array<vislib::math::Vector<float, 3> > rainbowColors;
		/** color table for protein atoms */
		vislib::Array<float> atomColorTable;
		
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
        // the average density value of the volume
        float meanDensityValue;
        // the first iso value
        float isoValue1;
        // the second iso value
        //float isoValue2;
		// the opacity of the isosurface
		float volIsoOpacity;

        vislib::math::Vector<float, 3> protrenTranslate;
        float protrenScale;

        // flag wether clipping planes are enabled
        bool volClipPlaneFlag;
        // the array of clipping planes
        vislib::Array<vislib::math::Vector<double, 4> > volClipPlane;
        // view aligned slicing
        ViewSlicing slices;
		// the opacity of the clipping plane
		float volClipPlaneOpacity;

        // interpolated atom positions
        float *atomPosInterPtr;
		int *hBondInterPtr;
		float lastUpdateVolumeTextureTime;

		// temporary atom array as member - do not use new-operator inside render()-routines!
		// temporary arrays for rendering operations ...
		bool getFrameData(megamol::protein_calls::MolecularDataCall *mol, int frameID, float *&interPosFramePtr, int *&interHBondFramePtr);

		vislib::Array<float> interpAtomPosTmpArray;
		vislib::Array<int> interpHBondTmpArray;
		unsigned int interpFrame0, interpFrame1;
		SIZE_T interpDataHash0, interpDataHash1;

		vislib::Array<float> update_vol, update_clr;
		vislib::Array<float> vertSpheres, vertCylinders, quatCylinders, inParaCylinders, color1Cylinders, color2Cylinders;

        bool forceUpdateVolumeTexture, forceUpdateColoringMode;

		megamol::protein::GridNeighbourFinder<float> gnf;

		// array for rendering the solvent molecules' atoms
		vislib::Array<float> solventAtomPos;
		// array for solvent molecules' atom residue type & the atom idx
		vislib::Array<float> solventAtomParams;
		// textures/FBOs for visible molecule counting
		vislib::graphics::gl::FramebufferObject molVisFbo;
        // dimensions of the molVisFbo
        unsigned int molVisFboWidth;
        unsigned int molVisFboHeight;
        // VBO for visible molecule counting
        GLuint molVisVbo;
        // framebuffer object for molecule type counting
        vislib::graphics::gl::FramebufferObject solTypeCountFbo;

	};


} /* end namespace protein */
} /* end namespace megamol */

#endif // MEGAMOLCORE_SOLVENTVOLRENDERER_H_INCLUDED
