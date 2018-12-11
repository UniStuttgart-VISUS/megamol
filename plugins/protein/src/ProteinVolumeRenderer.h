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

#include "slicing.h"
#include "CallProteinVolumeData.h"
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
		static const char *ClassName(void) {
			return "ProteinVolumeRenderer";
		}

		/**
		 * Answer a human readable description of this module.
		 *
		 * @return A human readable description of this module.
		 */
		static const char *Description(void) {
			return "Offers protein volume renderings.";
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
		ProteinVolumeRenderer(void);

		/** Dtor. */
		virtual ~ProteinVolumeRenderer(void);

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

        /**
         * Call callback to get the segmentation data
         *
         * @param c The calling call
         *
         * @return True on success
         */
        bool getSegmentationData( core::Call& call);

	   /**********************************************************************
		 * 'set'-functions
	    **********************************************************************/

		/** Set current coloring mode */
        inline void SetColoringMode( Color::ColoringMode cm) { currentColoringMode = cm; };

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
         * Volume rendering using volume data.
		*/
        bool RenderVolumeData( megamol::core::view::CallRender3D *call, CallProteinVolumeData *volume);
		
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
         * Render the segmented voxels as spheres.
         *
         * @param call The render call
         */
        void RenderSegmentedVoxels( megamol::core::view::CallRender3D *call);

        /**
         * Refresh all parameters.
		*/
        void ParameterRefresh( megamol::core::view::CallRender3D *call);
		

		/** 
         * Create a volume containing all molecule atoms.
		 *
		 * @param mol The data interface.
		*/
		void UpdateVolumeTexture(megamol::protein_calls::MolecularDataCall *mol);
		
		/**
         * Draw the volume.
		 *
		 * @param boundingbox The bounding box.
		 */
        void RenderVolume( vislib::math::Cuboid<float> boundingbox);
		
		/**
         * Write the parameters of the ray to the textures.
		 *
		 * @param boundingbox The bounding box.
		 */
        void RayParamTextures( vislib::math::Cuboid<float> boundingbox);

        /**
         * Create a volume containing the voxel map.
		 *
		 * @param volume The data interface.
         */
        void UpdateVolumeTexture( const CallProteinVolumeData *volume);

        /**
         * Draw the volume.
		 *
		 * @param volume The data interface.
         */
        void RenderVolume( const CallProteinVolumeData *volume);

        /**
         * Write the parameters of the ray to the textures.
		 *
		 * @param volume The data interface.
         */
        void RayParamTextures( const CallProteinVolumeData *volume);

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
         * Start the segmentation of the volume
         *
         * @param time The current time
         */
        void startVolumeSegmentation( float time);

        /**
         * Update the segmentation of the volume
         *
         * @param time The current time
         */
        void updateVolumeSegmentation( float time);

        /**
         * Update the segmentation of the volume
         *
         * @param time The current time
         */
        void updateVolumeSegmentation2( float time);

        /**
         * Update the segmentation of the volume
         *
         * @param time The current time
         */
        void updateVolumeSegmentationRmsd( float time);
        
        /**
         * Write the current volume as a raw file.
         */
        bool writeVolumeRAWUChar(core::param::ParamSlot& slot);
        /**
         * Write the current volume as a raw file.
         */
        bool writeVolumeRAWFloat(core::param::ParamSlot& slot);

		/**********************************************************************
		 * variables
		 **********************************************************************/
		
		/** caller slot */
		megamol::core::CallerSlot protDataCallerSlot;
		/** caller slot */
		megamol::core::CallerSlot protRendererCallerSlot;
        /** The volume data callee slot */
        megamol::core::CalleeSlot dataOutSlot;
        /** The segmentation data callee slot */
        megamol::core::CalleeSlot diagramDataOutSlot;

		// camera information
		vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;
        // scaling factor for the scene
        float scale;
        // translation of the scene
        vislib::math::Vector<float, 3> translation;
		
		megamol::core::param::ParamSlot coloringModeParam;
		// parameters for the volume rendering
		megamol::core::param::ParamSlot volIsoValue1Param;
		megamol::core::param::ParamSlot volIsoValue2Param;
		megamol::core::param::ParamSlot volFilterRadiusParam;
		megamol::core::param::ParamSlot volDensityScaleParam;
		megamol::core::param::ParamSlot volIsoOpacityParam;
        megamol::core::param::ParamSlot volClipPlaneFlagParam;
        megamol::core::param::ParamSlot volClipPlane0NormParam;
        megamol::core::param::ParamSlot volClipPlane0DistParam;
        megamol::core::param::ParamSlot volClipPlaneOpacityParam;
        /** parameter slot for positional interpolation */
        megamol::core::param::ParamSlot interpolParam;
        megamol::core::param::ParamSlot writeRawFileParam;

        /** parameter slot for color table filename */
        megamol::core::param::ParamSlot colorTableFileParam;
        /** parameter slot for min color of gradient color mode */
        megamol::core::param::ParamSlot minGradColorParam;
        /** parameter slot for mid color of gradient color mode */
        megamol::core::param::ParamSlot midGradColorParam;
        /** parameter slot for max color of gradient color mode */
        megamol::core::param::ParamSlot maxGradColorParam;

        /** parameter slot maximum inital segmentation size */
        megamol::core::param::ParamSlot initialSegmentationSizeParam;
        /** parameter slot segmentation stopping */
        megamol::core::param::ParamSlot stopSegmentationParam;
        /** parameter slot segmentation difference */
        megamol::core::param::ParamSlot segmentationDeltaParam;

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
        vislib::graphics::gl::GLSLShader dualIsosurfaceShader;
        vislib::graphics::gl::GLSLShader volRayStartShader;
        vislib::graphics::gl::GLSLShader volRayStartEyeShader;
        vislib::graphics::gl::GLSLShader volRayLengthShader;
		
		// current coloring mode
		Color::ColoringMode currentColoringMode;
		
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
        float isoValue2;
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
        float *posInter;

        // the mouse pos
        vislib::math::Vector<float, 3> mousePos;

        // the clicked mouse pos
        vislib::math::Vector<float, 3> clickedPos;
        // start volume segmentation
        bool startVolSeg;
        // segmentation time
        float segmentationTime;
        // flood filled voxels
        std::list<vislib::math::Vector<int, 3> > segmentedVoxels;
        unsigned int oldVoxelCount;
        bool stopSegmentation;

        bool fixedNumberOfVoxels;

        bool drawMarker;

        bool forceUpdateVolumeTexture;

        float callTime;
	};


} /* end namespace protein */
} /* end namespace megamol */

#endif // MEGAMOLCORE_PROTEINVOLRENDERER_H_INCLUDED
