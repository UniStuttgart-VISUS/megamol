/*
 * MoleculeCartoonRenderer.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MMPROTEINPLUGIN_MOLECULECARTOONRENDERER_H_INCLUDED
#define MMPROTEINPLUGIN_MOLECULECARTOONRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "protein_calls/MolecularDataCall.h"
#include "protein_calls/BindingSiteCall.h"
#include "mmcore/param/ParamSlot.h"
#include "BSpline.h"
#include "mmcore/CallerSlot.h"
#include "Color.h"
#include "mmcore/view/Renderer3DModuleDS.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/view/CallRenderDeferred3D.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/GLSLGeometryShader.h"
#include "vislib/graphics/gl/SimpleFont.h"
#include "vislib/Array.h"
#include <vector>
#include "mmcore/CallerSlot.h"
#include "CallColor.h"

namespace megamol {
namespace protein {

    /*
     * Protein Renderer class
     *
     * TODO:
     * - add Parameter:
     *    o number of segments per amino acids
     *    o number of tube segments for CARTOON_CPU
     * - add RenderMode CARTOON_GPU
     */

    class MoleculeCartoonRenderer : public megamol::core::view::Renderer3DModuleDS {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "MoleculeCartoonRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Offers protein cartoon renderings.";
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
        MoleculeCartoonRenderer(void);

        /** Dtor. */
        virtual ~MoleculeCartoonRenderer(void);

        enum CartoonRenderMode {
            CARTOON        = 0,
            CARTOON_SIMPLE = 1,
            CARTOON_CPU    = 2,
            CARTOON_GPU    = 3,
            CARTOON_LINE   = 4,
            CARTOON_TUBE_ONLY  = 5
        };

		enum RenderSource {
			RENDER_NORMAL = 0,
			RENDER_COMPARISON_BASE = 1
		};


        /**********************************************************************
        * 'get'-functions
        **********************************************************************/

        /** Get radius for cartoon rendering mode */
        inline float GetRadiusCartoon(void) const { return radiusCartoon; };

        /** Get number of spline segments per amino acid for cartoon rendering mode */
        inline unsigned int GetNumberOfSplineSegments(void) const { return numberOfSplineSeg; };

        /** Get number of tube segments per 390 degrees in CPU cartoon rendering mode */
        inline unsigned int GetNumberOfTubeSegments(void) const { return numberOfTubeSeg; };

        /** Get the color of a certain atom of the protein. */
        const float* GetProteinAtomColor( unsigned int idx);

        /**********************************************************************
         * 'set'-functions
         **********************************************************************/

        /** Set current render mode */
        void SetRenderMode( CartoonRenderMode rm) {
            currentRenderMode = rm;
            CallColor* col = this->molColorCallerSlot.CallAs<CallColor>(); // Try to get color call pointer
            if(col != NULL) {
                col->SetDirty(true);
            }
        };
        
        /** Set current coloring mode */
        void SetColoringMode0( Color::ColoringMode cm) {
            currentColoringMode0 = cm;
            CallColor* col = this->molColorCallerSlot.CallAs<CallColor>(); // Try to get color call pointer
            if(col != NULL) {
                col->SetDirty(true);
            }
        };

        /** Set current coloring mode */
        void SetColoringMode1( Color::ColoringMode cm) {
            currentColoringMode1 = cm;
            CallColor* col = this->molColorCallerSlot.CallAs<CallColor>(); // Try to get color call pointer
            if(col != NULL) {
                col->SetDirty(true);
            }
        };

        /** Set radius for cartoon rendering mode */
        inline void SetRadiusCartoon( float rad ) {
            radiusCartoon = rad;
            CallColor* col = this->molColorCallerSlot.CallAs<CallColor>(); // Try to get color call pointer
            if(col != NULL) {
                col->SetDirty(true);
            }
        };

        /** Set number of spline segments per amino acid for cartoon rendering mode */
        inline void SetNumberOfSplineSegments( unsigned int numSeg ) { numberOfSplineSeg = numSeg; };

        /** Set number of tube segments per 390 degrees in CPU cartoon rendering mode */
        inline void SetNumberOfTubeSegments( unsigned int numSeg ) { numberOfTubeSeg = numSeg; };

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
         * Render protein in hybrid CARTOON mode using the Geometry Shader.
         *
         * @param prot The data interface.
         */
		void RenderCartoonHybrid(const megamol::protein_calls::MolecularDataCall *mol, float* atomPos);
        
        /**
         * Render protein in CPU CARTOON mode using OpenGL primitives.
         *
         * @param prot The data interface.
         */
		void RenderCartoonCPU(const megamol::protein_calls::MolecularDataCall *mol, float* atomPos);

        /**
         * Render protein in CPU CARTOON mode using OpenGL lines.
         *
         * @param prot The data interface.
         */
		void RenderCartoonLineCPU(const megamol::protein_calls::MolecularDataCall *mol, float* atomPos);

         /**
         * Render protein in GPU CARTOON mode using OpenGL primitives.
         *
         * @param prot The data interface.
         */
		void RenderCartoonGPU(const megamol::protein_calls::MolecularDataCall *mol, float* atomPos);

        /**
        * Render protein in GPU CARTOON mode using OpenGL primitives.
        *
        * @param prot The data interface.
        */
		void RenderCartoonGPUTubeOnly(const megamol::protein_calls::MolecularDataCall *mol, float* atomPos);

        /**
         * Render the molecular data in stick mode.
         */
		void RenderStick(const megamol::protein_calls::MolecularDataCall *mol, const float *atomPos, const protein_calls::BindingSiteCall *bs = NULL);

         /**
          * Recompute all values.
          * This function has to be called after every change rendering attributes,
          * e.g. coloring or render mode.
          */
         void RecomputeAll(void);

        /**
         *	Update all parameter slots.
         *
         *	@param mol   Pointer to the data call.
		 *	@param frameID The current frame id used for the data call.
		 *	@param bs Pointer to the binding site call.
         */
		 void UpdateParameters(megamol::protein_calls::MolecularDataCall *mol, unsigned int frameID, const protein_calls::BindingSiteCall * bs = 0);

        /**********************************************************************
         * variables
         **********************************************************************/

        // caller slot
        megamol::core::CallerSlot molDataCallerSlot;
        // caller slot
        megamol::core::CallerSlot molRendererCallerSlot;
		// caller slot for offscreen rendering
        megamol::core::CallerSlot molRendererORCallerSlot;
        /** BindingSiteCall caller slot */
        megamol::core::CallerSlot bsDataCallerSlot;
		// caller slot for protein coloring
		megamol::core::CallerSlot molColorCallerSlot;
        
        /** camera information */
        vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;

        megamol::core::param::ParamSlot renderingModeParam;
        /** parameter slot for coloring mode */
        megamol::core::param::ParamSlot coloringModeParam0;
        /** parameter slot for coloring mode */
        megamol::core::param::ParamSlot coloringModeParam1;
        /** parameter slot for coloring mode weighting*/
        megamol::core::param::ParamSlot cmWeightParam;
        megamol::core::param::ParamSlot stickColoringModeParam;
        megamol::core::param::ParamSlot smoothCartoonColoringParam;
		megamol::core::param::ParamSlot compareParam;
        /** parameter slot for color table filename */
        megamol::core::param::ParamSlot colorTableFileParam;
        /** parameter slot for min color of gradient color mode */
        megamol::core::param::ParamSlot minGradColorParam;
        /** parameter slot for mid color of gradient color mode */
        megamol::core::param::ParamSlot midGradColorParam;
        /** parameter slot for max color of gradient color mode */
        megamol::core::param::ParamSlot maxGradColorParam;
        /** parameter slot for stick radius */
        megamol::core::param::ParamSlot stickRadiusParam;
		/** parameter slot for offscreen rendering */
        megamol::core::param::ParamSlot offscreenRenderingParam;
        /** parameter slot for positional interpolation */
        megamol::core::param::ParamSlot interpolParam;
        /** parameter slot for disabling rendering except protein */
        megamol::core::param::ParamSlot proteinOnlyParam;
        /** parameter slot for stick radius */
        megamol::core::param::ParamSlot tubeRadiusParam;
		/** parameter slot for refreshing in every frame*/
		megamol::core::param::ParamSlot recomputeAlwaysParam;

        // shader for per pixel lighting (polygonal view)
        vislib::graphics::gl::GLSLShader lightShader;
        // shader for tube generation (cartoon view)
        vislib::graphics::gl::GLSLGeometryShader cartoonShader;
        vislib::graphics::gl::GLSLGeometryShader tubeShader;
        vislib::graphics::gl::GLSLGeometryShader arrowShader;
        vislib::graphics::gl::GLSLGeometryShader helixShader;
        vislib::graphics::gl::GLSLGeometryShader tubeSimpleShader;
        vislib::graphics::gl::GLSLGeometryShader arrowSimpleShader;
        vislib::graphics::gl::GLSLGeometryShader helixSimpleShader;
        vislib::graphics::gl::GLSLGeometryShader tubeSplineShader;
        vislib::graphics::gl::GLSLGeometryShader arrowSplineShader;
        vislib::graphics::gl::GLSLGeometryShader helixSplineShader;
		vislib::graphics::gl::GLSLGeometryShader tubeORShader;
        vislib::graphics::gl::GLSLGeometryShader arrowORShader;
        vislib::graphics::gl::GLSLGeometryShader helixORShader;

		vislib::graphics::gl::GLSLShader sphereShader;
		vislib::graphics::gl::GLSLShader cylinderShader;
		vislib::graphics::gl::GLSLShader sphereShaderOR;
		vislib::graphics::gl::GLSLShader cylinderShaderOR;

        // current render mode
        CartoonRenderMode currentRenderMode;
        /** The current coloring mode */
        Color::ColoringMode currentColoringMode0;
        Color::ColoringMode currentColoringMode1;
        // smooth coloring of cartoon mode
        bool smoothCartoonColoringMode;

		// is comparison mode enabled?
		bool compare;

        // attribute locations for GLSL-Shader
        GLint attribLocInParams;
        GLint attribLocQuatC;
        GLint attribLocColor1;
        GLint attribLocColor2;

        // is the geometry shader (and OGL V2) supported?
        bool geomShaderSupported;

        // has the hybrid CARTOON render mode to be prepared?
        bool prepareCartoonHybrid;
        // has the CPU CARTOON render mode to be prepared?
        bool prepareCartoonCPU;
        // has the CARTOON LINE render mode to be prepared?
        bool prepareCartoonLine;

        // counters, vertex- and color-arrays for cartoon mode
        float *vertTube;
        float *normalTube;
        float *colorsParamsTube;
        unsigned int totalCountTube;
        float *vertArrow;
        float *normalArrow;
        float *colorsParamsArrow;
        unsigned int totalCountArrow;
        float *vertHelix;
        float *normalHelix;
        float *colorsParamsHelix;
        unsigned int totalCountHelix;

        // number of spline segments per amino acid
        unsigned int numberOfSplineSeg;
        // number of tube segments per 390 degrees (only used with cartoon GPU)
        unsigned int numberOfTubeSeg;
        // radius for secondary structure elements with CARTOON render modes
        float radiusCartoon;

        /** The color lookup table (for chains, amino acids,...) */
        vislib::Array<vislib::math::Vector<float, 3> > colorLookupTable;
        /** The color lookup table which stores the rainbow colors */
        vislib::Array<vislib::math::Vector<float, 3> > rainbowColors;

        /** The atom color table for rendering */
        vislib::Array<float> atomColorTable;

        // the Id of the current frame (for dynamic data)
        unsigned int currentFrameId;
        // the current call time
        float oldCallTime;

        unsigned int atomCount;
        
		// coordinates of the first (center) b-spline (result of the spline computation)
		std::vector<std::vector<vislib::math::Vector<float, 3> > > bSplineCoordsCPU;
		// coordinates of the second (direction) b-spline (result of the spline computation)
		std::vector<std::vector<vislib::math::Vector<float, 3> > > bSplineCoordsDirCPU;
		// color of secondary structure b-spline
		std::vector<std::vector<vislib::math::Vector<float, 3> > > cartoonColorCPU;

        vislib::Array<bool> atomVisible;
    };


} /* end namespace protein */
} /* end namespace megamol */

#endif // MMPROTEINPLUGIN_MOLECULECARTOONRENDERER_H_INCLUDED
