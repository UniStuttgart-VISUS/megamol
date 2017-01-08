/*
 * UncertaintyCartoonRenderer.h
 *
 * Author: Matthias Braun
 * Copyright (C) 2016 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 *
 * This module is based on the source code of "CartoonTessellationRenderer" in megamol protein plugin (svn revision 1511).
 *
 */


#ifndef MM_PROTEIN_UNCERTAINTY_PLUGIN_UNCERTAINTYCARTOONRENDERER_H_INCLUDED
#define MM_PROTEIN_UNCERTAINTY_PLUGIN_UNCERTAINTYCARTOONRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include <map>
#include <utility>

#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"

#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/GLSLTesselationShader.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/graphics/gl/IncludeAllGL.h"

#include "protein_calls/ResidueSelectionCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "UncertaintyDataCall.h"


// DEBUG output secondary strucutre type of first frame
//#define FIRSTFRAME_CHECK


namespace megamol {
	namespace protein_uncertainty {

	using namespace megamol::core;
	using namespace megamol::protein_calls;
	using namespace vislib::graphics::gl;

    /**
     * Renderer for simple sphere glyphs
     */
    class UncertaintyCartoonRenderer : public megamol::core::view::Renderer3DModule {
    public:


        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "UncertaintyCartoonRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Offers cartoon renderings for biomolecules (uses Tessellation Shaders).";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
#ifdef _WIN32
#if defined(DEBUG) || defined(_DEBUG)
            HDC dc = ::wglGetCurrentDC();
            HGLRC rc = ::wglGetCurrentContext();
            ASSERT(dc != NULL);
            ASSERT(rc != NULL);
#endif // DEBUG || _DEBUG
#endif // _WIN32
            return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable()
                && vislib::graphics::gl::GLSLTesselationShader::AreExtensionsAvailable()
                && isExtAvailable("GL_ARB_buffer_storage")
                && ogl_IsVersionGEQ(4,4);
        }

        /** Ctor. */
		UncertaintyCartoonRenderer(void);

        /** Dtor. */
		virtual ~UncertaintyCartoonRenderer(void);

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
        virtual bool GetCapabilities(Call& call);

        /**
        * The get extents callback. The module should set the members of
        * 'call' to tell the caller the extents of its data (bounding boxes
        * and times).
        *
        * @param call The calling call.
        *
        * @return The return value of the function.
        */
        virtual bool GetExtents(Call& call);

        /**
		* The ...
		*
		* @param t          The ...
		* @param outScaling The ...
		*
		* @return The pointer to the molecular data call.
        */
        MolecularDataCall *GetData(unsigned int t, float& outScaling);

        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Render(Call& call);

    private:

		/**
		* The ... .
		*
		* @param udc The uncertainty data call.
		* @param mol The molecular data call.
		*
		* @return The return value of the function.
		*/
		bool GetUncertaintyData(UncertaintyDataCall *udc, MolecularDataCall *mol);

        /**
         * The ... .
         *
         * @param mol        The ...
         * @param colBytes   The ...
         * @param vertBytes  The ...
         * @param colStride  The ...
         * @param vertStride The ...         
         */        
		void GetBytesAndStride(MolecularDataCall &mol, unsigned int &colBytes, unsigned int &vertBytes, unsigned int &colStride, unsigned int &vertStride);
        
        /**
         * The ... .
         *
         * @param syncObj The ...        
         */  
        void QueueSignal(GLsync& syncObj);
        
        /**
         * The ... .
         *
         * @param syncObj The ...        
         */          
		void WaitSignal(GLsync& syncObj);
        
        /** Strucutre to hold C-alpha data */
		struct CAlpha
		{
			float pos[4];                                               // position of the C-alpha atom
			float dir[3];                                               // direction of the amino-acid
			int   colIdx;                                               // UNUSED - don't delete ... shader is corrupt otherwise - WHY?
			float col[4];                                               // the color of the amino-acid or chain (depending on coloring mode)
			float diff;                                                 // the uncertainty difference
			int   flag;                                                 // UNUSED - the amino-acid flag (none, missing, heterogen)
			float unc[UncertaintyDataCall::secStructure::NOE];          // the uncertainties of the sctructure assignments                    - used for dithering
			int   sortedStruct[UncertaintyDataCall::secStructure::NOE]; // the sorted structure assignments: max=[0] to min=[NOE]             - used for dithering
		};

		/**
		* ... .
		*
		* @return The ... .
		*/
		bool loadTubeShader(void);

		/**
		* structure color for uncertain structure assignment
		*/
		enum coloringModes {
			COLOR_MODE_STRUCT        = 0,
			COLOR_MODE_UNCERTAIN     = 1,
			COLOR_MODE_CHAIN         = 2,
			COLOR_MODE_AMINOACID     = 3,
			COLOR_MODE_RESIDUE_DEBUG = 4
		};

		/**
		* structure for uncertain visualisations
		*/
		enum uncVisualisations {
			UNC_VIS_NONE   = 0,
			UNC_VIS_SIN_U  = 1,
			UNC_VIS_SIN_V  = 2,
			UNC_VIS_SIN_UV = 3,

		};

        /**********************************************************************
         * variables
         **********************************************************************/
         
#ifdef FIRSTFRAME_CHECK
		bool firstFrame;
#endif
        /** The call for PDB data */
        core::CallerSlot getPdbDataSlot;
	    /** The call for uncertainty data */
        core::CallerSlot uncertaintyDataSlot;	
		/** residue selection caller slot */
		core::CallerSlot resSelectionCallerSlot;
                
        // paramter
        core::param::ParamSlot scalingParam;
		core::param::ParamSlot sphereParam;
		core::param::ParamSlot backboneParam;
		core::param::ParamSlot backboneWidthParam;
		core::param::ParamSlot materialParam;
		core::param::ParamSlot lineDebugParam;
		core::param::ParamSlot buttonParam;
		core::param::ParamSlot colorInterpolationParam;
		core::param::ParamSlot tessLevelParam;
		core::param::ParamSlot colorModeParam;
		core::param::ParamSlot onlyTubesParam;
		core::param::ParamSlot colorTableFileParam;
		core::param::ParamSlot lightPosParam;
		core::param::ParamSlot uncVisParam;
		core::param::ParamSlot uncDistorParam;
        core::param::ParamSlot ditherParam;
                
        GLuint              vertArray;
        std::vector<GLsync> fences;           // (?)
		GLuint              theSingleBuffer;
        unsigned int        currBuf;
        GLuint              colIdxAttribLoc;  // (?)
        GLsizeiptr          bufSize;
		int                 numBuffers;
		void               *theSingleMappedMem;
		GLuint              singleBufferCreationBits;
        GLuint              singleBufferMappingBits;

		int                            currentTessLevel;
		uncVisualisations              currentUncVis;
	    coloringModes                  currentColoringMode;
		vislib::math::Vector<float, 4> currentLightPos;
		float                          currentScaling;
		float                          currentBackboneWidth;
		vislib::math::Vector<float, 4> currentMaterial;
		vislib::math::Vector<float, 4> currentUncDist;
        int                            currentDitherMode;

		/** shader for the tubes */
		vislib::graphics::gl::GLSLTesselationShader tubeShader;
		/** shader for the spheres (raycasting view) */
		vislib::graphics::gl::GLSLShader            sphereShader;

		vislib::SmartPtr<ShaderSource> vert;
        vislib::SmartPtr<ShaderSource> tessCont;
        vislib::SmartPtr<ShaderSource> tessEval;
        vislib::SmartPtr<ShaderSource> geom;
        vislib::SmartPtr<ShaderSource> frag;
		vislib::SmartPtr<ShaderSource> tubeVert;
        vislib::SmartPtr<ShaderSource> tubeTessCont;
        vislib::SmartPtr<ShaderSource> tubeTessEval;
        vislib::SmartPtr<ShaderSource> tubeGeom;
        vislib::SmartPtr<ShaderSource> tubeFrag;

        // the number of different structure types
        unsigned int structCount;
        
        // C-alpha main chain
		std::vector<CAlpha> mainChain;
        
		// the total number of amino-acids defined in molecular data
		unsigned int molAtomCount;

		// the total number of amino-acids defined in uncertainty data
		unsigned int aminoAcidCount;
		// The original PDB index
		vislib::Array<vislib::StringA> pdbIndex;
		// The synchronized index between molecular data and uncertainty data
		vislib::Array<unsigned int> synchronizedIndex;
		// the array for the residue flag
		vislib::Array<unsigned int> residueFlag;
		/** The uncertainty difference of secondary structure types */
		vislib::Array<float> diffUncertainty;
		// The secondary structure assignment methods and their secondary structure type assignments
		vislib::Array<vislib::Array<UncertaintyDataCall::secStructure> > secStructAssignment;
		// The values of the secondary structure uncertainty for each amino-acid 
		vislib::Array<vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> > secUncertainty;
		// The sorted structure types of the uncertainty values
		vislib::Array<vislib::math::Vector<unsigned int, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> > sortedUncertainty;

		// color table for chain id per amino acid
		vislib::Array<vislib::math::Vector<float, 3> > chainColors;
		// color table for amino acid per amino acid
		vislib::Array<vislib::math::Vector<float, 3> > aminoAcidColors;
		// secondary structure type colors as RGB(A)
		vislib::Array<vislib::math::Vector<float, 4> > secStructColorRGB;
		// color table
		vislib::Array<vislib::math::Vector<float, 3> > colorTable;

		// positions of C-alpha-atoms and O-atoms
		vislib::Array<vislib::Array<float> > positionsCa;
		vislib::Array<vislib::Array<float> > positionsO;

		// selection 
		vislib::Array<bool> selection;                                     // unused so far ...
		protein_calls::ResidueSelectionCall *resSelectionCall;             // unused so far ...
    };

	} /* end namespace protein_uncertainty */
} /* end namespace megamol */

#endif /* MM_PROTEIN_UNCERTAINTY_PLUGIN_UNCERTAINTYCARTOONRENDERER_H_INCLUDED */
