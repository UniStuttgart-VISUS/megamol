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
		*
		* @return The return value of the function.
		*/
		bool GetUncertaintyData(UncertaintyDataCall *udc);

        /**
		 * UNUSED ...
		 *
         * The ... .
         *
         * @param mol     The ...
         * @param vertBuf The ...
         * @param vertPtr The ...
         * @param colBuf  The ...
         * @param colPtr  The ...         
         */
        // void SetPointers(MolecularDataCall &mol, GLuint vertBuf, const void *vertPtr, GLuint colBuf, const void *colPtr);
        
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
         * @param mol        The ...
         * @param colBytes   The ...
         * @param vertBytes  The ...
         * @param colStride  The ...
         * @param vertStride The ...         
         */  
		void GetBytesAndStrideLines(MolecularDataCall &mol, unsigned int &colBytes, unsigned int &vertBytes, unsigned int &colStride, unsigned int &vertStride);

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
			float pos[4];
			float dir[3];
			int type;
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
		core::param::ParamSlot lineParam;
		core::param::ParamSlot backboneParam;
		core::param::ParamSlot backboneWidthParam;
		core::param::ParamSlot materialParam;
		core::param::ParamSlot lineDebugParam;
		core::param::ParamSlot buttonParam;
		core::param::ParamSlot colorInterpolationParam;
                
        GLuint              vertArray;
        std::vector<GLsync> fences;           // (?)
		GLuint              theSingleBuffer;
        unsigned int        currBuf;
        GLuint              colIdxAttribLoc;
        GLsizeiptr          bufSize;
		int                 numBuffers;
		void               *theSingleMappedMem;
		GLuint              singleBufferCreationBits;
        GLuint              singleBufferMappingBits;
        
        //typedef std::map<std::pair<int, int>, std::shared_ptr<GLSLShader> > shaderMap; // unused
        
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

        // positions of C-alpha-atoms and O-atoms
        vislib::Array<vislib::Array<float> > positionsCa;
        vislib::Array<vislib::Array<float> > positionsO;

        // C-alpha main chain
		std::vector<CAlpha> mainChain;
        
        /** shader for the spheres (raycasting view) */
        vislib::graphics::gl::GLSLShader            sphereShader;
        /** shader for spline rendering */
        vislib::graphics::gl::GLSLTesselationShader splineShader;
		/** shader for the tubes */
		vislib::graphics::gl::GLSLTesselationShader tubeShader;


		// the total number of amino-acids 
		unsigned int aminoAcidCount;
		// the array for the residue flag
		vislib::Array<UncertaintyDataCall::addFlags> residueFlag;
		// The secondary structure assignment methods and their secondary structure type assignments
		vislib::Array<vislib::Array<UncertaintyDataCall::secStructure> > secStructAssignment;
		// The values of the secondary structure uncertainty for each amino-acid 
		vislib::Array<vislib::math::Vector<float, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> > secUncertainty;
		// The sorted structure types of the uncertainty values
		vislib::Array<vislib::math::Vector<UncertaintyDataCall::secStructure, static_cast<int>(UncertaintyDataCall::secStructure::NOE)> > sortedUncertainty;

		// secondary structure type colors as RGB(A)
		vislib::Array<vislib::math::Vector<float, 4> > secStructColorRGB;
		// secondary structure type colors as HSL(A)
		vislib::Array<vislib::math::Vector<float, 4> > secStructColorHSL;  // unused so far ...


		// selection 
		vislib::Array<bool> selection;
		protein_calls::ResidueSelectionCall *resSelectionCall;
    };

	} /* end namespace protein_uncertainty */
} /* end namespace megamol */

#endif /* MM_PROTEIN_UNCERTAINTY_PLUGIN_UNCERTAINTYCARTOONRENDERER_H_INCLUDED */
