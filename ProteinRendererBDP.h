/*
 * ProteinRendererBDP.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_BUCKETDEPTHPEELING_H_INCLUDED
#define MEGAMOL_BUCKETDEPTHPEELING_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#include "CallProteinData.h"
#include "CallerSlot.h"
#include "CallFrame.h"
#include "ReducedSurface.h"
#include "ReducedSurfaceSimplified.h"
#include "view/Renderer3DModule.h"
#include "view/CallRender3D.h"
#include "param/ParamSlot.h"
#include "vislib/FpsCounter.h"
#include "vislib/GLSLShader.h"
#include "vislib/GLSLGeometryShader.h"
#include "vislib/Quaternion.h"
#include "vislib/SimpleFont.h"
#include <vector>
#include <set>
#include <algorithm>
#include <list>

#define NUM_BUFFERS 8  // DON'T CHANGE !

namespace megamol {
namespace protein {

    /**
     * Molecular Surface Renderer using Bucket Sort Depth Peeling (BDP).
     *
     * Computes the solvent excluded (Connolly) surface and 
     * renders the SES with Bucket Sort Depth Peeling.
     */
	class ProteinRendererBDP : public megamol::core::view::Renderer3DModule
    {
    public:

        /** depth peeling modi */
        enum DepthPeelingMode
        {
            BDP = 0,
            ADAPTIVE_BDP = 1,
            DUAL_DEPTHPEELING = 2
        };
                
        /** postprocessing modi */
        enum PostprocessingMode
        {
            NONE = 0,
            AMBIENT_OCCLUSION = 1,
            SILHOUETTE = 2,
            TRANSPARENCY = 3
        };

        /** render modi */
        enum RenderMode
        {
            GPU_RAYCASTING = 0,
            //POLYGONAL = 1,
            //POLYGONAL_GPU = 2,
            GPU_SIMPLIFIED = 3
        };

        /** coloring modi for the atoms */
        enum ColoringMode
        {
            ELEMENT   = 0,
            AMINOACID = 1,
            STRUCTURE = 2,
            CHAIN_ID  = 3,
            VALUE     = 4,
            RAINBOW   = 5,
            CHARGE    = 6
        };

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void)
        {
            return "ProteinRendererBDP";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) 
        {
            return "Offers protein surface renderings.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) 
        {
            //return true;
            return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable();
        }
        
        /** ctor */
        ProteinRendererBDP(void);
        
        /** dtor */
        virtual ~ProteinRendererBDP(void);

       /**********************************************************************
         * 'get'-functions
         **********************************************************************/
        
        /** Get probe radius */
        const float GetProbeRadius() const { return probeRadius; };

        /**********************************************************************
         * 'set'-functions
         **********************************************************************/

        /** Set probe radius */
        void SetProbeRadius( const float rad) { probeRadius = rad; };

        /** set the color of the silhouette */
        void SetSilhouetteColor( float r, float g, float b) { silhouetteColor.Set( r, g, b);
            codedSilhouetteColor = int( r * 255.0f)*1000000 + int( g * 255.0f)*1000 + int( b * 255.0f); };
        void SetSilhouetteColor( vislib::math::Vector<float, 3> color) { 
            SetSilhouetteColor( color.GetX(), color.GetY(), color.GetZ()); };

        /** set the color for minimum value (VALUE coloring mode) */
        void SetMinValueColor( float r, float g, float b) { minValueColor.Set( r, g, b);
            codedMinValueColor = int( r * 255.0f)*1000000 + int( g * 255.0f)*1000 + int( b * 255.0f); };
        void SetMinValueColor( vislib::math::Vector<float, 3> color) { 
            SetMinValueColor( color.GetX(), color.GetY(), color.GetZ()); };
        /** set the color for maximum value (VALUE coloring mode) */
        void SetMaxValueColor( float r, float g, float b) { maxValueColor.Set( r, g, b);
            codedMaxValueColor = int( r * 255.0f)*1000000 + int( g * 255.0f)*1000 + int( b * 255.0f); };
        void SetMaxValueColor( vislib::math::Vector<float, 3> color) { 
            SetMaxValueColor( color.GetX(), color.GetY(), color.GetZ()); };
        /** set the color for mean value (VALUE coloring mode) */
        void SetMeanValueColor( float r, float g, float b) { meanValueColor.Set( r, g, b);
            codedMeanValueColor = int( r * 255.0f)*1000000 + int( g * 255.0f)*1000 + int( b * 255.0f); };
        void SetMeanValueColor( vislib::math::Vector<float, 3> color) { 
            SetMeanValueColor( color.GetX(), color.GetY(), color.GetZ()); };
        
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
        
        /**
         * Render atoms as spheres using GLSL Raycasting shaders.
         *
         * @param protein The protein data interface.
         * @param scale The scale factor for the atom radius.
         */
        void RenderAtomsGPU( const CallProteinData *protein, 
            const float scale = 1.0f);

        /**
         * Renders the probe atom at position 'm'.
         *
         * @param m The probe position.
         */
        void RenderProbe( const vislib::math::Vector<float, 3> m);
        void RenderProbeGPU( const vislib::math::Vector<float, 3> m);

        /**
         * Compute all vertex, attribute and color arrays used for ray casting 
         * all molecular surfaces (spheres, spherical triangles, tori).
         */
        void ComputeRaycastingArrays();

        /**
         * Compute all vertex, attribute and color arrays used for ray casting 
         * the molecular surface 'ixdRS' (spheres, spherical triangles, tori).
         * @param idxRS The index of the reduced surface.
         */
        void ComputeRaycastingArrays( unsigned int idxRS);

        /**
         * Compute all vertex, attribute and color arrays used for ray casting 
         * the simplified molecular surface.
         */
        void ComputeRaycastingArraysSimple();

        /**
         * Code a RGB-color into one float.
         * For each color channel, its representation in range 0..255 is computed
         * and stores as follows:
         * rrrgggbbb.0
         * Note that the minimum value for the coded color is 0 and the maximum
         * value is 255255255.0 .
         *
         * @param col Vector containing the color as float [0.0]..[1.0] .
         * @return The coded color value.
         */
        float CodeColor( const vislib::math::Vector<float, 3> &col) const;

        /**
         * Decode a coded color to the original RGB-color.
         *
         * @param codedColor Integer value containing the coded color (rrrgggbbb).
         * @return The RGB-color value vector.
         */
        vislib::math::Vector<float, 3> DecodeColor( int codedColor) const;

        /**
         * Creates the frame buffer object and textures needed for offscreen rendering.
         */
        void CreatePostProcessFBO();

        /**
         * Render the molecular surface using GPU raycasting.
         *
         * @param protein Pointer to the protein data interface.
         */
        void RenderSESGpuRaycasting( const CallProteinData *protein);

        /**
         * Render the molecular surface using GPU raycasting.
         *
         * @param protein Pointer to the protein data interface.
         */
        void RenderSESGpuRaycastingSimple( const CallProteinData *protein);
        
        /**
         * Render debug stuff --- THIS IS ONLY FOR DEBUGGING PURPOSES, REMOVE IN FINAL VERSION!!!
         *
         * @param protein Pointer to the protein data interface.
         */
        void RenderDebugStuff( const CallProteinData *protein);

        /**
         * Postprocessing: use screen space ambient occlusion
         */
        void PostprocessingSSAO();

        /**
         * Postprocessing: use silhouette shader
         */
        void PostprocessingSilhouette();
        
        /**
         * Postprocessing: transparency (blend two images)
         */
        void PostprocessingTransparency( float transparency);
        
        /**
         * Fill amino acid color table.
         */
        void FillAminoAcidColorTable();

        /**
         * Creates a rainbow color table with 'num' entries.
         *
         * @param num The number of color entries.
         */
        void MakeRainbowColorTable( unsigned int num);
        
        /**
         * Returns the color of the atom 'idx' for the current coloring mode
         *
         * @param idx The index of the atom.
         * @return The color of the atom with the index 'idx'.
         */
        vislib::math::Vector<float, 3> GetProteinAtomColor( unsigned int idx);

        /**
         * Make color table for all atoms
         *
         * @param prot The protein data interface.
         * @param forceRecompute If 'true', the color Table is recomputed, otherwise only if necessary.
         */
        void MakeColorTable( const CallProteinData *prot, bool forceRecompute = true);

        /*
         * Create the cutting planes textures for sphere interior clipping
         * which store for every RS-vertex (of all molecular surfaces) the positions 
         * (and implicit the normals) of the planes cutting tori and spheres.
         */
        void CreateCutPlanesTextures();

        /*
         * Creates the cutting planes texture for sphere interior clipping
         * which stores for every RS-vertex (of the reduced surface 'idxRS') 
         * the positions (and implicit the normals) of the planes cutting 
         * tori and spheres.
         * 
         * @param idxRS Index of reduced surface
         */
        void CreateCutPlanesTexture( unsigned int idxRS);

        /**
         * Create the singularity textureS which stores for every RS-edge (of all
         * molecular surfaces) the positions of the probes that cut it.
         */
        void CreateSingularityTextures();
        
        /**
         * Create the singularity textureS which stores for every RS-edge (of all
         * molecular surfaces) the positions of the probes that cut it.
         */
        void CreateSingularityTexturesSimple();
        
        /**
         * Create the singularity texture for the reduced surface 'idxRS' which
         * stores for every RS-edge the positions of the probes that cut it.
         */
        void CreateSingularityTexture( unsigned int idxRS);


        /**
         * Creates the frame buffer object and textures needed for depth peeling.
         */
        void CreateDepthPeelingFBO();

        /**
         * Creates a display list for the bounding box vertices
         * 
         * @param bbox The bounding box of the protein
         */
        inline void createBBoxDisplayList(megamol::core::BoundingBoxes& bbox);

        /**
         * Creates the min max depth buffer
         */
        void createMinMaxDepthBuffer(void);

        /**
         * Render depth peeling result (blending)
         */
        void renderDepthPeeling(void);

        /**
         * Render the min max depth buffer (DEBUG output)
         */
        void renderMinMaxDepthBuffer(void);

        /**
         * Creates display list for a fullscreen quad
         */
        void createFullscreenQuadDisplayList(void);

    private:

        /**
         * The get capabilities callback. The module should set the members
         * of 'call' to tell the caller its capabilities.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetCapabilities(megamol::core::Call& call);

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetExtents(megamol::core::Call& call);

        /**
         * Open GL Render call.
         *
         * @param call The calling call.
         * @return The return value of the function.
         */
        virtual bool Render( megamol::core::Call& call);

        /**
         * Deinitialises this renderer. This is only called if there was a 
         * successful call to "initialise" before.
         */
        virtual void deinitialise(void);
        
        /**********************************************************************
         * variables
         **********************************************************************/
        
        // caller slot
        megamol::core::CallerSlot protDataCallerSlot;
        
        // camera information
        vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;
        
		megamol::core::param::ParamSlot postprocessingParam;
        megamol::core::param::ParamSlot rendermodeParam;
        megamol::core::param::ParamSlot coloringmodeParam;
        megamol::core::param::ParamSlot silhouettecolorParam;
        megamol::core::param::ParamSlot sigmaParam;
        megamol::core::param::ParamSlot lambdaParam;
        megamol::core::param::ParamSlot minvaluecolorParam;
        megamol::core::param::ParamSlot maxvaluecolorParam;
        megamol::core::param::ParamSlot meanvaluecolorParam;
        megamol::core::param::ParamSlot fogstartParam;
        megamol::core::param::ParamSlot debugParam;
        megamol::core::param::ParamSlot drawSESParam;
        megamol::core::param::ParamSlot drawSASParam;

        // not used yet ...
        megamol::core::param::ParamSlot depthPeelingParam;

        bool drawRS;
        bool drawSES;
        bool drawSAS;

        /** the reduced surface(s) */
        std::vector<ReducedSurface*> reducedSurface;
        /** the simplified reduced surface(s) */
        std::vector<ReducedSurfaceSimplified*> simpleRS;
        
        // shader for the cylinders (raycasting view)
        vislib::graphics::gl::GLSLShader cylinderShader;
        // shader for the spheres (raycasting view)
        vislib::graphics::gl::GLSLShader sphereShader;
        // shader for the spheres with clipped interior (raycasting view)
        vislib::graphics::gl::GLSLShader sphereClipInteriorShader;
        // shader for the spherical triangles (raycasting view)
        vislib::graphics::gl::GLSLShader sphericalTriangleShader;
        // shader for torus (raycasting view)
        vislib::graphics::gl::GLSLShader torusShader;
        // shader for per pixel lighting (polygonal view)
        vislib::graphics::gl::GLSLShader lightShader;
        // shader for 1D gaussian filtering (postprocessing)
        vislib::graphics::gl::GLSLShader hfilterShader;
        vislib::graphics::gl::GLSLShader vfilterShader;
        // shader for silhouette drawing (postprocessing)
        vislib::graphics::gl::GLSLShader silhouetteShader;
        // shader for cheap transparency (postprocessing/blending)
        vislib::graphics::gl::GLSLShader transparencyShader;
        // shader creating min max depth buffer
        vislib::graphics::gl::GLSLShader createDepthBufferShader;
        // shader for blending depth peeling result
        vislib::graphics::gl::GLSLShader renderDepthPeelingShader;

        // DEBUG: render min max depth buffer
        vislib::graphics::gl::GLSLShader renderDepthBufferShader;

        // epsilon value for float-comparison
        float epsilon;

        // radius of the probe atom
        float probeRadius;

        std::vector<vislib::math::Vector<float, 3> > atomColor;
        unsigned int currentArray;

        /** 'true' if the data for the current render mode is computed, 'false' otherwise */
        bool preComputationDone;

        /** current render mode */
        RenderMode currentRendermode;
        /** current coloring mode */
        ColoringMode currentColoringMode;
        /** postprocessing mode */
        PostprocessingMode postprocessing;
        /** current depth peeeling mode*/
        DepthPeelingMode depthpeeling;
        
        /** vertex and attribute arrays for raycasting the tori */
        std::vector<vislib::Array<float> > torusVertexArray;
        std::vector<vislib::Array<float> > torusInParamArray;
        std::vector<vislib::Array<float> > torusQuatCArray;
        std::vector<vislib::Array<float> > torusInSphereArray;
        std::vector<vislib::Array<float> > torusColors;
        std::vector<vislib::Array<float> > torusInCuttingPlaneArray;
        /** vertex and attribute arrays for raycasting the spherical triangles */
        std::vector<vislib::Array<float> > sphericTriaVertexArray;
        std::vector<vislib::Array<float> > sphericTriaVec1;
        std::vector<vislib::Array<float> > sphericTriaVec2;
        std::vector<vislib::Array<float> > sphericTriaVec3;
        std::vector<vislib::Array<float> > sphericTriaProbe1;
        std::vector<vislib::Array<float> > sphericTriaProbe2;
        std::vector<vislib::Array<float> > sphericTriaProbe3;
        std::vector<vislib::Array<float> > sphericTriaTexCoord1;
        std::vector<vislib::Array<float> > sphericTriaTexCoord2;
        std::vector<vislib::Array<float> > sphericTriaTexCoord3;
        std::vector<vislib::Array<float> > sphericTriaColors;
        /** vertex, color and attribute arrays for raycasting the spheres */
        std::vector<vislib::Array<float> > sphereTexCoord;
        std::vector<vislib::Array<float> > sphereVertexArray;
        std::vector<vislib::Array<float> > sphereColors;
        std::vector<vislib::Array<float> > sphereSurfVector;

        // FBOs
        GLuint colorFBO;
        GLuint blendFBO;
        GLuint horizontalFilterFBO;
        GLuint verticalFilterFBO;
        GLuint depthBufferFBO;
        GLuint depthPeelingFBO;

        // textures for FBOs
        GLuint texture0;
        GLuint depthTex0;
        GLuint texture1;
        GLuint depthTex1;
        GLuint hFilter;
        GLuint vFilter;
        GLuint depthBuffer;
        GLuint depthPeelingTex[NUM_BUFFERS];

        // FBO color buffer indices
        GLenum colorBufferIndex[NUM_BUFFERS];

        // display list for bbox
        GLuint bboxList;
        // display list for fullscreen quad
        GLuint fsQuadList;

        // width and height of view
        unsigned int width;
        unsigned int height;
        // sigma factor for screen space ambient occlusion
        float sigma;
        // lambda factor for screen space ambient occlusion
        float lambda;
        
        // color table for amino acids
        vislib::Array<vislib::math::Vector<float, 3> > aminoAcidColorTable;
        // color table for rainbow colors
        std::vector<vislib::math::Vector<float,3> > rainbowColors;

        // texture for singularity handling (concave triangles)
        std::vector<GLuint> singularityTexture;
        // sizes of singularity textures
        std::vector<unsigned int> singTexWidth, singTexHeight;
        // data of the singularity texture
        float *singTexData;
        
        // texture indices for interior clipping cutting planes 
        // (convex spherical cutouts)
        std::vector<GLuint> cutPlanesTexture;
        // sizes of the cutting planes textures
        std::vector<unsigned int> cutPlanesTexWidth, cutPlanesTexHeight;
        // texture data of the cutting planes textures
        std::vector<vislib::Array<float> > cutPlanesTexData;

        // silhouette color
        vislib::math::Vector<float, 3> silhouetteColor;
        int codedSilhouetteColor;

        // minimum and maximum value for VALUE coloring mode
        float minValue, maxValue;
        // colors for min, max and mean value
        vislib::math::Vector<float, 3> minValueColor, maxValueColor, meanValueColor;
        int codedMinValueColor, codedMaxValueColor, codedMeanValueColor;
        
        // start value for fogging
        float fogStart;
        // transparency value
        float transparency;
        
        // fps counter
        vislib::graphics::FpsCounter fpsCounter;

    };

} /* end namespace protein */
} /* end namespace megamol */

#endif /* MEGAMOL_BUCKETDEPTHPEELING_H_INCLUDED */
