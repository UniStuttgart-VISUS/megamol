/*
 * ProteinRendererSESGPU.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_PROTRENSESGPU_H_INCLUDED
#define MEGAMOL_PROTRENSESGPU_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */

#include "CallProteinData.h"
#include "param/ParamSlot.h"
#include "CallerSlot.h"
#include "CallFrame.h"
#include "view/Renderer3DModule.h"
#include "view/CallRender3D.h"
#include "vislib/FpsCounter.h"
#include "vislib/GLSLShader.h"
#include "vislib/GLSLGeometryShader.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include "vislib/FramebufferObject.h"

#define CHECK_FOR_OGL_ERROR() do { GLenum err; err = glGetError();if (err != GL_NO_ERROR) { fprintf(stderr, "%s(%d) glError: %s\n", __FILE__, __LINE__, gluErrorString(err)); } } while(0)

namespace megamol {
namespace protein {

    /**
     * Molecular Surface Renderer class.
     * Computes and renders the Solvent Excluded Surface on the GPU.
     */
    class ProteinRendererSESGPU : public megamol::core::view::Renderer3DModule
    {
    public:
        
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void)
        {
            return "ProteinRendererSESGPU";
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
        ProteinRendererSESGPU(void);
        
        /** dtor */
        virtual ~ProteinRendererSESGPU(void);

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
         * Creates a FBO for visibility tests with the given size.
         * @param size      The maximum size for the FBO textures.
         * @param width     Out parameter: texture width.
         * @param height    Out parameter: texture height.
         */
        void createVisibilityFBO( unsigned int size, unsigned int &width,
            unsigned int &height );

        /**
         * Renders all atoms using GPU ray casting and write atom ID to red color channel.
         * @param protein The protein call.
         */
        void RenderAtomIdGPU( const CallProteinData *protein);

        /**
         * Create the FBO for visibility test.
         * @param maxSize The maximum dimension for width/height.
         */
        void CreateVisibilityFBO( unsigned int maxSize);

        /**
         * Create the FBO for visible atoms.
         * @param atomCount The number of protein atoms.
         */
        void CreateVisibleAtomsFBO( unsigned int atomCount);

        /**
         * Find all visible atoms
         * @param protein The protein call.
         */
        void FindVisibleAtoms( const CallProteinData *protein);

        /**
         * Find for each visible atom all atoms that are in the proximity.
         * @param protein The protein call.
         */
        void ComputeVisibleAtomsVicinityTable( const CallProteinData *protein);
        
        /**
         * Compute the Reduced Surface using the Geometry Shader.
         * @param protein The protein call.
         */
        void ComputeRSGeomShader( const CallProteinData *protein);
        
        /**
         * Compute the Reduced Surface using the Fragment Shader.
         * @param protein The protein call.
         */
        void ComputeRSFragShader( const CallProteinData *protein);

        /**
         * Creates the FBO for reduced surface triangle generation.
         * @param atomCount The total number of atoms.
         * @param vicinityCount The maximum number of vicinity atoms.
         */
        void CreateTriangleFBO( unsigned int atomCount, unsigned int vicinityCount);

        /**
         * Creates the FBO for visible reduced surface triangles.
         */
        void CreateVisibleTriangleFBO( unsigned int atomCount, unsigned int vicinityCount);

        /**
         * Render all potential RS-faces as triangles using a vertex shader.
         * @param protein The protein call.
         */
        void RenderTriangles( const CallProteinData *protein);

        /**
         * Find all visible triangles (i.e. visible RS-faces).
         * @param protein The protein call.
         */
        void FindVisibleTriangles( const CallProteinData *protein);

        /**
         * Create fbo for adjacent triangles of visible triangles
         * @param atomCount The total number of atoms.
         * @param vicinityCount The maximum number of vicinity atoms.
         */
        void CreateAdjacentTriangleFBO( unsigned int atomCount, unsigned int vicinityCount);

        /**
         * Find the adjacent triangles to all visible triangles.
         * @param protein The protein call.
         */
        void FindAdjacentTriangles( const CallProteinData *protein);

        /**
         * Create the VBO for transform feedback.
         */
        void CreateTransformFeedbackVBO();

        /**
         * Create geometric primitives for ray casting.
         * @param protein The protein call.
         * @return The number of primitives which were read back.
         */
        unsigned int CreateGeometricPrimitives( const CallProteinData *protein);

        /**
         * Find all intersecting probes for each probe and create the singularity texture.
         * @param protein The protein call.
         * @param numProbes The number of RS-faces (i.e. probes in fixed positions).
         */
        void CreateSingularityTexture( const CallProteinData *protein, unsigned int numProbes);

        /**
         * Render the SES using GPU ray casting.
         * @param protein The protein call.
         * @param primitiveCount The number of primitives.
         */
        void RenderSES( const CallProteinData *protein, unsigned int primitiveCount);

        /**
         * Render all visible atoms using GPU ray casting.
         * @param protein The protein call.
         */
        void RenderVisibleAtomsGPU( const CallProteinData *protein);

        /**
         * Mark all atoms which are vertices of adjacent triangles as visible
         */
        void MarkAdjacentAtoms( const CallProteinData *protein);

    private:
        
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
        
        // shaders
        vislib::graphics::gl::GLSLShader writeSphereIdShader;
        vislib::graphics::gl::GLSLShader drawPointShader;
        vislib::graphics::gl::GLSLShader sphereShader;
        vislib::graphics::gl::GLSLGeometryShader reducedSurfaceGeomShader;
        vislib::graphics::gl::GLSLShader reducedSurfaceShader;
        vislib::graphics::gl::GLSLShader drawTriangleShader;
        vislib::graphics::gl::GLSLGeometryShader drawVisibleTriangleShader;
        vislib::graphics::gl::GLSLShader sphericalTriangleShader;
        vislib::graphics::gl::GLSLShader torusShader;
        vislib::graphics::gl::GLSLShader adjacentTriangleShader;
        vislib::graphics::gl::GLSLShader adjacentAtomsShader;

        // camera information
        vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;
        
        // the bounding box of the protein
        vislib::math::Cuboid<float> bBox;

        // fps counter
        vislib::graphics::FpsCounter fpsCounter;
        unsigned int printFps;

        // current width and height of camera view
        unsigned int width, height;

        // the GL clear color
        float clearCol[4];

		// start value for fogging
		float fogStart;
		// transparency value
		float transparency;

        // the radius of the probe defining the SES
        float probeRadius;

        // ----- variables for atom visibility test -----
        // the ID array (contains IDs from 0 .. n)
        float *proteinAtomId;
        unsigned int proteinAtomCount;
        // visibility FBO, textures and parameters
        GLuint visibilityFBO;
        GLuint visibilityColor;
        GLuint visibilityDepth;
        unsigned int visibilityTexWidth, visibilityTexHeight;
        // variables for drawing visibility texture as vertices (data + VBO)
        float *visibilityVertex;
        GLuint visibilityVertexVBO;
        // FBO, textures and parameters for visible atoms
        GLuint visibleAtomsFBO;
        GLuint visibleAtomsColor;
        GLuint visibleAtomsDepth;
        // array for visible atoms maks
        float *visibleAtomMask;
        // ----- vicinity table for visible atoms -----
        float *vicinityTable;
        unsigned int voxelLength;
        unsigned int *voxelMap;
        unsigned int voxelMapSize;
        unsigned int numAtomsPerVoxel;
        GLuint vicinityTableTex;
        float *visibleAtomsList;
        float *visibleAtomsIdList;
        unsigned int visibleAtomCount;
        GLuint visibleAtomsTex;
        GLuint visibleAtomsIdTex;
        // vertex and color arrays for geometry shader implementation
        float* atomPosRSGS;
        float* atomColRSGS;
        // ----- FBO and textures for triangle (i.e. potential RS-faces) computation
        GLuint triangleFBO;
        GLuint triangleColor0;
        GLuint triangleColor1;
        GLuint triangleColor2;
        GLuint triangleNormal;
        GLuint triangleDepth;
        // VBO and data arrays for triangle drawing
        GLuint triangleVBO;
        float* triangleVertex;
        // ----- FBO and textures for visible triangles (i.e. visible RS-faces)
        GLuint visibleTriangleFBO;
        GLuint visibleTriangleColor;
        GLuint visibleTriangleDepth;
        // ----- render to VBO -----
        GLuint visibilityTexVBO;
        // ----- FBO and textures for finding adjacent triangles -----
        GLuint adjacentTriangleFBO;
        GLuint adjacentTriangleColor;
        GLuint adjacentTriangleDepth;
        float *adjacentTriangleVertex;
        GLuint adjacentTriangleVBO;
        GLuint adjacentTriangleTexVBO;

        // ----- vicinity table for probes -----
        unsigned int *probeVoxelMap;
        // ----- singularity handling -----
        float *singTexData;
        float *singTexCoords;
        GLuint singTex;

        // VBO for spherical triangle center transform feedback
        GLuint sphericalTriaVBO[4];
        // query for transform feedback
        GLuint query;

        float delta;

		bool first;
};

} /* end namespace protein */
} /* end namespace megamol */

#endif /* MEGAMOL_PROTRENSESGPU_H_INCLUDED */