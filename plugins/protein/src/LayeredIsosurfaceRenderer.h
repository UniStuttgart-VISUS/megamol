/*
 * LayeredIsosurfaceRenderer.h
 *
 * Copyright (C) 2013 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MEGAMOLCORE_LAYEREDISORENDERER_H_INCLUDED
#define MEGAMOLCORE_LAYEREDISORENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "slicing.h"
#include "mmcore/moldyn/VolumeDataCall.h"
#include "protein_calls/VTIDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/FramebufferObject.h"

#define CHECK_FOR_OGL_ERROR() do { GLenum err; err = glGetError();if (err != GL_NO_ERROR) { fprintf(stderr, "%s(%d) glError: %s\n", __FILE__, __LINE__, gluErrorString(err)); } } while(0)

namespace megamol {
namespace protein {

    /**
     * Protein Renderer class
     */
    class LayeredIsosurfaceRenderer : public megamol::core::view::Renderer3DModule
    {
    public:
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "LayeredIsosurfaceRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Offers layered isosurface rendering.";
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
        LayeredIsosurfaceRenderer(void);

        /** Dtor. */
        virtual ~LayeredIsosurfaceRenderer(void);
        
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
        virtual bool GetExtents(megamol::core::Call& call);

        /**
        * The Open GL Render callback.
        *
        * @param call The calling call.
        * @return The return value of the function.
        */
        virtual bool Render(megamol::core::Call& call);
        
        /**
         * Volume rendering using volume data.
        */
        bool RenderVolumeData(megamol::core::view::CallRender3D *call, core::moldyn::VolumeDataCall *volume);
        
        /**
         * Volume rendering using volume data.
        */
		bool RenderVolumeData(megamol::core::view::CallRender3D *call, protein_calls::VTIDataCall *volume);
        
        /**
         * Initialize parameters for the LIC calculation and setup random texture.
         *
         * @return 'True' on success, 'false' otherwise
         */
        void InitLIC(unsigned int licRandBuffSize);

        /**
         * Refresh all parameters.
        */
        void ParameterRefresh(megamol::core::view::CallRender3D *call);
        
        /**
         * Draw the volume.
         *
         * @param boundingbox The bounding box.
         */
        void RenderVolume(vislib::math::Cuboid<float> boundingbox);
        
        /**
         * Write the parameters of the ray to the textures.
         *
         * @param boundingbox The bounding box.
         */
        void RayParamTextures(vislib::math::Cuboid<float> boundingbox);
        
        /**
         * Create a volume containing the voxel map.
         *
         * @param volume The data interface.
         */
        void UpdateVolumeTexture(const core::moldyn::VolumeDataCall *volume);

        /**
         * Create a volume containing the voxel map.
         *
         * @param volume The data interface.
         */
		void UpdateVolumeTexture(const protein_calls::VTIDataCall *volume);

        /**
         * Draw the bounding box of the protein around the origin.
         *
         * @param boundingbox The bounding box.
         */
        void DrawBoundingBox(vislib::math::Cuboid<float> boundingbox);

        /**
         * Draw the clipped polygon for correct clip plane rendering.
         *
         * @param boundingbox The bounding box.
         */
        void drawClippedPolygon(vislib::math::Cuboid<float> boundingbox);
        
        /**********************************************************************
         * variables
         **********************************************************************/
        
        /** caller slot volume data */
        megamol::core::CallerSlot volDataCallerSlot;
        /** caller slot renderer */
        megamol::core::CallerSlot rendererCallerSlot;
        /** caller slot for clip plane 0 */
        megamol::core::CallerSlot clipPlane0Slot;
        /** caller slot for clip plane 0 */
        megamol::core::CallerSlot clipPlane1Slot;
        /** caller slot for clip plane 0 */
        megamol::core::CallerSlot clipPlane2Slot;

        // camera information
        vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;
        // scaling factor for the scene
        float scale;
        // translation of the scene
        vislib::math::Vector<float, 3> translation;
        
        // parameters for the volume rendering
        megamol::core::param::ParamSlot volIsoValue0Param;
        megamol::core::param::ParamSlot volIsoValue1Param;
        megamol::core::param::ParamSlot volIsoValue2Param;
        megamol::core::param::ParamSlot volIsoOpacityParam;
        megamol::core::param::ParamSlot volClipPlaneFlagParam;
        // parameters for shader-based LIC
        megamol::core::param::ParamSlot volLicDirSclParam;
        megamol::core::param::ParamSlot volLicLenParam;
        megamol::core::param::ParamSlot volLicContrastStretchingParam;
        megamol::core::param::ParamSlot volLicBrightParam;
        megamol::core::param::ParamSlot volLicTCSclParam;
        megamol::core::param::ParamSlot doVolumeRenderingParam;
        megamol::core::param::ParamSlot doVolumeRenderingToggleParam;
        // shader for volume rendering
        vislib::graphics::gl::GLSLShader volumeShader;
        vislib::graphics::gl::GLSLShader volRayStartShader;
        vislib::graphics::gl::GLSLShader volRayStartEyeShader;
        vislib::graphics::gl::GLSLShader volRayLengthShader;
                
        // FBO for rendering opaque
        vislib::graphics::gl::FramebufferObject opaqueFBO;
        
        // volume texture
        GLuint volumeTex;
        // volume texture for vector field
        GLuint vectorfieldTex;
        unsigned int volumeSize;
        int currentFrameId;
        // FBO for volume generation
        GLuint volFBO;
        // volume parameters
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
        float isoValue0;
        // the second iso value
        float isoValue1;
        // the third iso value
        float isoValue2;
        // the opacity of the isosurface
        float volIsoOpacity;

        // flag wether clipping planes are enabled
        bool volClipPlaneFlag;
        // the array of clipping planes
        vislib::Array<vislib::math::Vector<float, 4> > volClipPlane;
        // view aligned slicing
        ViewSlicing slices;
            
        /// Uniform grid containing random buffer
        vislib::Array<float> licRandBuff;
        /// Random noise texture
        GLuint randNoiseTex;
        
        size_t lastHash;

    };

} /* end namespace protein */
} /* end namespace megamol */

#endif // MEGAMOLCORE_LAYEREDISORENDERER_H_INCLUDED
