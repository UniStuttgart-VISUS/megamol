/*
 * DirectVolumeRenderer.h
 *
 * Copyright (C) 2012 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMSTD_VOLUME_DIRVOLRENDERER_H_INCLUDED
#define MMSTD_VOLUME_DIRVOLRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "slicing.h"
#include "mmcore/moldyn/VolumeDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/math/Vector.h"

#define CHECK_FOR_OGL_ERROR() do { GLenum err; err = glGetError();if (err != GL_NO_ERROR) { fprintf(stderr, "%s(%d) glError: %s\n", __FILE__, __LINE__, gluErrorString(err)); } } while(0)

namespace megamol {
namespace stdplugin {
namespace volume {

    /**
     * Protein Renderer class
     */
    class DirectVolumeRenderer : public megamol::core::view::Renderer3DModule
    {
    public:
        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "DirectVolumeRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Offers direct volume rendering.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return vislib::graphics::gl::FramebufferObject::AreExtensionsAvailable()
                && vislib::graphics::gl::GLSLShader::AreExtensionsAvailable()
                && isExtAvailable("GL_ARB_texture_float")
                && isExtAvailable("GL_EXT_gpu_shader4")
                && isExtAvailable("GL_EXT_bindable_uniform");
        }

        /** Ctor. */
        DirectVolumeRenderer(void);

        /** Dtor. */
        virtual ~DirectVolumeRenderer(void);
        
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
        bool RenderVolumeData(megamol::core::view::CallRender3D *call, megamol::core::moldyn::VolumeDataCall *volume);
        
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
        void UpdateVolumeTexture(const megamol::core::moldyn::VolumeDataCall *volume);

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
        
        /** caller slot */
        megamol::core::CallerSlot volDataCallerSlot;
        /** caller slot for additional renderer */
        megamol::core::CallerSlot secRenCallerSlot;

        // camera information
        vislib::SmartPtr<vislib::graphics::CameraParameters> cameraInfo;
        // scaling factor for the scene
        float scale;
        // translation of the scene
        vislib::math::Vector<float, 3> translation;
        vislib::math::Vector<float, 3> bboxCenter;
        vislib::math::Cuboid<float> unionBBox;
        
        // parameters for the volume rendering
        megamol::core::param::ParamSlot volIsoValueParam;
        megamol::core::param::ParamSlot volIsoOpacityParam;
        megamol::core::param::ParamSlot volClipPlaneFlagParam;
        megamol::core::param::ParamSlot volClipPlane0NormParam;
        megamol::core::param::ParamSlot volClipPlane0DistParam;
        megamol::core::param::ParamSlot volClipPlaneOpacityParam;
        megamol::core::param::ParamSlot opaqRenWorldScaleParam;
        megamol::core::param::ParamSlot toggleVolBBoxParam;

        megamol::core::param::ParamSlot togglePbcXParam;
        megamol::core::param::ParamSlot togglePbcYParam;
        megamol::core::param::ParamSlot togglePbcZParam;

        // shader for volume rendering
        vislib::graphics::gl::GLSLShader volumeShader;
        vislib::graphics::gl::GLSLShader volRayStartShader;
        vislib::graphics::gl::GLSLShader volRayStartEyeShader;
        vislib::graphics::gl::GLSLShader volRayLengthShader;
                
        // FBO for rendering opaque
        vislib::graphics::gl::FramebufferObject opaqueFBO;

        // volume texture
        GLuint volumeTex;
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
        float isoValue;
        // the opacity of the isosurface
        float volIsoOpacity;

        // flag wether clipping planes are enabled
        bool volClipPlaneFlag;
        // the array of clipping planes
        vislib::Array<vislib::math::Vector<double, 4> > volClipPlane;
        // view aligned slicing
        ViewSlicing slices;
        // the opacity of the clipping plane
        float volClipPlaneOpacity;

        // Hash value of the volume data
        size_t hashValVol;
    
    };

} /* end namespace volume */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif // MMSTD_VOLUME_DIRVOLRENDERER_H_INCLUDED
