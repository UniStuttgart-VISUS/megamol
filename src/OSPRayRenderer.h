/*
* OSPRayRenderer.h
* Copyright (C) 2009-2015 by MegaMol Team
* Alle Rechte vorbehalten.
*/
#ifndef OSPRAY_RENDERER_H_INCLUDED
#define OSPRAY_RENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <stdint.h>
#include "vislib/graphics/gl/GLSLShader.h"
#include "ospray/ospray.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/CallerSlot.h"
#include "CallOSPRayLight.h"
#include <map>



namespace megamol {
namespace ospray {

    class OSPRayRenderer : public core::view::Renderer3DModule {
    protected:
        // Ctor
        OSPRayRenderer(void);

        // Dtor
        ~OSPRayRenderer(void);

        /**
        * initializes OSPRay
        */
        void initOSPRay(OSPDevice &dvce);

        /**
        * helper function for rendering the OSPRay texture
        * @param GLSL shader
        * @param GL texture object
        * @param OSPRay texture
        * @param GL vertex array object
        * @param image/window width
        * @param image/window heigth
        */
        void renderTexture2D(vislib::graphics::gl::GLSLShader &shader, const uint32_t * fb, int &width, int &height);

        /**
        * helper function for setting up the OSPRay screen
        * @param GL vertex array
        * @param GL vertex buffer object
        * @param GL texture object
        */
        void setupTextureScreen();

        /**
        * Releases the OGL content created by setupTextureScreen
        */
        void releaseTextureScreen();

        /**
        * helper function for initializing OSPRay
        * @param OSPRay renderer object
        * @param OSPRay camera object
        * @param OSPRay world object
        * @param OSPRay geometry object
        * @param geometry name/type
        * @param renderer type
        */
        void setupOSPRay(OSPRenderer &renderer, OSPCamera &camera, OSPModel &world, OSPGeometry &geometry, const char * geometry_name, const char * renderer_name);

        /**
        * helper function for initializing OSPRay
        * @param OSPRay renderer object
        * @param OSPRay camera object
        * @param OSPRay world object
        * @param OSPRay volume object
        * @param volume name/type
        * @param renderer type
        */
        void setupOSPRay(OSPRenderer &renderer, OSPCamera &camera, OSPModel &world, OSPVolume &volume, const char * volume_name, const char * renderer_name);

        /**
        * helper function for initializing OSPRay
        * @param OSPRay renderer object
        * @param OSPRay camera object
        * @param OSPRay world object
        * @param volume name/type
        * @param renderer type
        */
        void setupOSPRay(OSPRenderer &renderer, OSPCamera &camera, OSPModel &world, const char * volume_name, const char * renderer_name);

        /**
        * helper function for initializing OSPRays Camera
        * @param OSPRay camera object
        * @param CallRenderer3D object
        */
        void setupOSPRayCamera(OSPCamera& cam, core::view::CallRender3D* cr);

        /** 
        * color transfer helper
        * @param array with gray scales
        * @param transferfunction table/texture
        * @param transferfunction table/texture size
        * @param target array (rgba)
        */
        void colorTransferGray(std::vector<float> &grayArray, float const* transferTable, unsigned int tableSize, std::vector<float> &rgbaArray);
        
        /**
        * Texture from file importer
        * @param file path
        * @return 2
        */
        OSPTexture2D TextureFromFile(vislib::TString fileName);
        // helper function to write the rendered image as PPM file
        void writePPM(const char *fileName, const osp::vec2i &size, const uint32_t *pixel);

        // TODO: Documentation

        bool AbstractIsDirty();
        void AbstractResetDirty();
        void RendererSettings(OSPRenderer &renderer);
        OSPFrameBuffer newFrameBuffer(osp::vec2i& imgSize, const OSPFrameBufferFormat format = OSP_FB_RGBA8, const uint32_t frameBufferChannels = OSP_FB_COLOR);

        // vertex array, vertex buffer object, texture
        GLuint vaScreen, vbo, tex;

        // API Variables
        core::param::ParamSlot AOtransparencyEnabled;
        core::param::ParamSlot AOsamples;
        core::param::ParamSlot AOdistance;
        core::param::ParamSlot extraSamles;



        core::param::ParamSlot rd_epsilon;
        core::param::ParamSlot rd_spp;
        core::param::ParamSlot rd_maxRecursion;
        core::param::ParamSlot rd_type;
        core::param::ParamSlot rd_ptBackground;
        core::param::ParamSlot shadows;



        // renderer type
        enum rdenum {
            SCIVIS,
            PATHTRACER
        };

        // light
        std::vector<OSPLight> lightArray;
        OSPData lightsToRender;
        /** The call for ligtht */
        core::CallerSlot getLightSlot;

        // framebuffer dirtyness
        bool framebufferIsDirty;

        // device
        OSPDevice device;

        // renderer
        OSPRenderer renderer;

        // Light map
        std::map<CallOSPRayLight*, OSPRayLightContainer> lightMap;

        // Module dirtyness
        bool ModuleIsDirty;


        void fillLightArray();

    };

} // end namespace ospray
} // end namespace megamol

#endif /* OSPRAY_RENDERER_H_INCLUDED */
