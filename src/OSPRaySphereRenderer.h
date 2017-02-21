/*
 * OSPRaySphereRenderer.h
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#ifndef OSPRAY_SPHERERENDERER_H_INCLUDED
#define OSPRAY_SPHERERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/moldyn/AbstractSimpleSphereRenderer.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "OSPRayRenderer.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"


namespace megamol {
namespace ospray {

    class OSPRaySphereRenderer : public OSPRayRenderer {
    public:

        /**
        * Answer the name of this module.
        *
        * @return The name of this module.
        */
        static const char *ClassName(void) {
            return "OSPRaySphereRenderer";
        }

        /**
        * Answer a human readable description of this module.
        *
        * @return A human readable description of this module.
        */
        static const char *Description(void) {
            return "Renderer for sphere glyphs.";
        }

        /**
        * Answers whether this module is available on the current system.
        *
        * @return 'true' if the module is available, 'false' otherwise.
        */
        static bool IsAvailable(void) {
            return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable();
        }

        /** Dtor. */
        virtual ~OSPRaySphereRenderer(void);

        /** Ctor. */
        OSPRaySphereRenderer(void);

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
        * The render callback.
        *
        * @param call The calling call.
        *
        * @return The return value of the function.
        */
        virtual bool Render(megamol::core::Call& call);

        //virtual bool Resize();



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

        /** The call for data */
        core::CallerSlot getDataSlot;

        /** The call for clipping plane */
        core::CallerSlot getClipPlaneSlot;

        /** The call for Transfer function */
        core::CallerSlot getTFSlot;

        /**
        * TODO: Document
        */
        core::moldyn::MultiParticleDataCall *getData(unsigned int t, float& outScaling);

        /**
        * TODO: Document
        *
        * @param clipDat points to four floats
        * @param clipCol points to four floats
        */
        void getClipData(float *clipDat, float *clipCol);



        /** The texture shader */
        vislib::graphics::gl::GLSLShader osprayShader;

        // API VARS
        core::param::ParamSlot mat_Kd;
        core::param::ParamSlot mat_Ks;
        core::param::ParamSlot mat_Ns;
        core::param::ParamSlot mat_d;
        core::param::ParamSlot mat_Tf;
        core::param::ParamSlot mat_type;
        core::param::ParamSlot particleList;

        //tmp variable
        unsigned int number;

        // Interface dirty flag
        bool InterfaceIsDirty();

        // rendering conditions
        bool data_has_changed;
        bool cam_has_changed;
        SIZE_T m_datahash;
        vislib::SmartPtr<vislib::graphics::CameraParameters> camParams;
        float time;

        //data objects
        std::vector<float> cd_rgba;
        std::vector<float> vd;

        // OSP objects
        OSPFrameBuffer framebuffer;
        OSPCamera camera;
        OSPModel world;
        OSPGeometry spheres;
        OSPPlane pln;

        osp::vec2i imgSize;

        // OSPData 
        OSPData vertexData, colorData;

        // OSPRay texture
        const uint32_t * fb;




        bool renderer_changed;

        // material
        //OSPMaterial material;
        enum material_type {
            OBJMATERIAL,
            GLASS,
            MATTE,
            METAL,
            METALLICPAINT,
            PLASTIC,
            THINGLASS,
            VELVET
        };

        // data conversion
        size_t vertexLength;
        OSPDataType vertexType;
        size_t colorLength;
        OSPDataType convertedColorType;

        // color transfer data
        unsigned int tex_size;


	};

} /*end namespace ospray*/
} /*end namespace megamol*/



#endif /* OSPRAY_SPHERERENDERER_H_INCLUDED */
