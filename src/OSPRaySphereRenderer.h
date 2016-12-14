/*
 * OSPRaySphereRenderer.h
 * Copyright (C) 2009-2016 by MegaMol Team
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


namespace megamol {
namespace ospray {

    class OSPRaySphereRenderer : public core::moldyn::AbstractSimpleSphereRenderer, OSPRayRenderer {
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

        /** Ctor. */
        OSPRaySphereRenderer(void);

        /** Dtor. */
        virtual ~OSPRaySphereRenderer(void);

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
        /** The texture shader */
        vislib::graphics::gl::GLSLShader osprayShader;
        core::param::ParamSlot AOweight;
        core::param::ParamSlot AOsamples;
        core::param::ParamSlot AOdistance;
        core::param::ParamSlot extraSamles;

        core::param::ParamSlot lightIntensity;
        core::param::ParamSlot lightType;
        core::param::ParamSlot lightColor;
        core::param::ParamSlot shadows;

        core::param::ParamSlot dl_angularDiameter;
        core::param::ParamSlot dl_direction;
        core::param::ParamSlot dl_eye_direction;

        core::param::ParamSlot pl_position;
        core::param::ParamSlot pl_radius;

        core::param::ParamSlot sl_position;
        core::param::ParamSlot sl_direction;
        core::param::ParamSlot sl_openingAngle;
        core::param::ParamSlot sl_penumbraAngle;
        core::param::ParamSlot sl_radius;

        core::param::ParamSlot ql_position;
        core::param::ParamSlot ql_edgeOne;
        core::param::ParamSlot ql_edgeTwo;

        core::param::ParamSlot hdri_up;
        core::param::ParamSlot hdri_direction;
        core::param::ParamSlot hdri_evnfile;

        core::param::ParamSlot rd_epsilon;
        core::param::ParamSlot rd_spp;
        core::param::ParamSlot rd_maxRecursion;
        core::param::ParamSlot rd_type;
        core::param::ParamSlot rd_ptBackground;

        core::param::ParamSlot mat_Kd;
        core::param::ParamSlot mat_Ks;
        core::param::ParamSlot mat_Ns;
        core::param::ParamSlot mat_d;
        core::param::ParamSlot mat_Tf;
        core::param::ParamSlot mat_type;


        // Interface dirty flag
        bool InterfaceIsDirty();

        // rendering conditions
        bool data_has_changed;
        bool cam_has_changed;
        SIZE_T m_datahash;
        vislib::SmartPtr<vislib::graphics::CameraParameters> camParams;
        int extra_samples;
        float time;


        // OSP objects
        OSPRenderer renderer;
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

        // light
        OSPLight light;
        OSPData lightArray;
        enum lightenum { NONE,
            DISTANTLIGHT,
            POINTLIGHT,
            SPOTLIGHT,
            QUADLIGHT,
            AMBIENTLIGHT,
            HDRILIGHT };

        // renderer type
        enum rdenum {
            SCIVIS,
            PATHTRACER
        };
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
        OSPDataType colorType;

        // color transfer data
        unsigned int tex_size;

	};
} /*end namespace ospray*/
} /*end namespace megamol*/



#endif /* OSPRAY_SPHERERENDERER_H_INCLUDED */
