/*
* OSPRayVolumeRenderer.h
* Copyright (C) 2009-2016 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#ifndef OSPRAY_VOLUMERENDERER_H_INCLUDED
#define OSPRAY_VOLUMERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/CallRender3D.h"
#include "mmcore/view/Renderer3DModule.h"
#include "OSPRayRenderer.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"


namespace megamol {
namespace ospray {

    class OSPRayVolumeRenderer : public OSPRayRenderer {
    public:

        /**
        * Answer the name of this module.
        *
        * @return The name of this module.
        */
        static const char *ClassName(void) {
            return "OSPRayVolumeRenderer";
        }

        /**
        * Answer a human readable description of this module.
        *
        * @return A human readable description of this module.
        */
        static const char *Description(void) {
            return "Offers OSPRay volume rendering functionality";
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
        OSPRayVolumeRenderer(void);

        /** Dtor. */
        virtual ~OSPRayVolumeRenderer(void);

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
        * The render callback.
        *
        * @param call The calling call.
        *
        * @return The return value of the function.
        */
        virtual bool Render(megamol::core::Call& call);

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

        bool InterfaceIsDirty();


        vislib::graphics::gl::GLSLShader osprayShader;

        // API VARIABLES
        core::param::ParamSlot showVolume;
        core::param::ParamSlot showIsosurface;
        core::param::ParamSlot showSlice;

        core::param::ParamSlot sliceNormal;
        core::param::ParamSlot sliceDist;

        core::param::ParamSlot clippingBoxActive;
        core::param::ParamSlot clippingBoxLower;
        core::param::ParamSlot clippingBoxUpper;

        


        // rendering conditions
        bool data_has_changed;
        bool cam_has_changed;
        SIZE_T m_datahash;
        vislib::SmartPtr<vislib::graphics::CameraParameters> camParams;
        int extra_samples;
        float time;
        bool renderer_changed;

        // OSPRay objects
        OSPRenderer renderer;
        OSPFrameBuffer framebuffer;
        OSPCamera camera;
        OSPModel world;
        OSPVolume volume;
        osp::vec2i imgSize;
        OSPData voxels;
        OSPGeometry slice;
        OSPGeometry isosurface;

        // OSPRay texture
        const uint32_t * fb;

        /** caller slot */
        core::CallerSlot volDataCallerSlot;
        /** caller slot for additional renderer */
        core::CallerSlot secRenCallerSlot;
        /** The call for Transfer function */
        core::CallerSlot TFSlot;


        // scaling factor for the scene
        float scaling;
        };
    
} // end namespace ospray
} // end namespace megamol
#endif // !OSPRAY_VOLUMERENDERER_H_INCLUDED