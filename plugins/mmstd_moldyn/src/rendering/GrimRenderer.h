/*
 * GrimRenderer.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_MOLDYN_GRIMRENDERER_H_INCLUDED
#define MEGAMOL_MOLDYN_GRIMRENDERER_H_INCLUDED

#include "ParticleGridDataCall.h"

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/view/Renderer3DModule.h"

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/graphics/CameraParameters.h"
#include "vislib/math/Cuboid.h"
#include "vislib/forceinline.h"
#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/Pair.h"
#include "vislib/math/Point.h"
#include "vislib/SmartPtr.h"
#include "vislib/Array.h"
#include "vislib/assert.h"
#include "vislib/graphics/Camera.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/math/mathtypes.h"
#include "vislib/Pair.h"
#include "vislib/math/Plane.h"
#include "vislib/sys/sysfunctions.h"
#include "vislib/Trace.h"
#include "vislib/math/Vector.h"
#include "vislib/graphics/gl/IncludeAllGL.h"

#include <climits>

#include <glm/glm.hpp>


namespace megamol {
namespace stdplugin {
namespace moldyn {
namespace rendering {


    /**
        * Renderer for gridded imposters
        */
    class GrimRenderer : public core::view::Renderer3DModule {
    public:

        /**
            * Answer the name of this module.
            *
            * @return The name of this module.
            */
        static const char *ClassName(void) {
            return "GrimRenderer";
        }

        /**
            * Answer a human readable description of this module.
            *
            * @return A human readable description of this module.
            */
        static const char *Description(void) {
            return "Renderer of gridded imposters.";
        }

        /**
            * Answers whether this module is available on the current system.
            *
            * @return 'true' if the module is available, 'false' otherwise.
            */
        static bool IsAvailable(void) {
            return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable()
                && vislib::graphics::gl::FramebufferObject::AreExtensionsAvailable()
                && (isExtAvailable("GL_NV_occlusion_query") != GL_FALSE)
                && (isExtAvailable("GL_ARB_multitexture") != GL_FALSE)
                && (isExtAvailable("GL_ARB_vertex_buffer_object") != GL_FALSE);
        }

        /** Ctor. */
        GrimRenderer(void);

        /** Dtor. */
        virtual ~GrimRenderer(void);

    protected:

        /**
            * Implementation of 'Create'.
            *
            * @return 'true' on success, 'false' otherwise.
            */
        virtual bool create(void);

        /**
            * The get extents callback. The module should set the members of
            * 'call' to tell the caller the extents of its data (bounding boxes
            * and times).
            *
            * @param call The calling call.
            *
            * @return The return value of the function.
            */
        virtual bool GetExtents(core::view::CallRender3D &call);

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
        virtual bool Render(core::view::CallRender3D &call);

    private:

        /**
            * Utility class storing additional rendering information about a
            * given cell
            */
        class CellInfo {
        public:

            class CacheItem {
            public:
                GLuint data[2];

                CacheItem() {
                    this->data[0] = 0;
                    this->data[1] = 0;
                }

                ~CacheItem() {
                    if (this->data[0] != 0) {
                        ::glDeleteBuffersARB(2, this->data);
                    }
                    this->data[0] = 0;
                    this->data[1] = 0;
                }

                inline bool operator==(const CacheItem &rhs) {
                    return (this->data[0] == rhs.data[0])
                        && (this->data[1] == rhs.data[1]);
                }
            };

            /** Flag if the cell is visible now (inside the frustum; not occluded) */
            bool isvisible;

            /** Flag if the cell was visible in the last render */
            bool wasvisible;

            /** Flag if the cell is far enought to be rendered with dots */
            bool dots;

            /** The max radius found in this cell */
            float maxrad;

            /** The occlusion query object */
            unsigned int oQuery;

            /** gpu-ram caching variables */
            std::vector<CacheItem> cache;

            /**
                * Ctor
                */
            CellInfo(void);

            /**
                * Dtor
                */
            ~CellInfo(void);

            /**
                * Test for equality
                *
                * @param rhs The right hand side operand
                *
                * @return 'true' if 'this' and 'rhs' are equal.
                */
            inline bool operator==(const CellInfo &rhs) const {
                return (this->isvisible == rhs.isvisible)
                    && (this->wasvisible == rhs.wasvisible)
                    && (this->dots == rhs.dots)
                    && (vislib::math::IsEqual(this->maxrad, rhs.maxrad));
            }

        };

        /**
            * Sorts the grid cells by their distance to the viewer
            *
            * @param lhs The left hand side operand
            * @param rhs The right hand side operand
            *
            * @return The distance sort info
            */
        static bool depthSort(const vislib::Pair<unsigned int, float> &lhs,
            const vislib::Pair<unsigned int, float> &rhs);

        /** The sphere shader */
        vislib::graphics::gl::GLSLShader sphereShader;

        /** The vanilla sphere shader */
        vislib::graphics::gl::GLSLShader vanillaSphereShader;

        /** The shader to init the depth fbo */
        vislib::graphics::gl::GLSLShader initDepthShader;

        /** The shader to init the depth mip-map generation */
        vislib::graphics::gl::GLSLShader initDepthMapShader;

        /** The shader for the depth mip-mapping ping-ponging */
        vislib::graphics::gl::GLSLShader depthMipShader;

        /** The shader to render far-away, solid-coloured points */
        vislib::graphics::gl::GLSLShader pointShader;

        /** The shader to init the depth buffer with points */
        vislib::graphics::gl::GLSLShader initDepthPointShader;

        /** Von Guido aus */
        vislib::graphics::gl::GLSLShader vertCntShader;

        /** Von Guido aus */
        vislib::graphics::gl::GLSLShader vertCntShade2r;

        /** The frame buffer object for the depth estimate */
        vislib::graphics::gl::FramebufferObject fbo;

        /** buffers for depth-max mip map */
        vislib::graphics::gl::FramebufferObject depthmap[2];

        /** The call for data */
        core::CallerSlot getDataSlot;

        /** The call for Transfer function */
        core::CallerSlot getTFSlot;

        /** Flag to activate per cell culling */
        core::param::ParamSlot useCellCullSlot;

        /** Flag to activate per vertex culling */
        core::param::ParamSlot useVertCullSlot;

        /** Flag to activate output of percentage of culled cells */
        core::param::ParamSlot speakCellPercSlot;

        /** Flag to activate output of number of vertices */
        core::param::ParamSlot speakVertCountSlot;

        /** De-/Activates deferred shading with normal generation */
        core::param::ParamSlot deferredShadingSlot;

        /** A simple black-to-white transfer function texture as fallback */
        unsigned int greyTF;

        /** Cell distances */
        std::vector<vislib::Pair<unsigned int, float>> cellDists;

        /** Cell rendering informations */
        std::vector<CellInfo> cellInfos;

        /** Bytes of the GPU-Memory available for caching */
        SIZE_T cacheSize;

        /** Bytes of the GPU-Memory used by the caching */
        SIZE_T cacheSizeUsed;

        /** Frame buffer object used for deferred shading */
        vislib::graphics::gl::FramebufferObject dsFBO;

        /** The sphere shader */
        vislib::graphics::gl::GLSLShader deferredSphereShader;

        /** The vanilla sphere shader */
        vislib::graphics::gl::GLSLShader deferredVanillaSphereShader;

        /** The shader to render far-away, solid-coloured points */
        vislib::graphics::gl::GLSLShader deferredPointShader;

        /** The deferred shader */
        vislib::graphics::gl::GLSLShader deferredShader;

        /** The hash of the incoming data */
        SIZE_T inhash;

    };

} /* end namespace rendering */
} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOL_MOLDYN_GRIMRENDERER_H_INCLUDED */
