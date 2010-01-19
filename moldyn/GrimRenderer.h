/*
 * GrimRenderer.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_GRIMRENDERER_H_INCLUDED
#define MEGAMOLCORE_GRIMRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "view/Renderer3DModule.h"
#include "Call.h"
#include "CallerSlot.h"
#include "param/ParamSlot.h"
#include "vislib/CameraParameters.h"
#include "vislib/Cuboid.h"
#include "vislib/forceinline.h"
#include "vislib/FramebufferObject.h"
#include "vislib/GLSLShader.h"
#include "vislib/Pair.h"
#include "vislib/Point.h"
#include "vislib/SmartPtr.h"


namespace megamol {
namespace core {
namespace moldyn {

    /**
     * Renderer for gridded imposters
     */
    class GrimRenderer : public view::Renderer3DModule {
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
                && (glh_extension_supported("GL_NV_occlusion_query") != 0)
                && (glh_extension_supported("GL_ARB_multitexture") != 0);
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
        virtual bool Render(Call& call);

    private:

        /**
         * Utility class storing additional rendering information about a
         * given cell
         */
        class CellInfo {
        public:

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
            inline bool operator==(const CellInfo& rhs) const {
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
        static int depthSort(const vislib::Pair<unsigned int, float>& lhs,
            const vislib::Pair<unsigned int, float>& rhs);

        /** The sphere shader */
        vislib::graphics::gl::GLSLShader sphereShader;

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

        /** The frame buffer object for the depth estimate */
        vislib::graphics::gl::FramebufferObject fbo;

        /** buffers for depth-max mip map */
        vislib::graphics::gl::FramebufferObject depthmap[2];

        /** The call for data */
        CallerSlot getDataSlot;

        /** The call for Transfer function */
        CallerSlot getTFSlot;

        /** A simple black-to-white transfer function texture as fallback */
        unsigned int greyTF;

        /** Cell distances */
        vislib::Array<vislib::Pair<unsigned int, float> > cellDists;

        /** Cell rendering informations */
        vislib::Array<CellInfo> cellInfos;

    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_GRIMRENDERER_H_INCLUDED */
