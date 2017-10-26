/*
* WatermarkRenderer.h
*
* Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
* Alle Rechte vorbehalten.
*/

#ifndef MEGAMOLCORE_WATERMARK_RENDERER_H_INCLUDED
#define MEGAMOLCORE_WATERMARK_RENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/param/ParamSlot.h"

#include "vislib/graphics/gl/OpenGLTexture2D.h"
#include "vislib/math/Vector.h"

using namespace megamol::core;

namespace megamol {
    namespace core {
        namespace misc {

        /**
        * Render watermarks using PNG-files in all four corners of the viewport.
        */
        class WatermarkRenderer : public view::Renderer3DModule {
        public:

            /**
            * Answer the name of this module.
            *
            * @return The name of this module.
            */
            static const char *ClassName(void) {
                return "WatermarkRenderer";
            }

            /**
            * Answer a human readable description of this module.
            *
            * @return A human readable description of this module.
            */
            static const char *Description(void) {
                return "Render watermarks.";
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
            WatermarkRenderer(void);

            /** Dtor. */
            virtual ~WatermarkRenderer(void);

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

            /**********************************************************************
            * variables
            **********************************************************************/

            enum corner {
                TOP_LEFT     = 0,
                TOP_RIGHT    = 1,
                BOTTOM_LEFT  = 2,
                BOTTOM_RIGHT = 3,
            };

            vislib::graphics::gl::OpenGLTexture2D textureTopLeft;
            vislib::graphics::gl::OpenGLTexture2D textureTopRight;
            vislib::graphics::gl::OpenGLTexture2D textureBottomLeft;
            vislib::graphics::gl::OpenGLTexture2D textureBottomRight;

            vislib::math::Vector<float, 2> sizeTopLeft;
            vislib::math::Vector<float, 2> sizeTopRight;
            vislib::math::Vector<float, 2> sizeBottomLeft;
            vislib::math::Vector<float, 2> sizeBottomRight;

            float lastScaleAll;
            bool  firstParamChange;

            /**********************************************************************
            * functions
            **********************************************************************/

            /*   */
            SIZE_T  WatermarkRenderer::loadFile(const vislib::StringA & name, void **outData);

            /**  */
            bool loadTexture(WatermarkRenderer::corner cor, vislib::StringA filename);

            /**  */
            bool renderWatermark(WatermarkRenderer::corner cor, float vpH, float vpW);

            /**********************************************************************
            * parameters
            **********************************************************************/

            /**  */
            core::param::ParamSlot paramImgTopLeft;
            core::param::ParamSlot paramImgTopRight;
            core::param::ParamSlot paramImgBottomLeft;
            core::param::ParamSlot paramImgBottomRight;

            /**  */
            core::param::ParamSlot paramScaleAll;
            core::param::ParamSlot paramScaleTopLeft;
            core::param::ParamSlot paramScaleTopRight;
            core::param::ParamSlot paramScaleBottomLeft;
            core::param::ParamSlot paramScaleBottomRight;

            /**  */
            core::param::ParamSlot paramAlpha;
        };

        } /* end namespace misc */
    } /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_WATERMARK_RENDERER_H_INCLUDED */
