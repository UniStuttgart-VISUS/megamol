/*
 * TrackerRendererTransform.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_TRACKERRENDERERTRANSFORM_H_INCLUDED
#define MEGAMOLCORE_TRACKERRENDERERTRANSFORM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "view/Renderer3DModule.h"
#include "Call.h"
#include "CallerSlot.h"
#include "param/ParamSlot.h"


namespace megamol {
namespace trisoup {


    /**
     * (Meta-)Renderer applying a transformation got from tracker information
     */
    class TrackerRendererTransform : public core::view::Renderer3DModule {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "TrackerRendererTransform";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "(Meta-)Renderer applying a transformation got from tracker information";
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
        TrackerRendererTransform(void);

        /** Dtor. */
        virtual ~TrackerRendererTransform(void);

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
        virtual bool GetCapabilities(core::Call& call);

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetExtents(core::Call& call);

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
        virtual bool Render(core::Call& call);

    private:

        /** The slot to call the real renderer */
        core::CallerSlot outRenderSlot;

        /** The translation applied */
        core::param::ParamSlot translateSlot;

        /** The rotation applied */
        core::param::ParamSlot rotateSlot;

        /** The scale applied */
        core::param::ParamSlot scaleSlot;

        /** The minimum vector of the bounding box */
        core::param::ParamSlot bboxMinSlot;

        /** The maximum vector of the bounding box */
        core::param::ParamSlot bboxMaxSlot;

    };


} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_TRACKERRENDERERTRANSFORM_H_INCLUDED */
