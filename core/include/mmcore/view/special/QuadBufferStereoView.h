/*
 * QuadBufferStereoView.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_QUADBUFFERSTEREOVIEW_H_INCLUDED
#define MEGAMOLCORE_QUADBUFFERSTEREOVIEW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/special/AbstractStereoView.h"


namespace megamol {
namespace core {
namespace view {
namespace special {


    /**
     * Abstract base class of override rendering views
     */
    class QuadBufferStereoView : public AbstractStereoView {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "QuadBufferStereoView";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Override View Module for quad-buffer stereo output";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /**
         * Disallow usage in quickstarts
         *
         * @return false
         */
        static bool SupportQuickstart(void) {
            return false;
        }

        /** Ctor. */
        QuadBufferStereoView(void);

        /** Dtor. */
        virtual ~QuadBufferStereoView(void);

        /**
         * Renders this AbstractView3D in the currently active OpenGL context.
         */
        virtual void Render(const mmcRenderViewContext& context);

    protected:

        /**
         * Initializes the module directly after instanziation
         *
         * @return 'true' on success
         */
        virtual bool create(void);

        /**
         * Releases all resources of the module
         */
        virtual void release(void);

    private:

        /** Flag indicating if quad-buffers are qvailable */
        bool hasQuadBuffer;

    };

} /* end namespace special */
} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_QUADBUFFERSTEREOVIEW_H_INCLUDED */
