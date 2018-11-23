/*
 * LinesRenderer.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_TRISOUP_LINESRENDERER_H_INCLUDED
#define MEGAMOL_TRISOUP_LINESRENDERER_H_INCLUDED
#pragma once

#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/CallerSlot.h"


namespace megamol {
namespace trisoup {

    /**
     * Mesh-based renderer for bézier curve tubes
     */
    class LinesRenderer : public core::view::Renderer3DModule {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "LinesRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Renderer for lines data";
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
        LinesRenderer(void);

        /** Dtor. */
        virtual ~LinesRenderer(void);

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

        /** The call for data */
        core::CallerSlot getDataSlot;

    };

} /* end namespace trisoup */
} /* end namespace megamol */

#endif /* MEGAMOL_TRISOUP_LINESRENDERER_H_INCLUDED */
