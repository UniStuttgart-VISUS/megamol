/*
 * SwitchRenderer3D.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_SWITCHRENDERER3D_H_INCLUDED
#define MEGAMOLCORE_SWITCHRENDERER3D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CallerSlot.h"
#include "CallRender3D.h"
#include "mmcore/param/ParamSlot.h"
#include "Renderer3DModule.h"
#include "vislib/forceinline.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * A simple switch between two renderer 3d modules
     */
    class SwitchRenderer3D : public Renderer3DModule {
    public:

        /**
         * Gets the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "SwitchRenderer3D";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "A simple switch between two renderer 3d modules";
        }

        /**
         * Gets whether this module is available on the current system.
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
        SwitchRenderer3D(void);

        /** Dtor. */
        virtual ~SwitchRenderer3D(void);

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

    protected:

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
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Render(Call& call);

    private:

        /**
         * Callback for the switch renderer button.
         *
         * @param param Must be switchRendererBtnSlot
         *
         * @return true
         */
        bool onSwitchRenderer(param::ParamSlot& param);

        /**
         * Answer which renderer is selected.
         *
         * @return 1, 2, or 0 in case of no renderer is available.
         */
        VISLIB_FORCEINLINE int whichRenderer(void);

        /**
         * Answer the call to the selected renderer
         *
         * @param src The source call.
         *
         * @return The call to the selected renderer or NULL if no renderer is
         *         available
         */
        VISLIB_FORCEINLINE CallRender3D *callToRenderer(view::CallRender3D *src);

        /** Call to the first renderer */
        CallerSlot renderer1Slot;

        /** Call to the second renderer */
        CallerSlot renderer2Slot;

        /** Enum which renderer is active */
        param::ParamSlot selectedRendererSlot;

        /** Switches to the other renderer */
        param::ParamSlot switchRendererBtnSlot;

        /** The selection of the renderer */
        int selection;

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SWITCHRENDERER3D_H_INCLUDED */
