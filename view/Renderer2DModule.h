/*
 * Renderer2DModule.h
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_RENDERER2DMODULE_H_INCLUDED
#define MEGAMOLCORE_RENDERER2DMODULE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "api/MegaMolCore.std.h"
#include "Module.h"
#include "CalleeSlot.h"
#include "CallRender2D.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * Base class of rendering graph 2D renderer modules.
     */
    class MEGAMOLCORE_API Renderer2DModule : public Module {
    public:

        /** Ctor. */
        Renderer2DModule(void);

        /** Dtor. */
        virtual ~Renderer2DModule(void);

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
        virtual bool GetExtents(CallRender2D& call) = 0;

        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Render(CallRender2D& call) = 0;

    private:

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        bool onGetExtentsCallback(Call& call);

        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        bool onRenderCallback(Call& call);

        /** The render callee slot */
        CalleeSlot renderSlot;

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_RENDERERMODULE_H_INCLUDED */
