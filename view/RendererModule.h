/*
 * RendererModule.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_RENDERERMODULE_H_INCLUDED
#define MEGAMOLCORE_RENDERERMODULE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "Module.h"
#include "CalleeSlot.h"
#include "vislib/graphicstypes.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * Base class of rendering graph renderer modules.
     *
     * A renderer must be able to use a data source modules scale and offset.
     * Note that the data should first be scaled and then be offsetted.
     */
    class MEGAMOLCORE_API RendererModule : public Module {
    public:

        /** Ctor. */
        RendererModule(void);

        /** Dtor. */
        virtual ~RendererModule(void);

    protected:

        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Render(Call& call) = 0;

    private:

        /**
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        bool RenderCallback(Call& call) {
            return this->Render(call);
        }

        /** The render callee slot */
        CalleeSlot renderSlot;

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_RENDERERMODULE_H_INCLUDED */
