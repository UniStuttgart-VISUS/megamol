/*
 * Renderer3DModuleDS.h
 *
 * Copyright (C) 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_RENDERER3DMODULEDS_H_INCLUDED
#define MEGAMOLCORE_RENDERER3DMODULEDS_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/view/Renderer3DModule.h"
#include "vislib/graphics/graphicstypes.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * Base class of rendering graph 3D renderer modules.
     * Note: Provides an additional callee slot for deferred shading.
     */
    class MEGAMOLCORE_API Renderer3DModuleDS : public Renderer3DModule {
    public:

        /** Ctor. */
        Renderer3DModuleDS(void);

        /** Dtor. */
        virtual ~Renderer3DModuleDS(void);

    private:

        /**
         * The get capabilities callback. The module should set the members
         * of 'call' to tell the caller its capabilities.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        bool GetCapabilitiesCallback(Call& call) {
            return this->GetCapabilities(call);
        }

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        bool GetExtentsCallback(Call& call) {
            return this->GetExtents(call);
        }

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

        /** The render callee slot for deferred shading */
        CalleeSlot renderSlotDS;

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_RENDERERMODULEDS_H_INCLUDED */
