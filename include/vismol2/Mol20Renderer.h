/*
 * Mol20Renderer.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MOL20RENDERER_H_INCLUDED
#define MEGAMOLCORE_MOL20RENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vismol2/Mol2Data.h"
#include "view/Renderer3DModule.h"
#include "Call.h"
#include "CallerSlot.h"
#include "param/ParamSlot.h"


namespace megamol {
namespace core {
namespace vismol2 {

    /**
     * Renderer for mol 2.0 (VIS) data
     */
    class Mol20Renderer : public view::Renderer3DModule {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "Mol20Renderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Renderer for Mol 2.0 (VIS) data.";
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
        Mol20Renderer(void);

        /** Dtor. */
        virtual ~Mol20Renderer(void);

    protected:

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
         * The render callback.
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool Render(Call& call);

    private:

        /**
         * Simple recursive rendering call.
         *
         * @param cluster The cluster to render
         * @param ox The origin x coordinate
         * @param oy The origin y coordinate
         * @param oz The origin z coordinate
         * @param alpha The first temporal interpolation coordinate
         * @param beta The second temporal interpolation coordinate
         */
        void render(cluster_t& cluster, float ox, float oy, float oz,
            float alpha, float beta);

        /** The call for data */
        CallerSlot getDataSlot;

    };

} /* end namespace vismol2 */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MOL20RENDERER_H_INCLUDED */
