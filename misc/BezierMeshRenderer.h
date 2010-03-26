/*
 * BezierMeshRenderer.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_BEZIERMESHRENDERER_H_INCLUDED
#define MEGAMOLCORE_BEZIERMESHRENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "view/Renderer3DModule.h"
#include "CallerSlot.h"
#include "param/ParamSlot.h"
//#include "vislib/GLSLShader.h"


namespace megamol {
namespace core {
namespace misc {

    /**
     * Mesh-based renderer for bézier curve tubes
     */
    class BezierMeshRenderer : public view::Renderer3DModule {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "BezierMeshRenderer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Renderer for bézier curve";
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
        BezierMeshRenderer(void);

        /** Dtor. */
        virtual ~BezierMeshRenderer(void);

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

        /** The call for data */
        CallerSlot getDataSlot;

        /** The number of linear sections along the curve */
        param::ParamSlot curveSectionsSlot;

        /** The number of section along the profile */
        param::ParamSlot profileSectionsSlot;

        /** Controlls the type of the curve caps */
        param::ParamSlot capTypeSlot;

        /** The display list storing the objects */
        unsigned int objs;

        /** The data hash of the objects stored in the list */
        SIZE_T objsHash;

    };

} /* end namespace misc */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_BEZIERMESHRENDERER_H_INCLUDED */
