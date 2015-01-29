/*
 * AbstractSimpleSphereRenderer.h
 *
 * Copyright (C) 2012 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTSIMPLESPHERERENDERER_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTSIMPLESPHERERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/Renderer3DModule.h"
#include "Call.h"
#include "CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "MultiParticleDataCall.h"


namespace megamol {
namespace core {
namespace moldyn {

    /**
     * Renderer for simple sphere glyphs
     */
    class MEGAMOLCORE_API AbstractSimpleSphereRenderer : public view::Renderer3DModule {
    public:

    protected:

        /** Ctor. */
        AbstractSimpleSphereRenderer(void);

        /** Dtor. */
        virtual ~AbstractSimpleSphereRenderer(void);

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

        ///**
        // * The render callback.
        // *
        // * @param call The calling call.
        // *
        // * @return The return value of the function.
        // */
        //virtual bool Render(Call& call);

        ///** The sphere shader */
        //vislib::graphics::gl::GLSLShader sphereShader;

        /**
         * TODO: Document
         */
        MultiParticleDataCall *getData(unsigned int t, float& outScaling);

        /**
         * TODO: Document
         *
         * @param clipDat points to four floats
         * @param clipCol points to four floats
         */
        void getClipData(float *clipDat, float *clipCol);

    private:

        /** The call for data */
        CallerSlot getDataSlot;

        /** The call for clipping plane */
        CallerSlot getClipPlaneSlot;

    protected:

        /** The call for Transfer function */
        CallerSlot getTFSlot;

        /** A simple black-to-white transfer function texture as fallback */
        unsigned int greyTF;

    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTSIMPLESPHERERENDERER_H_INCLUDED */
