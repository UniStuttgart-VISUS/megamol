/*
 * AbstractSimpleSphereRenderer.h
 *
 * Copyright (C) 2012 by CGV (TU Dresden)
 * Copyright (C) 2018 by MegaMol Dev Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTSIMPLESPHERERENDERER_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTSIMPLESPHERERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
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

        /**
         * Answer the value of the forceTimeSlot parameter
         *
         * @return True if the time value requested from the data source should be forced
         */
        bool isTimeForced(void) const;

        /** The call for Transfer function */
        CallerSlot getTFSlot;

        /** A simple black-to-white transfer function texture as fallback */
        unsigned int greyTF;

        /** Bool parameter slot to force time */
        param::ParamSlot forceTimeSlot;

        /** Determine whether global or local bbox should be used */
        param::ParamSlot useLocalBBoxParam;

    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTSIMPLESPHERERENDERER_H_INCLUDED */
