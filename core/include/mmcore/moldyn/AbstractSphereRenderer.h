/*
 * AbstractSphereRenderer.h
 *
 * Copyright (C) 2012 by CGV (TU Dresden)
 * Copyright (C) 2018 by MegaMol Dev Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTSPHERERENDERER_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTSPHERERENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/view/Renderer3DModule.h"
#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/param/BoolParam.h"

#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/assert.h"


namespace megamol {
namespace core {
namespace moldyn {

    /**
     * Abstract renderer for simple sphere glyphs.
     */
    class MEGAMOLCORE_API AbstractSphereRenderer : public view::Renderer3DModule {
    public:

    protected:

        /** Ctor. */
        AbstractSphereRenderer(void);

        /** Dtor. */
        virtual ~AbstractSphereRenderer(void);

        /**
         * The get extents callback. The module should set the members of
         * 'call' to tell the caller the extents of its data (bounding boxes
         * and times).
         *
         * @param call The calling call.
         *
         * @return The return value of the function.
         */
        virtual bool GetExtents(megamol::core::view::CallRender3D& call);

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
         * TODO: Document
         *
         * @param t           ...
         * @param outScaling  ...
         *
         * @return Pointer to MultiParticleDataCall ...
         */
        MultiParticleDataCall *getData(unsigned int t, float& outScaling);

        /**
         * TODO: Document
         *
         * @param clipDat  Points to four floats ...
         * @param clipCol  Points to four floats ....
         */
        void getClipData(float *clipDat, float *clipCol);

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

    private:

        /** The call for data */
        CallerSlot getDataSlot;

        /** The call for clipping plane */
        CallerSlot getClipPlaneSlot;
    };

} /* end namespace moldyn */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTSPHERERENDERER_H_INCLUDED */
