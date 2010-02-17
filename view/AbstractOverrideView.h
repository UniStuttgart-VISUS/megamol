/*
 * AbstractOverrideView.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTOVERRIDEVIEW_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTOVERRIDEVIEW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CallerSlot.h"
#include "view/AbstractView.h"
#include "view/CallRenderView.h"
#include "vislib/CameraParameters.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * Abstract base class of override rendering views
     */
    class AbstractOverrideView : public AbstractView {
    public:

        /** Ctor. */
        AbstractOverrideView(void);

        /** Dtor. */
        virtual ~AbstractOverrideView(void);

    protected:

        /**
         * Answer the call connected to the render view slot.
         *
         * @return The call connected to the render view slot.
         */
        inline CallRenderView *getCallRenderView(void) {
            return this->renderViewSlot.CallAs<CallRenderView>();
        }

        /**
         * Packs the mouse coordinates, which are relative to the virtual
         * viewport size.
         *
         * @param x The x coordinate of the mouse position
         * @param y The y coordinate of the mouse position
         */
        virtual void packMouseCoordinates(float &x, float &y);

    private:

        /** Slot for outgoing rendering requests to other views */
        CallerSlot renderViewSlot;

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTOVERRIDEVIEW_H_INCLUDED */
