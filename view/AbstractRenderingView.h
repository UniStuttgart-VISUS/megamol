/*
 * AbstractRenderingView.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTRENDERINGVIEW_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTRENDERINGVIEW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "param/ParamSlot.h"
#include "view/AbstractView.h"
#include "vislib/Serialiser.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * Abstract base class of rendering views
     */
    class AbstractRenderingView : public AbstractView {
    public:

        /** Ctor. */
        AbstractRenderingView(void);

        /** Dtor. */
        virtual ~AbstractRenderingView(void);

        /**
         * Answer the camera synchronization number.
         *
         * @return The camera synchronization number
         */
        virtual unsigned int GetCameraSyncNumber(void) const = 0;

        /**
         * Serialises the camera of the view
         *
         * @param serialiser Serialises the camera of the view
         */
        virtual void SerialiseCamera(vislib::Serialiser& serialiser) const = 0;

    protected:

        /**
         * Answer the background colour for the view
         *
         * @return The background colour for the view
         */
        const float *bkgndColour(void) const;

        /**
         * Answer if the soft cursor should be shown
         *
         * @return 'true' if the soft cursor should be shown
         */
        bool showSoftCursor(void) const;

        /** Pointer to the override background colour */
        float *overrideBkgndCol;

        /** Pointer to the override viewport */
        int *overrideViewport;

    private:

        /** The background colour for the view */
        mutable float bkgndCol[3];

        /** The background colour for the view */
        mutable param::ParamSlot bkgndColSlot;

        /** Bool flag to activate software cursor rendering */
        mutable bool softCursor;

        /** Bool flag to activate software cursor rendering */
        mutable param::ParamSlot softCursorSlot;

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTRENDERINGVIEW_H_INCLUDED */
