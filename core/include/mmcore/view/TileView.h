/*
 * TileView.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_OVERRIDEVIEW_H_INCLUDED
#define MEGAMOLCORE_OVERRIDEVIEW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/view/AbstractTileView.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * Abstract base class of override rendering views
     */
    class TileView : public AbstractTileView {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "TileView";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Override View Module";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /**
         * Disallow usage in quickstarts
         *
         * @return false
         */
        static bool SupportQuickstart(void) {
            return false;
        }

        /** Ctor. */
        TileView(void);

        /** Dtor. */
        virtual ~TileView(void);

        /**
         * Renders this AbstractView3D in the currently active OpenGL context.
         */
        virtual void Render(const mmcRenderViewContext& context);

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
         * Callback requesting a rendering of this view
         *
         * @param call The calling call
         *
         * @return The return value
         */
        virtual bool OnRenderView(Call& call);

        /**
         * Unpacks the mouse coordinates, which are relative to the virtual
         * viewport size.
         *
         * @param x The x coordinate of the mouse position
         * @param y The y coordinate of the mouse position
         */
        virtual void unpackMouseCoordinates(float &x, float &y);

    private:

        /** Flag to identify the first frame */
        bool firstFrame;

        /** TODO */
        RenderOutputOpenGL * outCtrl;

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTOVERRIDEVIEW_H_INCLUDED */
