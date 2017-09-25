/*
 * DemoRenderer2D.h
 *
 * Copyright (C) 2009 - 2010 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_DEMORENDERER2D_H_INCLUDED
#define MEGAMOLCORE_DEMORENDERER2D_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/factories/ModuleAutoDescription.h"
#include "mmcore/view/Renderer2DModule.h"
#include "mmcore/view/MouseFlags.h"


namespace megamol {
namespace core {
namespace view {
namespace special {


    /**
     * A simple 2d renderer which just creates a circle
     */
    class DemoRenderer2D : public view::Renderer2DModule {
    public:

        /**
         * The class name for the factory
         *
         * @return The class name
         */
        static const char *ClassName(void) {
            return "DemoRenderer2D";
        }

        /**
         * A human-readable description string for the module
         *
         * @return The description string
         */
        static const char *Description(void) {
            return "Demo 2D-Renderer";
        }

        /**
         * Test if the module can be instanziated
         *
         * @return 'true'
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

        /**
         * ctor
         */
        DemoRenderer2D();

        /**
         * dtor
         */
        virtual ~DemoRenderer2D();

    protected:

        /**
         * Initializes the module directly after instanziation
         *
         * @return 'true' on success
         */
        virtual bool create(void);

        /**
         * Sets the extents (animation and bounding box) into the call object
         *
         * @param call The incoming call
         *
         * @return 'true' on success
         */
        virtual bool GetExtents(view::CallRender2D& call);

        /**
         * Renders the scene
         *
         * @param call The incoming call
         *
         * @return 'true' on success
         */
        virtual bool Render(view::CallRender2D& call);

        /**
         * Releases all resources of the module
         */
        virtual void release(void);

        /**
         * Callback for mouse events (move, press, and release)
         *
         * @param x The x coordinate of the mouse in world space
         * @param y The y coordinate of the mouse in world space
         * @param flags The mouse flags
         */
        virtual bool MouseEvent(float x, float y, view::MouseFlags flags);

    private:

        /** The mouse coordinate */
        float mx, my;

        /** The coordinates to draw the test line from */
        float fromx, fromy;

        /** The coordinates to draw the test line to */
        float tox, toy;

        /** Flag if the test line is being spanned */
        bool drag;

    };


} /* end namespace special */
} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_DEMORENDERER2D_H_INCLUDED */
