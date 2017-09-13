/*
 * ChronoGraph.h
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CHRONOGRAPH_H_INCLUDED
#define MEGAMOLCORE_CHRONOGRAPH_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/factories/ModuleAutoDescription.h"
#include "mmcore/view/Renderer2DModule.h"


namespace megamol {
namespace core {
namespace view {
namespace special {


    /**
     * A simple 2d renderer which just creates a circle
     */
    class ChronoGraph : public view::Renderer2DModule {
    public:

        /**
         * The class name for the factory
         *
         * @return The class name
         */
        static const char *ClassName(void) {
            return "ChronoGraph";
        }

        /**
         * A human-readable description string for the module
         *
         * @return The description string
         */
        static const char *Description(void) {
            return "ChronoGraph renderer displaying the core instance time";
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
        ChronoGraph();

        /**
         * dtor
         */
        virtual ~ChronoGraph();

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

    private:

        /**
         * Renders the info grid into the given rectangle
         *
         * @param time The time code to show
         * @param x The x coordinate
         * @param y The y coordinate
         * @param w The width
         * @param h The height
         */
        void renderInfoGrid(float time, float x, float y, float w, float h);

        /**
         * Renders the info circle into the given rectangle
         *
         * @param time The time code to show
         * @param x The x coordinate
         * @param y The y coordinate
         * @param w The width
         * @param h The height
         */
        void renderInfoCircle(float time, float x, float y, float w, float h);

    };


} /* end namespace special */
} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CHRONOGRAPH_H_INCLUDED */
