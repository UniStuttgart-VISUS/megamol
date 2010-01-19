/*
 * TimeView.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_TIMEVIEW_H_INCLUDED
#define MEGAMOLCORE_TIMEVIEW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "view/AbstractView.h"
#include "Module.h"


namespace megamol {
namespace core {
namespace special {


    /**
     * Class of rendering views
     */
    class TimeView : public view::AbstractView,
        public Module {
    public:

        /**
         * Gets the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "TimeView";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Special view showing the core instance time code.";
        }

        /**
         * Gets whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /**
         * Renders this view in the currently active OpenGL context.
         *
         * @param time The time to be rendered.
         */
        static void RenderView(double time);

        /** Ctor. */
        TimeView(void);

        /** Dtor. */
        virtual ~TimeView(void);

        /**
         * Renders this view in the currently active OpenGL context.
         */
        virtual void Render(void);

        /**
         * Resets the view. This normally sets the camera parameters to
         * default values.
         */
        virtual void ResetView(void) {
        }

        /**
         * Resizes the AbstractView3D.
         *
         * @param width The new width.
         * @param height The new height.
         */
        virtual void Resize(unsigned int width, unsigned int height) {
            this->width = width;
            this->height = height;
        }

        /**
         * Sets the button state of a button of the 2d cursor. See
         * 'vislib::graphics::Cursor2D' for additional information.
         *
         * @param button The button.
         * @param down Flag whether the button is pressed, or not.
         */
        virtual void SetCursor2DButtonState(unsigned int btn, bool down) {
            // intentionally empty. Disallow interactions!
        }

        /**
         * Sets the position of the 2d cursor. See 'vislib::graphics::Cursor2D'
         * for additional information.
         *
         * @param x The x coordinate
         * @param y The y coordinate
         */
        virtual void SetCursor2DPosition(float x, float y) {
            // intentionally empty. Disallow interactions!
        }

        /**
         * Sets the state of an input modifier.
         *
         * @param mod The input modifier to be set.
         * @param down The new state of the input modifier.
         */
        virtual void SetInputModifier(mmcInputModifier mod, bool down) {
            // intentionally empty. Disallow interactions!
        }

        /**
         * Freezes, updates, or unfreezes the view onto the scene (not the
         * rendering, but camera settings, timing, etc).
         *
         * @param freeze true means freeze or update freezed settings,
         *               false means unfreeze
         */
        virtual void UpdateFreeze(bool freeze);

    protected:

        /**
         * Implementation of 'Module::Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Implementation of 'Module::Release'.
         */
        virtual void release(void);

        /**
         * Callback requesting a rendering of this view
         *
         * @param call The calling call
         *
         * @return The return value
         */
        virtual bool onRenderView(Call& call);

    private:

        /** The viewport width */
        unsigned int width;

        /** The viewport height */
        unsigned int height;

    };


} /* end namespace special */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_TIMEVIEW_H_INCLUDED */
