/*
 * ClusterDisplay.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_CLUSTERDISPLAY_H_INCLUDED
#define MEGAMOLCORE_CLUSTERDISPLAY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CallerSlot.h"
#include "special/RenderSlave.h"


namespace megamol {
namespace core {
namespace special {


    /**
     * Cluster display module for displaying slave images rendered locally
     */
    class ClusterDisplay : public RenderSlave {
    public:

        /**
         * Gets the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "ClusterDisplay";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Special view module for displaying images on a local display";
        }

        /**
         * Gets whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

        /** Ctor. */
        ClusterDisplay(void);

        /** Dtor. */
        virtual ~ClusterDisplay(void);

        /**
         * Renders this AbstractView3D in the currently active OpenGL context.
         */
        virtual void Render(void);

        /**
         * Resets the view. This normally sets the camera parameters to
         * default values.
         */
        virtual void ResetView(void);

        /**
         * Sets the button state of a button of the 2d cursor. See
         * 'vislib::graphics::Cursor2D' for additional information.
         *
         * @param button The button.
         * @param down Flag whether the button is pressed, or not.
         */
        virtual void SetCursor2DButtonState(unsigned int btn, bool down);

        /**
         * Sets the position of the 2d cursor. See 'vislib::graphics::Cursor2D'
         * for additional information.
         *
         * @param x The x coordinate
         * @param y The y coordinate
         */
        virtual void SetCursor2DPosition(float x, float y);

        /**
         * Sets the state of an input modifier.
         *
         * @param mod The input modifier to be set.
         * @param down The new state of the input modifier.
         */
        virtual void SetInputModifier(mmcInputModifier mod, bool down);

        /**
         * Answers the desired window position configuration of this view.
         *
         * @param x To receive the coordinate of the upper left corner
         * @param y To recieve the coordinate of the upper left corner
         * @param w To receive the width
         * @param h To receive the height
         * @param nd To receive the flag deactivating window decorations
         *
         * @return 'true' if this view has a desired window position
         *         configuration, 'false' if not. In the latter case the value
         *         the parameters are pointing to are not altered.
         */
        virtual bool DesiredWindowPosition(int *x, int *y, int *w, int *h, bool *nd);

    private:

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
         * Answer the most likely instance name of the display
         *
         * @return The most likely instance name.
         */
        const vislib::StringA& instName(void);

        /** The view plane id */
        param::ParamSlot viewplane;

        /** The tile rectangle on the view plane */
        param::ParamSlot viewTile;

        /** caller slot for sending the cursor input */
        CallerSlot cursorInputSlot;

    };


} /* end namespace special */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_CLUSTERDISPLAY_H_INCLUDED */
