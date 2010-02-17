/*
 * OverrideView.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_OVERRIDEVIEW_H_INCLUDED
#define MEGAMOLCORE_OVERRIDEVIEW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "param/ParamSlot.h"
#include "view/AbstractOverrideView.h"
#include "vislib/CameraParameters.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * Abstract base class of override rendering views
     */
    class OverrideView : public AbstractOverrideView {
    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "OverrideView";
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

        /** Ctor. */
        OverrideView(void);

        /** Dtor. */
        virtual ~OverrideView(void);

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
         * Resizes the AbstractView3D.
         *
         * @param width The new width.
         * @param height The new height.
         */
        virtual void Resize(unsigned int width, unsigned int height);

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
         * Freezes, updates, or unfreezes the view onto the scene (not the
         * rendering, but camera settings, timing, etc).
         *
         * @param freeze true means freeze or update freezed settings,
         *               false means unfreeze
         */
        virtual void UpdateFreeze(bool freeze);

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
         * Packs the mouse coordinates, which are relative to the virtual
         * viewport size.
         *
         * @param x The x coordinate of the mouse position
         * @param y The y coordinate of the mouse position
         */
        virtual void packMouseCoordinates(float &x, float &y);

    private:

        /** The stereo projection eye */
        vislib::graphics::CameraParameters::StereoEye eye;

        /** The stereo projection eye */
        param::ParamSlot eyeSlot;

        /** The stereo projection type */
        vislib::graphics::CameraParameters::ProjectionType projType;

        /** The stereo projection type */
        param::ParamSlot projTypeSlot;

        /** The height of the rendering tile */
        float tileH;

        /** The rendering tile */
        param::ParamSlot tileSlot;

        /** The width of the rendering tile */
        float tileW;

        /** The x coordinate of the rendering tile */
        float tileX;

        /** The y coordinate of the rendering tile */
        float tileY;

        /** The height of the virtual viewport */
        float virtHeight;

        /** The virtual viewport size */
        param::ParamSlot virtSizeSlot;

        /** The width of the virtual viewport */
        float virtWidth;

        /** The width of the actual viewport in pixels */
        unsigned int viewportWidth;

        /** The height of the actual viewport in pixels */
        unsigned int viewportHeight;

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTOVERRIDEVIEW_H_INCLUDED */
