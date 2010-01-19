/*
 * TitleSceneView.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_TITLESCENEVIEW_H_INCLUDED
#define MEGAMOLCORE_TITLESCENEVIEW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "view/AbstractView.h"
#include "Module.h"
#include "vislib/AbstractVISLogo.h"
#include "vislib/CameraOpenGL.h"
#include "vislib/deprecated.h"
#include "vislib/GLSLShader.h"


namespace megamol {
namespace core {
namespace special {


    /**
     * Class of rendering views
     */
    class TitleSceneView : public view::AbstractView,
        public Module {
    public:

        /**
         * Gets the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "TitleSceneView";
        }

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        static const char *Description(void) {
            return "Special view showing the logo of MegaMol(TM)";
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
        VLDEPRECATED TitleSceneView(void);

        /** Dtor. */
        virtual ~TitleSceneView(void);

        /**
         * Renders this view in the currently active OpenGL context.
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
         * Callback requesting a rendering of this view
         *
         * @param call The calling call
         *
         * @return The return value
         */
        virtual bool onRenderView(Call& call);

    private:

        /** The VIS logo */
        static vislib::graphics::AbstractVISLogo *visLogo;

        /** The MegaMol logo */
        static vislib::graphics::AbstractVISLogo *megamolLogo;

        /** The logo usage counter */
        static unsigned int usageCount;

        /** A GPU shader for fancy lighting of the vis logo */
        static vislib::graphics::gl::GLSLShader *fancyShader;

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
         * Answer the current rotation angle for the vis logo.
         *
         * @return The current totation angle.
         */
        inline float getVISAngle(void);

        /** The scene camera */
        vislib::graphics::gl::CameraOpenGL cam;

    };


} /* end namespace special */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_TITLESCENEVIEW_H_INCLUDED */
