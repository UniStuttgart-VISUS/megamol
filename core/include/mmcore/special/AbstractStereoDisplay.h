/*
 * AbstractStereoDisplay.h
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_ABSTRACTSTEREODISPLAY_H_INCLUDED
#define MEGAMOLCORE_ABSTRACTSTEREODISPLAY_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/special/ClusterDisplayPlane.h"
#include "mmcore/special/ClusterDisplayTile.h"
#include "mmcore/view/AbstractView.h"
#include "mmcore/view/CallRenderViewGL.h"


namespace megamol {
namespace core {
namespace special {

    /**
     * Special view module used for column based stereo displays
     */
    class AbstractStereoDisplay : public view::AbstractView, public Module {
    public:

        /** Dtor. */
        virtual ~AbstractStereoDisplay(void);

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

        /** Ctor. */
        AbstractStereoDisplay(void);

        /**
         * Answer the width of the actual viewport in pixel
         *
         * @return The width of the actual viewport in pixel
         */
        inline unsigned int width(void) const {
            return this->viewportWidth;
        }

        /**
         * Answer the height of the actual viewport in pixel
         *
         * @return The height of the actual viewport in pixel
         */
        inline unsigned int height(void) const {
            return this->viewportHeight;
        }

        /**
         * Gets the connected call from the renderViewSlot.
         *
         * @return The connected call from the renderViewSlot or NULL
         */
        inline view::CallRenderViewGL *getCallRenderView(void) {
            return this->renderViewSlot.CallAs<view::CallRenderViewGL>();
        }

        /**
         * Callback requesting a rendering of this view
         *
         * @param call The calling call
         *
         * @return The return value
         */
        virtual bool onRenderView(Call& call);

    private:

        /** The width of the actual viewport in pixels */
        unsigned int viewportWidth;

        /** The height of the actual viewport in pixels */
        unsigned int viewportHeight;

        /** caller slot connected to the view to be rendered */
        CallerSlot renderViewSlot;

        /** caller slot for sending the cursor input */
        CallerSlot cursorInputSlot;

    };


} /* end namespace special */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTSTEREODISPLAY_H_INCLUDED */
