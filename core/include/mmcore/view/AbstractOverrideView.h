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

#include "mmcore/CallerSlot.h"
#include "mmcore/view/AbstractView.h"
#include "mmcore/view/CallRenderViewGL.h"
#include "vislib/forceinline.h"


namespace megamol {
namespace core {
namespace view {


    /**
     * Abstract base class of override rendering views
     */
    class MEGAMOLCORE_API AbstractOverrideView : public AbstractView {
    public:

        /** Ctor. */
        AbstractOverrideView(void);

        /** Dtor. */
        virtual ~AbstractOverrideView(void);

        /**
         * Answer the default time for this view
         *
         * @return The default time
         */
        virtual float DefaultTime(double instTime) const;

        /**
         * Answer the camera synchronization number.
         *
         * @return The camera synchronization number
         */
        virtual unsigned int GetCameraSyncNumber(void) const;

        /**
         * Serialises the camera of the view
         *
         * @param serialiser Serialises the camera of the view
         */
        virtual void SerialiseCamera(vislib::Serialiser& serialiser) const;

        /**
         * Deserialises the camera of the view
         *
         * @param serialiser Deserialises the camera of the view
         */
        virtual void DeserialiseCamera(vislib::Serialiser& serialiser);

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

        virtual bool OnKey(Key key, KeyAction action, Modifiers mods) override;

        virtual bool OnChar(unsigned int codePoint) override;

        virtual bool OnMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) override;

        virtual bool OnMouseMove(double x, double y) override;

        virtual bool OnMouseScroll(double dx, double dy) override;

        CallerSlot* GetCallerSlot() { return &renderViewSlot; }

    protected:

        /**
         * Answer the call connected to the render view slot.
         *
         * @return The call connected to the render view slot.
         */
        inline CallRenderViewGL *getCallRenderView(void) {
            return this->renderViewSlot.CallAs<CallRenderViewGL>();
        }

        /**
         * Packs the mouse coordinates, which are relative to the virtual
         * viewport size.
         *
         * @param x The x coordinate of the mouse position
         * @param y The y coordinate of the mouse position
         */
        virtual void packMouseCoordinates(float &x, float &y);

        /**
         * Answer the width of the actual viewport in pixels
         *
         * @return The width of the actual viewport in pixels
         */
        VISLIB_FORCEINLINE unsigned int getViewportWidth(void) const {
            return this->viewportWidth;
        }

        /**
         * Answer the height of the actual viewport in pixels
         *
         * @return The height of the actual viewport in pixels
         */
        VISLIB_FORCEINLINE unsigned int getViewportHeight(void) const {
            return this->viewportHeight;
        }

        /**
         * Disconnects the outgoing render call
         */
        void disconnectOutgoingRenderCall(void);

        /**
         * Answer the connected view
         *
         * @return The connected view or NULL if no view is connected
         */
        view::AbstractView *getConnectedView(void) const;

    private:

        /** Slot for outgoing rendering requests to other views */
        CallerSlot renderViewSlot;

        /** The width of the actual viewport in pixels */
        unsigned int viewportWidth;

        /** The height of the actual viewport in pixels */
        unsigned int viewportHeight;

    };


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTOVERRIDEVIEW_H_INCLUDED */
