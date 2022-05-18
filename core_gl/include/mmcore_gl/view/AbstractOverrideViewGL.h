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
#include "mmcore_gl/view/AbstractViewGL.h"
#include "mmcore_gl/view/CallRenderViewGL.h"
#include "vislib/forceinline.h"


namespace megamol {
namespace core_gl {
namespace view {


/**
 * Abstract base class of override rendering views
 */
class AbstractOverrideViewGL : public core_gl::view::AbstractViewGL {
public:
    /** Ctor. */
    AbstractOverrideViewGL(void);

    /** Dtor. */
    virtual ~AbstractOverrideViewGL(void);

    /**
     * Answer the default time for this view
     *
     * @return The default time
     */
    virtual float DefaultTime(double instTime) const;

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

    virtual bool OnKey(
        frontend_resources::Key key, frontend_resources::KeyAction action, frontend_resources::Modifiers mods) override;

    virtual bool OnChar(unsigned int codePoint) override;

    virtual bool OnMouseButton(frontend_resources::MouseButton button, frontend_resources::MouseButtonAction action,
        frontend_resources::Modifiers mods) override;

    virtual bool OnMouseMove(double x, double y) override;

    virtual bool OnMouseScroll(double dx, double dy) override;

    core::CallerSlot* GetCallerSlot() {
        return &renderViewSlot;
    }

protected:
    /**
     * Answer the call connected to the render view slot.
     *
     * @return The call connected to the render view slot.
     */
    inline CallRenderViewGL* getCallRenderView(void) {
        return this->renderViewSlot.CallAs<CallRenderViewGL>();
    }

    /**
     * Packs the mouse coordinates, which are relative to the virtual
     * viewport size.
     *
     * @param x The x coordinate of the mouse position
     * @param y The y coordinate of the mouse position
     */
    virtual void packMouseCoordinates(float& x, float& y);

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
    core::view::AbstractView* getConnectedView(void) const;

private:
    /** Slot for outgoing rendering requests to other views */
    core::CallerSlot renderViewSlot;

    /** The width of the actual viewport in pixels */
    unsigned int viewportWidth;

    /** The height of the actual viewport in pixels */
    unsigned int viewportHeight;
};


} /* end namespace view */
} // namespace core_gl
} /* end namespace megamol */

#endif /* MEGAMOLCORE_ABSTRACTOVERRIDEVIEW_H_INCLUDED */
