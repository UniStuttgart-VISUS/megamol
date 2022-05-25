/**
 * MegaMol
 * Copyright (c) 2010, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/CallerSlot.h"
#include "mmstd_gl/renderer/CallRenderViewGL.h"
#include "mmstd_gl/view/AbstractViewGL.h"

namespace megamol::mmstd_gl::view {

/**
 * Abstract base class of override rendering views
 */
class AbstractOverrideViewGL : public mmstd_gl::view::AbstractViewGL {
public:
    /** Ctor. */
    AbstractOverrideViewGL();

    /** Dtor. */
    ~AbstractOverrideViewGL() override;

    /**
     * Answer the default time for this view
     *
     * @return The default time
     */
    float DefaultTime(double instTime) const override;

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
    void ResetView() override;

    /**
     * Resizes the AbstractView3D.
     *
     * @param width The new width.
     * @param height The new height.
     */
    void Resize(unsigned int width, unsigned int height) override;

    bool OnKey(
        frontend_resources::Key key, frontend_resources::KeyAction action, frontend_resources::Modifiers mods) override;

    bool OnChar(unsigned int codePoint) override;

    bool OnMouseButton(frontend_resources::MouseButton button, frontend_resources::MouseButtonAction action,
        frontend_resources::Modifiers mods) override;

    bool OnMouseMove(double x, double y) override;

    bool OnMouseScroll(double dx, double dy) override;

    core::CallerSlot* GetCallerSlot() {
        return &renderViewSlot;
    }

protected:
    /**
     * Answer the call connected to the render view slot.
     *
     * @return The call connected to the render view slot.
     */
    inline CallRenderViewGL* getCallRenderView() {
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
    inline unsigned int getViewportWidth() const {
        return this->viewportWidth;
    }

    /**
     * Answer the height of the actual viewport in pixels
     *
     * @return The height of the actual viewport in pixels
     */
    inline unsigned int getViewportHeight() const {
        return this->viewportHeight;
    }

    /**
     * Disconnects the outgoing render call
     */
    void disconnectOutgoingRenderCall();

    /**
     * Answer the connected view
     *
     * @return The connected view or NULL if no view is connected
     */
    core::view::AbstractView* getConnectedView() const;

private:
    /** Slot for outgoing rendering requests to other views */
    core::CallerSlot renderViewSlot;

    /** The width of the actual viewport in pixels */
    unsigned int viewportWidth;

    /** The height of the actual viewport in pixels */
    unsigned int viewportHeight;
};

} // namespace megamol::mmstd_gl::view
