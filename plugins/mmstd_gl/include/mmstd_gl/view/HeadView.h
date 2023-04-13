/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/view/Input.h"
#include "mmstd_gl/renderer/CallRenderViewGL.h"
#include "mmstd_gl/view/AbstractViewGL.h"
#include "vislib/Serialiser.h"

namespace megamol::mmstd_gl::view {

/**
 * Abstract base class of rendering views
 */
class HeadView : public mmstd_gl::view::AbstractViewGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "HeadView";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Head View Module";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor. */
    HeadView();

    /** Dtor. */
    ~HeadView() override;

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
     * Renders this AbstractView3D in the currently active OpenGL context.
     *
     * @param context
     */
    ImageWrapper Render(double time, double instanceTime) override;

    ImageWrapper GetRenderingResult() const override;

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

    /**
     * Callback requesting a rendering of this view
     *
     * @param call The calling call
     *
     * @return The return value
     */
    bool OnRenderView(core::Call& call) override;

    bool OnKey(
        frontend_resources::Key key, frontend_resources::KeyAction action, frontend_resources::Modifiers mods) override;

    bool OnChar(unsigned int codePoint) override;

    bool OnMouseButton(frontend_resources::MouseButton button, frontend_resources::MouseButtonAction action,
        frontend_resources::Modifiers mods) override;

    bool OnMouseMove(double x, double y) override;

    bool OnMouseScroll(double dx, double dy) override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * Unpacks the mouse coordinates, which are relative to the virtual
     * viewport size.
     *
     * @param x The x coordinate of the mouse position
     * @param y The y coordinate of the mouse position
     */
    virtual void unpackMouseCoordinates(float& x, float& y);

private:
    /** Connection to a view */
    core::CallerSlot viewSlot;

    /** Connection to a module in desperate need for an invocation */
    core::CallerSlot tickSlot;

    /** Window width and height */
    unsigned int width, height;

    /** Incoming call */
    CallRenderViewGL* override_view_call;
};

} // namespace megamol::mmstd_gl::view
