/*
 * render_to_file.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AbstractView.h"
#include "mmcore/view/CallRenderView.h"
#include "mmcore/view/Input.h"

#include "vislib/Serialiser.h"

#include <memory>

namespace megamol {
namespace flowvis {

/**
 * Abstract base class of rendering views
 */
class render_to_file : public core::view::AbstractView {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "render_to_file"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Render input to file and also to screen"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    /**
     * Answer whether or not this module supports being used in a
     * quickstart. Overwrite if you don't want your module to be used in
     * quickstarts.
     *
     * This default implementation returns 'true'
     *
     * @return Whether or not this module supports being used in a
     *         quickstart.
     */
    static bool SupportQuickstart(void) { return false; }

    /** Ctor. */
    render_to_file(void);

    /** Dtor. */
    virtual ~render_to_file(void);

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
     * Renders this AbstractView3D in the currently active OpenGL context.
     *
     * @param context
     */
    virtual void Render(const mmcRenderViewContext& context);

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
     * Callback requesting a rendering of this view
     *
     * @param call The calling call
     *
     * @return The return value
     */
    virtual bool OnRenderView(core::Call& call);

    /**
     * Freezes, updates, or unfreezes the view onto the scene (not the
     * rendering, but camera settings, timing, etc).
     *
     * @param freeze true means freeze or update freezed settings,
     *               false means unfreeze
     */
    virtual void UpdateFreeze(bool freeze);

    virtual bool OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) override;

    virtual bool OnChar(unsigned int codePoint) override;

    virtual bool OnMouseButton(
        core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) override;

    virtual bool OnMouseMove(double x, double y) override;

    virtual bool OnMouseScroll(double dx, double dy) override;

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
     * Unpacks the mouse coordinates, which are relative to the virtual
     * viewport size.
     *
     * @param x The x coordinate of the mouse position
     * @param y The y coordinate of the mouse position
     */
    virtual void unpackMouseCoordinates(float& x, float& y);

private:
    /**
     * Record the screen and save to file
     */
    bool record(core::param::ParamSlot&);

    /** Connection to a view */
    core::CallerSlot viewSlot;

    /** Window width and height */
    unsigned int width, height;

    /** Viewport width and height */
    unsigned int vp_width, vp_height;

    /** Incoming call */
    core::view::CallRenderView* override_view_call;

    /** Store time info */
    float time;
    double instance_time;

    /** Parameters for the framebuffer */
    core::param::ParamSlot fbo_width;
    core::param::ParamSlot fbo_height;
    core::param::ParamSlot fbo_viewport;

    /** Parameters for file output */
    core::param::ParamSlot output_file;

    /** Trigger */
    core::param::ParamSlot trigger;
};

} // namespace flowvis
} /* end namespace megamol */
