/*
 * HeadView.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/view/Input.h"

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore_gl/view/AbstractViewGL.h"
#include "mmcore_gl/view/CallRenderViewGL.h"

#include "vislib/Serialiser.h"

namespace megamol {
namespace core_gl {
namespace view {

/**
 * Abstract base class of rendering views
 */
class HeadView : public core_gl::view::AbstractViewGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "HeadView";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Head View Module";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

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
    static bool SupportQuickstart(void) {
        return false;
    }

    /** Ctor. */
    HeadView(void);

    /** Dtor. */
    virtual ~HeadView(void);

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
     * Renders this AbstractView3D in the currently active OpenGL context.
     *
     * @param context
     */
    virtual ImageWrapper Render(double time, double instanceTime) override;

    ImageWrapper GetRenderingResult() const override;

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

    virtual bool OnKey(
        frontend_resources::Key key, frontend_resources::KeyAction action, frontend_resources::Modifiers mods) override;

    virtual bool OnChar(unsigned int codePoint) override;

    virtual bool OnMouseButton(frontend_resources::MouseButton button, frontend_resources::MouseButtonAction action,
        frontend_resources::Modifiers mods) override;

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
    /** Connection to a view */
    core::CallerSlot viewSlot;

    /** Connection to a module in desperate need for an invocation */
    core::CallerSlot tickSlot;

    /** Window width and height */
    unsigned int width, height;

    /** Incoming call */
    view::CallRenderViewGL* override_view_call;
};

} /* end namespace view */
} // namespace core_gl
} /* end namespace megamol */
