/*
 * SplitView.h
 *
 * Copyright (C) 2012 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_SPLITVIEW_H_INCLUDED
#define MEGAMOLCORE_SPLITVIEW_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AbstractView.h"
#include "mmcore/view/CallRenderView.h"
#include "mmcore/param/ColorParam.h"
#include "vislib/graphics/ColourRGBAu8.h"
#include "vislib/graphics/gl/FramebufferObject.h"


namespace megamol {
namespace core {
namespace view {


/**
 * Abstract base class of rendering views
 */
class SplitView : public AbstractView {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "SplitView"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Split View Module"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return vislib::graphics::gl::FramebufferObject::AreExtensionsAvailable(); }

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
    SplitView(void);

    /** Dtor. */
    virtual ~SplitView(void);

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
    virtual bool OnRenderView(Call& call);

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
     * Answer the renderer 1 call
     *
     * @return The renderer 1 call
     */
    inline CallRenderView* render1(void) const { return this->render1Slot.CallAs<CallRenderView>(); }

    /**
     * Answer the renderer 2 call
     *
     * @return The renderer 2 call
     */
    inline CallRenderView* render2(void) const { return this->render2Slot.CallAs<CallRenderView>(); }

    /**
     * Returns the focused (keyboard input) renderer.
     */
    inline CallRenderView* renderFocused() const {
        if (this->focus == 1) {
            return this->render1();
        } else if (this->focus == 2) {
            return this->render2();
        } else {
            return nullptr;
		}
    }

    /**
     * Returns the hovered (mouse input) renderer.
     */
    inline CallRenderView* renderHovered() const {
        auto mousePos = vislib::math::Point<float, 2>(this->mouseX, this->mouseY);
        if (this->client1Area.Contains(mousePos)) {
            return this->render1();
        } else if (this->client2Area.Contains(mousePos)) {
            return this->render2();
        } else {
            return nullptr;
		}
    }

    /**
     * Callback whenever the splitter colour gets updated
     *
     * @param sender The sending slot
     *
     * @return true
     */
    bool splitColourUpdated(param::ParamSlot& sender);

    /**
     * Calculates the client areas
     */
    void calcClientAreas(void);

    /** Connection to a overlay renderer (e.g. GUIRenderer) */
    mutable CallerSlot overlaySlot;

    /** Connection to the renderer 1 (left, top) */
    mutable CallerSlot render1Slot;

    /** Connection to the renderer 2 (right, bottom) */
    mutable CallerSlot render2Slot;

    /** The split orientation slot */
    param::ParamSlot splitOriSlot;

    /** The splitter distance slot */
    param::ParamSlot splitSlot;

    /** The splitter width slot */
    param::ParamSlot splitWidthSlot;

    /** The splitter colour slot */
    param::ParamSlot splitColourSlot;

    /** The splitter colour */
    param::ColorParam::ColorType splitColour;

    /** The override call */
    CallRenderView* overrideCall;

    vislib::math::Rectangle<float> clientArea;

    vislib::math::Rectangle<float> client1Area;

    vislib::math::Rectangle<float> client2Area;

    vislib::graphics::gl::FramebufferObject fbo1;

    vislib::graphics::gl::FramebufferObject fbo2;

    int focus;

    float mouseX;

    float mouseY;
};


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SPLITVIEW_H_INCLUDED */
