/*
 * SplitViewGL.h
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
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AbstractView.h"
#include "mmcore_gl/view/CallRenderViewGL.h"
#include "mmcore/view/TimeControl.h"
#include "vislib_gl/graphics/gl/FramebufferObject.h"
#include "vislib/math/Rectangle.h"

namespace megamol {
namespace core_gl {
namespace view {


/**
 * Abstract base class of rendering views
 */
class SplitViewGL : public core::view::AbstractView {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() { return "SplitViewGL"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() { return "A view embedding two other views separated by a splitter"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() { return vislib_gl::graphics::gl::FramebufferObject::AreExtensionsAvailable(); }

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
    static bool SupportQuickstart() { return false; }

    /** Ctor. */
    SplitViewGL();

    /** Dtor. */
    ~SplitViewGL() override;

    /**
     * Answer the default time for this view
     *
     * @return The default time
     */
    float DefaultTime(double instTime) const override;

    /**
     * Renders this SplitView.
     *
     * @param time ...
     * @param instanceTime ...
     */
    virtual ImageWrapper Render(double time, double instanceTime) override;

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
    virtual void Resize(unsigned int width, unsigned int height) override;

    /**
     * Callback requesting a rendering of this view
     *
     * @param call The calling call
     *
     * @return The return value
     */
    bool OnRenderView(core::Call& call) override;

    bool OnKey(frontend_resources::Key key, frontend_resources::KeyAction action, frontend_resources::Modifiers mods) override;

    bool OnChar(unsigned int codePoint) override;

    bool OnMouseButton(frontend_resources::MouseButton button, frontend_resources::MouseButtonAction action, frontend_resources::Modifiers mods) override;

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

    /** Override of GetExtents */
    virtual bool GetExtents(core::Call& call) override;

private:
    /**
     * Answer the renderer 1 call
     *
     * @return The renderer 1 call
     */
    inline CallRenderViewGL* render1() const { return this->_render1Slot.CallAs<CallRenderViewGL>(); }

    /**
     * Answer the renderer 2 call
     *
     * @return The renderer 2 call
     */
    inline CallRenderViewGL* render2() const { return this->_render2Slot.CallAs<CallRenderViewGL>(); }

    /**
     * Returns the focused (keyboard input) renderer.
     */
    inline CallRenderViewGL* renderFocused() const {
        if (this->_focus == 1) {
            return this->render1();
        } else if (this->_focus == 2) {
            return this->render2();
        } else {
            return nullptr;
        }
    }

    /**
     * Returns the hovered (mouse input) renderer.
     */
    inline CallRenderViewGL* renderHovered() const {
        auto mousePos = vislib::math::Point<float, 2>(this->_mouseX, this->_mouseY);
        if (this->_clientArea1.Contains(mousePos)) {
            return this->render1();
        } else if (this->_clientArea2.Contains(mousePos)) {
            return this->render2();
        } else {
            return nullptr;
        }
    }

    void updateSize(size_t width, size_t height);

    /**
     * Calculates the client areas
     */
    void adjustClientAreas();

    /** Connection to the renderer 1 (left, top) */
    mutable core::CallerSlot _render1Slot;

    /** Connection to the renderer 2 (right, bottom) */
    mutable core::CallerSlot _render2Slot;

    /** The split orientation slot */
    core::param::ParamSlot _splitOrientationSlot;

    /** The splitter distance slot */
    core::param::ParamSlot _splitPositionSlot;

    /** The splitter width slot */
    core::param::ParamSlot _splitWidthSlot;

    /** The splitter colour slot */
    core::param::ParamSlot _splitColourSlot;

    /** Slot enabling time synchronization */
    core::param::ParamSlot _enableTimeSyncSlot;

    /** Option for forwarding mouse and keyboard events to both child views */
    core::param::ParamSlot _inputToBothSlot;

    vislib::math::Rectangle<float> _clientArea;

    vislib::math::Rectangle<float> _clientArea1;

    vislib::math::Rectangle<float> _clientArea2;

    std::shared_ptr<glowl::FramebufferObject> _fboFull;

    std::shared_ptr<glowl::FramebufferObject> _fbo1;

    std::shared_ptr<glowl::FramebufferObject> _fbo2;

    int _focus;

    float _mouseX;

    float _mouseY;

    bool _dragSplitter;
};


} /* end namespace view */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SPLITVIEW_H_INCLUDED */
