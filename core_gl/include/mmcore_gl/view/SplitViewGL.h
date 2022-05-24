/**
 * MegaMol
 * Copyright (c) 2012, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <glowl/FramebufferObject.hpp>

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/view/AbstractViewGL.h"
#include "mmcore_gl/view/CallRenderViewGL.h"
#include "vislib/math/Rectangle.h"

namespace megamol::core_gl::view {

/**
 * Abstract base class of rendering views
 */
class SplitViewGL : public core_gl::view::AbstractViewGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "SplitViewGL";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "A view embedding two other views separated by a splitter";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
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
    static bool SupportQuickstart() {
        return false;
    }

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

    /** Override of GetExtents */
    bool GetExtents(core::Call& call) override;

private:
    /**
     * Answer the renderer 1 call
     *
     * @return The renderer 1 call
     */
    inline CallRenderViewGL* render1() const {
        return _render1Slot.CallAs<CallRenderViewGL>();
    }

    /**
     * Answer the renderer 2 call
     *
     * @return The renderer 2 call
     */
    inline CallRenderViewGL* render2() const {
        return _render2Slot.CallAs<CallRenderViewGL>();
    }

    /**
     * Returns the focused (keyboard input) renderer.
     */
    inline CallRenderViewGL* renderFocused() const {
        if (_focus == 1) {
            return render1();
        } else if (_focus == 2) {
            return render2();
        } else {
            return nullptr;
        }
    }

    /**
     * Returns the hovered (mouse input) renderer.
     */
    inline CallRenderViewGL* renderHovered() const {
        auto mousePos = vislib::math::Point<int, 2>(static_cast<int>(_mouseX), static_cast<int>(_mouseY));
        if (_clientArea1.Contains(mousePos)) {
            return render1();
        } else if (_clientArea2.Contains(mousePos)) {
            return render2();
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

    vislib::math::Rectangle<int> _clientArea;

    vislib::math::Rectangle<int> _clientArea1;

    vislib::math::Rectangle<int> _clientArea2;

    std::shared_ptr<glowl::FramebufferObject> _fboFull;

    std::shared_ptr<glowl::FramebufferObject> _fbo1;

    std::shared_ptr<glowl::FramebufferObject> _fbo2;

    int _focus;

    double _mouseX;

    double _mouseY;

    bool _dragSplitter;
};


} // namespace megamol::core_gl::view
