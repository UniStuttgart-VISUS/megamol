/*
 * GUIRenderer.cpp
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "GUIRenderer.h"


using namespace megamol;
using namespace megamol::gui;


/**
 * GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::GUIRenderer
 */
template <>
GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::GUIRenderer()
    : core::view::Renderer2DModule()
    , decorated_renderer_slot("decoratedRenderer", "Connects to another 2D Renderer being decorated")
    , overlay_slot("overlayRender", "Connected with SplitView for special overlay rendering")
    , imgui_context(nullptr)
    , window_manager()
    , tf_editor()
    , last_instance_time(0.0)
    , font_utf8_ranges()
    , load_new_profile()
    , load_new_font_filename()
    , load_new_font_size(13.0f)
    , load_new_font_index(-1)
    , loaded_profile_list()
    , delete_window() {

    this->decorated_renderer_slot.SetCompatibleCall<core::view::CallRender2DDescription>();
    this->MakeSlotAvailable(&this->decorated_renderer_slot);

    // InputCall
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::CallSplitViewOverlay::FunctionName(core::view::CallSplitViewOverlay::FnOverlay),
        &GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::OnOverlayCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnKey),
        &GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::OnKeyCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnChar),
        &GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::OnCharCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseButton),
        &GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::OnMouseButtonCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseMove),
        &GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::OnMouseMoveCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseScroll),
        &GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::OnMouseScrollCallback);
    this->MakeSlotAvailable(&this->overlay_slot);
}


/**
 * GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::GUIRenderer
 */
template <>
GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::GUIRenderer()
    : core::view::Renderer3DModule()
    , decorated_renderer_slot("decoratedRenderer", "Connects to another 2D Renderer being decorated")
    , overlay_slot("overlayRender", "Connected with SplitView for special overlay rendering")
    , imgui_context(nullptr)
    , window_manager()
    , tf_editor()
    , last_instance_time(0.0)
    , font_utf8_ranges()
    , load_new_profile()
    , load_new_font_filename()
    , load_new_font_size(13.0f)
    , load_new_font_index(-1)
    , loaded_profile_list()
    , delete_window() {

    this->decorated_renderer_slot.SetCompatibleCall<core::view::CallRender3DDescription>();
    this->MakeSlotAvailable(&this->decorated_renderer_slot);

    // Overlay Call
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::CallSplitViewOverlay::FunctionName(core::view::CallSplitViewOverlay::FnOverlay),
        &GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::OnOverlayCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnKey),
        &GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::OnKeyCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnChar),
        &GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::OnCharCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseButton),
        &GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::OnMouseButtonCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseMove),
        &GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::OnMouseMoveCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseScroll),
        &GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::OnMouseScrollCallback);
    this->MakeSlotAvailable(&this->overlay_slot);
}


/**
 * GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::ClassName
 */
template <> const char* GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::ClassName(void) {

    return "GUIRenderer2D";
}


/**
 * GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::ClassName
 */
template <> const char* GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::ClassName(void) {

    return "GUIRenderer3D";
}


/**
 * GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::GetExtents
 */
template <>
bool GUIRenderer<core::view::Renderer2DModule, core::view::CallRender2D>::GetExtents(core::view::CallRender2D& call) {

    auto* cr = this->decorated_renderer_slot.CallAs<core::view::CallRender2D>();
    if (cr != nullptr) {
        (*cr) = call;
        if ((*cr)(core::view::AbstractCallRender::FnGetExtents)) {
            call = (*cr);
        }
    } else {
        call.SetBoundingBox(vislib::math::Rectangle<float>(0, 1, 1, 0));
    }

    return true;
}


/**
 * GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::GetExtents
 */
template <>
bool GUIRenderer<core::view::Renderer3DModule, core::view::CallRender3D>::GetExtents(core::view::CallRender3D& call) {

    auto* cr = this->decorated_renderer_slot.CallAs<core::view::CallRender3D>();
    if (cr != nullptr) {
        (*cr) = call;
        if ((*cr)(core::view::AbstractCallRender::FnGetExtents)) {
            call = (*cr);
        }
    } else {
        call.AccessBoundingBoxes().Clear();
        call.AccessBoundingBoxes().SetWorldSpaceBBox(
            vislib::math::Cuboid<float>(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f));
    }

    return true;
}
