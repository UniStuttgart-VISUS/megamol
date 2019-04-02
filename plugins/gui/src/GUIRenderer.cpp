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
    , LinearTransferFunctionEditor()
    , imgui_context(nullptr)
    , decorated_renderer_slot("decoratedRenderer", "Connects to another 2D Renderer being decorated")
    , overlay_slot("overlayRender", "Connected with SplitView for special overlay rendering")
    , float_print_prec(3) // INIT: Float string format precision
    , windows()
    , lastInstTime(0.0)
    , main_reset_window(false)
    , param_file()
    , active_tf_param(nullptr)
    , show_fps_ms_options(false)
    , current_delay(0.0f)
    , max_delay(2.0f)
    , fps_values()
    , ms_values()
    , fps_value_scale(0.0f)
    , ms_value_scale(0.0f)
    , fps_ms_mode(0)
    , max_value_count(50)
    , font_new_load(false)
    , font_new_filename()
    , font_new_size(13.0f)
    , inst_name()
    , utf8_ranges() {

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
    , LinearTransferFunctionEditor()
    , imgui_context(nullptr)
    , decorated_renderer_slot("decoratedRenderer", "Connects to another 2D Renderer being decorated")
    , overlay_slot("overlayRender", "Connected with SplitView for special overlay rendering")
    , float_print_prec(3) // INIT: Float string format precision
    , windows()
    , lastInstTime(0.0)
    , main_reset_window(false)
    , param_file()
    , active_tf_param(nullptr)
    , show_fps_ms_options(false)
    , current_delay(0.0f)
    , max_delay(2.0f)
    , fps_values()
    , ms_values()
    , fps_value_scale(0.0f)
    , ms_value_scale(0.0f)
    , fps_ms_mode(0)
    , max_value_count(50)
    , font_new_load(false)
    , font_new_filename()
    , font_new_size(13.0f)
    , inst_name()
    , utf8_ranges() {

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
 * GUIRenderer<core::nextgen::Renderer3DModule_2, core::nextgen::CallRender3D_2>::GUIRenderer
 */
template <>
GUIRenderer<core::view::RendererModule<core::nextgen::CallRender3D_2>, core::nextgen::CallRender3D_2>::GUIRenderer()
    : core::view::RendererModule<core::nextgen::CallRender3D_2>()
    , LinearTransferFunctionEditor()
    , imgui_context(nullptr)
    , decorated_renderer_slot("decoratedRenderer", "Connects to another 3D Renderer being decorated")
    , overlay_slot("overlayRender", "Connected with SplitView for special overlay rendering")
    , float_print_prec(3) // INIT: Float string format precision
    , windows()
    , lastInstTime(0.0)
    , main_reset_window(false)
    , param_file()
    , active_tf_param(nullptr)
    , show_fps_ms_options(false)
    , current_delay(0.0f)
    , max_delay(2.0f)
    , fps_values()
    , ms_values()
    , fps_value_scale(0.0f)
    , ms_value_scale(0.0f)
    , fps_ms_mode(0)
    , max_value_count(50)
    , font_new_load(false)
    , font_new_filename()
    , font_new_size(13.0f)
    , inst_name()
    , utf8_ranges() {

    this->decorated_renderer_slot.SetCompatibleCall<core::nextgen::CallRender3D_2Description>();
    this->MakeSlotAvailable(&this->decorated_renderer_slot);

    this->MakeSlotAvailable(&this->renderSlot);

    // Overlay Call
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::CallSplitViewOverlay::FunctionName(core::view::CallSplitViewOverlay::FnOverlay),
        &GUIRenderer<core::view::RendererModule<core::nextgen::CallRender3D_2>,
            core::nextgen::CallRender3D_2>::OnOverlayCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnKey),
        &GUIRenderer<core::view::RendererModule<core::nextgen::CallRender3D_2>,
            core::nextgen::CallRender3D_2>::OnKeyCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnChar),
        &GUIRenderer<core::view::RendererModule<core::nextgen::CallRender3D_2>,
            core::nextgen::CallRender3D_2>::OnCharCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseButton),
        &GUIRenderer<core::view::RendererModule<core::nextgen::CallRender3D_2>,
            core::nextgen::CallRender3D_2>::OnMouseButtonCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseMove),
        &GUIRenderer<core::view::RendererModule<core::nextgen::CallRender3D_2>,
            core::nextgen::CallRender3D_2>::OnMouseMoveCallback);
    this->overlay_slot.SetCallback(core::view::CallSplitViewOverlay::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseScroll),
        &GUIRenderer<core::view::RendererModule<core::nextgen::CallRender3D_2>,
            core::nextgen::CallRender3D_2>::OnMouseScrollCallback);
    this->MakeSlotAvailable(&this->overlay_slot);
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


/**
 * GUIRenderer<core::view::RendererModule<core::nextgen::CallRender3D_2>, core::nextgen::CallRender3D_2>::GetExtents
 */
template <>
bool GUIRenderer<core::view::RendererModule<core::nextgen::CallRender3D_2>, core::nextgen::CallRender3D_2>::GetExtents(
    core::nextgen::CallRender3D_2& call) {

    auto* cr = this->decorated_renderer_slot.CallAs<core::nextgen::CallRender3D_2>();
    if (cr != nullptr) {
        (*cr) = call;
        if ((*cr)(core::view::AbstractCallRender::FnGetExtents)) {
            call = (*cr);
        }
    } else {
        call.AccessBoundingBoxes().Clear();
        call.AccessBoundingBoxes().SetBoundingBox(vislib::math::Cuboid<float>(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f));
    }

    return true;
}

/**
 * GUIRenderer<M, C>::Render
 */
template <>
bool GUIRenderer<core::view::RendererModule<core::nextgen::CallRender3D_2>, core::nextgen::CallRender3D_2>::Render(
    core::nextgen::CallRender3D_2& call) {

    if (this->overlay_slot.GetStatus() == core::AbstractSlot::SlotStatus::STATUS_CONNECTED) {
        vislib::sys::Log::DefaultLog.WriteError("[GUIRenderer][Render] Only one connected callee slot is allowed!");
        return false;
    }

    auto leftSlotParent = call.PeekCallerSlot()->Parent();
    std::shared_ptr<const core::view::AbstractView> viewptr =
        std::dynamic_pointer_cast<const core::view::AbstractView>(leftSlotParent);

    if (viewptr != nullptr) {
        auto vp = call.GetViewport();
        glViewport(vp.Left(), vp.Bottom(), vp.Width(), vp.Height());
        auto backCol = call.BackgroundColor();
        glClearColor(backCol.x, backCol.y, backCol.z, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    auto* cr = this->decorated_renderer_slot.template CallAs<core::nextgen::CallRender3D_2>();
    if (cr != nullptr) {
        (*cr) = call;
        if ((*cr)(core::view::AbstractCallRender::FnRender)) {
            call = (*cr);
        }
    }
    return this->renderGUI(call.GetViewport(), call.InstanceTime());
}