/*
 * OverlayInterfaceLayer.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "stdafx.h"
#include "OverlayInterfaceLayer.h"


using namespace megamol::gui;

/**
 * OverlayInterfaceLayer::OverlayInterfaceLayer
 */
template <> OverlayInterfaceLayer<megamol::core::view::Renderer2DModule>::OverlayInterfaceLayer(void) :
    overlay_slot("overlayRender", "Connected with SplitView for special overlay rendering (e.g. by gui::GUIRenderer)")
{

    this->overlay_slot.SetCallback(megamol::core::view::Renderer2DModule::ClassName(),
    core::view::InputCall::FunctionName(core::view::InputCall::FnOnKey),
    &OverlayInterfaceLayer<megamol::core::view::Renderer2DModule>::OnGUIKeyCallback);

    this->overlay_slot.SetCallback(megamol::core::view::Renderer2DModule::ClassName(),
    core::view::InputCall::FunctionName(core::view::InputCall::FnOnChar),
        &OverlayInterfaceLayer<megamol::core::view::Renderer2DModule>::OnGUICharCallback);

    this->overlay_slot.SetCallback(megamol::core::view::Renderer2DModule::ClassName(),
    core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseButton),
        &OverlayInterfaceLayer<megamol::core::view::Renderer2DModule>::OnGUIMouseButtonCallback);

    this->overlay_slot.SetCallback(megamol::core::view::Renderer2DModule::ClassName(),
    core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseMove),
        &OverlayInterfaceLayer<megamol::core::view::Renderer2DModule>::OnGUIMouseMoveCallback);

    this->overlay_slot.SetCallback(megamol::core::view::Renderer2DModule::ClassName(),
    core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseScroll),
        &OverlayInterfaceLayer<megamol::core::view::Renderer2DModule>::OnGUIMouseScrollCallback);

    this->overlay_slot.SetCallback(megamol::core::view::Renderer2DModule::ClassName(),
    core::view::CallSplitViewOverlay::FunctionName(core::view::CallSplitViewOverlay::FnRender),
        &OverlayInterfaceLayer<megamol::core::view::Renderer2DModule>::OnGUIRenderCallback);

    this->MakeSlotAvailable(&this->overlay_slot);

}

template <> OverlayInterfaceLayer<megamol::core::view::Renderer3DModule>::OverlayInterfaceLayer(void) :
    overlay_slot("overlayRender", "Connected with SplitView for special overlay rendering (e.g. by gui::GUIRenderer)")
{

    this->overlay_slot.SetCallback(megamol::core::view::Renderer3DModule::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnKey),
        &OverlayInterfaceLayer<megamol::core::view::Renderer3DModule>::OnGUIKeyCallback);

    this->overlay_slot.SetCallback(megamol::core::view::Renderer3DModule::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnChar),
        &OverlayInterfaceLayer<megamol::core::view::Renderer3DModule>::OnGUICharCallback);

    this->overlay_slot.SetCallback(megamol::core::view::Renderer3DModule::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseButton),
        &OverlayInterfaceLayer<megamol::core::view::Renderer3DModule>::OnGUIMouseButtonCallback);

    this->overlay_slot.SetCallback(megamol::core::view::Renderer3DModule::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseMove),
        &OverlayInterfaceLayer<megamol::core::view::Renderer3DModule>::OnGUIMouseMoveCallback);

    this->overlay_slot.SetCallback(megamol::core::view::Renderer3DModule::ClassName(),
        core::view::InputCall::FunctionName(core::view::InputCall::FnOnMouseScroll),
        &OverlayInterfaceLayer<megamol::core::view::Renderer3DModule>::OnGUIMouseScrollCallback);

    this->overlay_slot.SetCallback(megamol::core::view::Renderer3DModule::ClassName(),
        core::view::CallSplitViewOverlay::FunctionName(core::view::CallSplitViewOverlay::FnRender),
        &OverlayInterfaceLayer<megamol::core::view::Renderer3DModule>::OnGUIRenderCallback);

    this->MakeSlotAvailable(&this->overlay_slot);

}

/**
 * OverlayInterfaceLayer::~OverlayInterfaceLayer
 */
template <class M> megamol::gui::OverlayInterfaceLayer<M>::~OverlayInterfaceLayer(void) {

    // nothing to do here ...
}


/**
 * OverlayInterfaceLayer::OnGUIKeyCallback
 */
template <class M> bool megamol::gui::OverlayInterfaceLayer<M>::OnGUIKeyCallback(megamol::core::Call& call) {
    try {
        core::view::CallSplitViewOverlay& cr = dynamic_cast<core::view::CallSplitViewOverlay&>(call);
        auto& evt = cr.GetInputEvent();
        ASSERT(evt.tag == core::view::InputEvent::Tag::Key && "Callback invocation mismatched input event");
        return this->OnKey(evt.keyData.key, evt.keyData.action, evt.keyData.mods);
    } catch (...) {
        ASSERT("OnGUIKeyCallback call cast failed\n");
    }
    return false;
}


/**
 * OverlayInterfaceLayer::OnGUICharCallback
 */
template <class M> bool megamol::gui::OverlayInterfaceLayer<M>::OnGUICharCallback(megamol::core::Call& call) {
    try {
        core::view::CallSplitViewOverlay& cr = dynamic_cast<core::view::CallSplitViewOverlay&>(call);
        auto& evt = cr.GetInputEvent();
        ASSERT(evt.tag == core::view::InputEvent::Tag::Char && "Callback invocation mismatched input event");
        return this->OnChar(evt.charData.codePoint);
    } catch (...) {
        ASSERT("OnGUICharCallback call cast failed\n");
    }
    return false;
}


/**
 * OverlayInterfaceLayer::OnGUIMouseButtonCallback
 */
template <class M> bool megamol::gui::OverlayInterfaceLayer<M>::OnGUIMouseButtonCallback(megamol::core::Call& call) {
    try {
        core::view::CallSplitViewOverlay& cr = dynamic_cast<core::view::CallSplitViewOverlay&>(call);
        auto& evt = cr.GetInputEvent();
        ASSERT(evt.tag == core::view::InputEvent::Tag::MouseButton && "Callback invocation mismatched input event"); 
        return this->OnMouseButton(evt.mouseButtonData.button, evt.mouseButtonData.action,
        evt.mouseButtonData.mods);
    } catch (...) {
        ASSERT("OnGUIMouseButtonCallback call cast failed\n");
    }
    return false;
}


/**
 * OverlayInterfaceLayer::OnGUIMouseMoveCallback
 */
template <class M> bool megamol::gui::OverlayInterfaceLayer<M>::OnGUIMouseMoveCallback(megamol::core::Call& call) {
    try {
        core::view::CallSplitViewOverlay& cr = dynamic_cast<core::view::CallSplitViewOverlay&>(call);
        auto& evt = cr.GetInputEvent();
        ASSERT(evt.tag == core::view::InputEvent::Tag::MouseMove && "Callback invocation mismatched input event");
        return this->OnMouseMove(evt.mouseMoveData.x, evt.mouseMoveData.y);
    } catch (...) {
        ASSERT("OnGUIMouseMoveCallback call cast failed\n");
    }
    return false;
}


/**
 * OverlayInterfaceLayer::OnGUIMouseScrollCallback
 */
template <class M> bool megamol::gui::OverlayInterfaceLayer<M>::OnGUIMouseScrollCallback(megamol::core::Call& call) {
    try {
        core::view::CallSplitViewOverlay& cr = dynamic_cast<core::view::CallSplitViewOverlay&>(call);
        auto& evt = cr.GetInputEvent();
        ASSERT(evt.tag == core::view::InputEvent::Tag::MouseScroll && "Callback invocation mismatched input event"); 
        return this->OnMouseScroll(evt.mouseScrollData.dx, evt.mouseScrollData.dy);
    } catch (...) {
        ASSERT("OnGUIMouseScrollCallback call cast failed\n");
    }
    return false;
}
