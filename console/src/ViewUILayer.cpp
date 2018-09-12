/*
 * ViewMouseUILayer.cpp
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten. All rights reserved.
 */
#include "stdafx.h"
#include "ViewUILayer.h"
#include "mmcore/api/MegaMolCore.h"

using namespace megamol;
using namespace megamol::console;

ViewUILayer::ViewUILayer(gl::Window& wnd, void* viewHandle) : AbstractUILayer(wnd), hView(viewHandle) {}

ViewUILayer::~ViewUILayer() {
    hView = nullptr; // handle memory is owned by Window and will be deleted there
}

void ViewUILayer::OnResize(int w, int h) {
    ::mmcResizeView(hView, static_cast<unsigned int>(w), static_cast<unsigned int>(h));
}

bool ViewUILayer::OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) {
    // TODO: NYI
    return false;
}

bool ViewUILayer::OnChar(unsigned int codePoint) {
    // TODO: NYI
    return false;
}

bool ViewUILayer::OnMouseMove(double x, double y) {
    ::mmcSet2DMousePosition(hView, static_cast<float>(x), static_cast<float>(y));
    return false;
}

bool ViewUILayer::OnMouseButton(
    core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {
    // modifiers
    ::mmcSetInputModifier(hView, static_cast<mmcInputModifiers>(core::view::Modifier::SHIFT), mods.test(core::view::Modifier::SHIFT));
    ::mmcSetInputModifier(hView, static_cast<mmcInputModifiers>(core::view::Modifier::CTRL), mods.test(core::view::Modifier::CTRL));
    ::mmcSetInputModifier(hView, static_cast<mmcInputModifiers>(core::view::Modifier::ALT), mods.test(core::view::Modifier::ALT));

    // button states and infos stuff
    unsigned int btn = static_cast<unsigned int>(button);
    ::mmcSet2DMouseButton(hView, btn, action == core::view::MouseButtonAction::PRESS);


    /*
    Idee
    mmcSendInput(hView, inputs, inputs.length() * sizeof(input))
    */
    return action == core::view::MouseButtonAction::PRESS;
}

bool ViewUILayer::OnMouseScroll(double x, double y) {
    // TODO: NYI
    return false;
}
