/*
 * ViewMouseUILayer.cpp
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten. All rights reserved.
 */
#include "stdafx.h"
#include "ViewMouseUILayer.h"
#include "mmcore/api/MegaMolCore.h"

using namespace megamol;
using namespace megamol::console;

ViewMouseUILayer::ViewMouseUILayer(gl::Window& wnd, void * viewHandle) : AbstractUILayer(wnd), hView(viewHandle) {
}

ViewMouseUILayer::~ViewMouseUILayer() {
    hView = nullptr; // handle memory is owned by Window and will be deleted there
}

void ViewMouseUILayer::OnResize(int w, int h) {
    ::mmcResizeView(hView, static_cast<unsigned int>(w), static_cast<unsigned int>(h));
}

bool ViewMouseUILayer::OnMouseMove(double x, double y) {
    ::mmcSet2DMousePosition(hView, static_cast<float>(x), static_cast<float>(y));
    return false;
}

bool ViewMouseUILayer::OnMouseButton(core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {
    // modifiers
    ::mmcSetInputModifier(hView, MMC_INMOD_ALT, (mods & core::view::Modifiers::ALT) == core::view::Modifiers::ALT);
    ::mmcSetInputModifier(hView, MMC_INMOD_CTRL, (mods & core::view::Modifiers::CTRL) == core::view::Modifiers::CTRL);
    ::mmcSetInputModifier(hView, MMC_INMOD_SHIFT, (mods & core::view::Modifiers::SHIFT) == core::view::Modifiers::SHIFT);

    // button states and infos stuff
    unsigned int btn = static_cast<unsigned int>(button);
    ::mmcSet2DMouseButton(hView, btn, action == core::view::MouseButtonAction::PRESS);

    return action == core::view::MouseButtonAction::PRESS;
}

bool ViewMouseUILayer::OnMouseScroll(double x, double y) {
    // currently not implemented by MegaMol core views ...
    return false;
}
