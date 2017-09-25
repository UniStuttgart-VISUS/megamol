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

void ViewMouseUILayer::onResize(int w, int h) {
    ::mmcResizeView(hView, static_cast<unsigned int>(w), static_cast<unsigned int>(h));
}

bool ViewMouseUILayer::onMouseMove(double x, double y) {
    ::mmcSet2DMousePosition(hView, static_cast<float>(x), static_cast<float>(y));
    return false;
}

bool ViewMouseUILayer::onMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) {
    // modifiers
    ::mmcSetInputModifier(hView, MMC_INMOD_ALT, (mods & KEY_MOD_ALT) == KEY_MOD_ALT);
    ::mmcSetInputModifier(hView, MMC_INMOD_CTRL, (mods & KEY_MOD_CTRL) == KEY_MOD_CTRL);
    ::mmcSetInputModifier(hView, MMC_INMOD_SHIFT, (mods & KEY_MOD_SHIFT) == KEY_MOD_SHIFT);

    // button states and infos stuff
    unsigned int btn = static_cast<unsigned int>(button);
    ::mmcSet2DMouseButton(hView, btn, action == MouseButtonAction::PRESS);

    return action == MouseButtonAction::PRESS;
}

bool ViewMouseUILayer::onMouseWheel(double x, double y) {
    // currently not implemented by MegaMol core views ...
    return false;
}
