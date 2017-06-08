/*
 * gl/ATBToggleHotKeyUILayer.cpp
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "gl/ATBToggleHotKeyUILayer.h"
#ifdef HAS_ANTTWEAKBAR

using namespace megamol;
using namespace megamol::console;

gl::ATBToggleHotKeyUILayer::ATBToggleHotKeyUILayer(Window& wnd, ATBUILayer& atbLayer)
    : AbstractUILayer(wnd), atbLayer(atbLayer) {
    // intentionally empty
}

gl::ATBToggleHotKeyUILayer::~ATBToggleHotKeyUILayer() {
    // intentionally empty
}

bool gl::ATBToggleHotKeyUILayer::Enabled() {
    return !atbLayer.Enabled();
}

bool gl::ATBToggleHotKeyUILayer::onKey(Key key, int scancode, KeyAction action, Modifiers mods) {
    if ((key == KEY_F12) && (action == KeyAction::PRESS) && (static_cast<int>(mods) == 0)) {
        atbLayer.ToggleEnable();
        return true;
    }
    return false;
}

#endif /* HAS_ANTTWEAKBAR */
