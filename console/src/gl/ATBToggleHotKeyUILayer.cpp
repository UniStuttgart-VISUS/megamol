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

gl::ATBToggleHotKeyUILayer::ATBToggleHotKeyUILayer(ATBUILayer& atbLayer)
    : atbLayer(atbLayer) {
    // intentionally empty
}

gl::ATBToggleHotKeyUILayer::~ATBToggleHotKeyUILayer() {
    // intentionally empty
}

bool gl::ATBToggleHotKeyUILayer::Enabled() {
    return !atbLayer.Enabled();
}

bool gl::ATBToggleHotKeyUILayer::OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) {
    if ((key == core::view::Key::KEY_F12) && (action == core::view::KeyAction::PRESS) && mods.none()) {
        atbLayer.ToggleEnable();
        return true;
    }
    return false;
}

#endif /* HAS_ANTTWEAKBAR */
