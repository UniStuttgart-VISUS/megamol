/*
 * WindowEscapeHotKeysUILayer.cpp
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "gl/WindowEscapeHotKeysUILayer.h"
#include "WindowManager.h"

using namespace megamol;
using namespace megamol::console;

gl::WindowEscapeHotKeysUILayer::WindowEscapeHotKeysUILayer(gl::Window &wnd) : AbstractUILayer(wnd) {}

gl::WindowEscapeHotKeysUILayer::~WindowEscapeHotKeysUILayer() {}

bool gl::WindowEscapeHotKeysUILayer::onKey(Key key, int scancode, KeyAction action, Modifiers mods) {
    if (((key == Key::KEY_ESCAPE) || (key == Key::KEY_Q)) && (action == KeyAction::PRESS)
        && ((mods & ~KEY_MOD_SHIFT) == 0)) {
        if ((mods & KEY_MOD_SHIFT) == KEY_MOD_SHIFT) {
            wnd.RequestClose();
        } else {
            WindowManager::Instance().Shutdown();
        }
        return true;
    }
    return false;
}
