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

bool gl::WindowEscapeHotKeysUILayer::OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) {
    bool isQuit = (key == core::view::Key::KEY_ESCAPE)  || (key == core::view::Key::KEY_Q);
    bool isPressed = (action == core::view::KeyAction::PRESS);
    bool isNotShift = (mods & ~core::view::Modifiers::SHIFT) == core::view::Modifiers::NONE;
	bool isShift = (mods & core::view::Modifiers::CTRL) == core::view::Modifiers::CTRL;
    if (isQuit && isPressed  && isNotShift) {
        if (isShift) {
            wnd.RequestClose();
        } else {
            WindowManager::Instance().Shutdown();
        }
        return true;
    }
    return false;
}
