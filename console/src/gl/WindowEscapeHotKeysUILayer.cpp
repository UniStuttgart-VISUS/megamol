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

gl::WindowEscapeHotKeysUILayer::WindowEscapeHotKeysUILayer(gl::Window& _wnd) : wndPtr(&_wnd) {}

gl::WindowEscapeHotKeysUILayer::WindowEscapeHotKeysUILayer(std::function<void()> func) : actionFunc{func} {}

gl::WindowEscapeHotKeysUILayer::~WindowEscapeHotKeysUILayer() {}

bool gl::WindowEscapeHotKeysUILayer::OnKey(
    core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) {
    bool isQuit = ((mods.equals(core::view::Modifier::ALT)) && (key == core::view::Key::KEY_F4));
//    ||   (key == core::view::Key::KEY_ESCAPE);
//    || (key == core::view::Key::KEY_Q);
    bool isPressed = (action == core::view::KeyAction::PRESS);
    bool isNotShift = !(mods ^ core::view::Modifier::SHIFT).none();

    if (isQuit && isPressed && isNotShift) {
        if (mods.test(core::view::Modifier::CTRL)) {
            if (wndPtr) {
                wndPtr->RequestClose(); // TODO: better solution?
            } else {
                actionFunc();
            }
        } else {
            WindowManager::Instance().Shutdown(); // TODO: WHY?!
        }
        return true;
    }
    return false;
}
