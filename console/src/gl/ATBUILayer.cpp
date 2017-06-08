/*
 * gl/ATBUILayer.cpp
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#ifdef HAS_ANTTWEAKBAR
#include "gl/Window.h"
#include "ATBUILayer.h"
#include "GLFW/glfw3.h"
#include <string>
#include <sstream>
#include "gl/ATWinBar.h"
#include "gl/ATParamBar.h"
#include "vislib/sys/Log.h"

using namespace megamol;
using namespace megamol::console;

int gl::ATBUILayer::nextWinID = 0;

gl::ATBUILayer::ATBUILayer(Window& wnd, const char* wndName, void* hView, void *hCore)
        : AbstractUILayer(wnd), atb(), atbWinID(-1), atbKeyMod(0), hView(hView), winBar(), paramBar(), enabled(true), lastParamUpdateTime() {
    atb = atbInst::Instance();
    if (atb->OK()) {
        atbWinID = nextWinID++;
        ::TwSetCurrentWindow(atbWinID);

        if (TwGetBarCount() > 0) {
            TwBar *helpBar = TwGetBarByIndex(0);
            std::stringstream def;
            def << TwGetBarName(helpBar) << " "
                << "position='80 80' ";
            ::TwDefine(def.str().c_str());
        }

        winBar = std::make_shared<ATWinBar>(wnd, *this, wndName);
        paramBar = std::make_shared<ATParamBar>(hCore);
    }
    lastParamUpdateTime = std::chrono::system_clock::now() - std::chrono::seconds(1);
}

gl::ATBUILayer::~ATBUILayer() {
    // reset ATBars first
    winBar.reset();
    paramBar.reset();
}

bool gl::ATBUILayer::Enabled() {
    return enabled;
}

void gl::ATBUILayer::ToggleEnable() {
    enabled = !enabled;
    vislib::sys::Log::DefaultLog.WriteInfo("ATB GUI Layer in Window '%s' %sabled. F12 to toggle.", wnd.Name(), enabled ? "en" : "dis");
}

#if 1 /* REGION INPUT CONTROL */

void gl::ATBUILayer::onResize(int w, int h) {
    ::TwSetCurrentWindow(atbWinID);
    ::TwWindowSize(w, h);
}

void gl::ATBUILayer::onDraw() {
    ::TwSetCurrentWindow(atbWinID);

    if ((std::chrono::system_clock::now() - lastParamUpdateTime) > std::chrono::milliseconds(100)) {
        static_cast<ATParamBar*>(paramBar.get())->Update();
        lastParamUpdateTime = std::chrono::system_clock::now();
    }

    ::TwDraw();
}

bool gl::ATBUILayer::onKey(Key key, int scancode, KeyAction action, Modifiers mods) {
    ::TwSetCurrentWindow(atbWinID);

    atbKeyMod = 0;
    if (mods & Modifiers::KEY_MOD_SHIFT) atbKeyMod |= TW_KMOD_SHIFT;
    if (mods & Modifiers::KEY_MOD_CTRL) atbKeyMod |= TW_KMOD_CTRL;
    if (mods & Modifiers::KEY_MOD_ALT) atbKeyMod |= TW_KMOD_ALT;

    // Process key pressed
    if (action == KeyAction::PRESS || action == KeyAction::REPEAT) {
        bool testkp = ((atbKeyMod & TW_KMOD_CTRL) || (atbKeyMod & TW_KMOD_ALT)) ? true : false;

        if ((atbKeyMod == TW_KMOD_CTRL) && (key > static_cast<Key>(0)) && (key < KEY_ESCAPE)) {
            // CTRL cases
            printf(" key %d + %d\n", key, atbKeyMod);
            if (::TwKeyPressed(key, atbKeyMod)) {
                return true;
            }

        } else if (key >= KEY_ESCAPE) {
            int k = 0;
            switch (key) {
            case KEY_SPACE: k = TW_KEY_SPACE; break;
            case KEY_APOSTROPHE: k = '\''; break;
            case KEY_COMMA: k = ','; break;
            case KEY_MINUS: k = '-'; break;
            case KEY_PERIOD: k = '.'; break;
            case KEY_SLASH: k = '/'; break;
            case KEY_0: k = '0'; break;
            case KEY_1: k = '1'; break;
            case KEY_2: k = '2'; break;
            case KEY_3: k = '3'; break;
            case KEY_4: k = '4'; break;
            case KEY_5: k = '5'; break;
            case KEY_6: k = '6'; break;
            case KEY_7: k = '7'; break;
            case KEY_8: k = '8'; break;
            case KEY_9: k = '9'; break;
            case KEY_SEMICOLON: k = ';'; break;
            case KEY_EQUAL: k = '='; break;
            case KEY_A: k = 'a'; break;
            case KEY_B: k = 'b'; break;
            case KEY_C: k = 'c'; break;
            case KEY_D: k = 'd'; break;
            case KEY_E: k = 'e'; break;
            case KEY_F: k = 'f'; break;
            case KEY_G: k = 'g'; break;
            case KEY_H: k = 'h'; break;
            case KEY_I: k = 'i'; break;
            case KEY_J: k = 'j'; break;
            case KEY_K: k = 'k'; break;
            case KEY_L: k = 'l'; break;
            case KEY_M: k = 'm'; break;
            case KEY_N: k = 'n'; break;
            case KEY_O: k = 'o'; break;
            case KEY_P: k = 'p'; break;
            case KEY_Q: k = 'q'; break;
            case KEY_R: k = 'r'; break;
            case KEY_S: k = 's'; break;
            case KEY_T: k = 't'; break;
            case KEY_U: k = 'u'; break;
            case KEY_V: k = 'v'; break;
            case KEY_W: k = 'w'; break;
            case KEY_X: k = 'x'; break;
            case KEY_Y: k = 'y'; break;
            case KEY_Z: k = 'z'; break;
            case KEY_LEFT_BRACKET: k = '('; break;
            case KEY_BACKSLASH: k = '\\'; break;
            case KEY_RIGHT_BRACKET: k = ')'; break;
            case KEY_GRAVE_ACCENT: k = '´'; break;
            //case KEY_WORLD_1: k = TW_KEY_WORLD_1; break;
            //case KEY_WORLD_2: k = TW_KEY_WORLD_2; break;
            case KEY_ESCAPE: k = TW_KEY_ESCAPE; break;
            case KEY_ENTER: k = TW_KEY_RETURN; break;
            case KEY_TAB: k = TW_KEY_TAB; break;
            case KEY_BACKSPACE: k = TW_KEY_BACKSPACE; break;
            case KEY_INSERT: k = TW_KEY_INSERT; break;
            case KEY_DELETE: k = TW_KEY_DELETE; break;
            case KEY_RIGHT: k = TW_KEY_RIGHT; break;
            case KEY_LEFT: k = TW_KEY_LEFT; break;
            case KEY_DOWN: k = TW_KEY_DOWN; break;
            case KEY_UP: k = TW_KEY_UP; break;
            case KEY_PAGE_UP: k = TW_KEY_PAGE_UP; break;
            case KEY_PAGE_DOWN: k = TW_KEY_PAGE_DOWN; break;
            case KEY_HOME: k = TW_KEY_HOME; break;
            case KEY_END: k = TW_KEY_END; break;
            //case KEY_CAPS_LOCK: k = TW_KEY_CAPS_LOCK; break;
            //case KEY_SCROLL_LOCK: k = TW_KEY_SCROLL_LOCK; break;
            //case KEY_NUM_LOCK: k = TW_KEY_NUM_LOCK; break;
            //case KEY_PRINT_SCREEN: k = TW_KEY_PRINT_SCREEN; break;
            case KEY_PAUSE: k = TW_KEY_PAUSE; break;
            case KEY_F1: k = TW_KEY_F1; break;
            case KEY_F2: k = TW_KEY_F2; break;
            case KEY_F3: k = TW_KEY_F3; break;
            case KEY_F4: k = TW_KEY_F4; break;
            case KEY_F5: k = TW_KEY_F5; break;
            case KEY_F6: k = TW_KEY_F6; break;
            case KEY_F7: k = TW_KEY_F7; break;
            case KEY_F8: k = TW_KEY_F8; break;
            case KEY_F9: k = TW_KEY_F9; break;
            case KEY_F10: k = TW_KEY_F10; break;
            case KEY_F11: k = TW_KEY_F11; break;
            case KEY_F12: k = TW_KEY_F12; break;
            case KEY_F13: k = TW_KEY_F13; break;
            case KEY_F14: k = TW_KEY_F14; break;
            case KEY_F15: k = TW_KEY_F15; break;
            case KEY_F16: k = TW_KEY_F10 + 6; break;
            case KEY_F17: k = TW_KEY_F10 + 7; break;
            case KEY_F18: k = TW_KEY_F10 + 8; break;
            case KEY_F19: k = TW_KEY_F10 + 9; break;
            case KEY_F20: k = TW_KEY_F10 + 10; break;
            case KEY_F21: k = TW_KEY_F10 + 11; break;
            case KEY_F22: k = TW_KEY_F10 + 12; break;
            case KEY_F23: k = TW_KEY_F10 + 13; break;
            case KEY_F24: k = TW_KEY_F10 + 14; break;
            case KEY_F25: k = TW_KEY_F10 + 15; break;
            case KEY_KP_0: if (testkp) k = '0'; break;
            case KEY_KP_1: if (testkp) k = '1'; break;
            case KEY_KP_2: if (testkp) k = '2'; break;
            case KEY_KP_3: if (testkp) k = '3'; break;
            case KEY_KP_4: if (testkp) k = '4'; break;
            case KEY_KP_5: if (testkp) k = '5'; break;
            case KEY_KP_6: if (testkp) k = '6'; break;
            case KEY_KP_7: if (testkp) k = '7'; break;
            case KEY_KP_8: if (testkp) k = '8'; break;
            case KEY_KP_9: if (testkp) k = '9'; break;
            case KEY_KP_DECIMAL: if (testkp) k = '.'; break;
            case KEY_KP_DIVIDE: if (testkp) k = '/'; break;
            case KEY_KP_MULTIPLY: if (testkp) k = '*'; break;
            case KEY_KP_SUBTRACT: if (testkp) k = '-'; break;
            case KEY_KP_ADD: if (testkp) k = '+'; break;
            case KEY_KP_ENTER: k = TW_KEY_RETURN; break;
            case KEY_KP_EQUAL: if (testkp) k = '='; break;
            //case KEY_LEFT_SHIFT: k = TW_KEY_LEFT_SHIFT; break;
            //case KEY_LEFT_CONTROL: k = TW_KEY_LEFT_CONTROL; break;
            //case KEY_LEFT_ALT: k = TW_KEY_LEFT_ALT; break;
            //case KEY_LEFT_SUPER: k = TW_KEY_LEFT_SUPER; break;
            //case KEY_RIGHT_SHIFT: k = TW_KEY_RIGHT_SHIFT; break;
            //case KEY_RIGHT_CONTROL: k = TW_KEY_RIGHT_CONTROL; break;
            //case KEY_RIGHT_ALT: k = TW_KEY_RIGHT_ALT; break;
            //case KEY_RIGHT_SUPER: k = TW_KEY_RIGHT_SUPER; break;
            //case KEY_MENU: k = TW_KEY_MENU; break;
            default: break;
            }

            if (k > 0) {
                if (::TwKeyPressed(k, atbKeyMod)) {
                    return true;
                }
            }
        }
    }

    // Problem fix:
    //   Originally ATB only consumes onChar for string input. However, the
    // corresponding onKey events are not consumed and are thus interpreted by
    // other UILayers. E.g. entering a string into the ATB containing spaces
    // would also trigger the animation button of MegaMol™ Core's View3D.
    //   This fix consumes all onKey events when the ATB is in text input 
    // mode. The basic idea is to ask ATB if it would consume a '0' character
    // key pressed.
    if (::TwKeyTest('0', TW_KMOD_NONE)) {
        // This captures all keyboard input for ATB only.
        return true;
    }

    return false;
}

bool gl::ATBUILayer::onChar(unsigned int charcode) {
    ::TwSetCurrentWindow(atbWinID);

    if ((charcode & 0xff00) == 0) {
        if (::TwKeyPressed(charcode, atbKeyMod)) {
            return true;
        }
    }
    return false;
}

bool gl::ATBUILayer::onMouseMove(double x, double y) {
    ::TwSetCurrentWindow(atbWinID);

    ::TwMouseMotion((int)x, (int)y);
    return false;
}

bool gl::ATBUILayer::onMouseButton(MouseButton button, MouseButtonAction action, Modifiers mods) {
    ::TwSetCurrentWindow(atbWinID);

    TwMouseAction act = (action == MouseButtonAction::PRESS) ? TW_MOUSE_PRESSED : TW_MOUSE_RELEASED;
    TwMouseButtonID btn;
    switch (button) {
    case MouseButton::Left: btn = TW_MOUSE_LEFT; break;
    case MouseButton::Right: btn = TW_MOUSE_RIGHT; break;
    case MouseButton::Middle: btn = TW_MOUSE_MIDDLE; break;
    default: return false;
    }

    if (::TwMouseButton(act, btn)) {
        // ATB consumed mouse button >> thus will capture
        return true;
    }
    return false;
}

bool gl::ATBUILayer::onMouseWheel(double x, double y) {
    ::TwSetCurrentWindow(atbWinID);
    static int scrollpos = 0;
    scrollpos += static_cast<int>(y);
    //vislib::sys::Log::DefaultLog.WriteInfo("scrolling by %f", y);
    if (::TwMouseWheel(scrollpos)) {
        return true;
    }
    return false;
}

#endif /* REGION INPUT CONTROL */

#endif /* HAS_ANTTWEAKBAR */
