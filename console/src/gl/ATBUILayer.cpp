/*
 * gl/ATBUILayer.cpp
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "vislib/sys/KeyCode.h"
#ifdef HAS_ANTTWEAKBAR
#include "gl/Window.h"
#include "ATBUILayer.h"
#include "GLFW/glfw3.h"
#include <string>
#include <sstream>
#include "gl/ATWinBar.h"
#include "gl/ATParamBar.h"
#include "vislib/sys/Log.h"
#include "utility/HotFixes.h"

using namespace megamol;
using namespace megamol::console;

int gl::ATBUILayer::nextWinID = 0;

gl::ATBUILayer::ATBUILayer(const char* wndName, void* hView, void *hCore)
        : atb(), atbWinID(-1), atbKeyMod(0), hView(hView), winBar(), paramBar(), enabled(true), lastParamUpdateTime(), wndName(wndName) {
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

        winBar = std::make_shared<ATWinBar>(*this, this->wndName.c_str());
        paramBar = std::make_shared<ATParamBar>(hCore);
    }
    lastParamUpdateTime = std::chrono::system_clock::now() - std::chrono::seconds(1);
    this->isCoreHotFixed = utility::HotFixes::Instance().IsHotFixed("atbCore");
    this->wasdHotfixed = utility::HotFixes::Instance().IsHotFixed("wasdnorepeat");
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
    vislib::sys::Log::DefaultLog.WriteInfo("ATB GUI Layer in Window '%s' %sabled. F12 to toggle.", this->wndName.c_str(), enabled ? "en" : "dis");
}

#if 1 /* REGION INPUT CONTROL */

void gl::ATBUILayer::OnResize(int w, int h) {
    ::TwSetCurrentWindow(atbWinID);
    ::TwWindowSize(w, h);
}

void gl::ATBUILayer::OnDraw() {

    if (this->wasdHotfixed) {
        CoreHandle hParam;
        if (this->fwd) {
            this->OnChar('w');
        }
        if (this->back) {
            this->OnChar('s');
        }
        if (this->left) {
            this->OnChar('a');
        }
        if (this->right) {
            this->OnChar('d');
        }
    }

    ::TwSetCurrentWindow(atbWinID);

    if ((std::chrono::system_clock::now() - lastParamUpdateTime) > std::chrono::milliseconds(100)) {
        static_cast<ATParamBar*>(paramBar.get())->Update();
        lastParamUpdateTime = std::chrono::system_clock::now();
    }

    ::TwDraw();

    if (isCoreHotFixed) {
        // dirty fix: clean up leftover ATB state
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        // this is deprecated and NSight hates it
        //glBindBuffer(GL_COLOR_ARRAY, 0);
    }
}

bool gl::ATBUILayer::OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) {
    ::TwSetCurrentWindow(atbWinID);

    atbKeyMod = 0;
    
    if (mods.test(core::view::Modifier::SHIFT)) {
        atbKeyMod |= TW_KMOD_SHIFT;
    }
    if (mods.test(core::view::Modifier::CTRL)) {
        atbKeyMod |= TW_KMOD_CTRL;
    }
    if (mods.test(core::view::Modifier::ALT)) {
        atbKeyMod |= TW_KMOD_ALT;
    }

    // Process key pressed
    if (action == core::view::KeyAction::PRESS || action == core::view::KeyAction::REPEAT) {

        bool testkp = ((atbKeyMod & TW_KMOD_CTRL) || (atbKeyMod & TW_KMOD_ALT)) ? true : false;

        if ((atbKeyMod == TW_KMOD_CTRL) && (key > static_cast<core::view::Key>(0)) && (key < core::view::Key::KEY_ESCAPE)) {
            // CTRL cases
            printf(" key %d + %d\n", key, atbKeyMod);
            if (::TwKeyPressed(static_cast<int>(key), atbKeyMod)) {
                return true;
            }
        }
        else if (key >= core::view::Key::KEY_ESCAPE) {

            int k = 0;
            switch (key) {
            case core::view::Key::KEY_SPACE: k = TW_KEY_SPACE; break;
            case core::view::Key::KEY_APOSTROPHE: k = '\''; break;
            case core::view::Key::KEY_COMMA: k = ','; break;
            case core::view::Key::KEY_MINUS: k = '-'; break;
            case core::view::Key::KEY_PERIOD: k = '.'; break;
            case core::view::Key::KEY_SLASH: k = '/'; break;
            case core::view::Key::KEY_0: k = '0'; break;
            case core::view::Key::KEY_1: k = '1'; break;
            case core::view::Key::KEY_2: k = '2'; break;
            case core::view::Key::KEY_3: k = '3'; break;
            case core::view::Key::KEY_4: k = '4'; break;
            case core::view::Key::KEY_5: k = '5'; break;
            case core::view::Key::KEY_6: k = '6'; break;
            case core::view::Key::KEY_7: k = '7'; break;
            case core::view::Key::KEY_8: k = '8'; break;
            case core::view::Key::KEY_9: k = '9'; break;
            case core::view::Key::KEY_SEMICOLON: k = ';'; break;
            case core::view::Key::KEY_EQUAL: k = '='; break;
            case core::view::Key::KEY_A: k = 'a'; break;
            case core::view::Key::KEY_B: k = 'b'; break;
            case core::view::Key::KEY_C: k = 'c'; break;
            case core::view::Key::KEY_D: k = 'd'; break;
            case core::view::Key::KEY_E: k = 'e'; break;
            case core::view::Key::KEY_F: k = 'f'; break;
            case core::view::Key::KEY_G: k = 'g'; break;
            case core::view::Key::KEY_H: k = 'h'; break;
            case core::view::Key::KEY_I: k = 'i'; break;
            case core::view::Key::KEY_J: k = 'j'; break;
            case core::view::Key::KEY_K: k = 'k'; break;
            case core::view::Key::KEY_L: k = 'l'; break;
            case core::view::Key::KEY_M: k = 'm'; break;
            case core::view::Key::KEY_N: k = 'n'; break;
            case core::view::Key::KEY_O: k = 'o'; break;
            case core::view::Key::KEY_P: k = 'p'; break;
            case core::view::Key::KEY_Q: k = 'q'; break;
            case core::view::Key::KEY_R: k = 'r'; break;
            case core::view::Key::KEY_S: k = 's'; break;
            case core::view::Key::KEY_T: k = 't'; break;
            case core::view::Key::KEY_U: k = 'u'; break;
            case core::view::Key::KEY_V: k = 'v'; break;
            case core::view::Key::KEY_W: k = 'w'; break;
            case core::view::Key::KEY_X: k = 'x'; break;
            case core::view::Key::KEY_Y: k = 'y'; break;
            case core::view::Key::KEY_Z: k = 'z'; break;
            case core::view::Key::KEY_LEFT_BRACKET: k = '('; break;
            case core::view::Key::KEY_BACKSLASH: k = '\\'; break;
            case core::view::Key::KEY_RIGHT_BRACKET: k = ')'; break;
            case core::view::Key::KEY_GRAVE_ACCENT: k = '´'; break;
                //case core::view::Key::KEY_WORLD_1: k = TW_KEY_WORLD_1; break;
                //case core::view::Key::KEY_WORLD_2: k = TW_KEY_WORLD_2; break;
            case core::view::Key::KEY_ESCAPE: k = TW_KEY_ESCAPE; break;
            case core::view::Key::KEY_ENTER: k = TW_KEY_RETURN; break;
            case core::view::Key::KEY_TAB: k = TW_KEY_TAB; break;
            case core::view::Key::KEY_BACKSPACE: k = TW_KEY_BACKSPACE; break;
            case core::view::Key::KEY_INSERT: k = TW_KEY_INSERT; break;
            case core::view::Key::KEY_DELETE: k = TW_KEY_DELETE; break;
            case core::view::Key::KEY_RIGHT: k = TW_KEY_RIGHT; break;
            case core::view::Key::KEY_LEFT: k = TW_KEY_LEFT; break;
            case core::view::Key::KEY_DOWN: k = TW_KEY_DOWN; break;
            case core::view::Key::KEY_UP: k = TW_KEY_UP; break;
            case core::view::Key::KEY_PAGE_UP: k = TW_KEY_PAGE_UP; break;
            case core::view::Key::KEY_PAGE_DOWN: k = TW_KEY_PAGE_DOWN; break;
            case core::view::Key::KEY_HOME: k = TW_KEY_HOME; break;
            case core::view::Key::KEY_END: k = TW_KEY_END; break;
                //case core::view::Key::KEY_CAPS_LOCK: k = TW_KEY_CAPS_LOCK; break;
                //case core::view::Key::KEY_SCROLL_LOCK: k = TW_KEY_SCROLL_LOCK; break;
                //case core::view::Key::KEY_NUM_LOCK: k = TW_KEY_NUM_LOCK; break;
                //case core::view::Key::KEY_PRINT_SCREEN: k = TW_KEY_PRINT_SCREEN; break;
            case core::view::Key::KEY_PAUSE: k = TW_KEY_PAUSE; break;
            case core::view::Key::KEY_F1: k = TW_KEY_F1; break;
            case core::view::Key::KEY_F2: k = TW_KEY_F2; break;
            case core::view::Key::KEY_F3: k = TW_KEY_F3; break;
            case core::view::Key::KEY_F4: k = TW_KEY_F4; break;
            case core::view::Key::KEY_F5: k = TW_KEY_F5; break;
            case core::view::Key::KEY_F6: k = TW_KEY_F6; break;
            case core::view::Key::KEY_F7: k = TW_KEY_F7; break;
            case core::view::Key::KEY_F8: k = TW_KEY_F8; break;
            case core::view::Key::KEY_F9: k = TW_KEY_F9; break;
            case core::view::Key::KEY_F10: k = TW_KEY_F10; break;
            case core::view::Key::KEY_F11: k = TW_KEY_F11; break;
            case core::view::Key::KEY_F12: k = TW_KEY_F12; break;
            case core::view::Key::KEY_F13: k = TW_KEY_F13; break;
            case core::view::Key::KEY_F14: k = TW_KEY_F14; break;
            case core::view::Key::KEY_F15: k = TW_KEY_F15; break;
            case core::view::Key::KEY_F16: k = TW_KEY_F10 + 6; break;
            case core::view::Key::KEY_F17: k = TW_KEY_F10 + 7; break;
            case core::view::Key::KEY_F18: k = TW_KEY_F10 + 8; break;
            case core::view::Key::KEY_F19: k = TW_KEY_F10 + 9; break;
            case core::view::Key::KEY_F20: k = TW_KEY_F10 + 10; break;
            case core::view::Key::KEY_F21: k = TW_KEY_F10 + 11; break;
            case core::view::Key::KEY_F22: k = TW_KEY_F10 + 12; break;
            case core::view::Key::KEY_F23: k = TW_KEY_F10 + 13; break;
            case core::view::Key::KEY_F24: k = TW_KEY_F10 + 14; break;
            case core::view::Key::KEY_F25: k = TW_KEY_F10 + 15; break;
            case core::view::Key::KEY_KP_0: if (testkp) k = '0'; break;
            case core::view::Key::KEY_KP_1: if (testkp) k = '1'; break;
            case core::view::Key::KEY_KP_2: if (testkp) k = '2'; break;
            case core::view::Key::KEY_KP_3: if (testkp) k = '3'; break;
            case core::view::Key::KEY_KP_4: if (testkp) k = '4'; break;
            case core::view::Key::KEY_KP_5: if (testkp) k = '5'; break;
            case core::view::Key::KEY_KP_6: if (testkp) k = '6'; break;
            case core::view::Key::KEY_KP_7: if (testkp) k = '7'; break;
            case core::view::Key::KEY_KP_8: if (testkp) k = '8'; break;
            case core::view::Key::KEY_KP_9: if (testkp) k = '9'; break;
            case core::view::Key::KEY_KP_DECIMAL: if (testkp) k = '.'; break;
            case core::view::Key::KEY_KP_DIVIDE: if (testkp) k = '/'; break;
            case core::view::Key::KEY_KP_MULTIPLY: if (testkp) k = '*'; break;
            case core::view::Key::KEY_KP_SUBTRACT: if (testkp) k = '-'; break;
            case core::view::Key::KEY_KP_ADD: if (testkp) k = '+'; break;
            case core::view::Key::KEY_KP_ENTER: k = TW_KEY_RETURN; break;
            case core::view::Key::KEY_KP_EQUAL: if (testkp) k = '='; break;
                //case core::view::Key::KEY_LEFT_SHIFT: k = TW_KEY_LEFT_SHIFT; break;
                //case core::view::Key::KEY_LEFT_CONTROL: k = TW_KEY_LEFT_CONTROL; break;
                //case core::view::Key::KEY_LEFT_ALT: k = TW_KEY_LEFT_ALT; break;
                //case core::view::Key::KEY_LEFT_SUPER: k = TW_KEY_LEFT_SUPER; break;
                //case core::view::Key::KEY_RIGHT_SHIFT: k = TW_KEY_RIGHT_SHIFT; break;
                //case core::view::Key::KEY_RIGHT_CONTROL: k = TW_KEY_RIGHT_CONTROL; break;
                //case core::view::Key::KEY_RIGHT_ALT: k = TW_KEY_RIGHT_ALT; break;
                //case core::view::Key::KEY_RIGHT_SUPER: k = TW_KEY_RIGHT_SUPER; break;
                //case core::view::Key::KEY_MENU: k = TW_KEY_MENU; break;
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

    if (wasdHotfixed) {
        const bool down = (action != core::view::KeyAction::RELEASE);
        switch (key) {
	        case core::view::Key::KEY_W:
	            this->fwd = down;
	            break;
	        case core::view::Key::KEY_A:
	            this->left = down;
	            break;
	        case core::view::Key::KEY_S:
	            this->back = down;
	            break;
	        case core::view::Key::KEY_D:
	            this->right = down;
	            break;
        }
    }

    return false;
}

bool gl::ATBUILayer::OnChar(unsigned int charcode) {
    ::TwSetCurrentWindow(atbWinID);

    if ((charcode & 0xff00) == 0) {
        if (::TwKeyPressed(charcode, atbKeyMod)) {
            return true;
        }
    }
    return false;
}

bool gl::ATBUILayer::OnMouseMove(double x, double y) {
    ::TwSetCurrentWindow(atbWinID);

    ::TwMouseMotion((int)x, (int)y);
    return false;
}

bool gl::ATBUILayer::OnMouseButton(core::view::MouseButton button, core::view::MouseButtonAction action, core::view::Modifiers mods) {
    ::TwSetCurrentWindow(atbWinID);

    TwMouseAction act = (action == core::view::MouseButtonAction::PRESS) ? TW_MOUSE_PRESSED : TW_MOUSE_RELEASED;
    TwMouseButtonID btn;
    switch (button) {
    case core::view::MouseButton::BUTTON_LEFT: btn = TW_MOUSE_LEFT; break;
    case core::view::MouseButton::BUTTON_RIGHT: btn = TW_MOUSE_RIGHT; break;
    case core::view::MouseButton::BUTTON_MIDDLE: btn = TW_MOUSE_MIDDLE; break;
    default: return false;
    }

    if (::TwMouseButton(act, btn)) {
        // ATB consumed mouse button >> thus will capture
        return true;
    }
    return false;
}

bool gl::ATBUILayer::OnMouseScroll(double x, double y) {
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
