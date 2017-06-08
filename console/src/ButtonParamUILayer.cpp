/*
 * ButtonParamUILayer.cpp
 *
 * Copyright (C) 2016 MegaMol Team
 * Alle Rechte vorbehalten. All rights reserved.
 */
#include "stdafx.h"
#include "ButtonParamUILayer.h"
#include "mmcore/api/MegaMolCore.h"
#include "CoreHandle.h"
#include <cstdint>
#include <vector>

using namespace megamol;
using namespace megamol::console;

ButtonParamUILayer::ButtonParamUILayer(gl::Window& wnd, void * coreHandle, void * viewHandle)
        : AbstractUILayer(wnd), hCore(coreHandle), hView(viewHandle), last_update(), hotkeys(), maskingLayer(nullptr){
    last_update = std::chrono::system_clock::now() - std::chrono::hours(1);
}

ButtonParamUILayer::~ButtonParamUILayer() {
    hCore = nullptr; // handle memory is owned by application and will be deleted there
    hView = nullptr; // handle memory is owned by Window and will be deleted there
    maskingLayer = nullptr; // is just an optional reference
}

bool ButtonParamUILayer::Enabled() {
    return (maskingLayer == nullptr) || !maskingLayer->Enabled();
}

bool ButtonParamUILayer::onKey(Key key, int scancode, KeyAction action, Modifiers mods) {
    if ((std::chrono::system_clock::now() - last_update) > std::chrono::seconds(1)) {
        // update list periodically
        updateHotkeyList();
        last_update = std::chrono::system_clock::now();
    }

    if (action == KeyAction::RELEASE) return false;

    uint16_t cleanKey;
    switch (key) {
    case KEY_ENTER: cleanKey = vislib::sys::KeyCode::KEY_ENTER; break;
    case KEY_ESCAPE: cleanKey = vislib::sys::KeyCode::KEY_ESC; break;
    case KEY_TAB: cleanKey = vislib::sys::KeyCode::KEY_TAB; break;
    case KEY_LEFT: cleanKey = vislib::sys::KeyCode::KEY_LEFT; break;
    case KEY_UP: cleanKey = vislib::sys::KeyCode::KEY_UP; break;
    case KEY_RIGHT: cleanKey = vislib::sys::KeyCode::KEY_RIGHT; break;
    case KEY_DOWN: cleanKey = vislib::sys::KeyCode::KEY_DOWN; break;
    case KEY_PAGE_UP: cleanKey = vislib::sys::KeyCode::KEY_PAGE_UP; break;
    case KEY_PAGE_DOWN: cleanKey = vislib::sys::KeyCode::KEY_PAGE_DOWN; break;
    case KEY_HOME: cleanKey = vislib::sys::KeyCode::KEY_HOME; break;
    case KEY_END: cleanKey = vislib::sys::KeyCode::KEY_END; break;
    case KEY_INSERT: cleanKey = vislib::sys::KeyCode::KEY_INSERT; break;
    case KEY_DELETE: cleanKey = vislib::sys::KeyCode::KEY_DELETE; break;
    case KEY_BACKSPACE: cleanKey = vislib::sys::KeyCode::KEY_BACKSPACE; break;
    case KEY_F1: cleanKey = vislib::sys::KeyCode::KEY_F1; break;
    case KEY_F2: cleanKey = vislib::sys::KeyCode::KEY_F2; break;
    case KEY_F3: cleanKey = vislib::sys::KeyCode::KEY_F3; break;
    case KEY_F4: cleanKey = vislib::sys::KeyCode::KEY_F4; break;
    case KEY_F5: cleanKey = vislib::sys::KeyCode::KEY_F5; break;
    case KEY_F6: cleanKey = vislib::sys::KeyCode::KEY_F6; break;
    case KEY_F7: cleanKey = vislib::sys::KeyCode::KEY_F7; break;
    case KEY_F8: cleanKey = vislib::sys::KeyCode::KEY_F8; break;
    case KEY_F9: cleanKey = vislib::sys::KeyCode::KEY_F9; break;
    case KEY_F10: cleanKey = vislib::sys::KeyCode::KEY_F10; break;
    case KEY_F11: cleanKey = vislib::sys::KeyCode::KEY_F11; break;
    case KEY_F12: cleanKey = vislib::sys::KeyCode::KEY_F12; break;
    default: cleanKey = static_cast<uint16_t>(key);
    }
    cleanKey = cleanKey & ~vislib::sys::KeyCode::KEY_MOD;
    if ((mods & KEY_MOD_ALT) == KEY_MOD_ALT) cleanKey |= vislib::sys::KeyCode::KEY_MOD_ALT;
    if ((mods & KEY_MOD_CTRL) == KEY_MOD_CTRL) cleanKey |= vislib::sys::KeyCode::KEY_MOD_CTRL;
    if ((mods & KEY_MOD_SHIFT) == KEY_MOD_SHIFT) cleanKey |= vislib::sys::KeyCode::KEY_MOD_SHIFT;

    vislib::sys::KeyCode keycode(cleanKey);

    auto hotkey = hotkeys.find(keycode);
    if (hotkey == hotkeys.end()) return false;

    CoreHandle hParam;
    if (!::mmcGetParameter(hCore, hotkey->second.PeekBuffer(), hParam)) return false;

    ::mmcSetParameterValue(hParam, _T("click"));

    return true;
}

namespace {

    struct enumData {
        void *hCore;
        void *hView;
        std::map<vislib::sys::KeyCode, vislib::TString> hotKeys;
    };

    void MEGAMOLCORE_CALLBACK enumParameters(const TCHAR *name, struct enumData *data) {
        // check only relevant parameters
        CoreHandle hParam;
        if (!::mmcGetParameter(data->hCore, name, hParam)) return;
        if (!::mmcIsParameterRelevant(data->hView, hParam)) return;

        // check only button parameter
        unsigned int descLen;
        ::mmcGetParameterTypeDescription(hParam, nullptr, &descLen);
        if (descLen < 4) return;
        std::vector<unsigned char> buffer(descLen);
        ::mmcGetParameterTypeDescription(hParam, buffer.data(), &descLen);
        buffer.resize(descLen);
        if (buffer.size() <= 6) return;
        if (memcmp(buffer.data(), "MMBUTN", 6) != 0) return;

        // extract key
        uint16_t key;
        if (buffer.size() == 7) key = *reinterpret_cast<char*>(buffer.data() + 6);
        else if (buffer.size() == 8) key = *reinterpret_cast<uint16_t*>(buffer.data() + 6);
        else return; // something is strange with the hotkey

        // store param handle as part of the key codes
        vislib::sys::KeyCode keyCode(key);
        data->hotKeys[keyCode] = name;
    }

}

void ButtonParamUILayer::updateHotkeyList() {
    struct enumData data;
    data.hCore = hCore;
    data.hView = hView;
    ::mmcEnumParameters(hCore, (mmcEnumStringFunction)&enumParameters, &data);

    hotkeys = std::move(data.hotKeys);
}
