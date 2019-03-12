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

ButtonParamUILayer::ButtonParamUILayer(void * coreHandle, void * viewHandle)
        : hCore(coreHandle), hView(viewHandle), last_update(), hotkeys(), maskingLayer(nullptr){
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

bool ButtonParamUILayer::OnKey(core::view::Key key, core::view::KeyAction action, core::view::Modifiers mods) {

    if ((std::chrono::system_clock::now() - last_update) > std::chrono::seconds(1)) {
        // update list periodically
        updateHotkeyList();
        last_update = std::chrono::system_clock::now();
    }

    if (action == core::view::KeyAction::RELEASE) return false;

    core::view::KeyCode keyCode(key, mods);

    auto hotkey = hotkeys.find(keyCode.ToString());
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
        std::map<std::string, vislib::TString> hotKeys;
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
        WORD key, mods;
        if (buffer.size() == descLen) {
            key = *reinterpret_cast<WORD*>(buffer.data() + 6);
            mods = *reinterpret_cast<WORD*>(buffer.data() + 6 + sizeof(WORD));
        }
        else return; // something is strange with the hotkey

        // store param handle as part of the key codes
        core::view::KeyCode keyCode(static_cast<core::view::Key>(static_cast<int>(key)), core::view::Modifiers(static_cast<int>(mods)));
        data->hotKeys[keyCode.ToString()] = name;
    }

}

void ButtonParamUILayer::updateHotkeyList() {
    struct enumData data;
    data.hCore = hCore;
    data.hView = hView;
    ::mmcEnumParameters(hCore, (mmcEnumStringFunction)&enumParameters, &data);

    hotkeys = std::move(data.hotKeys);
}
