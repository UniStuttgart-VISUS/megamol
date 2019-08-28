/*
 * WindowManager.cpp
 *
 * Copyright (C) 2008, 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "gl/Window.h"

#include "stdafx.h"
#include "WindowManager.h"
#include <cassert>
#include "mmcore/api/MegaMolCore.h"
#include "utility/ConfigHelper.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "vislib/sys/Log.h"
#include "JobManager.h"
#include "utility/HotFixFileName.h"
#include "utility/HotFixes.h"
#include "mmcore/utility/KHR.h"

#include "ButtonParamUILayer.h"
#include "ViewUILayer.h"
#include "gl/WindowEscapeHotKeysUILayer.h"

const char* const megamol::console::WindowManager::TitlePrefix = "MegaMol - ";
const int megamol::console::WindowManager::TitlePrefixLength = 10;

/*
 * megamol::console::WindowManager::Instance
 */
megamol::console::WindowManager& megamol::console::WindowManager::Instance(void) {
    static megamol::console::WindowManager inst;
    return inst;
}

/*
 * megamol::console::WindowManager::~WindowManager
 */
megamol::console::WindowManager::~WindowManager(void) { assert(windows.empty()); }

/*
 * megamol::console::WindowManager::WindowManager
 */
megamol::console::WindowManager::WindowManager(void) : windows() {}

bool megamol::console::WindowManager::IsAlive(void) const { return !windows.empty(); }

void megamol::console::WindowManager::Update(void) {
    if (windows.empty()) return;

    bool cleaning = false;

    // TODO: What is with window FreezUpdate ?

    for (std::shared_ptr<gl::Window> win : windows) {
        if (!win->IsAlive()) {
            cleaning = true;
            continue;
        }

        win->Update();
        if (windows.empty()) return;
    }

    if (!cleaning) return;

    std::vector<std::shared_ptr<gl::Window>>::iterator e = windows.end();
    for (std::vector<std::shared_ptr<gl::Window>>::iterator i = windows.begin(); i != e;) {
        if ((*i)->IsAlive())
            ++i;
        else {
            i = windows.erase(i);
            e = windows.end(); // because we potentially changed everything.
        }
    }
}

void megamol::console::WindowManager::Shutdown(void) {
    for (std::shared_ptr<gl::Window> win : windows) {
        if (win->IsAlive()) win->RequestClose();
    }
    megamol::console::JobManager::Instance().Shutdown();
    while (!windows.empty()) Update();
}

bool megamol::console::WindowManager::InstantiatePendingView(void *hCore) {

    // get instance name
    const char* pendInstName = ::mmcGetPendingViewInstanceName(hCore);
    if ((pendInstName == nullptr) || (pendInstName[0] == 0)) {
        vislib::sys::Log::DefaultLog.WriteError("Pending instance name empty");
        return false;
    }

    // get window placement info
    utility::WindowPlacement wp;
    const void* wpData = nullptr;
    ::mmcValueType wpDataType = MMC_TYPE_VOIDP;

	const auto getWindowPlacementFromCore = [&](auto const& queryString) {
        wpDataType = MMC_TYPE_VOIDP;
        wpData = ::mmcGetConfigurationValue(hCore, MMC_CFGID_VARIABLE, queryString, &wpDataType);
        if (wpData != nullptr) {
            if (wpDataType == MMC_TYPE_CSTR) {
                if (!wp.Parse(vislib::TString(static_cast<const char*>(wpData)))) wpData = nullptr;
            } else if (wpDataType == MMC_TYPE_WSTR) {
                if (!wp.Parse(vislib::TString(static_cast<const wchar_t*>(wpData)))) wpData = nullptr;
            } else {
                wpData = nullptr;
            }
        }
    };
    getWindowPlacementFromCore((vislib::TString(pendInstName) + _T("-window")));
    if (wpData == nullptr) {
        getWindowPlacementFromCore(_T("*-window"));
    }
    if (wpData == nullptr) {
        ::memset(&wp, 0, sizeof(utility::WindowPlacement));
    }
    { // should the window stay on top?
        auto value = ::mmcGetConfigurationValue(hCore, MMC_CFGID_VARIABLE, _T("topmost"), &wpDataType);
        wp.topMost = false;
        if (value != nullptr) {
            try {
                switch (wpDataType) {
                case MMC_TYPE_BOOL:
                    wp.topMost = *static_cast<const bool*>(value);
                    break;

                case MMC_TYPE_CSTR:
                    wp.topMost = vislib::CharTraitsA::ParseBool(static_cast<const char*>(value));
                    break;

                case MMC_TYPE_WSTR:
                    wp.topMost = vislib::CharTraitsW::ParseBool(static_cast<const wchar_t*>(value));
                    break;
                }
            } catch (...) {
            }
        }
    }

    // enable KHR debug
    bool activateKHR = false;
    ::mmcValueType khrDataType = MMC_TYPE_VOIDP;
    const void* khrData = ::mmcGetConfigurationValue(hCore, MMC_CFGID_VARIABLE, _T("useKHRdebug"), &khrDataType);
    if (khrData != nullptr) {
        try {
            if (khrDataType == MMC_TYPE_CSTR) {
                activateKHR = vislib::CharTraitsA::ParseBool(static_cast<const char*>(khrData));
            } else if (khrDataType == MMC_TYPE_WSTR) {
                activateKHR = vislib::CharTraitsW::ParseBool(static_cast<const wchar_t*>(khrData));
            }
        } catch (...) {
        }
    }

    bool vsync = false;
    ::mmcValueType vsyncDataType = MMC_TYPE_VOIDP;
    const void* vsyncData = ::mmcGetConfigurationValue(hCore, MMC_CFGID_VARIABLE, _T("vsync"), &vsyncDataType);
    if (vsyncData != nullptr) {
        switch (vsyncDataType) {
        case MMC_TYPE_INT32:
            vsync = *static_cast<const int32_t*>(vsyncData) != 0;
            break;
        case MMC_TYPE_UINT32:
            vsync = *static_cast<const uint32_t*>(vsyncData) != 0;
            break;
        case MMC_TYPE_INT64:
            vsync = *static_cast<const int64_t*>(vsyncData) != 0;
            break;
        case MMC_TYPE_UINT64:
            vsync = *static_cast<const uint64_t*>(vsyncData) != 0;
            break;
        case MMC_TYPE_BYTE:
            vsync = *static_cast<const unsigned char*>(vsyncData) != 0;
            break;
        case MMC_TYPE_BOOL:
            vsync = *static_cast<const bool*>(vsyncData);
            break;
        case MMC_TYPE_FLOAT:
            vsync = *static_cast<const float*>(vsyncData) != 0.0f;
            break; // this does not really make any sense, but who cares
        case MMC_TYPE_CSTR:
            vsync = vislib::CharTraitsA::ParseBool(static_cast<const char*>(vsyncData));
            break;
        case MMC_TYPE_WSTR:
            vsync = vislib::CharTraitsW::ParseBool(static_cast<const wchar_t*>(vsyncData));
            break;
        default:
            break;
        }
    }

    // get an existing window to share context resources
    GLFWwindow* share = nullptr;
    if (!windows.empty()) share = windows[0]->WindowHandle();
    // TODO: share GL context from outside? no GL at all? responsibility of WindowManager?

    // prepare window object
    std::shared_ptr<gl::Window> w =
        std::make_shared<gl::Window>((vislib::StringA(TitlePrefix) + pendInstName).PeekBuffer(), wp, share);
    if (!w || !w->IsAlive()) {
        vislib::sys::Log::DefaultLog.WriteError("Unable to create window");
        return false;
    }

    if (activateKHR) megamol::core::utility::KHR::startDebug();
    if (vsync) w->EnableVSync();

    ::mmcInstantiatePendingView(hCore, w->Handle()); // pending view is instantiated and linked to windows w->Handle(), which is a core handle hView
    ::mmcRegisterViewCloseRequestFunction(w->Handle(), &gl::Window::RequestCloseCallback, w.get());

    auto viewLayer = std::make_shared<ViewUILayer>(w->Handle());
    w->AddUILayer(viewLayer);

    auto btnLayer = std::make_shared<ButtonParamUILayer>(hCore, w->Handle());
    w->AddUILayer(btnLayer);

    auto wehkLayer = std::make_shared<gl::WindowEscapeHotKeysUILayer>(*w.get());
    w->AddUILayer(wehkLayer); // add as last. This allows MegaMol module buttons to use 'q' (and 'ESC') as hotkeys, overriding this hotkey

    w->ForceIssueResizeEvent();
    windows.push_back(w);

    return true;
}
