/*
 * CoreHandle.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Window.h"
#include "HotKeyButtonParam.h"
#include "mmcore/api/MegaMolCore.h"
#include "vislib/assert.h"
#include "vislib/sys/Log.h"
#include "vislib/RawStorage.h"
#include "vislib/types.h"
#include "WindowManager.h"
#include <chrono>
#include <vector>
#include <algorithm>

/****************************************************************************/


/* extern declaration of closeAllWindows. Ugly and I hate it, but it works. */
extern void closeAllWindows(megamol::console::Window *exception = false);


namespace megamol {
namespace console {

    /**
     * Utility structure for parameter enumeration
     */
    typedef struct _paramenumcontext_t {

        /** The calling window object */
        Window *wnd;

        /** Pointer to the core handle */
        CoreHandle *hCore;

    } ParamEnumContext;

} /* end namespace console */
} /* end namespace megamol */

/****************************************************************************/


/*
 * megamol::console::Window::InternalHotKey::InternalHotKey
 */
megamol::console::Window::InternalHotKey::InternalHotKey(
        megamol::console::Window *owner, 
        void (megamol::console::Window::*callback)(void)) : HotKeyAction(),
        owner(owner), callback(callback) {
    ASSERT(owner != NULL);
    ASSERT(callback != NULL);
}


/*
 * megamol::console::Window::InternalHotKey::~InternalHotKey
 */
megamol::console::Window::InternalHotKey::~InternalHotKey(void) {
    this->owner = NULL; // DO NOT DELETE
    this->callback = NULL; // DO NOT DELETE
}


/*
 * megamol::console::Window::InternalHotKey::Trigger
 */
void megamol::console::Window::InternalHotKey::Trigger(void) {
    // pointer-to-member operator rulz
    (this->owner->*this->callback)();
}

/****************************************************************************/


/*
 * megamol::console::Window::hasClosed
 */
bool megamol::console::Window::hasClosed = false;


/*
 * megamol::console::Window::CloseCallback
 */
void megamol::console::Window::CloseCallback(void *wnd, void *params) {
    Window *win = static_cast<Window *>(wnd);
    ASSERT(win != NULL);
    ASSERT(::mmcIsHandleValid(win->hView));
    ASSERT(::mmvIsHandleValid(win->hWnd));
    win->MarkToClose();
}


/*
 * megamol::console::Window::RenderCallback
 */
void megamol::console::Window::RenderCallback(void *wnd, void *params) {
    Window *win = static_cast<Window *>(wnd);
    ASSERT(win != NULL);
    ASSERT(::mmcIsHandleValid(win->hView));
    ASSERT(::mmvIsHandleValid(win->hWnd));
    if (::mmcIsViewRunning(win->hView)) {
        ::mmcRenderView(win->hView, static_cast<mmcRenderViewContext *>(params));
    } else {
        win->MarkToClose();
    }
#ifdef HAS_ANTTWEAKBAR
    win->gui.Draw();
#endif /* HAS_ANTTWEAKBAR */
}


/*
 * megamol::console::Window::ResizeCallback
 */
void megamol::console::Window::ResizeCallback(void *wnd,
        unsigned int *params) {
    Window *win = static_cast<Window *>(wnd);
    ASSERT((win != NULL) && (params != NULL));
    ASSERT(::mmcIsHandleValid(win->hView));
    ASSERT(::mmvIsHandleValid(win->hWnd));
#ifdef HAS_ANTTWEAKBAR
    win->gui.SetWindowSize(params[0], params[1]);
#endif /* HAS_ANTTWEAKBAR */
    ::mmcResizeView(win->hView, params[0], params[1]);
}


/*
 * megamol::console::Window::KeyCallback
 */
void megamol::console::Window::KeyCallback(void *wnd, mmvKeyParams *params) {
    Window *win = static_cast<Window *>(wnd);
    ASSERT((win != NULL) && (params != NULL));
    ASSERT(::mmcIsHandleValid(win->hView));
    ASSERT(::mmvIsHandleValid(win->hWnd));
    win->setModifierStates(params->modAlt, params->modCtrl, params->modShift);
    ::mmcSet2DMousePosition(win->hView,
        static_cast<float>(params->mouseX),
        static_cast<float>(params->mouseY));
#ifdef HAS_ANTTWEAKBAR
    if (!win->gui.KeyPressed(params->keycode, params->modShift, params->modAlt, params->modCtrl)) {
#endif /* HAS_ANTTWEAKBAR */
    vislib::sys::KeyCode key(params->keycode);
    if (win->hotkeys.Contains(key)) {
        vislib::SmartPtr<HotKeyAction> action = win->hotkeys[key];
        if (!action.IsNull()) {
            action->Trigger();
            return;
        }
    }
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO + 1000,
        "Unused key %i\n", int(key));
#ifdef HAS_ANTTWEAKBAR
    }
#endif /* HAS_ANTTWEAKBAR */

}


/*
 * megamol::console::Window::MouseButtonCallback
 */
void megamol::console::Window::MouseButtonCallback(void *wnd,
        mmvMouseButtonParams *params) {
    Window *win = static_cast<Window *>(wnd);
    ASSERT((win != NULL) && (params != NULL));
    ASSERT(::mmcIsHandleValid(win->hView));
    ASSERT(::mmvIsHandleValid(win->hWnd));
    win->setModifierStates(params->modAlt, params->modCtrl, params->modShift);
#ifdef HAS_ANTTWEAKBAR
    if (!win->gui.MouseButton(params->button, params->buttonDown) || !params->buttonDown) {
#endif /* HAS_ANTTWEAKBAR */
    switch (params->button) {
        case 0: case 1: case 2:
            if (win->mouseBtns[params->button] != params->buttonDown) {
                ::mmcSet2DMouseButton(win->hView, params->button,
                    params->buttonDown);
                win->mouseBtns[params->button] = params->buttonDown;
            }
            break;
        default:
            break;
    }
#ifdef HAS_ANTTWEAKBAR
    }
#endif /* HAS_ANTTWEAKBAR */
    ::mmcSet2DMousePosition(win->hView,
        static_cast<float>(params->mouseX),
        static_cast<float>(params->mouseY));
}


/*
 * megamol::console::Window::MouseMoveCallback
 */
void megamol::console::Window::MouseMoveCallback(void *wnd,
        mmvMouseMoveParams *params) {
    Window *win = static_cast<Window *>(wnd);
    ASSERT((win != NULL) && (params != NULL));
    ASSERT(::mmcIsHandleValid(win->hView));
    ASSERT(::mmvIsHandleValid(win->hWnd));
    win->setModifierStates(params->modAlt, params->modCtrl, params->modShift);
#ifdef HAS_ANTTWEAKBAR
    win->gui.MouseMove(params->mouseX, params->mouseY);
#endif /* HAS_ANTTWEAKBAR */
    ::mmcSet2DMousePosition(win->hView,
        static_cast<float>(params->mouseX),
        static_cast<float>(params->mouseY));
}


/*
 * megamol::console::Window::UpdateFreezeCallback
 */
void megamol::console::Window::UpdateFreezeCallback(void *wnd, int *params) {
    Window *win = static_cast<Window *>(wnd);
    ASSERT(win != NULL);
    ASSERT(::mmcIsHandleValid(win->hView));
    ASSERT(::mmvIsHandleValid(win->hWnd));

    ::mmcFreezeOrUpdateView(win->hView, params[0] != 0);

}


/*
 * megamol::console::Window::CloseRequestCallback
 */
void MMAPI_CALLBACK megamol::console::Window::CloseRequestCallback(void* data) {
    Window *win = static_cast<Window *>(data);
    ASSERT(win != NULL);
    win->MarkToClose();
}


/*
 * megamol::console::Window::Window
 */
megamol::console::Window::Window(void) : hView(), hWnd(),
#ifdef HAS_ANTTWEAKBAR
        gui(),
#endif /* HAS_ANTTWEAKBAR */
        isClosed(false), modAlt(false), modCtrl(false), modShift(false),
        hotkeys() {
    this->mouseBtns[0] = false;
    this->mouseBtns[1] = false;
    this->mouseBtns[2] = false;

    this->hotkeys[vislib::sys::KeyCode(vislib::sys::KeyCode::KEY_ESC)]
        = new InternalHotKey(this, &Window::doExitAction);
    this->hotkeys[vislib::sys::KeyCode(vislib::sys::KeyCode::KEY_ESC | vislib::sys::KeyCode::KEY_MOD_SHIFT)]
        = new InternalHotKey(this, &Window::doExitAction);
    this->hotkeys[vislib::sys::KeyCode('q')]
        = new InternalHotKey(this, &Window::doExitAction);
    this->hotkeys[vislib::sys::KeyCode('Q' | vislib::sys::KeyCode::KEY_MOD_SHIFT)]
        = new InternalHotKey(this, &Window::doExitAction);
}


/*
 * megamol::console::Window::~Window
 */
megamol::console::Window::~Window(void) {
    // Intentionally empty atm
}


/*
 * megamol::console::Window::RegisterHotKeys
 */
void megamol::console::Window::RegisterHotKeys(CoreHandle& hCore) {
    ParamEnumContext peContext;
    peContext.wnd = this;
    peContext.hCore = &hCore;
    ::mmcEnumParametersA(hCore, &Window::registerHotKeys, &peContext);
}


/*
 * megamol::console::Window::RegisterHotKeyAction
 */
void megamol::console::Window::RegisterHotKeyAction(
        const vislib::sys::KeyCode& key, HotKeyAction *action, const vislib::StringA& target) {

    if (!this->hotkeys.Contains(key)) {
        // register hotkey 'key' of this window with this parameter
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
            "Mapping Key %s to %s\n", key.ToStringA().PeekBuffer(), target.PeekBuffer());
        this->hotkeys[key] = action;
    } else {
        delete action;
    }
}


#ifdef HAS_ANTTWEAKBAR

namespace {
    void MEGAMOLCORE_CALLBACK collectParams(const char *paramName, void *contextPtr) {
        std::vector<vislib::StringA> *paramNames = static_cast<std::vector<vislib::StringA>* >(contextPtr);
        assert(paramNames != nullptr);
        paramNames->push_back(paramName);
    }
}

/*
 * megamol::console::Window::InitGUI
 */
void megamol::console::Window::InitGUI(CoreHandle& hCore) {
    this->gui.BeginInitialisation();

    std::vector<vislib::StringA> params;
    ::mmcEnumParametersA(hCore, &collectParams, &params);

    for (const vislib::StringA& paramName : params) {
        vislib::SmartPtr<megamol::console::CoreHandle> hParam = new megamol::console::CoreHandle();
        vislib::RawStorage desc;
        if (!::mmcGetParameterA(hCore, paramName, *hParam)) continue;

        unsigned int len = 0;
        ::mmcGetParameterTypeDescription(*hParam, NULL, &len);
        desc.AssertSize(len);
        ::mmcGetParameterTypeDescription(*hParam, desc.As<unsigned char>(), &len);

        this->gui.AddParameter(hParam, paramName, desc.As<unsigned char>(), len);
    }

    this->gui.EndInitialisation();
}


void megamol::console::Window::UpdateGUI(CoreHandle& hCore) {
    std::vector<vislib::StringA> params;
    std::vector<vislib::StringA> deadParams = gui.ParametersNames();

    ::mmcEnumParametersA(hCore, &collectParams, &params);

    for (const vislib::StringA& paramName : params) {

        // search if param already exist
        auto dpi = std::find(deadParams.begin(), deadParams.end(), vislib::StringA(paramName));
        if (dpi != deadParams.end()) {
            deadParams.erase(dpi); // this gui parameter is in use and will not be deleted
            continue;
        }

        // parameter does not yet exist
        vislib::SmartPtr<megamol::console::CoreHandle> hParam = new megamol::console::CoreHandle();
        vislib::RawStorage desc;
        if (!::mmcGetParameterA(hCore, paramName, *hParam)) continue;

        unsigned int len = 0;
        ::mmcGetParameterTypeDescription(*hParam, NULL, &len);
        desc.AssertSize(len);
        ::mmcGetParameterTypeDescription(*hParam, desc.As<unsigned char>(), &len);

        this->gui.AddParameter(hParam, paramName, desc.As<unsigned char>(), len);

    }

    // now we delete all the orphaned gui parameters
    for (const vislib::StringA& paramName : deadParams) {
        this->gui.RemoveParameter(paramName);
    }

}
#endif /* HAS_ANTTWEAKBAR */

/*
 * megamol::console::Window::Update
 */
void megamol::console::Window::Update(CoreHandle& hCore) {
#ifdef HAS_ANTTWEAKBAR
    // update GUI once a second
    static std::chrono::system_clock::time_point last = std::chrono::system_clock::now();
    if (gui.IsActive()) {
        std::chrono::system_clock::time_point n = std::chrono::system_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(n - last).count() > 0) {
            last = n;
            UpdateGUI(hCore);
        }
    }
#endif /* HAS_ANTTWEAKBAR */

}


/*
 * megamol::console::Window::registerHotKeys
 */
void MEGAMOLCORE_CALLBACK megamol::console::Window::registerHotKeys(const char *paramName, void *contextPtr) {
    ParamEnumContext *context = reinterpret_cast<ParamEnumContext *>(contextPtr);
    vislib::SmartPtr<megamol::console::CoreHandle> hParam = new megamol::console::CoreHandle();
    vislib::RawStorage desc;

    if (context == NULL) { return; }
    if (!::mmcGetParameterA(*context->hCore, paramName, *hParam)) { return; }

    unsigned int len = 0;
    ::mmcGetParameterTypeDescription(*hParam, NULL, &len);
    desc.AssertSize(len);
    ::mmcGetParameterTypeDescription(*hParam, desc.As<unsigned char>(), &len);
    if (!vislib::StringA(desc.As<char>(), 6).Equals("MMBUTN")) { return; }// I r only interested in buttons

    if (!::mmcIsParameterRelevant(context->wnd->HView(), *hParam)) { return; }
    WORD key = 0;
    if (desc.GetSize() == 7) {
        key = *desc.AsAt<char>(6);
    } else if (desc.GetSize() == 8) {
        key = *desc.AsAt<WORD>(6);
    }

    if (key != 0) {
        context->wnd->RegisterHotKeyAction(vislib::sys::KeyCode(key),
            new HotKeyButtonParam(hParam), paramName);
    }
}


/*
 * megamol::console::Window::setModifierStates
 */
void megamol::console::Window::setModifierStates(bool alt, bool ctrl,
        bool shift) {
    if (alt != this->modAlt) {
        ::mmcSetInputModifier(this->hView, MMC_INMOD_ALT, alt);
        this->modAlt = alt;
    }
    if (ctrl != this->modCtrl) {
        ::mmcSetInputModifier(this->hView, MMC_INMOD_CTRL, ctrl);
        this->modCtrl = ctrl;
    }
    if (shift != this->modShift) {
        ::mmcSetInputModifier(this->hView, MMC_INMOD_SHIFT, shift);
        this->modShift = shift;
    }
}


/*
 * megamol::console::Window::doExitAction
 */
void megamol::console::Window::doExitAction(void) {
    if (this->modShift) {
        CloseCallback(this, NULL);
    } else {
        WindowManager::Instance()->MarkAllForClosure();
    }
}
