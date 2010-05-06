/*
 * CoreHandle.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Window.h"
#include "HotKeyButtonParam.h"
#include "MegaMolCore.h"
#include "vislib/assert.h"
#include "vislib/Log.h"
#include "vislib/RawStorage.h"
#include "vislib/types.h"
#include "WindowManager.h"

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
        ::mmcRenderView(win->hView, static_cast<bool*>(params));
    } else {
        win->MarkToClose();
    }
#ifdef WITH_TWEAKBAR
    win->gui.Draw();
#endif /* WITH_TWEAKBAR */
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
#ifdef WITH_TWEAKBAR
    win->gui.SetWindowSize(params[0], params[1]);
#endif /* WITH_TWEAKBAR */
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
#ifdef WITH_TWEAKBAR
    if (!win->gui.KeyPressed(params->keycode, params->modShift, params->modAlt, params->modCtrl)) {
#endif /* WITH_TWEAKBAR */
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
#ifdef WITH_TWEAKBAR
    }
#endif /* WITH_TWEAKBAR */

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
#ifdef WITH_TWEAKBAR
    if (!win->gui.MouseButton(params->button, params->buttonDown) || !params->buttonDown) {
#endif /* WITH_TWEAKBAR */
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
#ifdef WITH_TWEAKBAR
    }
#endif /* WITH_TWEAKBAR */
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
#ifdef WITH_TWEAKBAR
    win->gui.MouseMove(params->mouseX, params->mouseY);
#endif /* WITH_TWEAKBAR */
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
#ifdef WITH_TWEAKBAR
        gui(),
#endif /* WITH_TWEAKBAR */
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


#ifdef WITH_TWEAKBAR
/*
 * megamol::console::Window::InitGUI
 */
void megamol::console::Window::InitGUI(CoreHandle& hCore) {
    ParamEnumContext peContext;
    peContext.wnd = this;
    peContext.hCore = &hCore;
    this->gui.BeginInitialisation();
    ::mmcEnumParametersA(hCore, &Window::initGUI, &peContext);
    this->gui.EndInitialisation();
}


/*
 * megamol::console::Window::initGUI
 */
void MEGAMOLCORE_CALLBACK megamol::console::Window::initGUI(const char *paramName, void *contextPtr) {
    ParamEnumContext *context = reinterpret_cast<ParamEnumContext *>(contextPtr);
    vislib::SmartPtr<megamol::console::CoreHandle> hParam = new megamol::console::CoreHandle();
    vislib::RawStorage desc;

    if (context == NULL) { return; }
    if (!::mmcGetParameterA(*context->hCore, paramName, *hParam)) { return; }
    //if (!::mmcIsParameterRelevant(context->wnd->HView(), *hParam)) { return; }

    unsigned int len = 0;
    ::mmcGetParameterTypeDescription(*hParam, NULL, &len);
    desc.AssertSize(len);
    ::mmcGetParameterTypeDescription(*hParam, desc.As<unsigned char>(), &len);

    context->wnd->gui.AddParameter(hParam, paramName, desc.As<unsigned char>(), len);
}
#endif /* WITH_TWEAKBAR */


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
