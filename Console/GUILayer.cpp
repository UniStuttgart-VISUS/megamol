/*
 * GUILayer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "GUILayer.h"
#ifdef WITH_TWEAKBAR
#define TW_STATIC
#define TW_NO_LIB_PRAGMA
#include "AntTweakBar.h"
#include "vislib/assert.h"
#include "vislib/memutils.h"
#include "vislib/Trace.h"
#include "vislib/KeyCode.h"


using namespace megamol::console;

#if defined(DEBUG) || defined(_DEBUG)
#define TW_VERIFY(call, line) if ((call) == 0) { VLTRACE(VISLIB_TRCELVL_ERROR, "TwGetLastError[%d]: %s\n", line, ::TwGetLastError()); }
#else /* defined(DEBUG) || defined(_DEBUG) */
#define TW_VERIFY(call, line) call;
#endif /* defined(DEBUG) || defined(_DEBUG) */

/****************************************************************************/


/*
 * GUILayer::GUIClient::GUIClient
 */
GUILayer::GUIClient::GUIClient(void) : width(256), height(256), myBar(NULL) {
    if (cntr == 0) {
        this->Activate();
    }
    cntr++;
}


/*
 * GUILayer::GUIClient::~GUIClient
 */
GUILayer::GUIClient::~GUIClient(void) {
    ASSERT(cntr > 0);
    cntr--;
    if (myBar != NULL) {
        TW_VERIFY(::TwDeleteBar(static_cast<TwBar*>(this->myBar)), __LINE__);
        this->myBar = NULL;
    }
    if (cntr == 0) {
        SAFE_DELETE(layer);
    }
    if (activeClient == this) {
        activeClient = NULL;
    }
}


/*
 * GUILayer::GUIClient::Layer
 */
GUILayer& GUILayer::GUIClient::Layer(void) {
    if (layer == NULL) {
        layer = new GUILayer();
        if (activeClient != NULL) {
            GUIClient *c = activeClient;
            activeClient = NULL;
            c->Activate();
        }
    }
    layer->active = (this == activeClient);
    return *layer;
}


/*
 * GUILayer::GUIClient::Activate
 */
void GUILayer::GUIClient::Activate(void) {
    if (activeClient == this) return;
    if (activeClient != NULL) {
        activeClient->Deactivate();
    }
    activeClient = this;
    if (layer != NULL) {
        vislib::StringA name;
        TW_VERIFY(::TwWindowSize(this->width, this->height), __LINE__);
        if (this->myBar == NULL) {
            name.Format("%d", reinterpret_cast<int>(this));
            this->myBar = static_cast<void*>(::TwNewBar(name));
            if (this->myBar != NULL) {
                name.Format("%d label='Parameters' position='10 10' text=dark alpha=192 color='128 192 255'", reinterpret_cast<int>(this));
                TW_VERIFY(::TwDefine(name), __LINE__);
            }
        }
        if (this->myBar != NULL) {
            name.Format("%d visible=true", reinterpret_cast<int>(this));
            TW_VERIFY(::TwDefine(name), __LINE__);
        }
    }
}


/*
 * GUILayer::GUIClient::Deactivate
 */
void GUILayer::GUIClient::Deactivate(void) {
    if (activeClient == this) {
        activeClient = NULL;
        if (this->myBar != NULL) {
            vislib::StringA name;
            name.Format("%d visible=false", reinterpret_cast<int>(this));
            TW_VERIFY(::TwDefine(name), __LINE__);
        }
    }
}


/*
 * GUILayer::GUIClient::SetWindowSize
 */
void GUILayer::GUIClient::SetWindowSize(unsigned int w, unsigned int h) {
    this->width = static_cast<int>(w);
    if (this->width <= 0) this->width = 1;
    this->height = static_cast<int>(h);
    if (this->height <= 0) this->height = 1;
    if (activeClient == this) {
        TW_VERIFY(::TwWindowSize(this->width, this->height), __LINE__);
    }
}


/*
 * GUILayer::GUIClient::layer
 */
GUILayer* GUILayer::GUIClient::layer = NULL;


/*
 * GUILayer::GUIClient::cntr
 */
SIZE_T GUILayer::GUIClient::cntr = 0;


/*
 * GUILayer::GUIClient::activeClient
 */
GUILayer::GUIClient* GUILayer::GUIClient::activeClient = NULL;

/****************************************************************************/


/*
 * GUILayer::Draw
 */
void GUILayer::Draw(void) {
    if (!this->active) return;
    TW_VERIFY(::TwDraw(), __LINE__);
}


/*
 * GUILayer::MouseMove
 */
bool GUILayer::MouseMove(int x, int y) {
    if (!this->active) return false;
    return (::TwMouseMotion(x, y) == 1);
}


/*
 * GUILayer::MouseButton
 */
bool GUILayer::MouseButton(int btn, bool down) {
    if (!this->active) return false;
    TwMouseButtonID b = TW_MOUSE_LEFT;
    switch (btn) {
        case 0:
            b = TW_MOUSE_LEFT;
            break;
        case 1:
            b = TW_MOUSE_MIDDLE;
            break;
        case 2:
            b = TW_MOUSE_RIGHT;
            break;
    }
    return (::TwMouseButton(down ? TW_MOUSE_PRESSED : TW_MOUSE_RELEASED, b) == 1);
}


/*
 * GUILayer::KeyPressed
 */
bool GUILayer::KeyPressed(unsigned short keycode, bool shift, bool alt, bool ctrl) {
    if (!this->active) return false;
    int key = (keycode & 0x00FF);
    int mod = TW_KMOD_NONE;

    if ((keycode & vislib::sys::KeyCode::KEY_SPECIAL) != 0) {
        switch (keycode) {
            case vislib::sys::KeyCode::KEY_ENTER: key = TW_KEY_RETURN; break;
            case vislib::sys::KeyCode::KEY_ESC: key = TW_KEY_ESCAPE; break;
            case vislib::sys::KeyCode::KEY_TAB: key = TW_KEY_TAB; break;
            case vislib::sys::KeyCode::KEY_LEFT: key = TW_KEY_LEFT; break;
            case vislib::sys::KeyCode::KEY_UP: key = TW_KEY_UP; break;
            case vislib::sys::KeyCode::KEY_RIGHT: key = TW_KEY_RIGHT; break;
            case vislib::sys::KeyCode::KEY_DOWN: key = TW_KEY_DOWN; break;
            case vislib::sys::KeyCode::KEY_PAGE_UP: key = TW_KEY_PAGE_UP; break;
            case vislib::sys::KeyCode::KEY_PAGE_DOWN: key = TW_KEY_PAGE_DOWN; break;
            case vislib::sys::KeyCode::KEY_HOME: key = TW_KEY_HOME; break;
            case vislib::sys::KeyCode::KEY_END: key = TW_KEY_END; break;
            case vislib::sys::KeyCode::KEY_INSERT: key = TW_KEY_INSERT; break;
            case vislib::sys::KeyCode::KEY_DELETE: key = TW_KEY_DELETE; break;
            case vislib::sys::KeyCode::KEY_BACKSPACE: key = TW_KEY_BACKSPACE; break;
            case vislib::sys::KeyCode::KEY_F1: key = TW_KEY_F1; break;
            case vislib::sys::KeyCode::KEY_F2: key = TW_KEY_F2; break;
            case vislib::sys::KeyCode::KEY_F3: key = TW_KEY_F3; break;
            case vislib::sys::KeyCode::KEY_F4: key = TW_KEY_F4; break;
            case vislib::sys::KeyCode::KEY_F5: key = TW_KEY_F5; break;
            case vislib::sys::KeyCode::KEY_F6: key = TW_KEY_F6; break;
            case vislib::sys::KeyCode::KEY_F7: key = TW_KEY_F7; break;
            case vislib::sys::KeyCode::KEY_F8: key = TW_KEY_F8; break;
            case vislib::sys::KeyCode::KEY_F9: key = TW_KEY_F9; break;
            case vislib::sys::KeyCode::KEY_F10: key = TW_KEY_F10; break;
            case vislib::sys::KeyCode::KEY_F11: key = TW_KEY_F11; break;
            case vislib::sys::KeyCode::KEY_F12: key = TW_KEY_F12; break;
        }
    }

    if (shift) mod |= TW_KMOD_SHIFT;
    if (alt) mod |= TW_KMOD_ALT;
    if (ctrl) mod |= TW_KMOD_CTRL;

    return (::TwKeyPressed(key, mod) == 1);
}


/*
 * GUILayer::GUILayer
 */
GUILayer::GUILayer(void) : active(false) {
    TW_VERIFY(::TwInit(TW_OPENGL, NULL), __LINE__);
    TW_VERIFY(::TwDeleteAllBars(), __LINE__);
}


/*
 * GUILayer::~GUILayer
 */
GUILayer::~GUILayer(void) {
    TW_VERIFY(::TwTerminate(), __LINE__);
}

#endif /* WITH_TWEAKBAR */
