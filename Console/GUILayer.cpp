/*
 * GUILayer.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#ifdef WITH_TWEAKBAR
#include "GUILayer.h"
#include "MegaMolCore.h"
#include "vislib/assert.h"
#include "vislib/Log.h"
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
 * GUILayer::GUIClient::Factory
 */
GUILayer::GUIClient::Parameter *GUILayer::GUIClient::Factory(
        TwBar *bar, vislib::SmartPtr<megamol::console::CoreHandle> hParam,
        const char *name, unsigned char *desc, unsigned int len) {
    if (bar == NULL) {
        return new PlaceboParameter(hParam, name, desc, len);
    }

    // TODO: Implement

    return new StringParameter(bar, hParam, name, desc, len);
}


/*
 * GUILayer::GUIClient::GUIClient
 */
GUILayer::GUIClient::GUIClient(void) : width(256), height(256), _myBar(NULL),
        params() {
    cntr++;
}


/*
 * GUILayer::GUIClient::~GUIClient
 */
GUILayer::GUIClient::~GUIClient(void) {
    ASSERT(cntr > 0);
    cntr--;

    vislib::SingleLinkedList<Parameter *>::Iterator iter = this->params.GetIterator();
    while (iter.HasNext()) {
        delete iter.Next();
    }
    this->params.Clear();

    if (this->_myBar != NULL) {
        TW_VERIFY(::TwDeleteBar(this->_myBar), __LINE__);
        this->_myBar = NULL;
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
    if ((layer == NULL) && !this->params.IsEmpty()) {
        this->Layer(); // initialise layer
        vislib::SingleLinkedList<Parameter*>::Iterator iter = this->params.GetIterator();
        while (iter.HasNext()) {
            Parameter *& param = iter.Next();
            PlaceboParameter *pp = dynamic_cast<PlaceboParameter*>(param);
            if (pp != NULL) {
                // lazy instantiation
                param = Factory(this->myBar(), pp->CoreHandle(), pp->Name(),
                    pp->Description(), pp->DescriptionLength());
                delete pp;
            }
        }
    }
    if (layer != NULL) {
        vislib::StringA name;
        TW_VERIFY(::TwWindowSize(this->width, this->height), __LINE__);
        if (this->_myBar != NULL) {
            vislib::StringA def = this->name();
            def.Append(" visible=true");
            TW_VERIFY(::TwDefine(def), __LINE__);
        }
    }
}


/*
 * GUILayer::GUIClient::Deactivate
 */
void GUILayer::GUIClient::Deactivate(void) {
    if (activeClient == this) {
        activeClient = NULL;
        if (this->_myBar != NULL) {
            vislib::StringA def = this->name();
            def.Append(" visible=false");
            TW_VERIFY(::TwDefine(def), __LINE__);
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
 * GUILayer::GUIClient::AddParameter
 */
void GUILayer::GUIClient::AddParameter(
        vislib::SmartPtr<megamol::console::CoreHandle> hParam,
        const char *name, unsigned char *desc, unsigned int len) {

    Parameter *param = Factory(this->_myBar, hParam, name, desc, len);
    if (param != NULL) {
        this->params.Add(param);
    }
}


/*
 * GUILayer::GUIClient::Draw
 */
void GUILayer::GUIClient::Draw(void) {
    if (layer != NULL) {
        this->Layer().Draw();
    }
}


/*
 * GUILayer::GUIClient::MouseMove
 */
bool GUILayer::GUIClient::MouseMove(int x, int y) {
    if (layer == NULL) return false;
    return this->Layer().MouseMove(x, y);
}


/*
 * GUILayer::GUIClient::MouseButton
 */
bool GUILayer::GUIClient::MouseButton(int btn, bool down) {
    if (layer == NULL) return false;
    return this->Layer().MouseButton(btn, down);
}


/*
 * GUILayer::GUIClient::KeyPressed
 */
bool GUILayer::GUIClient::KeyPressed(unsigned short keycode, bool shift, bool alt, bool ctrl) {
    if (layer == NULL) return false;
    return this->Layer().KeyPressed(keycode, shift, alt, ctrl);
}


/*
 * GUILayer::GUIClient::myBar
 */
TwBar *GUILayer::GUIClient::myBar(void) {
    if (this->_myBar == NULL) {
        GUILayer &l = this->Layer();
        vislib::StringA def = this->name();
        this->_myBar = ::TwNewBar(def);
        ASSERT(this->_myBar != NULL);
        def.Append(" label='Parameters' position='10 10' text=dark alpha=192 color='128 192 255'");
        def.Append(" visible=");
        def.Append((activeClient == this) ? "true" : "false");
        TW_VERIFY(::TwDefine(def), __LINE__);
    }
    return this->_myBar;
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
 * GUILayer::GUIClient::Parameter::paramName
 */
vislib::StringA GUILayer::GUIClient::Parameter::paramName(const char *name) {
    vislib::StringA str(name);
    vislib::StringA::Size pos = str.FindLast("::");
    if (pos != vislib::StringA::INVALID_POS) {
        return str.Substring(pos + 2);
    }
    return str;
}


/*
 * GUILayer::GUIClient::Parameter::paramGroup
 */
vislib::StringA GUILayer::GUIClient::Parameter::paramGroup(const char *name) {
    vislib::StringA str(name);
    if (str.StartsWith("::")) {
        str = str.Substring(2);
    }
    vislib::StringA::Size pos = str.FindLast("::");
    if (pos != vislib::StringA::INVALID_POS) {
        str.Truncate(pos);
    }
    return str;
}


/****************************************************************************/


/*
 * GUILayer::GUIClient::ValueParameter::ValueParameter
 */
GUILayer::GUIClient::ValueParameter::ValueParameter(TwBar *bar,
        vislib::SmartPtr<megamol::console::CoreHandle> hParam, TwType type,
        const char *name, unsigned char *desc, unsigned int len
        , const char *def) : Parameter(bar, hParam, name) {
    vislib::StringA defStr;
    defStr.Format("label='%s' group='%s' ", paramName(name), paramGroup(name));
    defStr.Append(def);
    TW_VERIFY(::TwAddVarCB(bar, this->objName(), type, &ValueParameter::Set,
        &ValueParameter::Get, this, defStr), __LINE__);
}


/*
 * GUILayer::GUIClient::ValueParameter::~ValueParameter
 */
GUILayer::GUIClient::ValueParameter::~ValueParameter() {
    TW_VERIFY(::TwRemoveVar(this->Bar(), this->objName()), __LINE__);
}


/*
 * GUILayer::GUIClient::ValueParameter::Set
 */
void TW_CALL GUILayer::GUIClient::ValueParameter::Set(const void *value, void *clientData) {
    ValueParameter *p = static_cast<ValueParameter *>(clientData);
    p->Set(value);
}


/*
 * GUILayer::GUIClient::ValueParameter::Get
 */
void TW_CALL GUILayer::GUIClient::ValueParameter::Get(void *value, void *clientData) {
    ValueParameter *p = static_cast<ValueParameter *>(clientData);
    p->Get(value);
}

/****************************************************************************/


/*
 * GUILayer::GUIClient::StringParameter::StringParameter
 */
GUILayer::GUIClient::StringParameter::StringParameter(TwBar *bar,
        vislib::SmartPtr<megamol::console::CoreHandle> hParam,
        const char *name, unsigned char *desc, unsigned int len) 
        : ValueParameter(bar, hParam, TW_TYPE_CDSTRING, name, desc, len, "") {
}


/*
 * GUILayer::GUIClient::StringParameter::~StringParameter
 */
GUILayer::GUIClient::StringParameter::~StringParameter() {
}


/*
 * GUILayer::GUIClient::StringParameter::Set
 */
void GUILayer::GUIClient::StringParameter::Set(const void *value) {
    ::mmcSetParameterValueA(this->Handle(),
        *reinterpret_cast<const char * const *>(value));
}


/*
 * GUILayer::GUIClient::ValueParameter::Get
 */
void GUILayer::GUIClient::StringParameter::Get(void *value) {
    const char *dat = ::mmcGetParameterValueA(this->Handle());
    char **destPtr = (char **)value;
    ::TwCopyCDStringToLibrary(destPtr, dat);
}

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
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "GUI Layer initialized");
}


/*
 * GUILayer::~GUILayer
 */
GUILayer::~GUILayer(void) {
    TW_VERIFY(::TwTerminate(), __LINE__);
    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
        "GUI Layer shutdown");
}

#endif /* WITH_TWEAKBAR */
