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
GUILayer::GUIClient::GUIClient(void) : width(256), height(256) {
    cntr++;
}


/*
 * GUILayer::GUIClient::~GUIClient
 */
GUILayer::GUIClient::~GUIClient(void) {
    ASSERT(cntr > 0);
    cntr--;
    if (cntr == 0) {
        SAFE_DELETE(layer);
    }
}


/*
 * GUILayer::GUIClient::Layer
 */
GUILayer& GUILayer::GUIClient::Layer(void) {
    if (layer == NULL) {
        layer = new GUILayer();
    }
    if ((lastWidth != this->width) || (lastHeight != this->height)) {
        lastWidth = this->width;
        lastHeight = this->height;
        TW_VERIFY(::TwWindowSize(this->width, this->height), __LINE__);
    }
    return *layer;
}


/*
 * GUILayer::GUIClient::SetWindowSize
 */
void GUILayer::GUIClient::SetWindowSize(unsigned int w, unsigned int h) {
    this->width = static_cast<int>(w);
    if (this->width <= 0) this->width = 1;
    this->height = static_cast<int>(h);
    if (this->height <= 0) this->height = 1;
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
 * GUILayer::GUIClient::lastWidth
 */
int GUILayer::GUIClient::lastWidth = 0;


/*
 * GUILayer::GUIClient::lastHeight
 */
int GUILayer::GUIClient::lastHeight = 0;

/****************************************************************************/


/*
 * GUILayer::Draw
 */
void GUILayer::Draw(void) {
    TW_VERIFY(::TwDraw(), __LINE__);
}


/*
 * GUILayer::MouseMove
 */
bool GUILayer::MouseMove(int x, int y) {
    return (::TwMouseMotion(x, y) == 1);
}


/*
 * GUILayer::MouseButton
 */
bool GUILayer::MouseButton(int btn, bool down) {
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
 * GUILayer::GUILayer
 */
GUILayer::GUILayer(void) {
    TW_VERIFY(::TwInit(TW_OPENGL, NULL), __LINE__);
}


/*
 * GUILayer::~GUILayer
 */
GUILayer::~GUILayer(void) {
    TW_VERIFY(::TwTerminate(), __LINE__);
}

#endif /* WITH_TWEAKBAR */
