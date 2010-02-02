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
GUILayer::GUIClient::GUIClient(void) {
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
    return *layer;
}


/*
 * GUILayer::GUIClient::layer
 */
GUILayer* GUILayer::GUIClient::layer = NULL;


/*
 * GUILayer::GUIClient::cntr
 */
SIZE_T GUILayer::GUIClient::cntr = 0;

/****************************************************************************/


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
