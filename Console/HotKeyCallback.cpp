/*
 * HotKeyCallback.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "HotKeyCallback.h"
#include "vislib/assert.h"


/*
 * megamol::console::HotKeyCallback::HotKeyCallback
 */
megamol::console::HotKeyCallback::HotKeyCallback(void (*callback)(void))
        : HotKeyAction(), callback(callback) {
    ASSERT(this->callback != NULL);
}


/*
 * megamol::console::HotKeyCallback::~HotKeyCallback
 */
megamol::console::HotKeyCallback::~HotKeyCallback(void) {
    // intentionally empty
}


/*
 * megamol::console::HotKeyCallback::Trigger
 */
void megamol::console::HotKeyCallback::Trigger(void) {
    this->callback();
}
