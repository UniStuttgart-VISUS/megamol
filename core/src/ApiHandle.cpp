/*
 * ApiHandle.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/ApiHandle.h"


/*
 * megamol::core::ApiHandle::magicMegaMolUUID
 */ 
const unsigned char megamol::core::ApiHandle::magicMegaMolUUID[
    megamol::core::ApiHandle::uuidSize] = { 
        0x3f, 0xf7, 0x5e, 0x30, 0x43, 0xfb, 0x11, 0xdb, 
        0xb0, 0xde, 0x08, 0x00, 0x20, 0x0c, 0x9a, 0x65 };


/*
 * megamol::core::ApiHandle::ApiHandle(void)
 */
megamol::core::ApiHandle::ApiHandle(void) {
}


/*
 * megamol::core::ApiHandle::~ApiHandle(void)
 */
megamol::core::ApiHandle::~ApiHandle(void) {
}
