/*
 * ApiHandle.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ApiHandle.h"


/*
 * megamol::viewer::ApiHandle::magicMegaMolUUID
 */ 
const unsigned char megamol::viewer::ApiHandle::magicMegaMolUUID[
    megamol::viewer::ApiHandle::uuidSize] = { 
        0xee, 0x45, 0x18, 0xc1, 0xe6, 0xbf, 0x11, 0xdc,
        0x95, 0xff, 0x08, 0x00, 0x20, 0x0c, 0x9a, 0x66 };


/*
 * megamol::viewer::ApiHandle::ApiHandle(void)
 */
megamol::viewer::ApiHandle::ApiHandle(void) : UserData(NULL) {
}


/*
 * megamol::viewer::ApiHandle::~ApiHandle(void)
 */
megamol::viewer::ApiHandle::~ApiHandle(void) {
    this->UserData = NULL; // DO NOT DELETE!
}
