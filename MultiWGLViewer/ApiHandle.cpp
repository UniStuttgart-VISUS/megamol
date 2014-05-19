/*
 * ApiHandle.cpp
 *
 * Copyright (C) 2006 - 2011 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ApiHandle.h"


/*
 * megamol::wgl::ApiHandle::magicMegaMolUUID
 */ 
const unsigned char megamol::wgl::ApiHandle::magicMegaMolUUID[
    megamol::wgl::ApiHandle::uuidSize] = { 
        0xee, 0x45, 0x19, 0xc1, 0xe6, 0xbf, 0x12, 0xdb,
        0x95, 0xff, 0x09, 0x00, 0x20, 0x0d, 0x9a, 0x65 };


/*
 * megamol::wgl::ApiHandle::ApiHandle(void)
 */
megamol::wgl::ApiHandle::ApiHandle(void) : UserData(NULL) {
}


/*
 * megamol::wgl::ApiHandle::~ApiHandle(void)
 */
megamol::wgl::ApiHandle::~ApiHandle(void) {
    this->UserData = NULL; // DO NOT DELETE!
}
