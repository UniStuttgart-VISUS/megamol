/*
 * clustermessages.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/clustermessages.h"


/*
 * vislib::net::cluster::InitialiseMessageHeader
 */
void vislib::net::cluster::InitialiseMessageHeader(
        MessageHeader& inOutHeader, const UINT32 blockId, 
        const UINT32 blockLength) {
    inOutHeader.MagicNumber = MAGIC_NUMBER;
    inOutHeader.Header.BlockId = blockId;
    inOutHeader.Header.BlockLength = blockLength;
}
