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
        MessageHeader& inOutHeader) {
    inOutHeader.MagicNumber = MAGIC_NUMBER;
}
