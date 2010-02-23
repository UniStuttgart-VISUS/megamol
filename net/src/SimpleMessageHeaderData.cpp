/*
 * SimpleMessageHeaderData.cpp
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/SimpleMessageHeaderData.h"


#include <climits>


/*
 * vislib::net::VLSNP1_FIRST_RESERVED_MESSAGE_ID
 */
const vislib::net::SimpleMessageID vislib::net::VLSNP1_FIRST_RESERVED_MESSAGE_ID
    = static_cast<vislib::net::SimpleMessageID>(UINT_MAX - 1024);
