/*
 * Serialisable.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/Serialisable.h"


/*
 * vislib::Serialisable::~Serialisable
 */
vislib::Serialisable::~Serialisable(void) {
    // Nothing to do.
}


/*
 * vislib::Serialisable::OnSerialisationDataReceived
 */
bool vislib::Serialisable::OnSerialisationDataReceived(BYTE *data) {
    return true;
}
