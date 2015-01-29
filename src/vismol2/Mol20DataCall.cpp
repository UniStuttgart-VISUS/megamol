/*
 * Mol20DataCall.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "Mol20DataCall.h"

using namespace megamol::core;


/*
 * vismol2::Mol20DataCall::Mol20DataCall
 */
vismol2::Mol20DataCall::Mol20DataCall(void) : Call(), bbox(),
        frame(NULL), time(UINT_MAX) {
}


/*
 * vismol2::Mol20DataCall::~Mol20DataCall
 */
vismol2::Mol20DataCall::~Mol20DataCall(void) {
    if (this->frame != NULL) {
        this->frame->Unlock();
        this->frame = NULL; // DO NOT DELETE!
    }
}
