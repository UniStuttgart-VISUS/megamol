/*
 * AsyncSocketContext.cpp
 *
 * Copyright (C) 2006 - 2009 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/AsyncSocketContext.h"

#include "vislib/AsyncSocket.h"


/*
 * vislib::net::AsyncSocketContext::AsyncSocketContext
 */
vislib::net::AsyncSocketContext::AsyncSocketContext(AsyncCallback callback,
        void *userContext) : Super(callback, userContext) {
}


/*
 * vislib::net::AsyncSocketContext::~AsyncSocketContext
 */
vislib::net::AsyncSocketContext::~AsyncSocketContext(void) {
}


/*
 * vislib::net::AsyncSocketContext::Wait
 */
void vislib::net::AsyncSocketContext::Wait(void) {
}


/*
 * islib::net::AsyncSocketContext::notifyCompleted
 */
void vislib::net::AsyncSocketContext::notifyCompleted(const DWORD cntData,
        const DWORD errorCode) {
    this->cntData = cntData;
    this->errorCode = errorCode;
}
