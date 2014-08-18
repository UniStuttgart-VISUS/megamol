/*
 * AsyncSocketContext.cpp
 *
 * Copyright (C) 2009 by Visualisierungsinstitut der Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2009 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "vislib/AsyncSocketContext.h"

#include "vislib/AsyncSocket.h"
#include "vislib/IllegalParamException.h"
#include "vislib/Trace.h"


/*
 * vislib::net::AsyncSocketContext::AsyncSocketContext
 */
vislib::net::AsyncSocketContext::AsyncSocketContext(AsyncCallback callback,
        void *userContext) : Super(callback, userContext), evt(true) {
    VLSTACKTRACE("AsyncSocketContext::AsyncSocketContext", __FILE__, __LINE__);
#if (!defined(_WIN32) || defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN))
    this->cntData = 0;
    this->errorCode = 0;
#endif /* (!defined(_WIN32) || defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN)) */
}


/*
 * vislib::net::AsyncSocketContext::~AsyncSocketContext
 */
vislib::net::AsyncSocketContext::~AsyncSocketContext(void) {
    VLSTACKTRACE("AsyncSocketContext::~AsyncSocketContext", __FILE__, __LINE__);
}


/*
 * vislib::net::AsyncSocketContext::Reset
 */
void vislib::net::AsyncSocketContext::Reset(void) {
    VLSTACKTRACE("AsyncSocketContext::Reset", __FILE__, __LINE__);
    Super::Reset();
    this->evt.Reset();
}


/*
 * vislib::net::AsyncSocketContext::Wait
 */
void vislib::net::AsyncSocketContext::Wait(void) {
    VLSTACKTRACE("AsyncSocketContext::Wait", __FILE__, __LINE__);
    this->evt.Wait();
}


/*
 * islib::net::AsyncSocketContext::notifyCompleted
 */
void vislib::net::AsyncSocketContext::notifyCompleted(const DWORD cntData,
        const DWORD errorCode) {
    VLSTACKTRACE("AsyncSocketContext::notifyCompleted", __FILE__, __LINE__);

    VLTRACE(Trace::LEVEL_VL_ANNOYINGLY_VERBOSE, "Signaling completion of "
        "asynchronous socket operation with return value %u for "
        "%u Bytes ...\n", errorCode, cntData);

#if (!defined(_WIN32) || defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN))
    /* Remember the result of the operation. */
    this->cntData = cntData;
    this->errorCode = errorCode;
#endif /* (!defined(_WIN32) || defined(VISLIB_ASYNCSOCKET_LIN_IMPL_ON_WIN)) */

    /* Call the callback if set. */
    if (this->callback != NULL) {
        this->callback(this);
    }

    /* Signal the event. */
    this->evt.Set();
}


/*
 * vislib::net::AsyncSocketContext::operator =
 */
vislib::net::AsyncSocketContext& vislib::net::AsyncSocketContext::operator =(
        const AsyncSocketContext& rhs) {
    VLSTACKTRACE("AsyncSocketContext::operator =", __FILE__, __LINE__);

    if (this != &rhs) {
        throw IllegalParamException("rhs", __FILE__, __LINE__);
    }

    return *this;
}
