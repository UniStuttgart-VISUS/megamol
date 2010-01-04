/*
 * AbstractBidiCommChannel.cpp
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "vislib/AbstractBidiCommChannel.h"

#include "vislib/StackTrace.h"


///*
// * vislib::net::AbstractBidiCommChannel::AddRef
// */
//UINT32 vislib::net::AbstractBidiCommChannel::AddRef(void) {
//    VLSTACKTRACE("AbstractBidiCommChannel::AddRef", __FILE__, __LINE__);
//    return ReferenceCounted::AddRef();
//}
//
//
///*
// * vislib::net::AbstractBidiCommChannel::Release
// */
//UINT32 vislib::net::AbstractBidiCommChannel::Release(void) {
//     VLSTACKTRACE("AbstractBidiCommChannel::Release", __FILE__, __LINE__);
//     return ReferenceCounted::Release();
//}


/*
 * vislib::net::AbstractBidiCommChannel::AbstractBidiCommChannel
 */
vislib::net::AbstractBidiCommChannel::AbstractBidiCommChannel(void) 
        : AbstractInboundCommChannel(), AbstractOutboundCommChannel() {
     VLSTACKTRACE("AbstractBidiCommChannel::AbstractBidiCommChannel", __FILE__, 
        __LINE__);
}


/*
 * vislib::net::AbstractBidiCommChannel::~AbstractBidiCommChannel
 */
vislib::net::AbstractBidiCommChannel::~AbstractBidiCommChannel(void) {
     VLSTACKTRACE("AbstractBidiCommChannel::~AbstractBidiCommChannel", __FILE__, 
        __LINE__);
}
