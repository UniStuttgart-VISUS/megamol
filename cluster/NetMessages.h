/*
 * NetMessages.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_NETMESSAGES_H_INCLUDED
#define MEGAMOLCORE_NETMESSAGES_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/types.h"


namespace megamol {
namespace core {
namespace cluster {

namespace netmessages {

    /** Message sent to shut down a node */
    const UINT32 MSG_SHUTDOWN = 1;

    /** Message sent to the head node to initize time sync */
    const UINT32 MSG_QUERY_TIMESYNC = 2;

    /** A ping message for time sync */
    const UINT32 MSG_PING_TIMESYNC = 3;

    /** Data for time sync messages */
    typedef struct _timesyncdata_t {

        /** The number of the current trip */
        UINT32 trip;

        /** The instance time of the server when sending this message */
        float sendTimes[10];

        /** The instance time of the server when receiving this message */
        float recvTimes[10];

    } TimeSyncData;

} /* end namespace netmessages */

} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_NETMESSAGES_H_INCLUDED */
