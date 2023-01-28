/*
 * NetMessages.h
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_NETMESSAGES_H_INCLUDED
#define MEGAMOLCORE_NETMESSAGES_H_INCLUDED
#pragma once

#include "vislib/types.h"


namespace megamol::core::cluster::netmessages {

/************************************************************************/

/** Message sent to shut down a node */
const UINT32 MSG_SHUTDOWN = 1;

/** Message sent to the head node to initize time sync */
const UINT32 MSG_REQUEST_TIMESYNC = 2;

/** A ping message for time sync */
const UINT32 MSG_PING_TIMESYNC = 3;

/** A ping message for time sync */
const UINT32 MSG_DONE_TIMESYNC = 4;

/**
 * A timing sanity check message, required because the performance
 * counter seems to be buggy on some machines.
 */
const UINT32 MSG_TIME_SANITYCHECK = 5;

/**
 * Requesting the name of the node on the other end of the communication
 * channel
 */
const UINT32 MSG_WHATSYOURNAME = 6;

/** Tells the own name */
const UINT32 MSG_MYNAMEIS = 7;

/** Message requesting graph setup information from the master node */
const UINT32 MSG_REQUEST_GRAPHSETUP = 8;

/** Message with graph setup information from the master node */
const UINT32 MSG_GRAPHSETUP = 9;

/** Informs all client nodes that they are required to perform a resetup */
const UINT32 MSG_REQUEST_RESETUP = 10;

/** Message requesting camera setup information */
const UINT32 MSG_REQUEST_CAMERASETUP = 11;

/** Sets the view to be shown by the cluster */
const UINT32 MSG_SET_CLUSTERVIEW = 12;

/** Message containing all values of the current camera */
const UINT32 MSG_SET_CAMERAVALUES = 13;

/** Message containing a parameter value pair */
const UINT32 MSG_SET_PARAMVALUE = 14;

/** Message requesting camera values */
const UINT32 MSG_REQUEST_CAMERAVALUES = 15;

/** Message to activate or deactive the view pause mode */
const UINT32 MSG_REMOTEVIEW_PAUSE = 16;

/** Message to force the network VSync option into a specific state */
const UINT32 MSG_FORCENETVSYNC = 17;

/** Message to join the network VSync barrier */
const UINT32 MSG_NETVSYNC_JOIN = 18;

/** Message to leave the network VSync barrier */
const UINT32 MSG_NETVSYNC_LEAVE = 19;

/** Message to cross the network VSync barrier */
const UINT32 MSG_NETVSYNC_CROSS = 20;

/** Message with verbatim lua code for graph setup from master */
const UINT32 MSG_GRAPHSETUP_LUA = 21;

/************************************************************************/

/** The number of pings used for time syncing */
const unsigned int MAX_TIME_SYNC_PING = 20;

/** Data for time sync messages */
typedef struct _timesyncdata_t {

    /** The number of the current trip */
    UINT32 trip;

    /** The instance time of the client */
    double clntTimes[MAX_TIME_SYNC_PING];

    /** The instance time of the server */
    double srvrTimes[MAX_TIME_SYNC_PING + 1];

} TimeSyncData;

} // namespace megamol::core::cluster::netmessages

#endif /* MEGAMOLCORE_NETMESSAGES_H_INCLUDED */
