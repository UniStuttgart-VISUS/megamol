/*
 * SimpleClusterCommUtil.h
 *
 * Copyright (C) 2009 - 2010 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_SIMPLECLUSTERDATAGRAM_H_INCLUDED
#define MEGAMOLCORE_SIMPLECLUSTERDATAGRAM_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "utility/Configuration.h"


namespace megamol {
namespace core {
namespace cluster {


    /**
     * Answer the default port used for simple cluster datagram communication
     *
     * @param cfg The configuration to load the port info from
     *
     * @return The default port for simple datagram communication
     */
    unsigned int GetDatagramPort(const utility::Configuration *cfg = NULL);


    /**
     * Answer the default port used for simple cluster TCP stream communication
     *
     * @param cfg The configuration to load the port info from
     *
     * @return The default port for simple TCP stream communication
     */
    unsigned int GetStreamPort(const utility::Configuration *cfg = NULL);

#define MSG_CONNECTTOSERVER 1
#define MSG_SHUTDOWN 2

#define MSG_HANDSHAKE_INIT 1
#define MSG_HANDSHAKE_BACK 2
#define MSG_HANDSHAKE_FORTH 3
#define MSG_HANDSHAKE_DONE 4

#define MSG_TIMESYNC 5
#define TIMESYNCDATACOUNT 10
    typedef struct _timesyncdata_t {
        unsigned short cnt;
        double time[TIMESYNCDATACOUNT];
    } TimeSyncData;

#define MSG_MODULGRAPH 6
#define MSG_VIEWCONNECT 7

    /**
     * Struct layout a simple cluster datagram
     */
    typedef struct _simpleclusterdatagram_t {

        /** The datagram message */
        unsigned short msg;

        /** The number of times this message should be echoed */
        unsigned char cntEchoed;

        /** The payload data */
        union _payload_t {

            /** raw data */
            char data[256];

            /** a string pair */
            struct _strings_t {

                /** The length (max 127 bytes!) */
                unsigned char len1;

                /** The string 1 data (terminating zero optional!!!) */
                char str1[127];

                /** The length (max 127 bytes!) */
                unsigned char len2;

                /** The string 2 data (terminating zero optional!!!) */
                char str2[127];

            } Strings;

        } payload;

    } SimpleClusterDatagram;


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SIMPLECLUSTERDATAGRAM_H_INCLUDED */
