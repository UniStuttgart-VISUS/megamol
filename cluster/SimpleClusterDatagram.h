/*
 * SimpleClusterDatagram.h
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

#define MSG_CONNECTTOSERVER 1
#define MSG_SHUTDOWN 2


    /**
     * Struct layout a simple cluster datagram
     */
    typedef struct _simpleclusterdatagram_t {

        /** The datagram message */
        unsigned int msg;

        /** The payload data */
        union _payload_t {

            /** raw data */
            char data[256];

        } payload;

    } SimpleClusterDatagram;


} /* end namespace cluster */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SIMPLECLUSTERDATAGRAM_H_INCLUDED */
