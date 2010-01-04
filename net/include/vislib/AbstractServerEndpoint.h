/*
 * AbstractServerEndpoint.h
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTSERVERENDPOINT_H_INCLUDED
#define VISLIB_ABSTRACTSERVERENDPOINT_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractCommChannel.h"
#include "vislib/ReferenceCounted.h"
#include "vislib/SmartRef.h"


namespace vislib {
namespace net {


    /**
     * This class defines the interface of a communication channel end point 
     * that acts as a server, i. e. waits for clients to connect on a specific
     * address.
     *
     * Rationale: The address is a string that must be parsed by implementing
     * classes. This choice takes into account that strings are the most 
     * flexible way of specifying an address, which allows the interface to be
     * the same for different network families. It is, however, encouraged that
     * subclasses provide additional methods using their native address format.
     */
    class AbstractServerEndpoint : public virtual ReferenceCounted {

    public:

        /**
         * Binds the server to a specified address.
         *
         * Note: The default implementation redirects the method to the UNICODE
         * version of the method.
         *
         * @param address The address to bind to.
         *
         * @throws Exception Or derived in case the operation fails.
         */
        virtual void Bind(const char *address);

        /**
         * Binds the server to a specified address.
         *
         * Note: The default implementation redirects the method to the UNICODE
         * version of the method.
         *
         * @param address The address to bind to.
         *
         * @throws Exception Or derived in case the operation fails.
         */
        virtual void Bind(const wchar_t *address) = 0;

        /**
         * Waits for a client to connect.
         *
         * This method blocks until a remote node connected.
         *
         * @param backlog Maximum length of the queue of pending connections.
         *
         * @return A communication channel for the new connection.
         *
         * @throws Exception Or derived in case the operation fails.
         */
        virtual SmartRef<AbstractCommChannel> WaitForClient(
            const int backlog) = 0;

    protected:

        /** Ctor. */
        AbstractServerEndpoint(void);

        /** Dtor. */
        virtual ~AbstractServerEndpoint(void);

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTSERVERENDPOINT_H_INCLUDED */

