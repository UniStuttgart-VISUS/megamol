/*
 * AbstractCommServerChannel.h
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTCOMMSERVERCHANNEL_H_INCLUDED
#define VISLIB_ABSTRACTCOMMSERVERCHANNEL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/AbstractCommChannel.h"


namespace vislib {
namespace net {


    /**
     * This class is a specialisation of the AbstractCommChannel, which adds
     * methods for server behaviour to the interface.
     */
    class AbstractCommServerChannel : public AbstractCommChannel {

    public:

        /**
         * Permit incoming connection attempt on the communication channel.
         *
         * @return The communcation channel for the new client connection.
         *
         * @throws Exception Or derived in case the operation fails.
         */
        virtual SmartRef<AbstractCommChannel> Accept(void) = 0;

        /**
         * Binds the server to a specified end point address.
         *
         * @param endPoint The end point address to bind to.
         *
         * @throws Exception Or derived in case the operation fails.
         */
        virtual void Bind(SmartRef<AbstractCommEndPoint> endPoint) = 0;

        /**
         * Bring the communication channel in a state in which it is listening
         * for an incoming connection. The method returns once a connection
         * to the server was made from a remote address.
         *
         * @param backlog Maximum length of the queue of pending connections.
         *
         * @throws Exception Or derived in case the operation fails.
         */
        virtual void Listen(const int backlog) = 0;

    protected:

        /** Superclass typedef. */
        typedef AbstractCommChannel Super;

        /** Ctor. */
        AbstractCommServerChannel(void);

        /** Dtor. */
        virtual ~AbstractCommServerChannel(void);

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTCOMMSERVERCHANNEL_H_INCLUDED */

