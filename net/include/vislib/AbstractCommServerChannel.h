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

#include "vislib/AbstractCommClientChannel.h"
#include "vislib/AbstractCommEndPoint.h"
#include "vislib/ReferenceCounted.h"
#include "vislib/SmartRef.h"
#include "vislib/types.h"


namespace vislib {
namespace net {


    /**
     * This class defines the interface for server end points in the VISlib
     * communication channel abstraction layer.
     *
     * Note for implementors: Subclasses should provide static Create() 
     * methods which create objects on the heap that must have been created 
     * with C++ new. The Release() method of this class assumes creation 
     * with C++ new and releases the object be calling delete once the last 
     * reference was released.
     *
     * Rationale: Due to the design-inherent polymorphism of this abstraction 
     * layer, we use reference counting for managing the objects. This is 
     * because some classes in the layer must return objects on the heap. Users
     * of AbstractCommChannel should use SmartRef for handling the reference
     * counting.
     *
     * Note for implementors: Subclasses that provide the server and the client
     * interface in one class should inherit from AbstractCommChannel.
     */
    class AbstractCommServerChannel : public virtual ReferenceCounted {

    public:

        /**
         * Permit incoming connection attempt on the communication channel.
         *
         * @return The communcation channel for the new client connection.
         *
         * @throws Exception Or derived in case the operation fails.
         */
        virtual SmartRef<AbstractCommClientChannel> Accept(void) = 0;

        /**
         * Binds the server to a specified end point address.
         *
         * @param endPoint The end point address to bind to.
         *
         * @throws Exception Or derived in case the operation fails.
         */
        virtual void Bind(SmartRef<AbstractCommEndPoint> endPoint) = 0;

        /**
         * Terminate the open connection if any and reset the communication
         * channel to initialisation state.
         *
         * @throws Exception Or derived class in case of an error.
         */
        virtual void Close(void) = 0;

        /**
         * Answer the address the channel is using locally.
         *
         * The object returned needs not necessarily to be identical with the
         * address and end point that the channel has been bound to. Subclasses 
         * must, however, guarantee that the returned end point is equal wrt. 
         * to the operator ==() of the end point object.
         *
         * @return The address of the local end point.
         *
         * @throws Exception Or derived in case the operation fails.
         */
        virtual SmartRef<AbstractCommEndPoint> GetLocalEndPoint(void) const = 0;

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
        typedef ReferenceCounted Super;

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

