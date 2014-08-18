/*
 * AbstractCommClientChannel.h
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTCOMMCLIENTCHANNEL_H_INCLUDED
#define VISLIB_ABSTRACTCOMMCLIENTCHANNEL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/AbstractCommEndPoint.h"
#include "vislib/ReferenceCounted.h"
#include "vislib/SmartRef.h"
#include "vislib/types.h"


namespace vislib {
namespace net {


    /**
     * The AbstractCommClientChannel represents a bidiectional, full duplex 
     * communication channel. Subclasses must implement this behaviour by
     * providing an implementation for all pure virtual methods defined by this
     * interface. 
     *
     * This class is part of the VISlib communication channel abstraction
     * layer. This layer is intended to provide a common class-based interface
     * for different network technologies.
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
    class AbstractCommClientChannel : public virtual ReferenceCounted {

    public:

        /** Constant for specifying an infinite timeout. */
        static const UINT TIMEOUT_INFINITE;

        /**
         * Terminate the open connection if any and reset the communication
         * channel to initialisation state.
         *
         * @throws Exception Or derived class in case of an error.
         */
        virtual void Close(void) = 0;

        /**
         * Connects the channel to the peer node at the specified end 
         * point address.
         *
         * @param endPoint The remote end point to connect to.
         *
         * @throws Exception Or derived in case the operation fails.
         */
        virtual void Connect(SmartRef<AbstractCommEndPoint> endPoint) = 0;

        /**
         * Answer the address the channel is using locally.
         *
         * The object returned needs not necessarily to be identical with the
         * address and end point has been bound or connected. Subclasses must,
         * however, guarantee that the returned end point is equal wrt. to the
         * == operator of the end point object.
         *
         * @return The address of the local end point.
         *
         * @throws Exception Or derived in case the operation fails.
         */
        virtual SmartRef<AbstractCommEndPoint> GetLocalEndPoint(void) const = 0;

        /**
         * Answer the address the remote peer of this channel is using.
         *
         * The object returned needs not necessarily to be identical with the
         * address and end point has been bound or connected. Subclasses must,
         * however, guarantee that the returned end point is equal wrt. to the
         * == operator of the end point object.
         *
         * @return The address of the remote end point.
         *
         * @throws Exception Or derived in case the operation fails.
         */
        virtual SmartRef<AbstractCommEndPoint> GetRemoteEndPoint(void) const = 0;

        /**
         * Receives 'cntBytes' over the communication channel and saves them to 
         * the memory designated by 'outData'. 'outData' must be large enough to 
         * receive at least 'cntBytes'.
         *
         * @param outData      The buffer to receive the data. The caller must
         *                     allocate this memory and remains owner.
         * @param cntBytes     The number of bytes to receive.
         * @param timeout      A timeout in milliseconds. 
         *                     Use AbstractCommChannel::TIMEOUT_INFINITE to 
         *                     specify an infinite timeout. If the operation 
         *                     timeouts, an exception will be thrown.
         * @param forceReceive If the data block cannot be received as a single 
         *                     packet, repeat the operation until all of 
         *                     'cntBytes' is received or a step fails.
         *
         * @return The number of bytes acutally received.
         *
         * @throws Exception Or any derived exception depending on the 
         *                   underlying layer in case of an error.
         */
        virtual SIZE_T Receive(void *outData, const SIZE_T cntBytes,
            const UINT timeout = TIMEOUT_INFINITE, 
            const bool forceReceive = true) = 0;

        /**
         * Sends 'cntBytes' from the location designated by 'data' over the
         * communication channel.
         *
         * @param data      The data to be sent. The caller remains owner of the
         *                  memory.
         * @param cntBytes  The number of bytes to be sent. 'data' must contain
         *                  at least this number of bytes.
         * @param timeout   A timeout in milliseconds. 
         *                  Use AbstractCommChannel::TIMEOUT_INFINITE to 
         *                  specify an infinite timeout. If the operation 
         *                  timeouts, an exception will be thrown.
         * @param forceSend If the data block cannot be sent as a single packet,
         *                  repeat the operation until all of 'cntBytes' is sent
         *                  or a step fails.
         *
         * @return The number of bytes acutally sent.
         *
         * @throws Exception Or any derived exception depending on the 
         *                   underlying layer in case of an error.
         */
        virtual SIZE_T Send(const void *data, const SIZE_T cntBytes,
            const UINT timeout = TIMEOUT_INFINITE, 
            const bool forceSend = true) = 0;

    protected:

        /** Superclass typedef. */
        typedef ReferenceCounted Super;

        /** Ctor. */
        AbstractCommClientChannel(void);

        /** Dtor. */
        virtual ~AbstractCommClientChannel(void);

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTCOMMCLIENTCHANNEL_H_INCLUDED */

