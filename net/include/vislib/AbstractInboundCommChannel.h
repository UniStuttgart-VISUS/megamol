/*
 * AbstractInboundCommChannel.h
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTINBOUNDCOMMCHANNEL_H_INCLUDED
#define VISLIB_ABSTRACTINBOUNDCOMMCHANNEL_H_INCLUDED
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
     * This class adds interface methods that inbound communication channels
     * must implement in addition to the methods of AbstractCommChannel.
     */
    class AbstractInboundCommChannel : public virtual AbstractCommChannel {

    public:

        /**
         * Receives 'cntBytes' over the communication channel and saves them to 
         * the memory designated by 'outData'. 'outData' must be large enough to 
         * receive at least 'cntBytes'.
         *
         * @param outData      The buffer to receive the data. The caller must
         *                     allocate this memory and remains owner.
         * @param cntBytes     The number of bytes to receive.
         * @param timeout      A timeout in milliseconds. A value less than 1 
         *                     specifies an infinite timeout. If the operation 
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
            const INT timeout, const bool forceReceive) = 0;

    protected:

        /** Superclass typedef. */
        typedef AbstractCommChannel Super;

        /** Ctor. */
        AbstractInboundCommChannel(void);

        /** Dtor. */
        virtual ~AbstractInboundCommChannel(void);

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTINBOUNDCOMMCHANNEL_H_INCLUDED */
