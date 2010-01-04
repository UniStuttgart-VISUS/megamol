/*
 * AbstractCommChannel.h
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTCOMMCHANNEL_H_INCLUDED
#define VISLIB_ABSTRACTCOMMCHANNEL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/ReferenceCounted.h"
#include "vislib/types.h"


namespace vislib {
namespace net {


    /**
     * This is the superclass of the VISlib communication channel abstraction
     * layer. This layer is intended to provide a common class-based interface
     * for different network technologies.
     *
     * Implementation note: Due to the design-inherent polymorphism of this
     * abstraction layer, we use reference counting for managing the objects.
     * This is because some classes in the layer must return objects on the 
     * heap.
     */
    class AbstractCommChannel : public virtual ReferenceCounted {

    public:

        /** Constant for specifying an infinite timeout. */
        static const UINT TIMEOUT_INFINITE;

        /**
         * Terminate the open connection if any and reset the communication
         * channel to initialisation state.
         *
         * @throws Exception or derived class in case of an error.
         */
        virtual void Close(void) = 0;

        /**
         * Answer whether the communication channel is an inbound channel.
         * In this case, it can safely be casted to AbstractInboundCommChannel.
         * Note that a communication channel can be inbound and outbound at the
         * same time.
         *
         * @return true If the communication channel is inbound.
         */
        virtual bool IsInbound(void) const;

        /**
         * Answer whether the communication channel is an outbound channel.
         * In this case, it can safely be casted to AbstractOutboundCommChannel.
         * Note that a communication channel can be inbound and outbound at the
         * same time.
         *
         * @return true If the communication channel is outbound.
         */
        virtual bool IsOutbound(void) const;

    protected:

        /** Ctor. */
        AbstractCommChannel(void);

        /** Dtor. */
        virtual ~AbstractCommChannel(void);

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTCOMMCHANNEL_H_INCLUDED */
