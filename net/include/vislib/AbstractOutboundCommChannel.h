/*
 * AbstractOutboundCommChannel.h
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTOUTBOUNDCOMMCHANNEL_H_INCLUDED
#define VISLIB_ABSTRACTOUTBOUNDCOMMCHANNEL_H_INCLUDED
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
     * This class adds interface methods that outbound communication channels
     * must implement in addition to the methods of AbstractCommChannel.
     */
    class AbstractOutboundCommChannel : public virtual AbstractCommChannel {

    public:

        /**
         * Send 'cntBytes' from the location designated by 'data' over the
         * communication channel.
         *
         * @param data      The data to be sent. The caller remains owner of the
         *                  memory.
         * @param cntBytes  The number of bytes to be sent. 'data' must contain
         *                  at least this number of bytes.
         * @param timeout   A timeout in milliseconds. A value less than 1 
         *                  specifies an infinite timeout. If the operation 
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
            const INT timeout, const bool forceSend) = 0;

    protected:

        /** Superclass typedef. */
        typedef AbstractCommChannel Super;

        /** Ctor. */
        AbstractOutboundCommChannel(void);

        /** Dtor. */
        virtual ~AbstractOutboundCommChannel(void);

    };

} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTOUTBOUNDCOMMCHANNEL_H_INCLUDED */
