/*
 * IbRdmaCommServerChannel.h
 *
 * Copyright (C) 2006 - 2012 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_IBRDMACOMMSERVERCHANNEL_H_INCLUDED
#define VISLIB_IBRDMACOMMSERVERCHANNEL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/Socket.h"                      // Must be first!
#include "vislib/AbstractCommServerChannel.h"
#include "vislib/IbRdmaException.h"

#include "rdma/rdma_cma.h"
#include "rdma/rdma_verbs.h"


namespace vislib {
namespace net {
namespace ib {


    /**
     * TODO: comment class
     */
    class IbRdmaCommServerChannel : public AbstractCommServerChannel {

    public:

        static SmartRef<IbRdmaCommServerChannel> Create(void);

        virtual SmartRef<AbstractCommClientChannel> Accept(void);

        virtual void Bind(SmartRef<AbstractCommEndPoint> endPoint);

        virtual void Close(void);

        virtual SmartRef<AbstractCommEndPoint> GetLocalEndPoint(void) const;

        virtual void Listen(const int backlog = SOMAXCONN);

    private:

        /** Superclass typedef. */
        typedef AbstractCommServerChannel Super;

        /** Ctor. */
        IbRdmaCommServerChannel(void);

        /** Dtor. */
        ~IbRdmaCommServerChannel(void);

        struct rdma_cm_id *id;

    };
    
} /* end namespace ib */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_IBRDMACOMMSERVERCHANNEL_H_INCLUDED */

