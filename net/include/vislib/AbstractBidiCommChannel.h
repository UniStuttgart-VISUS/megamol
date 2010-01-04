/*
 * AbstractBidiCommChannel.h
 *
 * Copyright (C) 2010 by Christoph Müller. Alle Rechte vorbehalten.
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_ABSTRACTBIDICOMMCHANNEL_H_INCLUDED
#define VISLIB_ABSTRACTBIDICOMMCHANNEL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/AbstractInboundCommChannel.h"
#include "vislib/AbstractOutboundCommChannel.h"


namespace vislib {
namespace net {


    /**
     * This class defines the interface of a bi-directional communication 
     * channel.
     *
     * The rationale for adding this class is that most communication channels
     * can be expected to be bi-directional. Therefore, we want to group the 
     * resolution of possible ambiguities due to multiple inheritance into one 
     * class.
     */
    class AbstractBidiCommChannel : public virtual AbstractInboundCommChannel,
            public virtual AbstractOutboundCommChannel {

    public:

        ///**
        // * Increment the reference count.
        // *
        // * @return The new value of the reference counter.
        // */
        //UINT32 AddRef(void);

        ///**
        // * Decrement the reference count. If the reference count reaches zero,
        // * the object is released using the allocator A.
        // *
        // * @return The new value of the reference counter.
        // */
        //UINT32 Release(void);

    protected:

        /** Ctor. */
        AbstractBidiCommChannel(void);

        /** Dtor. */
        virtual ~AbstractBidiCommChannel(void);

    };
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_ABSTRACTBIDICOMMCHANNEL_H_INCLUDED */
