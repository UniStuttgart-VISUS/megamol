/*
 * SimpleMessageHeaderData.h
 *
 * Copyright (C) 2006 - 2010 by Visualisierungsinstitut Universitaet Stuttgart. 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_SIMPLEMESSAGEHEADERDATA_H_INCLUDED
#define VISLIB_SIMPLEMESSAGEHEADERDATA_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/types.h"


namespace vislib {
namespace net {


    /** ID type for simple messages. */
    typedef UINT32 SimpleMessageID;

    /** Size type for simple messages. */
    typedef UINT32 SimpleMessageSize;


    /**
     * This is the message header that goes over the wire. It is not recommended
     * using this structure directly, but rather than via the object-oriented 
     * wrappers. We use this structure to have a definite layout of the data 
     * that go over the write and that can be exchanged with other programming
     * languages.
     */
    typedef struct SimpleMessageHeaderData_t {
        SimpleMessageID MessageID;  ///< User-defined message ID.
        SimpleMessageSize BodySize; ///< Size of the body to follow in bytes.
    } SimpleMessageHeaderData;


    /**
     * First VISlib-reserved message ID. User programmes shall not use message
     * IDs equal or larger than this number.
     */
    extern const SimpleMessageID VLSNP1_FIRST_RESERVED_MESSAGE_ID;
    
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_SIMPLEMESSAGEHEADERDATA_H_INCLUDED */
