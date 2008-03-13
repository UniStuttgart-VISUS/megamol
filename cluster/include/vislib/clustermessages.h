/*
 * clustermessages.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CLUSTERMESSAGES_H_INCLUDED
#define VISLIB_CLUSTERMESSAGES_H_INCLUDED
#if (_MSC_VER > 1000)
#pragma once
#endif /* (_MSC_VER > 1000) */
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include "vislib/types.h"


namespace vislib {
namespace net {
namespace cluster {

    /**
     * The block header describes the content of a cluster network message.
     * The 'BlockId' specifies the message to follow in the next 'BlockLength'
     * bytes.
     *
     * The 'BlockId' of the first block may be MSGID_MULTIPLE. In this case,
     * multiple messages follow directly each other, each introduced with a
     * new BlockHeader.
     */
    typedef struct BlockHeader_t {
        UINT32 BlockId;
        UINT32 BlockLength;
    } BlockHeader;

    /**
     * This is the message header that is used by all the VISlib cluster
     * node classes for network communication.
     */
    typedef struct MessageHeader_t {
        UINT32 MagicNumber;
        BlockHeader Header;
    } MessageHeader;


    /**
     * Initialises the message header 'inOutHeader'.
     *
     * Actually, the magic number is set.
     *
     * @param inOutHeader The header to be initialised.
     */
    void InitialiseMessageHeader(MessageHeader& inOutHeader);


    /** The magic number which must be at the begin of all network messages. */
    extern const UINT32 MAGIC_NUMBER;

    /** This is the first message a client must send to a server node. */
    extern const UINT32 MSGID_INTRODUCE;

    /** 
     * This message ID indicates that the message consists of multiple other
     * messages. The BlockHeader of the first message directly follows the
     * BlockHeader of the MSGID_MULTIPLE message. The message size that is
     * specified for MSGID_MULTIPLE is the size of all messages to follow in
     * bytes.
     */
    extern const UINT32 MSGID_MULTIPLE;

    extern const UINT32 MSGID_APERTUREANGLE;

    extern const UINT32 MSGID_EYE;

    extern const UINT32 MSGID_FARCLIP;

    extern const UINT32 MSGID_FOCALDISTANCE;

    extern const UINT32 MSGID_LIMITS;

    extern const UINT32 MSGID_LOOKAT;

    extern const UINT32 MSGID_NEARCLIP;

    extern const UINT32 MSGID_POSITION;
    
    extern const UINT32 MSGID_PROJECTION;

    extern const UINT32 MSGID_STEREODISPARITY;

    extern const UINT32 MSGID_TILERECT;

    extern const UINT32 MSGID_UP;
    
    extern const UINT32 MSGID_VIRTUALVIEWSIZE;
    
} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CLUSTERMESSAGES_H_INCLUDED */

