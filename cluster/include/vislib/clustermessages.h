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


#include "vislib/AbstractClusterNode.h"
#include "vislib/Array.h"
#include "vislib/RunnableThread.h"
#include "vislib/TcpServer.h"


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


    /** The magic number which must be at the begin of all network messages. */
    const UINT32 MAGIC_NUMBER = static_cast<UINT32>('v')
        | static_cast<UINT32>('l') << 8
        | static_cast<UINT32>('v') << 16
        | static_cast<UINT32>('c') << 24;

    
    /** This is the first message a client must send to a server node. */
    const UINT32 MSGID_INTRODUCE = 1;

    /** 
     * This message ID indicates that the message consists of multiple other
     * messages. The BlockHeader of the first message directly follows the
     * BlockHeader of the MSGID_MULTIPLE message. The message size that is
     * specified for MSGID_MULTIPLE is the size of all messages to follow in
     * bytes.
     */
    const UINT32 MSGID_MULTIPLE = 2;

    const UINT32 FIRST_CAMERAPARAMETER_MSGID = 64;

    const UINT32 MSGID_APERTUREANGLE = FIRST_CAMERAPARAMETER_MSGID + 0;

    const UINT32 MSGID_EYE = FIRST_CAMERAPARAMETER_MSGID + 1;

    const UINT32 MSGID_FARCLIP = FIRST_CAMERAPARAMETER_MSGID + 2;

    const UINT32 MSGID_FOCALDISTANCE = FIRST_CAMERAPARAMETER_MSGID + 3;

    const UINT32 MSGID_LIMITS = FIRST_CAMERAPARAMETER_MSGID + 4;

    const UINT32 MSGID_LOOKAT = FIRST_CAMERAPARAMETER_MSGID + 5;

    const UINT32 MSGID_NEARCLIP = FIRST_CAMERAPARAMETER_MSGID + 6;

    const UINT32 MSGID_POSITION = FIRST_CAMERAPARAMETER_MSGID + 7;
    
    const UINT32 MSGID_PROJECTION = FIRST_CAMERAPARAMETER_MSGID + 8;

    const UINT32 MSGID_STEREODISPARITY = FIRST_CAMERAPARAMETER_MSGID + 9;

    const UINT32 MSGID_TILERECT = FIRST_CAMERAPARAMETER_MSGID + 10;

    const UINT32 MSGID_UP = FIRST_CAMERAPARAMETER_MSGID + 11;
    
    const UINT32 MSGID_VIRTUALVIEWSIZE = FIRST_CAMERAPARAMETER_MSGID + 12;
    
} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CLUSTERMESSAGES_H_INCLUDED */

