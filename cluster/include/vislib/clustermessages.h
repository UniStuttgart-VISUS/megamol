/*
 * clustermessages.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef VISLIB_CLUSTERMESSAGES_H_INCLUDED
#define VISLIB_CLUSTERMESSAGES_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
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
     * @param blockId     The message ID of the first (required) block.
     * @param blockLength The size of the first (required) block.
     */
    void InitialiseMessageHeader(MessageHeader& inOutHeader, 
        const UINT32 blockId = 0, const UINT32 blockLength = 0);


    /** The magic number which must be at the begin of all network messages. */
    const UINT32 MAGIC_NUMBER = static_cast<UINT32>('v')
        | static_cast<UINT32>('l') << 8
        | static_cast<UINT32>('c') << 16
        | 1 << 24;


    /** 
     * Create a message ID of the base message set used for establishing a 
     * a connection etc.
     */
    #define VLC1_BASE_MSG_ID(id) (0 + (id))

    /**
     * Declare and define a variable 'name' that has the base message ID 'id'.
     * The string "MSGID_" is prepended to the name.
     */
    #define DEFINE_VLC1_BASE_MSG(name, id)                                     \
        const UINT32 MSGID_##name = VLC1_BASE_MSG_ID(id)

    /**
     * Declare and define a variable 'name' that has the camera parameter 
     * message ID 'id'.
     * This macro adds the default offset for camera parameter messages to 'id', 
     * i.e. the first camera parameter message should pass 0 here.
     * The string "MSGID_CAM_" is prepended to the name.
     */
    #define DEFINE_VLC1_CAM_MSG(name, id)                                      \
        DEFINE_VLC1_BASE_MSG(CAM_##name, 64 + (id))


    /** This is the first message a client must send to a server node. */
    DEFINE_VLC1_BASE_MSG(INTRODUCE, 1);

    /** 
     * This message ID indicates that the message consists of multiple other
     * messages. The BlockHeader of the first message directly follows the
     * BlockHeader of the MSGID_MULTIPLE message. The message size that is
     * specified for MSGID_MULTIPLE is the size of all messages to follow in
     * bytes.
     */
    DEFINE_VLC1_BASE_MSG(MULTIPLE, 2);

    DEFINE_VLC1_CAM_MSG(APERTUREANGLE, 0);

    DEFINE_VLC1_CAM_MSG(EYE, 1);

    DEFINE_VLC1_CAM_MSG(FARCLIP, 2);

    DEFINE_VLC1_CAM_MSG(FOCALDISTANCE, 3);

    DEFINE_VLC1_CAM_MSG(LIMITS, 4);

    DEFINE_VLC1_CAM_MSG(LOOKAT, 5);

    DEFINE_VLC1_CAM_MSG(NEARCLIP, 6);

    DEFINE_VLC1_CAM_MSG(POSITION, 7);

    DEFINE_VLC1_CAM_MSG(PROJECTION, 8);

    DEFINE_VLC1_CAM_MSG(STEREODISPARITY, 9);

    DEFINE_VLC1_CAM_MSG(TILERECT, 10);

    DEFINE_VLC1_CAM_MSG(UP, 11);
    
    DEFINE_VLC1_CAM_MSG(VIRTUALVIEWSIZE, 12);

    DEFINE_VLC1_CAM_MSG(SERIALISEDCAMPARAMS, 13);

    DEFINE_VLC1_CAM_MSG(REQUEST_ALL, 14);

    /** 
     * Create a message ID in the non-reserved user message range of VISlib
     * cluster message IDs. The 'id' is just offset.
     */
    #define VLC1_USER_MSG_ID(id) ((id) + 0x80000000)

} /* end namespace cluster */
} /* end namespace net */
} /* end namespace vislib */

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
#endif /* VISLIB_CLUSTERMESSAGES_H_INCLUDED */
