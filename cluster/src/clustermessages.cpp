/*
 * clustermessages.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/clustermessages.h"


/*
 * vislib::net::cluster::InitialiseMessageHeader
 */
void vislib::net::cluster::InitialiseMessageHeader(
        MessageHeader& inOutHeader) {
    inOutHeader.MagicNumber = MAGIC_NUMBER;
}


const UINT32 vislib::net::cluster::MAGIC_NUMBER 
    = static_cast<UINT32>('v')
    | static_cast<UINT32>('l') << 8
    | static_cast<UINT32>('c') << 16
    | 1 << 24;


/** 
 * Create a message ID of the base message set used for establishing a 
 * a connection etc.
 */
#define VLC1_BASE_MSG(id) (0 + (id))


/** Define a message of the camera parameter broadcasting message set. */
#define VLC1_CAMPARAM_MSG(id) (64 + (id))


const UINT32 vislib::net::cluster::MSGID_INTRODUCE = VLC1_BASE_MSG(1);

const UINT32 vislib::net::cluster::MSGID_MULTIPLE = VLC1_BASE_MSG(2);


const UINT32 vislib::net::cluster::MSGID_APERTUREANGLE = VLC1_CAMPARAM_MSG(0);

const UINT32 vislib::net::cluster::MSGID_EYE = VLC1_CAMPARAM_MSG(1);

const UINT32 vislib::net::cluster::MSGID_FARCLIP = VLC1_CAMPARAM_MSG(2);

const UINT32 vislib::net::cluster::MSGID_FOCALDISTANCE = VLC1_CAMPARAM_MSG(3);

const UINT32 vislib::net::cluster::MSGID_LIMITS = VLC1_CAMPARAM_MSG(4);

const UINT32 vislib::net::cluster::MSGID_LOOKAT = VLC1_CAMPARAM_MSG(5);

const UINT32 vislib::net::cluster::MSGID_NEARCLIP = VLC1_CAMPARAM_MSG(6);

const UINT32 vislib::net::cluster::MSGID_POSITION = VLC1_CAMPARAM_MSG(7);

const UINT32 vislib::net::cluster::MSGID_PROJECTION = VLC1_CAMPARAM_MSG(8);

const UINT32 vislib::net::cluster::MSGID_STEREODISPARITY = VLC1_CAMPARAM_MSG(9);

const UINT32 vislib::net::cluster::MSGID_TILERECT = VLC1_CAMPARAM_MSG(10);

const UINT32 vislib::net::cluster::MSGID_UP = VLC1_CAMPARAM_MSG(11);

const UINT32 vislib::net::cluster::MSGID_VIRTUALVIEWSIZE 
    = VLC1_CAMPARAM_MSG(12);
