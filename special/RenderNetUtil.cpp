/*
 * RenderNetUtil.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "RenderNetUtil.h"
#include "api/MegaMolCore.h"
#include "vislib/Exception.h"
#include "vislib/Log.h"
#include "vislib/PerformanceCounter.h"
#include "vislib/Process.h"
#include "vislib/SystemInformation.h"
#include "vislib/types.h"
#include "vislib/UnsupportedOperationException.h"

using namespace megamol::core;


namespace megamol {
namespace core {
namespace special {

    static const char HandshakeHeaderID[] = "MegaMolRenderNetworkHandshake";

#ifdef _WIN32
    static const UINT64 endianTestValue = 0x0123456789ABCDEFUL;
#else
    static const UINT64 endianTestValue = 0x0123456789ABCDEFLLU;
#endif

    static const UINT8 okBeit = 79;

    typedef struct _mmhandshakeheader {
        char id[sizeof(HandshakeHeaderID)];
        UINT64 endianTest;
        UINT16 coreversion[4];
    } HandshakeHeader;

} /* end namespace special */
} /* end namespace core */
} /* end namespace megamol */


/*
 * special::RenderNetUtil::DefaultPort
 */
const unsigned short special::RenderNetUtil::DefaultPort = 54321;


/*
 * special::RenderNetUtil::handshakeReceiveTimeout
 */
const int special::RenderNetUtil::handshakeReceiveTimeout = 5000;
    // vislib::net::Socket::TIMEOUT_INFINITE;


/*
 * special::RenderNetUtil::handshakeSendTimeout
 */
const int special::RenderNetUtil::handshakeSendTimeout = 5000;


/*
 * special::RenderNetUtil::HandshakeAsClient
 */
void special::RenderNetUtil::HandshakeAsClient(vislib::net::Socket& socket) {
    int position = 0;
    HandshakeHeader header;
    UINT16 u1, u2, u3, u4;

    try {
        ::memcpy(header.id, HandshakeHeaderID, sizeof(HandshakeHeaderID));
        header.endianTest = endianTestValue;
        ::mmcGetVersionInfo(&u1, &u2, &u3, &u4, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
        header.coreversion[0] = u1;
        header.coreversion[1] = u2;
        header.coreversion[2] = u3;
        header.coreversion[3] = u4;

        if (socket.Send(&header, sizeof(HandshakeHeader), handshakeSendTimeout, 0, true) != sizeof(HandshakeHeader)) {
            throw vislib::Exception("Communication size mismatch", __FILE__, __LINE__);
        }
        position = 1;

        if (socket.Receive(&header, sizeof(HandshakeHeader), handshakeReceiveTimeout, 0, true) != sizeof(HandshakeHeader)) {
            throw vislib::Exception("Communication size mismatch", __FILE__, __LINE__);
        }
        position = 2;

        if (::memcmp(header.id, HandshakeHeaderID, sizeof(HandshakeHeaderID)) != 0) {
            throw vislib::Exception("Handshake ID mismatch", __FILE__, __LINE__);
        }
        if (header.endianTest != endianTestValue) {
            throw vislib::Exception("Mixed endian unsupported", __FILE__, __LINE__);
        }
        if ((u1 != header.coreversion[0]) || (u2 != header.coreversion[1])
                || (u3 != header.coreversion[2]) || (u4 != header.coreversion[3])) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
                "MegaMol core version mismatch detected");
        }

        position = 3;
        if (socket.Send(&okBeit, 1, handshakeSendTimeout, 0, true) != 1) {
            throw vislib::Exception("Communication size mismatch", __FILE__, __LINE__);
        }

    } catch(vislib::Exception e) {
        vislib::StringA msg;
        msg.Format("Handshake (Client) failed at %d: %s\n", position, e.GetMsgA());
        throw vislib::Exception(msg.PeekBuffer(), e.GetFile(), e.GetLine());
    } catch(...) {
        vislib::StringA msg;
        msg.Format("Handshake (Client) failed at %d: Unknown error\n", position);
        throw vislib::Exception(msg.PeekBuffer(), __FILE__, __LINE__);
    }
}


/*
 * special::RenderNetUtil::HandshakeAsServer
 */
void special::RenderNetUtil::HandshakeAsServer(vislib::net::Socket& socket) {
    int position = 0;
    HandshakeHeader header;
    UINT16 u1, u2, u3, u4;
    UINT8 b;

    try {
        if (socket.Receive(&header, sizeof(HandshakeHeader), handshakeReceiveTimeout, 0, true) != sizeof(HandshakeHeader)) {
            throw vislib::Exception("Communication size mismatch", __FILE__, __LINE__);
        }
        position = 1;
        ::mmcGetVersionInfo(&u1, &u2, &u3, &u4, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);

        if (::memcmp(header.id, HandshakeHeaderID, sizeof(HandshakeHeaderID)) != 0) {
            throw vislib::Exception("Handshake ID mismatch", __FILE__, __LINE__);
        }
        if (header.endianTest != endianTestValue) {
            throw vislib::Exception("Mixed endian unsupported", __FILE__, __LINE__);
        }
        if ((u1 != header.coreversion[0]) || (u2 != header.coreversion[1])
                || (u3 != header.coreversion[2]) || (u4 != header.coreversion[3])) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
                "MegaMol core version mismatch detected");
        }

        header.coreversion[0] = u1;
        header.coreversion[1] = u2;
        header.coreversion[2] = u3;
        header.coreversion[3] = u4;

        position = 2;
        if (socket.Send(&header, sizeof(HandshakeHeader), handshakeSendTimeout, 0, true) != sizeof(HandshakeHeader)) {
            throw vislib::Exception("Communication size mismatch", __FILE__, __LINE__);
        }
        position = 3;

        if (socket.Receive(&b, 1, handshakeReceiveTimeout, 0, true) != 1) {
            throw vislib::Exception("Communication size mismatch", __FILE__, __LINE__);
        }
        position = 1;

        if (b != okBeit) {
            throw vislib::Exception("Final handshake acknowledgement failed", __FILE__, __LINE__);
        }

    } catch(vislib::Exception e) {
        vislib::StringA msg;
        msg.Format("Handshake (Server) failed at %d: %s\n", position, e.GetMsgA());
        throw vislib::Exception(msg.PeekBuffer(), e.GetFile(), e.GetLine());
    } catch(...) {
        vislib::StringA msg;
        msg.Format("Handshake (Server) failed at %d: Unknown error\n", position);
        throw vislib::Exception(msg.PeekBuffer(), __FILE__, __LINE__);
    }
}


/*
 * special::RenderNetUtil::WhoAreYou
 */
vislib::StringA special::RenderNetUtil::WhoAreYou(vislib::net::Socket& socket) {
    vislib::StringA str;
    UINT32 strLen;
    if (socket.Receive(&strLen, sizeof(UINT32), handshakeReceiveTimeout, 0, true) != sizeof(UINT32)) {
        throw vislib::Exception("Communication size mismatch", __FILE__, __LINE__);
    }
    if (socket.Receive(str.AllocateBuffer(strLen), strLen, handshakeReceiveTimeout, 0, true) != strLen) {
        throw vislib::Exception("Communication size mismatch", __FILE__, __LINE__);
    }
    return str;
}


/*
 * special::RenderNetUtil::ThisIsI
 */
void special::RenderNetUtil::ThisIsI(vislib::net::Socket& socket, const vislib::StringA& name) {
    UINT32 strLen = name.Length();
    if (socket.Send(&strLen, sizeof(UINT32), handshakeSendTimeout, 0, true) != sizeof(UINT32)) {
        throw vislib::Exception("Communication size mismatch", __FILE__, __LINE__);
    }
    if (socket.Send(name.PeekBuffer(), strLen, handshakeSendTimeout, 0, true) != strLen) {
        throw vislib::Exception("Communication size mismatch", __FILE__, __LINE__);
    }
}


/*
 * special::RenderNetUtil::MyName
 */
vislib::StringA special::RenderNetUtil::MyName(void) {
    vislib::StringA str;
    str.Format("MegaMol[%u;%s]@%s",
        static_cast<unsigned int>(vislib::sys::Process::CurrentID()),
        vislib::sys::SystemInformation::UserNameA().PeekBuffer(),
        vislib::sys::SystemInformation::ComputerNameA().PeekBuffer());
    return str;
}


/*
 * special::RenderNetUtil::ReceiveMessage
 */
void special::RenderNetUtil::ReceiveMessage(vislib::net::Socket& socket,
        special::RenderNetMsg& outMsg) {
    //vislib::sys::PerformanceCounter timer(true);

    //timer.SetMark();
    if (socket.Receive(outMsg.dat.As<void>(), RenderNetMsg::headerSize,
            vislib::net::Socket::TIMEOUT_INFINITE, 0, true) != RenderNetMsg::headerSize) {
        outMsg.SetDataSize(0); // paranoia
        throw vislib::Exception("Communication size mismatch", __FILE__, __LINE__);
    }
    SIZE_T size = outMsg.GetDataSize();
    outMsg.SetDataSize(size);

    if (size > 0) {
        if (socket.Receive(outMsg.Data(), size,
                vislib::net::Socket::TIMEOUT_INFINITE, 0, true) != size) {
            outMsg.SetDataSize(0); // paranoia
            throw vislib::Exception("Message truncated", __FILE__, __LINE__);
        }
    }
    //printf("msg receive: %f\n", vislib::sys::PerformanceCounter::ToMillis(timer.Difference()));

}


/*
 * special::RenderNetUtil::SendMessage
 */
void special::RenderNetUtil::SendMessage(vislib::net::Socket& socket,
        const special::RenderNetMsg& msg) {
    //vislib::sys::PerformanceCounter timer(true);

    //timer.SetMark();
    SIZE_T size = RenderNetMsg::headerSize + msg.GetDataSize();
    if (socket.Send(msg.dat.As<void>(), size,
            vislib::net::Socket::TIMEOUT_INFINITE, 0, true) != size) {
        throw vislib::Exception("Communication size mismatch", __FILE__, __LINE__);
    }
    //printf("msg send: %f\n", vislib::sys::PerformanceCounter::ToMillis(timer.Difference()));

}


/*
 * special::RenderNetUtil::RenderNetUtil
 */
special::RenderNetUtil::RenderNetUtil(void) {
    throw vislib::UnsupportedOperationException("RenderNetUtil::Ctor",
        __FILE__, __LINE__);
}


/*
 * special::RenderNetUtil::~RenderNetUtil
 */
special::RenderNetUtil::~RenderNetUtil(void) {
    throw vislib::UnsupportedOperationException("RenderNetUtil::Dtor",
        __FILE__, __LINE__);
}
