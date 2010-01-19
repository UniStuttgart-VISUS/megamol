/*
 * RenderSlave.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "RenderSlave.h"
#ifdef _WIN32
#include <windows.h>
#endif /* _WIN32 */
#include <climits>
#include <GL/gl.h>
#include <GL/glu.h>
#include "CoreInstance.h"
#include "CallDescriptionManager.h"
#include "ModuleDescriptionManager.h"
#include "param/BoolParam.h"
#include "param/IntParam.h"
#include "param/StringParam.h"
#include "special/CallRegisterAtController.h"
#include "special/ClusterSignRenderer.h"
#include "special/RenderNetUtil.h"
#include "view/CallRenderView.h"
#include "vislib/assert.h"
#include "vislib/AutoLock.h"
#ifndef USE_INTERLOCKED_WORKAROUND
#include "vislib/Interlocked.h"
#endif /* !USE_INTERLOCKED_WORKAROUND */
#include "vislib/Map.h"
#include "vislib/NetworkInformation.h"
#include "vislib/RawStorage.h"
#include "vislib/RawStorageSerialiser.h"
#include "vislib/StringConverter.h"
#include "vislib/UnsupportedOperationException.h"
#include "vislib/UTF8Encoder.h"

using namespace megamol::core;


/*
 * special::RenderSlave::Receiver::Receiver
 */
special::RenderSlave::Receiver::Receiver(void) : vislib::sys::Runnable(),
        owner(NULL), socket(NULL) {
    // intentionally empty
}


/*
 * special::RenderSlave::Receiver::~Receiver
 */
special::RenderSlave::Receiver::~Receiver(void) {
    ASSERT(this->socket == NULL);
}


/*
 * special::RenderSlave::Receiver::Setup
 */
void special::RenderSlave::Receiver::Setup(special::RenderSlave *owner, vislib::net::Socket *socket) {
    ASSERT((this->owner == NULL) || (this->owner == owner));
    ASSERT(this->socket == NULL);
    this->owner = owner;
    this->socket = socket;
}


/*
 * special::RenderSlave::Receiver::Run
 */
DWORD special::RenderSlave::Receiver::Run(void *userData) {
    ASSERT(this->owner != NULL);
    ASSERT(this->socket != NULL);
    vislib::net::Socket *s = NULL;
    RenderNetMsg msg;

    try {

        s = this->socket;
        if (s != NULL) {
            msg.SetType(RenderNetUtil::MSGTYPE_REQUEST_TIMESYNC);
            msg.SetID(0);
            msg.SetDataSize(0);
            RenderNetUtil::SendMessage(*s, msg);
        }

        while (true) {
            s = this->socket;
            if (s == NULL) break;
            RenderNetUtil::ReceiveMessage(*s, msg);

            if (!this->owner->HandleMessage(msg)) {
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_WARN,
                    "Unhandled Message (%u; %u; %u)",
                    static_cast<unsigned int>(msg.GetType()),
                    static_cast<unsigned int>(msg.GetID()),
                    static_cast<unsigned int>(msg.GetDataSize()));
            }
        }

    } catch(...) {
    }

    this->owner->serverConnectedSlot.Param<param::BoolParam>()->SetValue(false);
    this->owner->closeConnection(false);
    this->socket = NULL;
    return 0;
}


/*
 * special::RenderSlave::Receiver::Terminate
 */
bool special::RenderSlave::Receiver::Terminate(void) {
    this->socket = NULL; // signal termination request
    return true;
}

/****************************************************************************/


/*
 * special::RenderSlave::~RenderSlave
 */
special::RenderSlave::~RenderSlave(void) {
    this->Release();
    delete const_cast<vislib::sys::Runnable*>(
        this->receiveThread.GetRunnable());
    SAFE_DELETE(this->viewDesc);
}


/*
 * special::RenderSlave::Render
 */
void special::RenderSlave::Render(void) {
    using vislib::sys::Log;

    // update network connection ?
    if (this->serverAddrSlot.IsDirty()
            || this->serverConnectedSlot.IsDirty()
            || this->serverPortSlot.IsDirty()) {
        this->serverAddrSlot.ResetDirty();
        this->serverConnectedSlot.ResetDirty();
        this->serverPortSlot.ResetDirty();

        this->closeConnection();

        if (this->serverConnectedSlot.Param<param::BoolParam>()->Value()) {
            vislib::net::IPEndPoint ipep;

            vislib::TString epStr;
            epStr.Format(_T("%s:%d"), 
                this->serverAddrSlot.Param<param::StringParam>()->Value().PeekBuffer(),
                this->serverPortSlot.Param<param::IntParam>()->Value());

            float wildness = vislib::net::NetworkInformation::GuessRemoteEndPoint(ipep, epStr);

            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO,
                "Trying to connect to master at %s (from %s; wildness %.2f)\n",
                ipep.ToStringA().PeekBuffer(), vislib::StringA(epStr).PeekBuffer(), wildness);

            try {
                this->socket.Create(ipep,
                    vislib::net::Socket::TYPE_STREAM,
                    vislib::net::Socket::PROTOCOL_TCP);
                this->socket.SetNoDelay(true);
                this->socket.Connect(ipep);
                // connection established under reserve, perform handshake
                RenderNetUtil::HandshakeAsClient(this->socket);
                RenderNetUtil::ThisIsI(this->socket, RenderNetUtil::MyName());
                Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100, "Handshake complete.\n");

                Receiver *rcvr = dynamic_cast<Receiver *>(
                    const_cast<vislib::sys::Runnable *>(
                    this->receiveThread.GetRunnable()));

                rcvr->Setup(this, &this->socket);
                this->receiveThread.Start();

            } catch(vislib::Exception e) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "Unable to connect to master: %s\n",
                    e.GetMsgA());
                this->serverConnectedSlot.Param<param::BoolParam>()->SetValue(false);

            } catch(...) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "Unable to connect to master: Unknown error\n");
                this->serverConnectedSlot.Param<param::BoolParam>()->SetValue(false);

            }
        }
    }

    // update scene view
    ViewDescription *vd = NULL;
#ifndef USE_INTERLOCKED_WORKAROUND
    vd = reinterpret_cast<ViewDescription*>(vislib::sys::Interlocked::Exchange(
        reinterpret_cast<INT_PTR*>(&this->viewDesc), reinterpret_cast<INT_PTR>(vd)));
#else /* USE_INTERLOCKED_WORKAROUND */
    this->critSecInterlocked.Lock();
    vd = this->viewDesc;
    this->viewDesc = NULL;
    this->critSecInterlocked.Unlock();
#endif /* USE_INTERLOCKED_WORKAROUND */
    if (vd != NULL) {
        // initialize new scene

        // disconnects old view modules
        this->renderViewSlot.ConnectCall(NULL);
        this->GetCoreInstance()->CleanupModuleGraph();

        view::AbstractView * view
            = this->GetCoreInstance()->instantiateSubView(vd);
        Module *viewMod = dynamic_cast<Module *>(view);
        if (viewMod != NULL) {
            CalleeSlot *s1 = dynamic_cast<CalleeSlot*>(viewMod->FindSlot("renderView"));
            if (s1 != NULL) {
                Call *crv = CallDescriptionManager::Instance()->Find(
                    view::CallRenderView::ClassName())->CreateCall();
                s1->ConnectCall(crv);
                this->renderViewSlot.ConnectCall(crv);
            }
        }
        delete vd;
        this->GetCoreInstance()->CleanupModuleGraph();
    }

    view::CallRenderView *crv = this->renderViewSlot.CallAs<view::CallRenderView>();
    if (crv == NULL) {
        // no view is connected so we render a glyph representing the cluster
        // connection status
        ::glViewport(0, 0, this->viewportWidth, this->viewportHeight);
        bool isStereo = (this->plane.Type() != ClusterDisplayPlane::TYPE_VOID)
            ? ((this->plane.Type() == ClusterDisplayPlane::TYPE_STEREO_LEFT)
            || (this->plane.Type() == ClusterDisplayPlane::TYPE_STEREO_RIGHT))
            : false;
        bool isRightEye = (this->plane.Type() != ClusterDisplayPlane::TYPE_VOID)
            ? (this->plane.Type() == ClusterDisplayPlane::TYPE_STEREO_RIGHT)
            : false;

        if (this->receiveThread.IsRunning()) {
            special::ClusterSignRenderer::RenderYes(this->viewportWidth,
                this->viewportHeight, isStereo, isRightEye);
        } else {
            special::ClusterSignRenderer::RenderNo(this->viewportWidth,
                this->viewportHeight, isStereo, isRightEye);
        }

        return;
    }

    crv->ResetAll();
    if (this->plane.Type() != ClusterDisplayPlane::TYPE_VOID) {
        crv->SetProjection(
            ((this->plane.Type() == ClusterDisplayPlane::TYPE_STEREO_LEFT)
            || (this->plane.Type() == ClusterDisplayPlane::TYPE_STEREO_RIGHT))
                ? vislib::graphics::CameraParameters::STEREO_OFF_AXIS
                : vislib::graphics::CameraParameters::MONO_PERSPECTIVE,
            (this->plane.Type() == ClusterDisplayPlane::TYPE_STEREO_LEFT)
                ? vislib::graphics::CameraParameters::LEFT_EYE
                : vislib::graphics::CameraParameters::RIGHT_EYE);
        crv->SetTile(this->plane.Width(), this->plane.Height(),
            this->tile.X(), this->tile.Y(), this->tile.Width(), this->tile.Height());
    }
    crv->SetViewportSize(this->viewportWidth, this->viewportHeight);
    (*crv)();
}


/*
 * special::RenderSlave::Resize
 */
void special::RenderSlave::Resize(unsigned int width, unsigned int height) {
    this->viewportWidth = width;
    if (this->viewportWidth < 1) this->viewportWidth = 1;
    this->viewportHeight = height;
    if (this->viewportHeight < 1) this->viewportHeight = 1;
}


/*
 * special::RenderSlave::onRenderView
 */
bool special::RenderSlave::onRenderView(Call& call) {
    throw vislib::UnsupportedOperationException(
        "RenderSlave::onRenderView", __FILE__, __LINE__);
    return false;
}


/*
 * special::RenderSlave::create
 */
bool special::RenderSlave::create(void) {
    vislib::net::Socket::Startup();
    return true;
}


/*
 * special::RenderSlave::release
 */
void special::RenderSlave::release(void) {
    CallRegisterAtController *c2 = this->controllerSlot.CallAs<CallRegisterAtController>();
    if (c2 != NULL) {
        c2->SetClient(this);
        (*c2)(1);
    }
    this->closeConnection();
    vislib::net::Socket::Cleanup();
}


/*
 * special::RenderSlave::OnConnect
 */
void special::RenderSlave::OnConnect(AbstractSlot& slot) {
    if (&slot == &this->controllerSlot) {
        // connect to controller
        CallRegisterAtController *c = this->controllerSlot.CallAs<CallRegisterAtController>();
        if (c != NULL) {
            c->SetClient(this);
            (*c)(0);
        }
    }
}


/*
 * special::RenderSlave::RenderSlave
 */
special::RenderSlave::RenderSlave(void) : view::AbstractView(),
        Module(), ClusterControllerClient(), AbstractSlot::Listener(),
        viewportWidth(1), viewportHeight(1),
        plane(ClusterDisplayPlane::TYPE_VOID), tile(),
        controllerSlot("controller", "Slot connecting the display to the cluster controller"),
        renderViewSlot("renderView", "Slot connecting to the view to be rendered"),
        serverAddrSlot("serverAddr", "The server network address"),
        serverPortSlot("serverPort", "The server network port"),
        serverConnectedSlot("serverConnection", "The state of the server connection"),
        closeOnDisconnectSlot("closeOnDisconnect", "Controls if the view should close if the connection to the server closes"),
        socket(), receiveThread(new Receiver()), viewDesc(NULL) {

    this->controllerSlot.SetCompatibleCall<CallRegisterAtControllerDescription>();
    this->controllerSlot.AddListener(this);
    this->MakeSlotAvailable(&this->controllerSlot);

    this->renderViewSlot.SetCompatibleCall<view::CallRenderViewDescription>();
    this->MakeSlotAvailable(&this->renderViewSlot);

    this->serverAddrSlot << new param::StringParam("");
    this->MakeSlotAvailable(&this->serverAddrSlot);

    // Note: Port minval should be 49152, which is the lowest port for the
    //  private dynamic ip port range. However, we do not want to enforce
    //  using only these ports.
    this->serverPortSlot << new param::IntParam(
        RenderNetUtil::DefaultPort, 1, USHRT_MAX);
    this->MakeSlotAvailable(&this->serverPortSlot);

    this->serverConnectedSlot << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->serverConnectedSlot);

    this->closeOnDisconnectSlot << new param::BoolParam(true);
    this->MakeSlotAvailable(&this->closeOnDisconnectSlot);
}


/*
 * special::RenderSlave::setClusterDisplayTile
 */
void special::RenderSlave::setClusterDisplayTile(
        const special::ClusterDisplayPlane &plane,
        const special::ClusterDisplayTile &tile) {
    this->plane = plane;
    this->tile = tile;
}


/*
 * special::RenderSlave::resetClusterDisplayTile
 */
void special::RenderSlave::resetClusterDisplayTile(void) {
    ClusterDisplayPlane cdp(ClusterDisplayPlane::TYPE_VOID);
    this->plane = cdp;
}


/*
 * special::RenderSlave::HandleMessage
 */
bool special::RenderSlave::HandleMessage(special::RenderNetMsg &msg) {
    using vislib::sys::Log;

    switch (msg.GetType()) {
        case RenderNetUtil::MSGTYPE_TIMESYNC: {
            if (msg.GetDataSize() != sizeof(RenderNetUtil::TimeSyncData)) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
                    "Message data size mismatch. Ignoring message %u.\n", msg.GetID());
                return false;
            }

            RenderNetUtil::TimeSyncData *tsd = msg.DataAs<RenderNetUtil::TimeSyncData>();
            if (tsd->trip < RenderNetUtil::TimeSyncTripCnt) {
                RenderNetUtil::SendMessage(this->socket, msg);

            } else {
                // finally!
                double srvrTime;
                double now = this->GetCoreInstance()->GetInstanceTime();
                double latency = 0.0;

                for (unsigned int i = 0; i < RenderNetUtil::TimeSyncTripCnt - 1; i++) {
                    tsd->srvrTimes[i] = tsd->srvrTimes[i + 1] - tsd->srvrTimes[i];
                    latency += tsd->srvrTimes[i];
                    //printf("Dbl-Latency: %f\n", tsd->srvrTimes[i]);

                }
                latency /= static_cast<double>(RenderNetUtil::TimeSyncTripCnt - 1);
                //printf("Dbl-Latency*: %f\n", latency);
                latency *= 0.5; // because it is roundtrip latency
                //printf("Latency*: %f\n", latency);

                // the current time on the server
                srvrTime = tsd->srvrTimes[RenderNetUtil::TimeSyncTripCnt - 1] + latency;

                this->GetCoreInstance()->OffsetInstanceTime(srvrTime - now);
                // now time on client and server are (quite) synchrone
                Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100, "Core-Time synchronized.");

                // disconnects view modules
                this->renderViewSlot.ConnectCall(NULL);
                this->GetCoreInstance()->CleanupModuleGraph();

                // as next step of initialisation request module graph synchronisation
                RenderNetMsg nextMsg(RenderNetUtil::MSGTYPE_REQUEST_MODGRAPHSYNC, 0, 0);
                RenderNetUtil::SendMessage(this->socket, nextMsg);

            }
            return true;
        }
        case RenderNetUtil::MSGTYPE_SETUP_MODGRAPH: {
            try {
                this->setupModuleGraph(msg.Data(), msg.GetDataSize());

                // TODO: Next initialization schtepp

            } catch(vislib::Exception e) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "Unable to setup module graph: %s\n", e.GetMsgA());

                // disconnects view modules
                this->renderViewSlot.ConnectCall(NULL);
                this->GetCoreInstance()->CleanupModuleGraph();
            } catch(...) {
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "Unable to setup module graph: Unknown exception\n");

                // disconnects view modules
                this->renderViewSlot.ConnectCall(NULL);
                this->GetCoreInstance()->CleanupModuleGraph();
            }
            return true;
        }
    }

    // TODO: Implement

    return false;
}


/*
 * special::RenderSlave::UpdateFreeze
 */
void special::RenderSlave::UpdateFreeze(bool freeze) {
    view::CallRenderView *crv = this->renderViewSlot.CallAs<view::CallRenderView>();
    if (crv == NULL) return;
    (*crv)(freeze ? 1 : 2);
}


/*
 * special::RenderSlave::closeConnection
 */
void special::RenderSlave::closeConnection(bool andThread) {
    using vislib::sys::Log;
    if (this->socket.IsValid()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 250, "Closing socket");
        try {
            this->socket.Shutdown();
        } catch(...) {
        }
        try {
            this->socket.Close();
        } catch(...) {
        }
    }
    if (this->receiveThread.IsRunning()) {

        if (andThread) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 250,
                "Terminating receiver thread");
            this->receiveThread.Terminate(false);
        }

        // on disconnection

        // disconnects view modules
        this->renderViewSlot.ConnectCall(NULL);
        this->GetCoreInstance()->CleanupModuleGraph();

        if (this->closeOnDisconnectSlot.Param<param::BoolParam>()->Value()) {
            // shutdown view instance (or all instances)
            this->LockModuleGraph(true);
            AbstractNamedObject *ano = this;
            while (ano != NULL) {
                ViewInstance *vi = dynamic_cast<ViewInstance *>(ano);
                if (vi != NULL) {
                    vi->RequestClose();
                    break;
                }
                ano = ano->Parent();
            }
            this->UnlockModuleGraph();
        }

    }

}


/*
 * special::RenderSlave::setupModuleGraph
 */
void special::RenderSlave::setupModuleGraph(const void* data, SIZE_T size) {
    AbstractNamedObject::GraphLocker locker(this, true);
    vislib::sys::AutoLock lock(locker);

    // disconnect view (paranoia, since it should already have been done)
    this->renderViewSlot.ConnectCall(NULL);
    this->GetCoreInstance()->CleanupModuleGraph();

    if (size < sizeof(UINT32)) return; // empty scene ... oh, well, okey.
    vislib::RawStorageSerialiser reader(static_cast<const BYTE*>(data), size);

    vislib::Map<UINT32, vislib::StringA> modules;
    UINT32 modCnt;
    UINT32 callCnt;
    ViewDescription *vd = new ViewDescription(NULL);
    vislib::StringA viewName;

    reader >> viewName >> modCnt;
    vd->SetViewModuleID(viewName);
    //printf("Deserializing %u modules\n", modCnt);
    for (UINT32 mi = 0; mi < modCnt; mi++) {
        vislib::StringA className;
        vislib::StringA fullName;
        UINT32 paramCnt;

        reader >> className >> fullName >> paramCnt;
        vd->AddModule(ModuleDescriptionManager::Instance()->Find(className), fullName);
        modules[mi] = fullName;
        //printf("  %u: %s as %s\n", mi, className.PeekBuffer(), fullName.PeekBuffer());

        for (UINT32 pi = 0; pi < paramCnt; pi++) {
            vislib::StringA paramName;
            vislib::StringA paramValueUTF8;
            vislib::TString paramValue;

            reader >> paramName >> paramValueUTF8;
            if (!vislib::UTF8Encoder::Decode(paramValue, paramValueUTF8)) continue;
            vd->AddParamValue(fullName + "::" + paramName, paramValue);
            //printf("    param %s <= %s\n", paramName.PeekBuffer(), paramValueUTF8.PeekBuffer());

        }

    }

    reader >> callCnt;
    for (UINT32 ci = 0; ci < callCnt; ci++) {
        vislib::StringA className;
        UINT32 callerModID;
        vislib::StringA callerSlotName;
        UINT32 calleeModID;
        vislib::StringA calleeSlotName;

        reader >> className >> callerModID >> callerSlotName >> calleeModID >> calleeSlotName;
        vd->AddCall(CallDescriptionManager::Instance()->Find(className),
            modules[callerModID] + "::" + callerSlotName,
            modules[calleeModID] + "::" + calleeSlotName);
        //printf("  call %s [%u]::%s => [%u]::%s\n", className.PeekBuffer(),
        //    callerModID, callerSlotName.PeekBuffer(),
        //    calleeModID, calleeSlotName.PeekBuffer());
    }

#ifndef USE_INTERLOCKED_WORKAROUND
    vd = reinterpret_cast<ViewDescription*>(vislib::sys::Interlocked::Exchange(
        reinterpret_cast<INT_PTR*>(&this->viewDesc), reinterpret_cast<INT_PTR>(vd)));
#else /* USE_INTERLOCKED_WORKAROUND */
    this->critSecInterlocked.Lock();
    ViewDescription *vdt = vd;
    vd = this->viewDesc;
    this->viewDesc = vdt;
    this->critSecInterlocked.Unlock();
#endif /* USE_INTERLOCKED_WORKAROUND */
    delete vd;
}
