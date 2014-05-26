/*
 * View.coo
 *
 * Copyright (C) 2014 Visualisierungsinstitut der Universität Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include <vector>

#include "AbstractNamedObject.h"
#include "CoreInstance.h"
#ifdef WITH_MPI
#include "mpi.h"
#endif /* WITH_MPI */

#include "cluster/mpi/View.h"

#include "cluster/simple/Client.h"
#include "cluster/simple/ClientViewRegistration.h"
#include "cluster/simple/CommUtil.h"

#include "glh/glh_extensions.h"

#include "param/BoolParam.h"
#include "param/IntParam.h"
#include "param/StringParam.h"

#include "vislib/assert.h"
#include "vislib/AutoLock.h"
#include "vislib/CmdLineProvider.h"
#include "vislib/DNS.h"
#include "vislib/IPHostEntry.h"
#include "vislib/NetworkInformation.h"
#include "vislib/RawStorageSerialiser.h"
#include "vislib/ShallowSimpleMessage.h"
#include "vislib/tchar.h"
#include "vislib/Thread.h"
#include "vislib/Trace.h"
#include "vislib/UTF8Encoder.h"


#define _TRACE_INFO(fmt, ...)
#define _TRACE_MESSAGING(fmt, ...)
#define _TRACE_PACKAGING(fmt, ...)
#define _TRACE_BARRIERS(fmt, ...)

#ifndef _TRACE_INFO
#define _TRACE_INFO(fmt, ...) VLTRACE(vislib::Trace::LEVEL_INFO, fmt,\
    __VA_ARGS__)
#endif /* _TRACE_INFO */
#ifndef _TRACE_MESSAGING
#define _TRACE_MESSAGING(fmt, ...) VLTRACE(vislib::Trace::LEVEL_INFO + 10,\
    fmt, __VA_ARGS__)
#endif /* _TRACE_MESSAGING */
#ifndef _TRACE_PACKAGING
#define _TRACE_PACKAGING(fmt, ...) VLTRACE(vislib::Trace::LEVEL_INFO + 20,\
    fmt, __VA_ARGS__)
#endif /* _TRACE_PACKAGING */
#ifndef _TRACE_BARRIERS
#define _TRACE_BARRIERS(fmt, ...) VLTRACE(vislib::Trace::LEVEL_INFO + 1000,\
    fmt, __VA_ARGS__)
#endif /* _TRACE_BARRIERS */



/*
 * megamol::core::cluster::mpi::View::IsAvailable
 */
bool megamol::core::cluster::mpi::View::IsAvailable(void) {
    VLAUTOSTACKTRACE;
#ifdef WITH_MPI
    return true;
#else /* WITH_MPI */
    return false;
#endif /* WITH_MPI */
}


/*
 * megamol::core::cluster::mpi::View::View
 */
megamol::core::cluster::mpi::View::View(void) : Base1(), Base2(),
        bcastMaster(-1),
        isMpiInitialised(false),
        mpiRank(-1),
        mpiSize(-1),
        paramInitialiseMpi("initialiseMpi", "Enables the view to initialise the MPI library."),
        relayOffset(0) {
    VLAUTOSTACKTRACE;
    this->hasMasterConnection.store(false);

    this->paramInitialiseMpi << new param::BoolParam(true);
    this->paramInitialiseMpi.SetUpdateCallback(&View::onInitialiseMpiChanged);
    this->MakeSlotAvailable(&this->paramInitialiseMpi);
}


/*
 * megamol::core::cluster::mpi::View::~View
 */
megamol::core::cluster::mpi::View::~View(void) {
    VLAUTOSTACKTRACE;
    this->Release();
}


/*
 * megamol::core::cluster::mpi::View::OnDispatcherExited
 */
void megamol::core::cluster::mpi::View::OnDispatcherExited(
        vislib::net::SimpleMessageDispatcher& src) throw() {
    VLAUTOSTACKTRACE;
    _TRACE_INFO("Rank %d is no potential master any more.\n", this->mpiRank);
    this->hasMasterConnection = false;
}


/*
 * megamol::core::cluster::mpi::View::OnDispatcherStarted
 */
void megamol::core::cluster::mpi::View::OnDispatcherStarted(
        vislib::net::SimpleMessageDispatcher& src) throw() {
    VLAUTOSTACKTRACE;
    _TRACE_INFO("Rank %d is now a potential master.\n", this->mpiRank);
    this->hasMasterConnection = true;
}


/*
 * megamol::core::cluster::mpi::View::OnMessageReceived
 */
bool megamol::core::cluster::mpi::View::OnMessageReceived(
        vislib::net::SimpleMessageDispatcher& src,
        const vislib::net::AbstractSimpleMessage& msg) throw() {
    VLAUTOSTACKTRACE;

    switch (msg.GetHeader().GetMessageID()) {
        case MSG_TIMESYNC:
        case MSG_MODULGRAPH:
        case MSG_VIEWCONNECT:
        case MSG_PARAMUPDATE:
        case MSG_CAMERAUPDATE:
            //::DebugBreak();
            this->storeMessageForRelay(msg);
            break;

        case MSG_HANDSHAKE_BACK:
        case MSG_HANDSHAKE_DONE:
        case MSG_TCUPDATE:
        default:
            /* Ignore this. */
            break;
    }

    return true;
}

/*
 * megamol::core::cluster::mpi::View::Render
 */
void megamol::core::cluster::mpi::View::Render(float time, double instTime) {
    VLAUTOSTACKTRACE;
    bool canRender = false;
    view::CallRenderView *crv = nullptr;
    FrameState state;

    /* Perform some lazy initialisation (copied from simple::View). */
    if (this->viewState == ViewState::CREATED) {
        this->viewState = ViewState::LAZY_INITIALISED;
        this->initTileViewParameters();
        AbstractNamedObject *ano = this;
        while (ano != nullptr) {
            if (this->loadConfiguration(ano->Name())) break;
            ano = ano->Parent();
        }
    } /* end if (this->viewState = ViewState::CREATED) */

    /* Reset the state. */
    ::ZeroMemory(&state, sizeof(state));
    state.Time = time;
    state.InstanceTime = instTime;

    /* Ensure that we know where to get the status from. */
    canRender = this->knowsBcastMaster();
    if (canRender) {
        // We have a master, check whether it is still valid. It is OK to do
        // this everywhere, because only the data from the real master will
        // remain after the broadcast. Furthermore, nodes that are currently not
        // the master can never invalidate it.
        state.InvalidateMaster = (this->isBcastMaster()
            && !this->hasMasterConnection);
    } else {
        // We have no master, so we try to negotiate one.
        canRender = this->negotiateBcastMaster();
    }

    /* If we can render, synchronise the state now. */
    if (canRender && (this->mpiSize > 1)) {
        _TRACE_BARRIERS("Rank %d is before status synchronisation.\n",
            this->mpiRank);
#ifdef WITH_MPI
        ASSERT(this->knowsBcastMaster());

        {
            vislib::sys::AutoLock l(this->relayBufferLock);
            state.RelaySize = this->relayOffset;
            this->relayOffset = 0;  // Note: bcast master must do that in CS!

            ::MPI_Bcast(&state, sizeof(state), MPI_BYTE, this->getBcastMaster(),
                MPI_COMM_WORLD);

            if (!this->isBcastMaster()) {
                _TRACE_MESSAGING("Rank %d is preparing to receive %u bytes of "
                    "relayed messages...\n", this->mpiRank, state.RelaySize);
                this->relayBuffer.AssertSize(state.RelaySize);
            }

            if (state.RelaySize > 0) {
                _TRACE_MESSAGING("Rank %d is participating in relay of "
                    "%u bytes.\n", this->mpiRank, state.RelaySize);
                ::MPI_Bcast(static_cast<void *>(this->relayBuffer),
                    state.RelaySize, MPI_BYTE, this->getBcastMaster(),
                    MPI_COMM_WORLD);
            }

            ASSERT(this->relayOffset == 0);
        }
#endif /* WITH_MPI */

        // Post-process status
        if (!this->isBcastMaster() && (state.RelaySize > 0)) {
            this->ModuleGraphLock().LockExclusive();
            auto av = this->GetConnectedView();
            size_t offset = 0;

            while (offset < state.RelaySize) {
                vislib::net::ShallowSimpleMessage msg(this->relayBuffer.At(
                    offset));
                _TRACE_MESSAGING("Rank %d is processing relayed message %u "
                    "(%u bytes in body) at offset %u...\n", this->mpiRank,
                    msg.GetHeader().GetMessageID(),
                    msg.GetHeader().GetBodySize(),
                    offset);
                offset += msg.GetMessageSize();

                switch (msg.GetHeader().GetMessageID()) {
                    case MSG_TIMESYNC:
                        if (msg.GetBodyAs<simple::TimeSyncData>()->cnt
                                == TIMESYNCDATACOUNT) {
                            // Make the view prepare for the next graph.
                            this->DisconnectViewCall();
                            this->GetCoreInstance()->CleanupModuleGraph();
                        }
                        break;

                    case MSG_MODULGRAPH:
                        //::DebugBreak();
                        this->SetSetupMessage(msg);
                        this->processInitialisationMessage();
                        break;

                    case MSG_VIEWCONNECT: {
                        //::DebugBreak();
                        vislib::StringA name(msg.GetBodyAs<char>(),
                            msg.GetHeader().GetBodySize());
                        this->ConnectView(name);
                        // Client additionally does: this->views[0]->SetCamIniMessage();
                        } break;

                    case MSG_PARAMUPDATE: {
                        vislib::StringA name(msg.GetBodyAs<char>(),
                            msg.GetHeader().GetBodySize());
                        vislib::StringA::Size pos = name.Find('=');
                        vislib::TString value;
                        vislib::UTF8Encoder::Decode(value,
                            name.Substring(pos + 1));
                        name.Truncate(pos);
                        ////Log::DefaultLog.WriteInfo("Setting Parameter %s to %s\n", name.PeekBuffer(), vislib::StringA(value).PeekBuffer());
                        param::ParamSlot *ps = dynamic_cast<param::ParamSlot*>(
                            this->FindNamedObject(name, true));
                        if (ps != nullptr) {
                            ps->Param<param::AbstractParam>()->ParseValue(value);
                        }
                        } break;

                    case MSG_CAMERAUPDATE:
                        if ((av != nullptr) 
                                && (msg.GetHeader().GetBodySize() > 0)) {
                            vislib::RawStorageSerialiser ser(
                                msg.GetBodyAs<BYTE>(),
                                msg.GetHeader().GetBodySize());
                            av->DeserialiseCamera(ser);
                        }
                        break;
                } /* end switch (msg.GetHeader().GetMessageID()) */
            } /* end while (offset < state.RelaySize) */

            this->ModuleGraphLock().UnlockExclusive();

            _TRACE_MESSAGING("Rank %d has processed all relayed messages.\n",
                this->mpiRank);
        } /* end if (!this->isBcastMaster() && (state.RelaySize > 0)) */

        if (state.InvalidateMaster) {
            this->bcastMaster = -1;
            _TRACE_INFO("Rank %d invalidated the broadcast master.\n",
                this->mpiRank);
        }
    } /* end if (!this->isBcastMaster() && (state.RelaySize > 0)) */

    this->processInitialisationMessage();
    this->registerClient(true);

    /* Ensure that we have a rendering call that we can execute. */
    if (canRender) {
        crv = this->getCallRenderView();
        canRender = (crv != nullptr);
    }

    /* Render the view if any; do fallback rendering otherwise. */
    if (canRender) {
        ASSERT(crv != nullptr);
        this->checkParameters();

        crv->ResetAll();
        crv->SetTime(state.Time);
        crv->SetInstanceTime(state.InstanceTime);
        crv->SetProjection(this->getProjType(), this->getEye());

        if (this->hasTile()) {
            //::DebugBreak();
            crv->SetTile(this->getVirtWidth(), this->getVirtHeight(),
                this->getTileX(), this->getTileY(),
                this->getTileW(), this->getTileH());
        }

        crv->SetOutputBuffer(GL_BACK, this->getViewportWidth(),
            this->getViewportHeight());

        //view::AbstractView *view = NULL;
        //if (crv->PeekCalleeSlot() != NULL) view = dynamic_cast<view::AbstractView*>(
        //        const_cast<AbstractNamedObject*>(crv->PeekCalleeSlot()->Parent()));
        //if (view != NULL){
        //    if (this->frozenCam != NULL) view->DeserialiseCamera(*this->frozenCam);
        //    /* this forces to use this time */
        //    //view->SetFrameTime(static_cast<float>(this->frozenTime));
        //}

        //{
        //    vislib::sys::AutoLock lock(renderLock);

        if (!(*crv)(view::CallRenderView::CALL_RENDER)) {
            this->renderFallbackView();
        }

        ::glFlush();
#ifdef WITH_MPI
        _TRACE_BARRIERS("Rank %d is before swap barrier.\n", this->mpiRank);
        ::MPI_Barrier(MPI_COMM_WORLD);
#endif /* WITH_MPI */

    } else {
        this->renderFallbackView();
    } /* end if (canRender) */
}


/*
 * megamol::core::cluster::mpi::View::create
 */
bool megamol::core::cluster::mpi::View::create(void) {
    VLAUTOSTACKTRACE;
    bool retval = Base1::create();
    this->viewState = ViewState::CREATED;

    if (retval) {
        this->initialiseMpi();
    }

    return retval;
}


/*
 * megamol::core::cluster::mpi::View::finaliseMpi
 */
void megamol::core::cluster::mpi::View::finaliseMpi(void) {
    VLAUTOSTACKTRACE;
    if (this->isMpiInitialised) {
#ifdef WITH_MPI
        ::MPI_Finalize();
#endif /* WITH_MPI */
    }
}


/*
 * megamol::core::cluster::mpi::View::initialiseMpi
 */
bool megamol::core::cluster::mpi::View::initialiseMpi(void) {
    VLAUTOSTACKTRACE;
    bool isInit = this->paramInitialiseMpi.Param<param::BoolParam>()->Value();
    bool retval = !isInit;

    if (isInit) {
        retval = this->isMpiInitialised;
        if (!retval) {
#ifdef WITH_MPI
#ifdef _WIN32
            vislib::sys::CmdLineProviderA cmdLine(::GetCommandLineA());
            int argc = cmdLine.ArgC();
            char **argv = cmdLine.ArgV();
            ::MPI_Init(&argc, &argv);
            vislib::sys::Log::DefaultLog.WriteInfo(_T("MPI was initialised ")
                _T("by module %hs."), View::ClassName());
#else /* _WIN32 */
            vislib::sys::Log::DefaultLog.WriteError(_T("MPI cannot be ")
                _T("initialised lazily on platforms other than Windows. ")
                _T("Please initialise MPI before using this module.");
#endif /* _WIN32 */

            ::MPI_Comm_rank(MPI_COMM_WORLD, &this->mpiRank);
            ::MPI_Comm_size(MPI_COMM_WORLD, &this->mpiSize);
            vislib::sys::Log::DefaultLog.WriteInfo(_T("This view is %d of %d."),
                this->mpiRank, this->mpiSize);

#endif /* WITH_MPI */
        }
    }

    // TODO: Register data types as necessary

    return retval;
}


/*
 * megamol::core::cluster::mpi::View::negotiateBcastMaster
 */
bool megamol::core::cluster::mpi::View::negotiateBcastMaster(void) {
    VLAUTOSTACKTRACE;
    ASSERT(this->mpiRank >= 0);
    ASSERT(this->mpiSize > 0);
    ASSERT(this->bcastMaster < 0);

#if WITH_MPI
    std::vector<int> responses(this->mpiSize);

    //VLTRACE(vislib::Trace::LEVEL_INFO, "Negotiating master of %d "
    //    "ranks...\n", this->mpiSize);
    int myResponse = static_cast<int>(this->hasMasterConnection);
    ::MPI_Allgather(&myResponse, 1, MPI_INT, responses.data(), 1, MPI_INT,
        MPI_COMM_WORLD);

    auto end = responses.end();
    int master = 0;
    for (auto it = responses.begin(); it != end; ++it) {
        if (*it > 0) {
            // Use the first node that claims to have a connection.
            this->bcastMaster = master;
            _TRACE_INFO("Rank %d is the first having a connection to the "
                "controller.\n", this->bcastMaster);
            return true;
        }
        ++master;
    }
#endif /* WITH_MPI */

    return false;
}


/*
 * megamol::core::cluster::mpi::View::onInitialiseMpiChanged
 */
bool megamol::core::cluster::mpi::View::onInitialiseMpiChanged(
        param::ParamSlot& slot) {
    VLAUTOSTACKTRACE;
    bool isInit = this->paramInitialiseMpi.Param<param::BoolParam>()->Value();
    if (isInit) {
        this->initialiseMpi();
    } else {
        this->finaliseMpi();
    }
    return true;
}


/*
 * megamol::core::cluster::mpi::View::release
 */
void megamol::core::cluster::mpi::View::release(void) {
    VLAUTOSTACKTRACE;
    this->finaliseMpi();
    Base1::release();
}


/*
 * megamol::core::cluster::mpi::View::storeMessageForRelay
 */
void megamol::core::cluster::mpi::View::storeMessageForRelay(
        const vislib::net::AbstractSimpleMessage& msg) {
    VLAUTOSTACKTRACE;
    if (this->isBcastMaster()) {
// if (this->relayOffset == 8) ::DebugBreak();
//if (msg.GetHeader().GetBodySize() == 0) ::DebugBreak();
        vislib::sys::AutoLock l(this->relayBufferLock);
        size_t msgSize = msg.GetMessageSize();
        _TRACE_PACKAGING("Rank %d is storing message %u (%u bytes in body, "
            "%u bytes in total) for relaying at offset %u...\n", this->mpiRank,
            msg.GetHeader().GetMessageID(), msg.GetHeader().GetBodySize(),
            msgSize, this->relayOffset);

        this->relayBuffer.AssertSize(this->relayOffset + msgSize, true);
        ::memcpy(this->relayBuffer.At(this->relayOffset),
            static_cast<const void *>(msg), msgSize);
        this->relayOffset += msgSize;
    }
}


#if 0
/*
 * megamol::core::cluster::mpi::View::Render
 */
void megamol::core::cluster::mpi::View::Render(float time, double instTime) {
    if (this->firstFrame) {
        this->firstFrame = false;
        this->initTileViewParameters();
        AbstractNamedObject *ano = this;
        while (ano != NULL) {
            if (this->loadConfiguration(ano->Name())) break;
            ano = ano->Parent();
        }
        if (this->GetCoreInstance()->Configuration().IsConfigValueSet("scv-heartbeat-port")) {
            try {
                this->heartBeatPortSlot.Param<param::IntParam>()->SetValue(
                    vislib::CharTraitsW::ParseInt(
                        this->GetCoreInstance()->Configuration().ConfigValue("scv-heartbeat-port")));
            } catch(vislib::Exception e) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "Failed to load heartbeat port configuration: %s [%s, %d]\n",
                    e.GetMsgA(), e.GetFile(), e.GetLine());
            } catch(...) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "Failed to load heartbeat port configuration: Unknown exception\n");
            }
        }
        if (this->GetCoreInstance()->Configuration().IsConfigValueSet("scv-heartbeat-server")) {
            this->heartBeatServerSlot.Param<param::StringParam>()->SetValue(
                this->GetCoreInstance()->Configuration().ConfigValue("scv-heartbeat-server"));
        }
    }

    if (this->initMsg != NULL) {
        if (this->initMsg->GetHeader().GetMessageID() == MSG_MODULGRAPH) {
            this->GetCoreInstance()->SetupGraphFromNetwork(this->initMsg);
            this->client->ContinueSetup();
        } else {
            this->directCamSyncUpdated(this->directCamSyncSlot);
            if (this->initMsg->GetHeader().GetMessageID() == MSG_CAMERAUPDATE) {
                this->client->ContinueSetup(2);
            }
        }
        SAFE_DELETE(this->initMsg);
    }

    if (this->client == NULL) {
        ClientViewRegistration *sccvr = this->registerSlot.CallAs<ClientViewRegistration>();
        if (sccvr != NULL) {
            sccvr->SetView(this);
            if ((*sccvr)()) {
                this->client = sccvr->GetClient();
            }
        }
    }

    if (this->heartBeatPortSlot.IsDirty() || this->heartBeatServerSlot.IsDirty()) {
        this->heartBeatPortSlot.ResetDirty();
        this->heartBeatServerSlot.ResetDirty();

        try {
            this->heartbeat.Connect(
                this->heartBeatServerSlot.Param<param::StringParam>()->Value(),
                static_cast<unsigned int>(this->heartBeatPortSlot.Param<param::IntParam>()->Value()));

        } catch(vislib::Exception e) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Failed to configure heartbeat: %s [%s, %d]\n",
                e.GetMsgA(), e.GetFile(), e.GetLine());
        } catch(...) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Failed to configure heartbeat: Unknown exception\n");
        }
    }

    bool heartbeatOn = false;
    bool doSecondHeartbeat = false;
    try {
        heartbeatOn = this->heartbeat.Sync(1, this->heartbeatPayload);
    } catch(...) {
        heartbeatOn = false;
        doSecondHeartbeat = true;
    }
    if (heartbeatOn) {
        ASSERT(this->heartbeatPayload.GetSize() >= 13);
        unsigned char c = *this->heartbeatPayload.As<unsigned char>();
        doSecondHeartbeat = ((c & 0x01) == 0x01);
        instTime = *this->heartbeatPayload.AsAt<double>(1);
        time = *this->heartbeatPayload.AsAt<float>(1 + sizeof(double));
        view::AbstractView *view = this->GetConnectedView();
        if ((this->heartbeatPayload.GetSize() > 13) && (view != NULL)) {
            vislib::RawStorageSerialiser ser(&this->heartbeatPayload, 1 + sizeof(double) + sizeof(float));
            view->DeserialiseCamera(ser);
        }
    } else {
        doSecondHeartbeat = true;
    }

    view::CallRenderView *crv = this->getCallRenderView();
    this->checkParameters();

    if (!this->frozen) {
        this->frozenTime = instTime;
    }

    if (crv != NULL) {
        crv->ResetAll();
        crv->SetTime(time);
        crv->SetInstanceTime(instTime);
        crv->SetProjection(this->getProjType(), this->getEye());
        if ((this->getVirtWidth() != 0) && (this->getVirtHeight() != 0)
                && (this->getTileW() != 0) && (this->getTileH() != 0)) {
            crv->SetTile(this->getVirtWidth(), this->getVirtHeight(),
                this->getTileX(), this->getTileY(), this->getTileW(), this->getTileH());
        }
        crv->SetOutputBuffer(GL_BACK, this->getViewportWidth(), this->getViewportHeight());

        //if ((this->netVSyncBarrier != NULL) && (this->netVSyncBarrier->GetDataSize() > 0)) {
        //    //printf("Barrier with %u bytes data\n", this->netVSyncBarrier->GetDataSize());
        //    vislib::RawStorageSerialiser camera(
        //        this->netVSyncBarrier->GetData() + 4,
        //        this->netVSyncBarrier->GetDataSize() - 4);
        //}
        view::AbstractView *view = NULL;
        if (crv->PeekCalleeSlot() != NULL) view = dynamic_cast<view::AbstractView*>(
                const_cast<AbstractNamedObject*>(crv->PeekCalleeSlot()->Parent()));
        if (view != NULL){
            if (this->frozenCam != NULL) view->DeserialiseCamera(*this->frozenCam);
            /* this forces to use this time */
            //view->SetFrameTime(static_cast<float>(this->frozenTime));
        }

        {
            vislib::sys::AutoLock lock(renderLock);

            if (!(*crv)(view::CallRenderView::CALL_RENDER)) {
                this->renderFallbackView();
            }

        }

    } else {
        this->renderFallbackView();

    }

    ::glFlush();

    if (doSecondHeartbeat) {
        try {
            this->heartbeat.Sync(2, this->heartbeatPayload);
        } catch(...) {
        }
    }

#if 0 // TODO: activate with something else
    // HAZARD: requires a second message to ensure all nodes synchronize at the same point!!!
    // sync with second heartbeat ping 
    heartbeatOn = false;
    try {
        heartbeatOn = this->heartbeat.Sync(this->heartbeatPayload);
    } catch(...) {
        heartbeatOn = false;
    }
    if (heartbeatOn) {
        ASSERT(this->heartbeatPayload.GetSize() >= 12);
        instTime = *this->heartbeatPayload.As<double>();
        time = *this->heartbeatPayload.AsAt<float>(sizeof(double));
        view::AbstractView *view = this->GetConnectedView();
        if ((this->heartbeatPayload.GetSize() > 12) && (view != NULL)) {
            vislib::RawStorageSerialiser ser(&this->heartbeatPayload, sizeof(double) + sizeof(float));
            view->DeserialiseCamera(ser);
        }
    }
#endif

}
#endif
