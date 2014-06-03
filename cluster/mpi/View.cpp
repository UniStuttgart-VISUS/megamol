/*
 * View.cpp
 *
 * Copyright (C) 2014 Visualisierungsinstitut der Universität Stuttgart.
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <string>
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
#include "vislib/SystemInformation.h"
#include "vislib/tchar.h"
#include "vislib/Thread.h"
#include "vislib/Trace.h"
#include "vislib/UTF8Encoder.h"


#define _TRACE_INFO(fmt, ...)
#define _TRACE_MESSAGING(fmt, ...)
//#define _TRACE_PACKAGING(fmt, ...)
#define _TRACE_BARRIERS(fmt, ...)
#define _TRACE_GSYNC(fmt, ...)
#define _TRACE_LOCKS(fmt, ...)

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
#ifndef _TRACE_GSYNC
#define _TRACE_GSYNC(fmt, ...) VLTRACE(vislib::Trace::LEVEL_INFO + 2000,\
    fmt, __VA_ARGS__)
#endif /* _TRACE_GSYNC */
#ifndef _TRACE_LOCKS
#define _TRACE_LOCKS(fmt, ...) VLTRACE(vislib::Trace::LEVEL_INFO + 5000,\
    fmt, __VA_ARGS__)
#endif /* _TRACE_LOCKS */

#define _TRACE_ACQUIRE_LOCK(name) _TRACE_LOCKS("Rank %d acquiring "\
    name " lock for thread [%u].\n", this->mpiRank,\
    vislib::sys::Thread::CurrentID())
#define _TRACE_RELEASE_LOCK(name) _TRACE_LOCKS("Rank %d releasing "\
    name " lock in thread [%u].\n", this->mpiRank,\
    vislib::sys::Thread::CurrentID())


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
        mustNegotiateMaster(true),
        paramInitialiseMpi("initialiseMpi", "Enables the view to initialise the MPI library."),
        paramUseGsync("useGsync", "Try to synchronise buffer swaps if possible."),
        relayOffset(0) {
    VLAUTOSTACKTRACE;
    this->hasMasterConnection.store(false);

    this->paramInitialiseMpi << new param::BoolParam(true);
    this->paramInitialiseMpi.SetUpdateCallback(&View::onInitialiseMpiChanged);
    this->MakeSlotAvailable(&this->paramInitialiseMpi);

    this->paramUseGsync << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->paramUseGsync);
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
    if (!this->knowsBcastMaster()) {
        this->mustNegotiateMaster = true;
        _TRACE_INFO("Rank %d must negotiate the master, because it does not "
            "know one.\n", this->mpiRank);
    }
    if (this->mustNegotiateMaster) {
        // We have no master, so we try to negotiate one.
        this->mustNegotiateMaster = !this->negotiateBcastMaster();
        _TRACE_INFO("Rank %d has negotiated the master. Must negotiate "
            "again: %d\n", this->mpiRank, this->mustNegotiateMaster);

        // Request all nodes to enable Gsync (join the swap group) given that
        // - we have negotiated a master
        // - the master has a Gsync-capable graphics adapter
        // - Gsync has not yet been enabled
        // - the user did not disable Gsync
        state.InitSwapGroup = !this->mustNegotiateMaster
            && this->isBcastMaster() && this->hasGsync()
            && !this->isGsyncEnabled()
            && this->paramUseGsync.Param<param::BoolParam>()->Value();
    } else {
        // We have a master, check whether it is still valid. It is OK to do
        // this everywhere, because only the data from the real master will
        // remain after the broadcast. Furthermore, nodes that are currently not
        // the master can never invalidate it.
        state.InvalidateMaster = (this->isBcastMaster()
            && !this->hasMasterConnection);
        _TRACE_INFO("Rank %d thinks that the broadcast master %s be "
            "invalidated.\n", this->mpiRank, 
            (state.InvalidateMaster ? "should" : "should not"));
    }

    /* If we have a master, synchronise the state now. */
    if (this->knowsBcastMaster() && (this->mpiSize > 1)) {
        _TRACE_BARRIERS("Rank %d is before status synchronisation.\n",
            this->mpiRank);
#ifdef WITH_MPI
        ASSERT(this->knowsBcastMaster());

        state.RelaySize = this->filterRelayBuffer();
        // It is safe using this size without any lock, because filtering of the
        // relay buffer must only be triggered by the rendering thread, ie
        // cannot occur concurrently while the following code is executed.

        ::MPI_Bcast(&state, sizeof(state), MPI_BYTE, this->getBcastMaster(),
                MPI_COMM_WORLD);

        if (state.InitSwapGroup) {
            SwapGroupApi::GetInstance().JoinSwapGroup(1);
        }

        if (state.InvalidateMaster) {
            this->mustNegotiateMaster = true;
            _TRACE_INFO("Rank %d invalidated the broadcast master. Will "
                "be renegotiated in the next frame.\n", this->mpiRank);
        }

        if (!this->isBcastMaster()) {
            _TRACE_MESSAGING("Rank %d is preparing to receive %u bytes of "
                "relayed messages...\n", this->mpiRank, state.RelaySize);
            this->filteredRelayBuffer.AssertSize(state.RelaySize);
        }

        if (state.RelaySize > 0) {
            _TRACE_MESSAGING("Rank %d is participating in relay of %u bytes.\n",
                this->mpiRank, state.RelaySize);
            ::MPI_Bcast(static_cast<void *>(this->filteredRelayBuffer),
                static_cast<int>(state.RelaySize), MPI_BYTE,
                this->getBcastMaster(), MPI_COMM_WORLD);
        }
#endif /* WITH_MPI */
    } /* if (this->knowsBcastMaster() && (this->mpiSize > 1)) */


    // Post-process status
    if (!this->isBcastMaster() && (state.RelaySize > 0)) {
        this->ModuleGraphLock().LockExclusive();
        _TRACE_ACQUIRE_LOCK("module graph");
        // This code only runs on MPI slaves, ie there is no concurrent access
        // to the relay buffer. Everything runs in the rendering thread.
        //vislib::sys::AutoLock l(this->relayBufferLock); // TODO: Should be unnecessary.
        auto av = this->GetConnectedView();
        size_t offset = 0;

        while (offset < state.RelaySize) {
            vislib::net::ShallowSimpleMessage msg(this->filteredRelayBuffer.At(
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
                        _TRACE_INFO("Rank %d is disconnecting the "
                            "view call...", this->mpiRank);
                        this->DisconnectViewCall();
                        _TRACE_INFO("Rank %d is cleaning up the "
                            "module graph...", this->mpiRank);
                        this->GetCoreInstance()->CleanupModuleGraph();
                    }
                    break;

                case MSG_MODULGRAPH:
                    //::DebugBreak();
                    _TRACE_INFO("Rank %d is preparing the module "
                        "graph...", this->mpiRank);
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

        _TRACE_RELEASE_LOCK("module graph");
        this->ModuleGraphLock().UnlockExclusive();

        _TRACE_MESSAGING("Rank %d has processed all relayed messages.\n",
            this->mpiRank);
    } /* end if (!this->isBcastMaster() && (state.RelaySize > 0)) */

    this->processInitialisationMessage();
    this->registerClient(true);

    /* Ensure that we have a rendering call that we can execute. */
    crv = this->getCallRenderView();
    canRender = (crv != nullptr);

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

        //::glFlush();
        ::glFinish();
#ifdef WITH_MPI
        _TRACE_BARRIERS("Rank %d is before swap barrier.\n", this->mpiRank);
        ::MPI_Barrier(MPI_COMM_WORLD);
#endif /* WITH_MPI */

        if (state.InitSwapGroup && this->isBcastMaster()) {
            // Now all nodes should have joined the swap group, so the master
            // can enable the barrier.
            ASSERT(this->hasGsync());
            _TRACE_GSYNC("Broadcast master %d is binding swap barrier...\n",
                this->mpiRank);
            SwapGroupApi::GetInstance().BindSwapBarrier(1, 1);
        }

    } else {
        this->renderFallbackView();
    } /* end if (canRender) */
}


/*
 * megamol::core::cluster::mpi::View::SwapGroupApi::GetInstance
 */
megamol::core::cluster::mpi::View::SwapGroupApi&
megamol::core::cluster::mpi::View::SwapGroupApi::GetInstance(void) {
    VLAUTOSTACKTRACE;
    static SwapGroupApi instance;
    return instance;
}


/*
 * megamol::core::cluster::mpi::View::SwapGroupApi::BindSwapBarrier
 */
bool megamol::core::cluster::mpi::View::SwapGroupApi::BindSwapBarrier(
        const GLuint group, const GLuint barrier) {
    VLAUTOSTACKTRACE;
    bool retval = false;
#if (defined(_WIN32) && defined(MPI_VIEW_WITH_SWAPGROUP))
    if (this->IsAvailable()) {
        retval = (this->wglBindSwapBarrierNV(group, barrier) == TRUE);
    }
#endif /* (defined(_WIN32) && defined(MPI_VIEW_WITH_SWAPGROUP)) */
    return retval;
}


/*
 * megamol::core::cluster::mpi::View::SwapGroupApi::JoinSwapGroup
 */
bool megamol::core::cluster::mpi::View::SwapGroupApi::JoinSwapGroup(
        const GLuint group) {
    VLAUTOSTACKTRACE;
    bool retval = false;
#if (defined(_WIN32) && defined(MPI_VIEW_WITH_SWAPGROUP))
    if (this->IsAvailable()) {
        HDC hDC = ::wglGetCurrentDC();
        if (hDC != nullptr) {
            retval = (this->wglJoinSwapGroupNV(hDC, group) == TRUE);
        }
    }
#endif /* (defined(_WIN32) && defined(MPI_VIEW_WITH_SWAPGROUP)) */
    return retval;
}


/*
 * megamol::core::cluster::mpi::View::SwapGroupApi::QueryFrameCount
 */
bool megamol::core::cluster::mpi::View::SwapGroupApi::QueryFrameCount(
        unsigned int &count) {
    VLAUTOSTACKTRACE;
    bool retval = false;
#if (defined(_WIN32) && defined(MPI_VIEW_WITH_SWAPGROUP))
    if (this->IsAvailable()) {
        HDC hDC = ::wglGetCurrentDC();
        if (hDC != nullptr) {
            retval = (this->wglQueryFrameCountNV(hDC, &count) == TRUE);
        }
    }
#endif /* (defined(_WIN32) && defined(MPI_VIEW_WITH_SWAPGROUP)) */
    return retval;
}


/*
 * megamol::core::cluster::mpi::View::SwapGroupApi::QueryMaxSwapGroups
 */
bool megamol::core::cluster::mpi::View::SwapGroupApi::QueryMaxSwapGroups(
        GLuint& outMaxGroups, GLuint& outMaxBarriers) {
    VLAUTOSTACKTRACE;
    bool retval = false;
#if (defined(_WIN32) && defined(MPI_VIEW_WITH_SWAPGROUP))
    if (this->IsAvailable()) {
        HDC hDC = ::wglGetCurrentDC();
        if (hDC != nullptr) {
            retval = (this->wglQueryMaxSwapGroupsNV(hDC, &outMaxGroups,
                &outMaxBarriers) == TRUE);
        }
    }
#endif /* (defined(_WIN32) && defined(MPI_VIEW_WITH_SWAPGROUP)) */
    return retval;
}


/*
 * megamol::core::cluster::mpi::View::SwapGroupApi::QuerySwapGroup
 */
bool megamol::core::cluster::mpi::View::SwapGroupApi::QuerySwapGroup(
        GLuint& outGroup, GLuint& outBarrier) {
    VLAUTOSTACKTRACE;
    bool retval = false;
#if (defined(_WIN32) && defined(MPI_VIEW_WITH_SWAPGROUP))
    if (this->IsAvailable()) {
        HDC hDC = ::wglGetCurrentDC();
        if (hDC != nullptr) {
            retval = (this->wglQuerySwapGroupNV(hDC, &outGroup, &outBarrier)
                == TRUE);
        }
    }
#endif /* (defined(_WIN32) && defined(MPI_VIEW_WITH_SWAPGROUP)) */
    return retval;
}


/*
 * megamol::core::cluster::mpi::View::SwapGroupApi::ResetFrameCount
 */
bool megamol::core::cluster::mpi::View::SwapGroupApi::ResetFrameCount(void) {
    VLAUTOSTACKTRACE;
    bool retval = false;
#if (defined(_WIN32) && defined(MPI_VIEW_WITH_SWAPGROUP))
    if (this->IsAvailable()) {
        HDC hDC = ::wglGetCurrentDC();
        if (hDC != nullptr) {
            retval = (this->wglResetFrameCountNV(hDC) == TRUE);
        }
    }
#endif /* (defined(_WIN32) && defined(MPI_VIEW_WITH_SWAPGROUP)) */
    return retval;
}


/*
 * megamol::core::cluster::mpi::View::SwapGroupApi::SwapGroupApi
 */
megamol::core::cluster::mpi::View::SwapGroupApi::SwapGroupApi(void)
        : isAvailable(false) {
    VLAUTOSTACKTRACE;
#if (defined(_WIN32) && defined(MPI_VIEW_WITH_SWAPGROUP))
    this->wglJoinSwapGroupNV = reinterpret_cast<PFNWGLJOINSWAPGROUPNVPROC>(
        ::wglGetProcAddress("wglJoinSwapGroupNV"));
    this->wglBindSwapBarrierNV = reinterpret_cast<PFNWGLBINDSWAPBARRIERNVPROC>(
        ::wglGetProcAddress("wglBindSwapBarrierNV"));
    this->wglQuerySwapGroupNV = reinterpret_cast<PFNWGLQUERYSWAPGROUPNVPROC>(
        ::wglGetProcAddress("wglQuerySwapGroupNV"));
    this->wglQueryMaxSwapGroupsNV
        = reinterpret_cast<PFNWGLQUERYMAXSWAPGROUPSNVPROC>(
        ::wglGetProcAddress("wglQueryMaxSwapGroupsNV"));
    this->wglQueryFrameCountNV = reinterpret_cast<PFNWGLQUERYFRAMECOUNTNVPROC>(
        ::wglGetProcAddress("wglQueryFrameCountNV"));
    this->wglResetFrameCountNV = reinterpret_cast<PFNWGLRESETFRAMECOUNTNVPROC>(
        ::wglGetProcAddress("wglResetFrameCountNV"));

    this->isAvailable = ((this->wglJoinSwapGroupNV != nullptr)
        && (this->wglBindSwapBarrierNV != nullptr)
        && (this->wglQuerySwapGroupNV != nullptr)
        && (this->wglQueryMaxSwapGroupsNV != nullptr)
        && (this->wglQueryFrameCountNV != nullptr)
        && (this->wglResetFrameCountNV != nullptr));
#endif /* (defined(_WIN32) && defined(MPI_VIEW_WITH_SWAPGROUP)) */

    vislib::sys::Log::DefaultLog.WriteInfo(_T("Swap lock is%s available."),
        (this->isAvailable ? _T("") : _T(" not")));
}


/*
 * megamol::core::cluster::mpi::View::create
 */
bool megamol::core::cluster::mpi::View::create(void) {
    VLAUTOSTACKTRACE;
    bool retval = Base1::create();
    this->viewState = ViewState::CREATED;
    this->mustNegotiateMaster = true;

    if (retval) {
        this->initialiseMpi();
    }

    return retval;
}


/*
 * megamol::core::cluster::mpi::View::filterRelayBuffer
 */
size_t megamol::core::cluster::mpi::View::filterRelayBuffer(void) {
    VLAUTOSTACKTRACE;
    typedef std::pair<size_t, size_t> RangeType;

    size_t retval = 0;
    std::unordered_map<vislib::net::SimpleMessageID, RangeType> msgs;
    std::unordered_map<std::string, RangeType> params;
    std::vector<RangeType> ranges;

    _TRACE_ACQUIRE_LOCK("relay buffer");
    vislib::sys::AutoLock l(this->relayBufferLock);

    if (this->relayOffset > 0) {
        //::DebugBreak();

        /* Phase 1: Find the unique messages that we need to transmit. */
        for (size_t offset = 0; offset < this->relayOffset;) {
            vislib::net::ShallowSimpleMessage msg(this->relayBuffer.At(offset));

            if (msg.GetHeader().GetMessageID() == MSG_PARAMUPDATE) {
                /* Parameter messages need to be filtered by content. */
                static_assert(sizeof(char) == 1, "Character is a byte.");
                auto body = msg.GetBodyAs<char>();
                auto value = ::strchr(body, '=');
                ASSERT(value != nullptr);
                std::string name(body, value - body);

                params[name] = RangeType(offset, msg.GetMessageSize());

            } else {
                /* Only keep the last of these messages. */
                msgs[msg.GetHeader().GetMessageID()] = RangeType(offset,
                    msg.GetMessageSize());
            }

            offset += msg.GetMessageSize();
        } /* for (size_t offset = 0; offset < this->relayOffset;) */

        /*
         * Phase 2: Sort all required messages according to their original 
         * order in 'relayBuffer'.
         */
        ranges.reserve(msgs.size() + params.size());
        for (auto it = msgs.begin(); it != msgs.end(); ++it) {
            ranges.push_back(it->second);
            retval += it->second.second;
        }
        for (auto it = params.begin(); it != params.end(); ++it) {
            ranges.push_back(it->second);
            retval += it->second.second;
        }
        std::sort(ranges.begin(), ranges.end(), [](RangeType& l, RangeType& r) {
            return l.first < r.first;
        });

        /* Phase 3: Copy the data. */
        ASSERT(!ranges.empty());
        this->filteredRelayBuffer.AssertSize(retval);

        if (retval != this->relayOffset) {
            //::DebugBreak();
            size_t offset = 0;
            for (auto it = ranges.begin(); it != ranges.end(); ++it) {
                ::memcpy(this->filteredRelayBuffer.At(offset),
                    this->relayBuffer.At(it->first), it->second);
                offset += it->second;
            }
        } else {
            /* Can copy at once. */
            ::memcpy(this->filteredRelayBuffer.At(0),
                this->relayBuffer.At(0), retval);
        }

        _TRACE_PACKAGING("Rank %d repackaged %u bytes of messages for relay to "
            "%u of filtered messages for relay.\n", this->mpiRank,
            this->relayOffset, retval);
    } /* end if (this->relayOffset > 0) */

    this->relayOffset = 0;
    _TRACE_RELEASE_LOCK("relay buffer");
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
 * megamol::core::cluster::mpi::View::hasGsync
 */
bool megamol::core::cluster::mpi::View::hasGsync(void) const {
    VLAUTOSTACKTRACE;
    GLuint maxGroups, maxBarriers;
    if (SwapGroupApi::GetInstance().QueryMaxSwapGroups(maxGroups,
            maxBarriers)) {
        _TRACE_GSYNC("Device supports %u swap groups and %u swap barriers.\n",
            maxGroups, maxBarriers);
        return ((maxGroups > 0) && (maxBarriers > 0));

    } else {
        return false;
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
            vislib::sys::Log::DefaultLog.WriteInfo(_T("This view on %hs is %d ")
                _T("of %d."),
                vislib::sys::SystemInformation::ComputerNameA().PeekBuffer(),
                this->mpiRank, this->mpiSize);

#endif /* WITH_MPI */
        }
    }

    // TODO: Register data types as necessary

    return retval;
}


/*
 * megamol::core::cluster::mpi::View::isGsyncEnabled
 */
bool megamol::core::cluster::mpi::View::isGsyncEnabled(void) const {
    VLAUTOSTACKTRACE;
    GLuint group, barrier;
    if (SwapGroupApi::GetInstance().QuerySwapGroup(group, barrier)) {
        _TRACE_GSYNC("Swap group is %u, swap barrier is %u.\n", group, barrier);
        return ((group > 0) && (barrier > 0));

    } else {
        return false;
    }
}


/*
 * megamol::core::cluster::mpi::View::negotiateBcastMaster
 */
bool megamol::core::cluster::mpi::View::negotiateBcastMaster(void) {
    VLAUTOSTACKTRACE;
    ASSERT(this->mpiRank >= 0);
    ASSERT(this->mpiSize > 0);

#if WITH_MPI
    std::vector<int> responses(this->mpiSize);

    _TRACE_INFO("Negotiating master of %d ranks. Rank %d %s a candidate.\n",
        this->mpiSize, this->mpiRank,
        (this->hasMasterConnection ? "is" : "is not"));
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
    if (this->isGsyncEnabled()) {
        // Swap group ID 1 is because-i-know (we always use 1).
        SwapGroupApi::GetInstance().BindSwapBarrier(1, 0);
        SwapGroupApi::GetInstance().JoinSwapGroup(0);
    }
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
        _TRACE_ACQUIRE_LOCK("relay buffer");
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
        _TRACE_RELEASE_LOCK("relay buffer");
    }
}
