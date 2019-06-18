#include "stdafx.h"
#include "RendernodeView.h"

#include <array>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"

#include "vislib/RawStorageSerialiser.h"
#include "vislib/sys/Log.h"

#define RV_DEBUG_OUTPUT = 1


megamol::pbs::RendernodeView::RendernodeView() {
    
}


megamol::pbs::RendernodeView::~RendernodeView() { this->Release(); }


void megamol::pbs::RendernodeView::release(void) {}


bool megamol::pbs::RendernodeView::create(void) { return true; }


bool megamol::pbs::RendernodeView::process_msgs(MsgBody_t const& msgs) const {
    auto begin = msgs.cbegin();
    auto const end = msgs.cend();

    while (begin < end) {
        auto const type = get_msg_type(begin, end);
        switch (type) {
        case MessageType::PRJ_FILE_MSG:
        case MessageType::PARAM_UPD_MSG: {
            auto msg = get_msg(get_msg_size(begin, end), begin, end);
            std::string mg(msg.begin(), msg.end());
            std::string result;
            auto const success = this->GetCoreInstance()->GetLuaState()->RunString(mg, result);
            if (!success) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "RendernodeView: Could not queue project file: %s", result.c_str());
            }
        } break;
        case MessageType::CAM_UPD_MSG: {
            auto msg = get_msg(get_msg_size(begin, end), begin, end);
            vislib::RawStorageSerialiser ser(reinterpret_cast<unsigned char*>(msg.data()), msg.size());
            auto view = this->getConnectedView();
            if (view != nullptr) {
                view->DeserialiseCamera(ser);
            } else {
                vislib::sys::Log::DefaultLog.WriteError("RendernodeView: Cannot update camera. No view connected.");
            }
        } break;
        case MessageType::NULL_MSG:
            break;
        default:
            vislib::sys::Log::DefaultLog.WriteWarn("RendernodeView: Unknown msg type.");
        }
        begin = progress_msg(get_msg_size(begin, end), begin, end);
    }

    return true;
}


void megamol::pbs::RendernodeView::Render(const mmcRenderViewContext& context) {
#ifdef WITH_MPI
    // 0 time, 1 instanceTime
    std::array<double, 2> timestamps = {0.0, 0.0};

    auto crv = this->getCallRenderView();

    // if broadcastmaster, start listening thread
    auto isBCastMaster = isBCastMasterSlot_.Param<core::param::BoolParam>->Value();
    auto BCastRank = BCastRankSlot_.Param<core::param::IntParam>->Value();
    if (!threads_initialized_ && isBCastMaster) {
        if (!init_threads()) {
            vislib::sys::Log::DefaultLog.WriteError(
                "RendernodeView: Could not initialize receiver thread on BCast master.\n");
        }
    }

    // if listening thread announces new param, broadcast them
    MsgBody_t msg;
    uint64_t msg_size = 0;
    if (isBCastMaster) {
        timestamps[0] = context.Time;
        timestamps[1] = context.InstanceTime;
        if (data_has_changed_.load()) {
            msg = prepare_bcast_msg();
        } else {
            msg = prepare_null_msg();
        }
        msg_size = msg.size();
    }
    MPI_Bcast(timestamps.data(), 2, MPI_DOUBLE, BCastRank, this->comm_);
    MPI_Bcast(&msg_size, 1, MPI_UINT64_T, BCastRank, this->comm_);
    msg.resize(msg_size);
    MPI_Bcast(msg.data(), msg_size, MPI_UNSIGNED_CHAR, BCastRank, this->comm_);

    // handle new param from broadcast
    if (!process_msgs(msg)) {
        vislib::sys::Log::DefaultLog.WriteError(
            "RendernodeView: Error occured during processing of broadcasted messages.\n");
    }

    // initialize rendering
    // check whether rendering is possible in current state
    if (crv != nullptr) {
        crv->ResetAll();
        crv->SetTime(static_cast<float>(timestamps[0]));
        crv->SetInstanceTime(timestamps[1]);
        crv->SetGpuAffinity(context.GpuAffinity);
        crv->SetProjection(this->getProjType(), this->getEye());

        if (this->hasTile()) {
            crv->SetTile(this->getVirtWidth(), this->getVirtHeight(), this->getTileX(), this->getTileY(),
                this->getTileW(), this->getTileH());
        }

        crv->SetOutputBuffer(GL_BACK, this->getViewportWidth(), this->getViewportHeight());

        if (!crv->operator()(core::view::CallRenderView::CALL_RENDER)) {
            vislib::sys::Log::DefaultLog.WriteError("RendernodeView: Failed to call render on dependend view\n");
        }

        glFinish();
    } else {
#    ifdef RV_DEBUG_OUTPUT
        vislib::sys::Log::DefaultLog.WriteWarn("RendernodeView: crv_ is nullptr.\n");
#    endif
    }

    // sync barrier
    MPI_Barrier(this->comm_);
#endif
}
