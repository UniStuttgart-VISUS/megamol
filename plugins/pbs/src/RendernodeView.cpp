#include "stdafx.h"
#include "RendernodeView.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"

#include "vislib/sys/Log.h"


void megamol::pbs::RendernodeView::release(void) {}

void megamol::pbs::RendernodeView::Render(const mmcRenderViewContext& context) {
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
    std::vector<unsigned char> msg;
    uint64_t msg_size = 0;
    if (isBCastMaster) {
        if (data_has_changed_.load()) {
            msg = prepare_bcast_msg();
        } else {
            msg = prepare_null_msg();
        }
        msg_size = msg.size();
    }
    MPI_Bcast(&msg_size, 1, MPI_UINT64_T, BCastRank, this->comm_);
    msg.resize(msg_size);
    MPI_Bcast(msg.data(), msg_size, MPI_UNSIGNED_CHAR, BCastRank, this->comm_);

    // handle new param from broadcast
    if (!process_msgs()) {
        vislib::sys::Log::DefaultLog.WriteError(
            "RendernodeView: Error occured during processing of broadcasted messages.\n");
    }

    // initialize rendering
    // check whether rendering is possible in current state
    if (crv_ != nullptr) {
        if (!crv_->operator()(core::view::CallRenderView::CALL_RENDER)) {
            vislib::sys::Log::DefaultLog.WriteError("RendernodeView: Failed to call render on depended view\n");
        }
    }

    // sync barrier
    MPI_Barrier(this->comm_);
}

bool megamol::pbs::RendernodeView::create(void) { return false; }
