#include "stdafx.h"
#include "RendernodeView.h"

#include <array>

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"

#include "vislib/sys/Log.h"

#define RV_DEBUG_OUTPUT = 1


void megamol::pbs::RendernodeView::release(void) {}


bool megamol::pbs::RendernodeView::create(void) { return false; }


void megamol::pbs::RendernodeView::Render(const mmcRenderViewContext& context) {
    // 0 time, 1 instanceTime
    std::array<double, 2> timestamps = {0.0, 0.0};

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
    if (crv_ != nullptr) {
        crv_->ResetAll();
        crv_->SetTime(static_cast<float>(timestamps[0]));
        crv_->SetInstanceTime(timestamps[1]);
        crv_->SetGpuAffinity(context.GpuAffinity);
        crv_->SetProjection(this->getProjType(), this->getEye());

        if (this->hasTile()) {
            crv_->SetTile(this->getVirtWidth(), this->getVirtHeight(), this->getTileX(), this->getTileY(),
                this->getTileW(), this->getTileH());
        }

        crv_->SetOutputBuffer(GL_BACK, this->getViewportWidth(), this->getViewportHeight());

        if (!crv_->operator()(core::view::CallRenderView::CALL_RENDER)) {
            vislib::sys::Log::DefaultLog.WriteError("RendernodeView: Failed to call render on dependend view\n");
        }

        glFinish();
    } else {
#ifdef RV_DEBUG_OUTPUT
        vislib::sys::Log::DefaultLog.WriteWarn("RendernodeView: crv_ is nullptr.\n");
#endif
    }

    // sync barrier
    MPI_Barrier(this->comm_);
}
