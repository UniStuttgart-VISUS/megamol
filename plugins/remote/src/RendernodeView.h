#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <string>

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AbstractTileView.h"
#include "mmcore/LuaState.h"

#ifdef WITH_MPI
#include "mpi.h"
#endif

#include "DistributedProto.h"
#include "FBOCommFabric.h"
#include "mmcore/param/IntParam.h"

namespace megamol {
namespace remote {

class RendernodeView : public core::view::AbstractTileView {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "RendernodeView"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Simple MPI-based render view."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
#ifdef WITH_MPI
        return true;
#else
        return false;
#endif
    };

    /**
     * Disallow usage in quickstarts.
     *
     * @return false
     */
    static bool SupportQuickstart(void) { return false; }

    /**
     * Initializes a new instance.
     */
    RendernodeView(void);

    /**
     * Finalizes the instance.
     */
    virtual ~RendernodeView(void);

    void Render(const mmcRenderViewContext& context) override;

	bool OnRenderView(core::Call& call) override;

protected:
    bool create(void) override;

    void release(void) override;

private:
    bool init_threads();

    bool shutdown_threads();

    // Message_t prepare_bcast_msg();

    Message_t prepare_null_msg() const {
        Message_t msg(MessageHeaderSize);
        auto const type = MessageType::NULL_MSG;
        uint64_t const size = 0;
        auto const size_ptr = reinterpret_cast<char const*>(&size);
        msg[0] = type;
        std::copy(size_ptr, size_ptr + sizeof(uint64_t), msg.begin() + 1);
        return msg;
    }

    bool process_msgs(Message_t const& msgs);

    void recv_loop();

    static MessageType get_msg_type(Message_t::const_iterator const& begin, Message_t::const_iterator const& end) {
        return static_cast<MessageType>(*begin);
    }

    static uint64_t get_msg_size(Message_t::const_iterator const& begin, Message_t::const_iterator const& end) {
        uint64_t ret = 0;
        if (std::distance(begin, end) > MessageHeaderSize) {
            std::copy(begin + MessageTypeSize, begin + MessageHeaderSize, &ret);
        }
        return ret;
    }

    static Message_t get_msg(
        uint64_t size, Message_t::const_iterator const& begin, Message_t::const_iterator const& end) {
        Message_t msg;
        if (std::distance(begin, end) < MessageHeaderSize + size) {
            return msg;
        }

        msg.resize(size);
        std::copy(begin + MessageHeaderSize, begin + MessageHeaderSize + size, msg.begin());

        return msg;
    }

    static Message_t::const_iterator progress_msg(
        uint64_t size, Message_t::const_iterator const& begin, Message_t::const_iterator const& end) {
        if (std::distance(begin, end) > MessageHeaderSize + size) {
            return begin + MessageHeaderSize + size;
        }

        return end;
    }

    bool onBCastRankChanged(core::param::ParamSlot& p) {
        bcast_rank_ = BCastRankSlot_.Param<core::param::IntParam>()->Value();
        if (!isBCastMaster()) {
            shutdown_threads();
        }
        return true;
    }

    bool onAddressChanged(core::param::ParamSlot& p) {
        if (isBCastMaster()) init_threads();
        return true;
    }

    bool isBCastMaster() const { return rank_ == bcast_rank_; }

    bool initMPI();

    core::CallerSlot request_mpi_slot_;

    core::CallerSlot sync_data_slot_;

    // core::param::ParamSlot isBCastMasterSlot_;

    core::param::ParamSlot BCastRankSlot_;

    core::param::ParamSlot address_slot_;

    std::thread receiver_thread_;

    mutable std::mutex recv_msgs_mtx_;

    Message_t recv_msgs_;

    FBOCommFabric recv_comm_;

    bool threads_initialized_ = false;

    std::atomic<bool> data_has_changed_;

    bool run_threads;

    // core::view::CallRenderView* crv_ = nullptr;

#ifdef WITH_MPI
    MPI_Comm comm_;
#else
    int comm_;
#endif

    int rank_;

    int bcast_rank_;

    int comm_size_;
}; // end class RendernodeView

} // end namespace remote
} // end namespace megamol
