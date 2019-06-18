#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AbstractTileView.h"

#include "mpi.h"

#include "FBOCommFabric.h"
#include "DistributedProto.h"

namespace megamol {
namespace pbs {

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

protected:
    bool create(void) override;

    void release(void) override;

private:
    bool init_threads();

    bool shutdown_threads();

    MsgBody_t prepare_bcast_msg();

    MsgBody_t prepare_null_msg();

    bool process_msgs(MsgBody_t const& msgs) const;

    static MessageType get_msg_type(MsgBody_t::const_iterator const& begin, MsgBody_t::const_iterator const& end) {
        return static_cast<MessageType>(*begin);
    }

    static uint64_t get_msg_size(MsgBody_t::const_iterator const& begin, MsgBody_t::const_iterator const& end) {
        uint64_t ret = 0;
        if (begin + 5 <= end) {
            std::copy(begin + 1, begin + 5, &ret);
        }
        return ret;
    }

    static MsgBody_t get_msg(
        uint64_t size, MsgBody_t::const_iterator const& begin, MsgBody_t::const_iterator const& end) {
        MsgBody_t msg;
        if (begin + 1 + 4 + size >= end) {
            return msg;
        }

        msg.resize(size);
        std::copy(begin + 5, begin + 5 + size, msg.begin());

        return msg;
    }

    static MsgBody_t::const_iterator progress_msg(
        uint64_t size, MsgBody_t::const_iterator const& begin, MsgBody_t::const_iterator const& end) {
        if (begin + 1 + 4 + size <= end) {
            return begin + 1 + 4 + size;
        }

        return end;
    }

    core::param::ParamSlot isBCastMasterSlot_;

    core::param::ParamSlot BCastRankSlot_;

    std::thread receiver_thread_;

    std::unique_ptr<MessageList_t> recv_msgs_;

    FBOCommFabric recv_comm_;

    bool threads_initialized_ = false;

    std::atomic<bool> data_has_changed_;

    // core::view::CallRenderView* crv_ = nullptr;

    MPI_Comm comm_;

    int rank_;

    int comm_size_;
}; // end class RendernodeView

} // end namespace pbs
} // end namespace megamol
