#pragma once

#include <thread>
#include <memory>
#include <vector>
#include <atomic>

#include "mmcore/view/AbstractTileView.h"
#include "mmcore/param/ParamSlot.h"

#include "mpi.h"

#include "FBOCommFabric.h"

namespace megamol {
namespace pbs {
class RendernodeView : public core::view::AbstractTileView {
public:
    enum MessageType : unsigned char {
        NULL_MSG = 0u,
        PRJ_FILE_MSG,
        CAM_UPD_MSG,
        PARAM_UPD_MSG
    };

    struct Message {
        MessageType type;
        uint64_t size;
        std::vector<unsigned char> msg;
    };

    using Message_t = Message;

    using MessageList_t = std::vector<Message_t>;

    virtual void Render(const mmcRenderViewContext& context) override;

protected:
    virtual bool create(void) override;

    virtual void release(void) override;

private:
    bool init_threads();

    bool shutdown_threads();

    std::vector<unsigned char> prepare_bcast_msg();

    std::vector<unsigned char> prepare_null_msg();

    bool process_msgs();

    core::param::ParamSlot isBCastMasterSlot_;

    core::param::ParamSlot BCastRankSlot_;

    std::thread receiver_thread_;

    std::unique_ptr<MessageList_t> recv_msgs_;

    FBOCommFabric recv_comm_;

    bool threads_initialized_ = false;

    std::atomic<bool> data_has_changed_;

    core::view::CallRenderView* crv_ = nullptr;
}; // end class RendernodeView
} // end namespace pbs
} // end namespace megamol
