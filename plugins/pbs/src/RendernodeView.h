#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/AbstractTileView.h"

#include "mpi.h"

#include "FBOCommFabric.h"

namespace megamol {
namespace pbs {

class RendernodeView : public core::view::AbstractTileView {
public:
    enum MessageType : unsigned char { NULL_MSG = 0u, PRJ_FILE_MSG, CAM_UPD_MSG, PARAM_UPD_MSG };

    struct Message {
        MessageType type;
        uint64_t size;
        std::vector<unsigned char> msg;
    };

    using Message_t = Message;

    using MessageList_t = std::vector<Message_t>;

    void Render(const mmcRenderViewContext& context) override;

protected:
    bool create(void) override;

    void release(void) override;

private:
    bool init_threads();

    bool shutdown_threads();

    std::vector<unsigned char> prepare_bcast_msg();

    std::vector<unsigned char> prepare_null_msg();

    bool process_msgs(std::vector<unsigned char>& msgs);

    static MessageType get_msg_type(
        std::vector<unsigned char>::const_iterator begin, std::vector<unsigned char>::const_iterator end) {
        return static_cast<MessageType>(*begin);
    }

    static uint64_t get_msg_size(
        std::vector<unsigned char>::const_iterator begin, std::vector<unsigned char>::const_iterator end) {
        uint64_t ret = 0;
        if (begin + 5 <= end) {
            std::copy(begin + 1, begin + 5, &ret);
        }
        return ret;
    }

    static std::vector<unsigned char> get_msg(uint64_t size, std::vector<unsigned char>::const_iterator begin,
        std::vector<unsigned char>::const_iterator end) {
        std::vector<unsigned char> msg;
        if (begin + 1 + 4 + size >= end) {
            return msg;
        }

        msg.resize(size);
        std::copy(begin + 5, begin + 5 + size, msg.begin());

        return msg;
    }

    static std::vector<unsigned char>::const_iterator progress_msg(uint64_t size,
        std::vector<unsigned char>::const_iterator begin, std::vector<unsigned char>::const_iterator end) {
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

    core::view::CallRenderView* crv_ = nullptr;

    MPI_Comm comm_;

    int rank_;

    int comm_size_;
}; // end class RendernodeView

} // end namespace pbs
} // end namespace megamol
