#pragma once

#include <thread>
#include <mutex>

#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/job/AbstractJob.h"
#include "mmcore/param/ParamUpdateListener.h"
#include "mmcore/param/ParamSlot.h"

#include "DistributedProto.h"
#include "FBOCommFabric.h"

namespace megamol {
namespace pbs {

class HeadnodeServer : public core::Module, public core::param::ParamUpdateListener {
public:
    HeadnodeServer();
    ~HeadnodeServer() override;

protected:
    bool create() override;
    void release() override;

    void ParamUpdated(core::param::ParamSlot& slot) override;

private:
    bool init_threads();

    void shutdown_threads();

    bool onStartServer(core::param::ParamSlot& param);

    void sender_loop(FBOCommFabric& comm, core::CallerSlot& view);

    bool check_cam_upd(core::CallerSlot& view, unsigned int& syncnumber, MsgBody_t& msg) const;

    core::CallerSlot view_slot_;

    core::param::ParamSlot renderhead_address_slot_;

    core::param::ParamSlot start_server_slot_;

    FBOCommFabric comm_fabric_;

    std::thread sender_thread_;

    bool run_threads = false;

    mutable std::mutex send_buffer_guard_;

    MsgBody_t send_buffer_;
}; // end class HeadnodeServer

} // end namespace pbs
} // end namespace megamol
