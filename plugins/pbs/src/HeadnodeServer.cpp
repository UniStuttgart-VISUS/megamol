#include "stdafx.h"
#include "HeadnodeServer.h"

#include <chrono>
#include <string>

#include "mmcore/CalleeSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/AbstractParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/view/AbstractView.h"
#include "vislib/RawStorageSerialiser.h"


void megamol::pbs::HeadnodeServer::ParamUpdated(core::param::ParamSlot& slot) {
    MsgBody_t msg;

    std::string const name = std::string(slot.FullName());
    std::string const value = std::string(slot.Param<core::param::AbstractParam>()->ValueString());
    std::string mg = "mmSetParamValue(" + name + "," + value + ")";

    msg.resize(1 + 4 + mg.size());
    msg[0] = static_cast<std::byte>(MessageType::PARAM_UPD_MSG);
    auto size = mg.size();
    std::copy(reinterpret_cast<std::byte*>(&size), reinterpret_cast<std::byte*>(&size) + 4, msg.begin() + 1);
    std::copy(mg.begin(), mg.end(), msg.begin() + 5);

    // TODO add to send buffer
}


void megamol::pbs::HeadnodeServer::sender_loop(FBOCommFabric& comm, core::CallerSlot& view) {
    using namespace std::chrono_literals;
    unsigned int syncnumber = -1;

    // TODO Ensure that project is transmitted only upon request
    if (this->GetCoreInstance()->IsLuaProject()) {
        std::string const lua = std::string(this->GetCoreInstance()->GetMergedLuaProject());
        // TODO Setup MSG_PROJ_UPD
    } else {
        vislib::sys::Log::DefaultLog.WriteError("HeadnodeServer: Only LUA-based projects are supported.");
    }

    while (run_threads) {

        // check whether camera has been updated
        MsgBody_t cam_msg;
        auto const cam_updated = check_cam_upd(view, syncnumber, cam_msg);

        if (cam_updated) {
            // TODO Add camera msg to message queue
        }

        // send messages
        std::vector<char> buf;
        while (!comm.Recv(buf, recv_type::RECV) && run_threads) {
        }
        // request received
        // send stuff
        // clear send buffer
        {
            std::lock_guard<std::mutex> lock(send_buffer_guard_);
            comm.Send(send_buffer_, send_type::SEND);
            send_buffer_.clear();
        }

        std::this_thread::sleep_for(1000ms / 60);
    }
}


bool megamol::pbs::HeadnodeServer::check_cam_upd(
    core::CallerSlot& view, unsigned int& syncnumber, MsgBody_t& msg) const {
    AbstractNamedObject::const_ptr_type avp;
    const core::view::AbstractView* av = nullptr;
    core::Call* call = nullptr;
    unsigned int csn = 0;
    vislib::RawStorage mem;
    vislib::RawStorageSerialiser serialiser(&mem);

    av = nullptr;
    call = view.CallAs<core::Call>();
    if ((call != nullptr) && (call->PeekCalleeSlot() != nullptr) && (call->PeekCalleeSlot()->Parent() != nullptr)) {
        avp = call->PeekCalleeSlot()->Parent();
        av = dynamic_cast<const core::view::AbstractView*>(avp.get());
    }
    if (av == nullptr) return false;

    csn = av->GetCameraSyncNumber();
    if ((csn != syncnumber)) {
        syncnumber = csn;
        av->SerialiseCamera(serialiser);

        msg.resize(1 + 4 + mem.GetSize());
        msg[0] = static_cast<std::byte>(MessageType::CAM_UPD_MSG);
        auto size = mem.GetSize();
        std::copy(reinterpret_cast<std::byte*>(&size), reinterpret_cast<std::byte*>(&size) + 4, msg.begin() + 1);
        std::copy(mem.AsAt<std::byte>(0), mem.AsAt<std::byte>(0) + mem.GetSize(), msg.begin() + 5);

        return true;
    }

    return false;
}


bool megamol::pbs::HeadnodeServer::onStartServer(core::param::ParamSlot& param) {
    shutdown_threads();
    init_threads();
    return true;
}


bool megamol::pbs::HeadnodeServer::init_threads() {
    this->comm_fabric_ = FBOCommFabric(std::make_unique<ZMQCommFabric>(zmq::socket_type::rep));
    auto const address = std::string(this->renderhead_address_slot_.Param<core::param::StringParam>()->Value());
    this->comm_fabric_.Bind(address);

    this->sender_thread_ =
        std::thread(&HeadnodeServer::sender_loop, this, std::ref(this->comm_fabric_), std::ref(this->view_slot_));

    return true;
}
