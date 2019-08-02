#include "stdafx.h"
#include "HeadnodeServer.h"

#include <array>
#include <chrono>
#include <string>

#include "mmcore/CalleeSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/AbstractParam.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/view/AbstractView.h"
#include "mmcore/view/CallRenderView.h"
#include "vislib/RawStorageSerialiser.h"


megamol::pbs::HeadnodeServer::HeadnodeServer()
    : view_slot_("viewSlot", "Connects to the view.")
    , renderhead_port_slot_("port", "Sets to port to listen to.")
    , start_server_slot_("start", "Start listening to port.")
    , lua_command_slot_("LUACommand", "Sends custom lua command to the RendernodeView")
    , deploy_project_slot_("deployProject", "Sends project file on connect")
    , comm_fabric_(std::make_unique<ZMQCommFabric>(zmq::socket_type::rep))
    , run_threads_(false)
    , is_job_running_(true) {


    lua_command_slot_ << new megamol::core::param::StringParam("");
    lua_command_slot_.SetUpdateCallback(&HeadnodeServer::onLuaCommand);
    this->MakeSlotAvailable(&this->lua_command_slot_);

    renderhead_port_slot_ << new megamol::core::param::IntParam(52000);
    this->MakeSlotAvailable(&this->renderhead_port_slot_);

    deploy_project_slot_ << new megamol::core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->deploy_project_slot_);

    start_server_slot_ << new megamol::core::param::ButtonParam(core::view::Key::KEY_F8);
    start_server_slot_.SetUpdateCallback(&HeadnodeServer::onStartServer);
    this->MakeSlotAvailable(&this->start_server_slot_);


    this->view_slot_.SetCompatibleCall<core::view::CallRenderViewDescription>();
    this->MakeSlotAvailable(&this->view_slot_);

    buffer_has_changed_.store(false);
}


megamol::pbs::HeadnodeServer::~HeadnodeServer() { this->Release(); }


bool megamol::pbs::HeadnodeServer::IsRunning(void) const { return is_job_running_; }


bool megamol::pbs::HeadnodeServer::Start() { return true; }


bool megamol::pbs::HeadnodeServer::Terminate() {
    shutdown_threads();
    return true;
}


bool megamol::pbs::HeadnodeServer::create() {
    this->GetCoreInstance()->RegisterParamUpdateListener(this);
    return true;
}


void megamol::pbs::HeadnodeServer::release() {
    this->is_job_running_ = false;
    shutdown_threads();
    if (this->GetCoreInstance() != nullptr) {
        this->GetCoreInstance()->UnregisterParamUpdateListener(this);
    }
}


void megamol::pbs::HeadnodeServer::ParamUpdated(core::param::ParamSlot& slot) {
    // if (!running_) return;
    if (!run_threads_) return;

    std::vector<char> msg;

    std::string const name = std::string(slot.FullName());
    std::string const value = std::string(slot.Param<core::param::AbstractParam>()->ValueString());
    std::string mg = "mmSetParamValue(\"" + name + "\", \"" + value + "\")";

    msg.resize(MessageHeaderSize + mg.size());
    msg[0] = static_cast<char>(MessageType::PARAM_UPD_MSG);
    auto size = mg.size();
    std::copy(reinterpret_cast<char*>(&size), reinterpret_cast<char*>(&size) + MessageSizeSize,
        msg.begin() + MessageTypeSize);
    std::copy(mg.begin(), mg.end(), msg.begin() + MessageHeaderSize);


    std::lock_guard<std::mutex> guard(send_buffer_guard_);
    send_buffer_.insert(send_buffer_.end(), msg.begin(), msg.end());
    buffer_has_changed_.store(true);
}


bool megamol::pbs::HeadnodeServer::get_cam_upd(std::vector<char>& msg) {

    AbstractNamedObject::const_ptr_type avp;
    const core::view::AbstractView* av = nullptr;
    core::Call* call = nullptr;
    unsigned int csn = 0;

    av = nullptr;
    call = this->view_slot_.CallAs<core::Call>();
    if ((call != nullptr) && (call->PeekCalleeSlot() != nullptr) && (call->PeekCalleeSlot()->Parent() != nullptr)) {
        avp = call->PeekCalleeSlot()->Parent();
        av = dynamic_cast<const core::view::AbstractView*>(avp.get());
    }
    if (av == nullptr) return false;

    csn = av->GetCameraSyncNumber();
    if ((csn != syncnumber)) {
        syncnumber = csn;
        vislib::RawStorage mem;
        vislib::RawStorageSerialiser serialiser(&mem);
        av->SerialiseCamera(serialiser);

        msg.resize(MessageHeaderSize + mem.GetSize());
        msg[0] = static_cast<char>(MessageType::CAM_UPD_MSG);
        auto size = mem.GetSize();
        std::copy(reinterpret_cast<char*>(&size), reinterpret_cast<char*>(&size) + MessageSizeSize,
            msg.begin() + MessageTypeSize);
        std::copy(mem.AsAt<char>(0), mem.AsAt<char>(0) + mem.GetSize(), msg.begin() + MessageHeaderSize);

        return true;
    }

    return false;
}


bool megamol::pbs::HeadnodeServer::init_threads() {
    try {
        shutdown_threads();
        this->comm_fabric_ = FBOCommFabric(std::make_unique<ZMQCommFabric>(zmq::socket_type::rep));
        auto const port = std::to_string(this->renderhead_port_slot_.Param<core::param::IntParam>()->Value());
        vislib::sys::Log::DefaultLog.WriteInfo("HeadnodeServer: Starting listener on port %s.\n", port.c_str());
        std::string const address = "tcp://*:" + port;
        this->comm_fabric_.Bind(address);
        run_threads_ = true;
        this->comm_thread_ = std::thread(&HeadnodeServer::do_communication, this);
        vislib::sys::Log::DefaultLog.WriteInfo("HeadnodeServer: Communication thread started.\n");
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("HeadnodeServer: Could not initialize threads\n");
        return false;
    }

    return true;
}


bool megamol::pbs::HeadnodeServer::shutdown_threads() {
    run_threads_ = false;
    if (comm_thread_.joinable()) {
        vislib::sys::Log::DefaultLog.WriteInfo("HeadnodeServer: Joining thread.");
        comm_thread_.join();
    }
    return true;
}


void megamol::pbs::HeadnodeServer::do_communication() {
    using namespace std::chrono_literals;

    std::vector<char> const null_buf(MessageHeaderSize, 0);
    std::vector<char> buf(3);
    std::vector<char> cam_msg;

    // retrieve modulgraph
    if (this->deploy_project_slot_.Param<core::param::BoolParam>()->Value()) {
    if (this->GetCoreInstance()->IsLuaProject()) {
        auto const lua = std::string(this->GetCoreInstance()->GetMergedLuaProject());
        std::vector<char> msg(MessageHeaderSize + lua.size());
        msg[0] = MessageType::PRJ_FILE_MSG;
        auto size = lua.size();
        std::copy(reinterpret_cast<char*>(&size), reinterpret_cast<char*>(&size) + MessageSizeSize,
            msg.begin() + MessageTypeSize);
        std::copy(lua.begin(), lua.end(), msg.begin() + MessageHeaderSize);
        {
            std::lock_guard<std::mutex> lock(send_buffer_guard_);
            send_buffer_.insert(send_buffer_.end(), msg.begin(), msg.end());
        }
    }
    }
    try {
        while (run_threads_) {
            // Wait for message
            while (!comm_fabric_.Recv(buf, recv_type::RECV) && run_threads_) {
            }
            if (!run_threads_) break;

            // check whether camera has been updated
            auto cam_updated = false;

            while (!(cam_updated || buffer_has_changed_.load()) && run_threads_) {
                std::this_thread::sleep_for(1000ms / 120);
                cam_updated = get_cam_upd(cam_msg);
            }

            {
                std::lock_guard<std::mutex> lock(send_buffer_guard_);
                if (cam_updated) {
                    send_buffer_.insert(send_buffer_.end(), cam_msg.begin(), cam_msg.end());
                }

                if (!send_buffer_.empty()) {
		    // vislib::sys::Log::DefaultLog.WriteInfo("HeadnodeServer: Sending parameter update.\n");
                    comm_fabric_.Send(send_buffer_, send_type::SEND);
                    send_buffer_.clear();
                    buffer_has_changed_.store(false);
                } else {
                    comm_fabric_.Send(null_buf, send_type::SEND);
                }
            }

            /*if (cam_updated) {
                while (!comm_fabric_.Recv(buf, recv_type::RECV) && run_threads_) {
                }
                if (!run_threads_) break;
                comm_fabric_.Send(cam_msg, send_type::SEND);
            }*/
            // std::this_thread::sleep_for(1000ms / 120);
        }
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("HeadnodeServer: Error during communication;");
    }
    vislib::sys::Log::DefaultLog.WriteInfo("HeadnodeServer: Exiting sender loop.");
}


bool megamol::pbs::HeadnodeServer::onStartServer(core::param::ParamSlot& param) {
    init_threads();

    return true;
}


bool megamol::pbs::HeadnodeServer::onLuaCommand(core::param::ParamSlot& param) {
    if (!run_threads_) return true;

    std::vector<char> msg;
    std::string mg = std::string(param.Param<core::param::StringParam>()->ValueString());

    msg.resize(MessageHeaderSize + mg.size());
    msg[0] = static_cast<char>(MessageType::PARAM_UPD_MSG);
    auto size = mg.size();
    std::copy(reinterpret_cast<char*>(&size), reinterpret_cast<char*>(&size) + MessageSizeSize,
        msg.begin() + MessageTypeSize);
    std::copy(mg.begin(), mg.end(), msg.begin() + MessageHeaderSize);


    std::lock_guard<std::mutex> guard(send_buffer_guard_);
    send_buffer_.insert(send_buffer_.end(), msg.begin(), msg.end());
    buffer_has_changed_.store(true);

    return true;
}
