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
#include "mmcore/param/ButtonParam.h"
#include "mmcore/view/CallRenderView.h"
#include "mmcore/param/IntParam.h"


megamol::pbs::HeadnodeServer::HeadnodeServer()
    : start_server_slot_("start", "Start listening to port.")
    , renderhead_port_slot_("port", "Sets to port to listen to.")
    , view_slot_("viewSlot", "Connects to the view.")
    , comm_fabric_(std::make_unique<ZMQCommFabric>(zmq::socket_type::rep))
    , running_(false) {


    renderhead_port_slot_ << new megamol::core::param::IntParam(52000);
    this->MakeSlotAvailable(&this->renderhead_port_slot_);

    start_server_slot_ << new megamol::core::param::ButtonParam(core::view::Key::KEY_F8);
    start_server_slot_.SetUpdateCallback(&HeadnodeServer::onStartServer);
    this->MakeSlotAvailable(&this->start_server_slot_);

    this->view_slot_.SetCompatibleCall<core::view::CallRenderViewDescription>();
    this->MakeSlotAvailable(&this->view_slot_);
}

megamol::pbs::HeadnodeServer::~HeadnodeServer() {
    this->Release();
}

bool megamol::pbs::HeadnodeServer::IsRunning(void) const { return true; }

bool megamol::pbs::HeadnodeServer::Start() { return true; }

bool megamol::pbs::HeadnodeServer::Terminate() { return true; }

bool megamol::pbs::HeadnodeServer::create() {
    this->GetCoreInstance()->RegisterParamUpdateListener(this);
    return true;
}

void megamol::pbs::HeadnodeServer::release() {
    if (this->GetCoreInstance() != nullptr) {
        this->GetCoreInstance()->UnregisterParamUpdateListener(this);
    }
}

void megamol::pbs::HeadnodeServer::ParamUpdated(core::param::ParamSlot& slot) {

    if (!running_) return;

    std::vector<std::byte> msg;

    std::string const name = std::string(slot.FullName());
    std::string const value = std::string(slot.Param<core::param::AbstractParam>()->ValueString());
    std::string mg = "mmSetParamValue(" + name + "," + value + ")";

    msg.resize(1 + 4 + mg.size());
    msg[0] = static_cast<std::byte>(MessageType::PARAM_UPD_MSG);
    auto size = mg.size();
    std::copy(reinterpret_cast<std::byte*>(&size), reinterpret_cast<std::byte*>(&size) + 4, msg.begin() + 1);
    std::vector<char> char_mg(mg.begin(), mg.end());
    std::vector<std::byte> byte_mg = reinterpret_cast<std::vector<std::byte>&>(char_mg);
    std::copy(byte_mg.begin(), byte_mg.end(), msg.begin() + 5);


    std::lock_guard<std::mutex> guard(send_buffer_guard_);

    if (send_buffer_.empty()) {
        send_buffer_.resize(msg.size());
        send_buffer_.insert(send_buffer_.begin(),msg.begin(),msg.end());
    } else {
        std::vector<std::byte> byte_old_size(sizeof(uint64_t));
        byte_old_size.insert(byte_old_size.begin(), send_buffer_.begin() + 1, send_buffer_.begin() + 5);
        auto old_size = reinterpret_cast<uint64_t>(&byte_old_size);
        mg = ";" + mg;
        auto new_size = old_size + mg.size();
        std::copy(
            reinterpret_cast<std::byte*>(&new_size), reinterpret_cast<std::byte*>(&new_size) + 4, send_buffer_.begin() + 1);
        send_buffer_.resize(new_size);
        send_buffer_.insert(send_buffer_.begin() + 5 + old_size, byte_mg.begin(), byte_mg.end());
    }
}



bool megamol::pbs::HeadnodeServer::get_cam_upd(std::vector<std::byte>& msg) {

    AbstractNamedObject::const_ptr_type avp;
    const core::view::AbstractView* av = nullptr;
    core::Call* call = nullptr;
    unsigned int csn = 0;
    vislib::RawStorage mem;
    vislib::RawStorageSerialiser serialiser(&mem);

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

bool megamol::pbs::HeadnodeServer::init_threads() {
    try {

        this->comm_fabric_ = FBOCommFabric(std::make_unique<ZMQCommFabric>(zmq::socket_type::rep));
        auto const port = std::to_string(this->renderhead_port_slot_.Param<core::param::IntParam>()->Value());
        std::string address = "tcp://*:" + port;
        this->comm_fabric_.Bind(address);

        this->comm_thread_ = std::thread(&HeadnodeServer::do_communication, this);
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("HeadnodeServer: Could not initialize threads");
        return false;
    }

    return true;
}


void megamol::pbs::HeadnodeServer::do_communication() {
    try {
        while (true) {
            // Wait for message
            std::vector<char> buf;
            while (!comm_fabric_.Recv(buf, recv_type::RECV)) {
            }

            // check whether camera has been updated
            std::vector<std::byte> cam_msg;
            auto const cam_updated = get_cam_upd(cam_msg);

            std::lock_guard<std::mutex> lock(send_buffer_guard_);
            auto char_sendbuff = reinterpret_cast<std::vector<char>&>(send_buffer_);
            comm_fabric_.Send(char_sendbuff, send_type::SEND);
            send_buffer_.clear();

            if (cam_updated) {
                auto char_cam_msg = reinterpret_cast<std::vector<char>&>(cam_msg);
                comm_fabric_.Send(char_cam_msg, send_type::SEND);
            }
        }
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("HeadnodeServer: Error during communication;");
    }
}


bool megamol::pbs::HeadnodeServer::onStartServer(core::param::ParamSlot& param) {
    using namespace std::chrono_literals;


    if (running_) {
        vislib::sys::Log::DefaultLog.WriteError("HeadnodeServer: Communication already running.");
        return true;
    }

    running_ = init_threads();

    return true;
}

