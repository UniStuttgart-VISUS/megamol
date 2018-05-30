#include "stdafx.h"
#include "FBOTransmitter2.h"

#include "glad/glad.h"

#include "vislib/sys/Log.h"

#include "mmcore/CallerSlot.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/view/CallRender3D.h"

#ifdef __unix__
#include <unistd.h>
#include <limits.h>
#endif


megamol::pbs::FBOTransmitter2::FBOTransmitter2()
    : address_slot_{"address", "The address the transmitter should connect to"}
    , commSelectSlot_{"communicator", "Select the communicator to use"}
    , view_name_slot_{"view", "The name of the view instance to be used"}
    , trigger_button_slot_{"trigger", "Triggers transmission"}
    , frame_id_{0}
    , thread_stop_{false}
    , fbo_msg_read_{new fbo_msg_header_t}
    , fbo_msg_send_{new fbo_msg_header_t}
    , color_buf_read_{new std::vector<char>}
    , depth_buf_read_{new std::vector<char>}
    , color_buf_send_{new std::vector<char>}
    , depth_buf_send_{new std::vector<char>}
    , col_buf_el_size_{4}
    , depth_buf_el_size_{4}
    , connected_{false} {
    this->address_slot_ << new megamol::core::param::StringParam{"tcp://*:34242"};
    this->MakeSlotAvailable(&this->address_slot_);
    auto ep = new megamol::core::param::EnumParam(FBOCommFabric::ZMQ_COMM);
    ep->SetTypePair(FBOCommFabric::ZMQ_COMM, "ZMQ");
    ep->SetTypePair(FBOCommFabric::MPI_COMM, "MPI");
    commSelectSlot_ << ep;
    this->MakeSlotAvailable(&commSelectSlot_);
    this->view_name_slot_ << new megamol::core::param::StringParam{"inst"};
    this->MakeSlotAvailable(&this->view_name_slot_);
    this->trigger_button_slot_ << new megamol::core::param::ButtonParam{vislib::sys::KeyCode::KEY_MOD_ALT | 't'};
    this->trigger_button_slot_.SetUpdateCallback(&FBOTransmitter2::triggerButtonClicked);
    this->MakeSlotAvailable(&this->trigger_button_slot_);
}


megamol::pbs::FBOTransmitter2::~FBOTransmitter2() { this->Release(); }


bool megamol::pbs::FBOTransmitter2::create() { return true; }


void megamol::pbs::FBOTransmitter2::release() {
    this->thread_stop_ = true;

    this->transmitter_thread_.join();
}


void megamol::pbs::FBOTransmitter2::AfterRender(megamol::core::view::AbstractView* view) {
    if (!connected_) {
        auto const address = std::string(T2A(this->address_slot_.Param<megamol::core::param::StringParam>()->Value()));

        FBOCommFabric registerComm = FBOCommFabric{std::make_unique<ZMQCommFabric>(zmq::socket_type::req)};
        registerComm.Connect("tcp://127.0.0.1:42000");

        std::string hostname;
#if _WIN32
        DWORD buf_size = 32767;
        hostname.resize(buf_size);
        GetComputerNameA(hostname.data(), &buf_size);
#else
        hostname.resize(HOST_NAME_MAX);
        gethostname(hostname.data(), HOST_NAME_MAX);
#endif
        char stuff[1024];
        sprintf(stuff, "tcp://%s:%s", hostname.c_str(), address.c_str());
        auto name = std::string{stuff};
        std::vector<char> buf(name.begin(), name.end()); //<TODO there should be a better way
#if _DEBUG
        vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter2: Sending client name %s\n", name.c_str());
#endif
        registerComm.Send(buf);
#if _DEBUG
        vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter2: Sent client name\n");
#endif
#if _DEBUG
        vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter2: Receiving client ack\n");
#endif
        registerComm.Recv(buf);
#if _DEBUG
        vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter2: Received client ack\n");
#endif


#if _DEBUG
        vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter2: Connecting comm\n");
#endif

        auto const comm_type = static_cast<FBOCommFabric::commtype>(
            this->commSelectSlot_.Param<megamol::core::param::EnumParam>()->Value());
        //auto const address = std::string(T2A(this->address_slot_.Param<megamol::core::param::StringParam>()->Value()));
        switch (comm_type) {
        case FBOCommFabric::MPI_COMM: {
            int const rank = atoi(address.c_str());
            this->comm_.reset(new FBOCommFabric{std::make_unique<MPICommFabric>(rank, rank)});
        } break;
        case FBOCommFabric::ZMQ_COMM:
        default:
            this->comm_.reset(new FBOCommFabric(std::make_unique<ZMQCommFabric>(zmq::socket_type::rep)));
        }

        this->comm_->Bind(std::string{"tcp://*:"} + address);

        this->thread_stop_ = false;

        this->transmitter_thread_ = std::thread(&FBOTransmitter2::transmitterJob, this);

        connected_ = true;
    }

    // get viewport of current render context
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    auto const width = viewport[2] - viewport[0];
    auto const height = viewport[3] - viewport[1];

    // read FBO
    std::vector<char> col_buf(width * height * col_buf_el_size_);
    std::vector<char> depth_buf(width * height * depth_buf_el_size_);

    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, col_buf.data());
    glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, depth_buf.data());

    float bbox[6];
    if (!this->extractBoundingBox(bbox)) {
        vislib::sys::Log::DefaultLog.WriteError("FBOTransmitter2: could not extract bounding box");
    }

    // copy data to read buffer, if possible
    {
        std::lock_guard<std::mutex> read_guard{this->buffer_read_guard_}; //< maybe try_lock instead

        float lower[] = {viewport[0], viewport[2]};
        float upper[] = {viewport[1], viewport[3]};
        for (int i = 0; i < 4; ++i) {
            this->fbo_msg_read_->screen_area[i] = this->fbo_msg_read_->updated_area[i] = viewport[i];
        }
        // this->fbo_msg_read_->screen_area = {viewport[0], viewport[1], viewport[2], viewport[3]};
        // this->fbo_msg_read_->updated_area = viewp_t{lower, upper};
        this->fbo_msg_read_->color_type = fbo_color_type::RGBAu8;
        this->fbo_msg_read_->depth_type = fbo_depth_type::Df;
        for (int i = 0; i < 6; ++i) {
            this->fbo_msg_read_->os_bbox[i] = this->fbo_msg_read_->cs_bbox[i] = bbox[i];
        }
        // this->fbo_msg_read_->os_bbox = bbox;
        // this->fbo_msg_read_->cs_bbox = bbox;

        this->color_buf_read_->resize(col_buf.size());
        std::copy(col_buf.begin(), col_buf.end(), this->color_buf_read_->begin());
        this->depth_buf_read_->resize(depth_buf.size());
        std::copy(depth_buf.begin(), depth_buf.end(), this->depth_buf_read_->begin());

        this->fbo_msg_read_->frame_id = this->frame_id_.fetch_add(1);
    }

    this->swapBuffers();
}


void megamol::pbs::FBOTransmitter2::transmitterJob() {
    while (!this->thread_stop_) {
        // transmit only upon request
        std::vector<char> buf;
        try {
#if _DEBUG
            vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter2: Waiting for request\n");
#endif
            if (!this->comm_->Recv(buf, recv_type::RECV)) {
                vislib::sys::Log::DefaultLog.WriteError("FBOTransmitter2: Error during recv in 'transmitterJob'\n");
            }
#if _DEBUG
            else {
                vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter2: Request received\n");
            }
#endif
        } catch (zmq::error_t const& e) {
            vislib::sys::Log::DefaultLog.WriteError(
                "FBOTransmitter2: Exception during recv in 'transmitterJob': %s\n", e.what());
        } catch (...) {
            vislib::sys::Log::DefaultLog.WriteError("FBOTransmitter2: Exception during recv in 'transmitterJob'\n");
        }

        // wait for request
        {
            std::lock_guard<std::mutex> send_lock(this->buffer_send_guard_);
            // compose message from header, color_buf, and depth_buf
            buf.resize(sizeof(fbo_msg_header_t) + this->color_buf_send_->size() + this->depth_buf_send_->size());
            std::copy(reinterpret_cast<char*>(&(*fbo_msg_send_)),
                reinterpret_cast<char*>(&(*fbo_msg_send_)) + sizeof(fbo_msg_header_t), buf.data());
            std::copy(
                this->color_buf_send_->begin(), this->color_buf_send_->end(), buf.data() + sizeof(fbo_msg_header_t));
            std::copy(this->depth_buf_send_->begin(), this->depth_buf_send_->end(),
                buf.data() + sizeof(fbo_msg_header_t) + this->color_buf_send_->size());

            // send data
            try {
#if _DEBUG
                vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter2: Sending answer\n");
#endif
                if (!this->comm_->Send(buf, send_type::SEND)) {
                    vislib::sys::Log::DefaultLog.WriteError("FBOTransmitter2: Error during send in 'transmitterJob'\n");
                }
#if _DEBUG
                else {
                    vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter2: Answer sent\n");
                }
#endif
            } catch (zmq::error_t const& e) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "FBOTransmitter2: Exception during send in 'transmitterJob': %s\n", e.what());
            } catch (...) {
                vislib::sys::Log::DefaultLog.WriteError("FBOTransmitter2: Exception during send in 'transmitterJob'\n");
            }
        }
    }
}


bool megamol::pbs::FBOTransmitter2::triggerButtonClicked(megamol::core::param::ParamSlot& slot) {
    // happy trigger finger hit button action happend
    using vislib::sys::Log;

    std::string mvn(view_name_slot_.Param<megamol::core::param::StringParam>()->Value());
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 100, "Transmission of \"%s\" requested", mvn.c_str());

    this->ModuleGraphLock().LockExclusive();
    auto anoc = AbstractNamedObjectContainer::dynamic_pointer_cast(this->RootModule());
    auto ano = anoc->FindNamedObject(mvn.c_str());
    auto vi = dynamic_cast<megamol::core::view::AbstractView*>(ano.get());
    if (vi != nullptr) {
        vi->RegisterHook(this);
    } else {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to find view \"%s\" for transmission", mvn.c_str());
    }
    this->ModuleGraphLock().UnlockExclusive();

    return true;
}


bool megamol::pbs::FBOTransmitter2::extractBoundingBox(float bbox[6]) {
    bool success = true;
    std::string mvn(view_name_slot_.Param<megamol::core::param::StringParam>()->Value());
    this->ModuleGraphLock().LockExclusive();
    auto anoc = AbstractNamedObjectContainer::dynamic_pointer_cast(this->RootModule());
    auto ano = anoc->FindNamedObject(mvn.c_str());
    auto vi = dynamic_cast<core::view::AbstractView*>(ano.get());
    if (vi != nullptr) {
        for (auto c = vi->ChildList_Begin(); c != vi->ChildList_End(); c++) {
            auto sl = dynamic_cast<megamol::core::CallerSlot*>((*c).get());
            if (sl != nullptr) {
                auto r = sl->CallAs<megamol::core::view::CallRender3D>();
                if (r != nullptr) {
                    bbox[0] = r->AccessBoundingBoxes().ObjectSpaceBBox().GetLeft();
                    bbox[1] = r->AccessBoundingBoxes().ObjectSpaceBBox().GetBottom();
                    bbox[2] = r->AccessBoundingBoxes().ObjectSpaceBBox().GetBack();
                    bbox[3] = r->AccessBoundingBoxes().ObjectSpaceBBox().GetRight();
                    bbox[4] = r->AccessBoundingBoxes().ObjectSpaceBBox().GetTop();
                    bbox[5] = r->AccessBoundingBoxes().ObjectSpaceBBox().GetFront();
                    break;
                }
            }
        }
    } else {
        success = false;
    }
    this->ModuleGraphLock().UnlockExclusive();
    return success;
}
