#include "stdafx.h"
#include "FBOTransmitter2.h"

#include "glad/glad.h"

#include "vislib/sys/Log.h"

#include "mmcore/param/StringParam.h"


megamol::pbs::FBOTransmitter2::FBOTransmitter2()
    : megamol::core::Module{}
    , megamol::core::view::AbstractView::Hooks{}
    , address_slot_{"address", "The address the transmitter should connect to"}
    , frame_id_{0}
    , thread_stop_{false}
    , col_buf_el_size_{16}
    , depth_buf_el_size_{4} {
    this->address_slot_ << new megamol::core::param::StringParam{"tcp:\\127.0.0.1:34242"};
    this->MakeSlotAvailable(&this->address_slot_);
}


megamol::pbs::FBOTransmitter2::~FBOTransmitter2(){ this->Release(); }


bool megamol::pbs::FBOTransmitter2::create() {
    this->comm_.reset(new FBOCommFabric(std::make_unique<ZMQCommFabric>(zmq::socket_type::rep)));

    auto const address = std::string(T2A(this->address_slot_.Param<megamol::core::param::StringParam>()->Value()));
    this->comm_->Connect(address);

    this->thread_stop_ = false;

    this->transmitter_thread_ = std::thread(&FBOTransmitter2::transmitterJob, this);

    return true;
}


void megamol::pbs::FBOTransmitter2::release() {
    this->thread_stop_ = true;

    this->transmitter_thread_.join();
}


void megamol::pbs::FBOTransmitter2::AfterRender(megamol::core::view::AbstractView* view) {
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

    // copy data to read buffer, if possible
    {
        std::lock_guard<std::mutex> read_guard{this->buffer_read_guard_}; //< maybe try_lock instead

        float lower[] = {viewport[0], viewport[2]};
        float upper[] = {viewport[1], viewport[3]};
        this->fbo_msg_read_->screen_area = viewp_t{lower, upper};
        this->fbo_msg_read_->updated_area = viewp_t{lower, upper};
        this->fbo_msg_read_->color_type = fbo_color_type::RGBAu8;
        this->fbo_msg_read_->depth_type = fbo_depth_type::Df;

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
            if (!this->comm_->Recv(buf, recv_type::RECV)) {
                vislib::sys::Log::DefaultLog.WriteError("FBOTransmitter2: Error during recv in 'transmitterJob'\n");
            }
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
                if (!this->comm_->Send(buf, send_type::SEND)) {
                    vislib::sys::Log::DefaultLog.WriteError("FBOTransmitter2: Error during send in 'transmitterJob'\n");
                }
            } catch (...) {
                vislib::sys::Log::DefaultLog.WriteError("FBOTransmitter2: Exception during send in 'transmitterJob'\n");
            }
        }
    }
}

