#include "stdafx.h"
#include "FBOCompositor2.h"

#include <sstream>

#include "mmcore/param/StringParam.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/sys/Log.h"


megamol::pbs::FBOCompositor2::FBOCompositor2()
    : addressesSlot_{"addresses", "Put all addresses of FBOTransmitter2s separated by a ';'"}, connected_{false} {
    addressesSlot_ << new megamol::core::param::StringParam("tcp://localhost:324242");
    this->MakeSlotAvailable(&addressesSlot_);
}


megamol::pbs::FBOCompositor2::~FBOCompositor2() { this->Release(); }


bool megamol::pbs::FBOCompositor2::create() { return true; }


void megamol::pbs::FBOCompositor2::release() {
    close_promise_.set_value(true);
    collector_thread_.join();
}


bool megamol::pbs::FBOCompositor2::GetCapabilities(megamol::core::Call& call) {
    auto* cr = dynamic_cast<megamol::core::view::CallRender3D*>(&call);
    if (cr == nullptr) return false;

    cr->SetCapabilities(megamol::core::view::CallRender3D::CAP_RENDER |
                        megamol::core::view::CallRender3D::CAP_LIGHTING |
                        megamol::core::view::CallRender3D::CAP_ANIMATION);

    return true;
}


bool megamol::pbs::FBOCompositor2::GetExtents(megamol::core::Call& call) { return true; }


bool megamol::pbs::FBOCompositor2::Render(megamol::core::Call& call) {
    /*GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, reinterpret_cast<GLint*>(viewport));

    auto const width = viewport[2] - viewport[0];
    auto const height = viewport[3] - viewport[1];

    if (this->width_ != width || this->height_ != height) {
        this->width_ = width;
        this->height_ = height;
        this->resize(this->width_, this->height_);
    }*/

    if (!connected_) {
        auto close_future = close_promise_.get_future();

        auto const addresses =
            std::string{T2A(this->addressesSlot_.Param<megamol::core::param::StringParam>()->Value())};

        auto comms = this->connectComms(this->getAddresses(addresses));
        this->collector_thread_ =
            std::thread{&FBOCompositor2::collectorJob, this, std::move(comms), std::move(close_future)};

        connected_ = true;
    }

    // if data changed check is size has changed
    // if no directly upload
    // it yes resize textures and upload afterward
    if (data_has_changed_.load()) {
        std::lock_guard<std::mutex> write_guard(this->buffer_write_guard_);

        auto const width = (*this->fbo_msg_write_)[0].fbo_msg_header.screen_area.upper_[0] -
                           (*this->fbo_msg_write_)[0].fbo_msg_header.screen_area.lower_[0];
        auto const height = (*this->fbo_msg_write_)[0].fbo_msg_header.screen_area.upper_[1] -
                            (*this->fbo_msg_write_)[0].fbo_msg_header.screen_area.lower_[1];

        for (size_t i = 0; i < this->color_textures_.size(); ++i) {
            auto const& fbo = (*this->fbo_msg_recv_)[i];
            glBindTexture(GL_TEXTURE_2D, this->color_textures_[i]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, fbo.color_buf.data());
            glBindTexture(GL_TEXTURE_2D, this->depth_textures_[i]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, fbo.depth_buf.data());
        }

        data_has_changed_.store(false);
    }


    // constantly render current texture set
    for (size_t i = 0; i < this->color_textures_.size(); ++i) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, this->color_textures_[i]);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, this->depth_textures_[i]);

        // render command
    }

    return true;
}


void megamol::pbs::FBOCompositor2::receiverJob(
    FBOCommFabric& comm, std::promise<fbo_msg_t>&& fbo_msg_promise, std::future<bool>&& close) const {
    while (!close.valid()) {
        // send a request for data
        std::vector<char> buf{'r', 'e', 'q'};
        try {
            if (!comm.Send(buf, send_type::SEND)) {
                vislib::sys::Log::DefaultLog.WriteError("FBOCompositor2: Exception during send in 'receiverJob'\n");
            }
        } catch (...) {
            vislib::sys::Log::DefaultLog.WriteError("FBOCompositor2: Exception during send in 'receiverJob'\n");
        }

        // receive requested frame info
        try {
            if (!comm.Recv(buf, recv_type::RECV)) {
                vislib::sys::Log::DefaultLog.WriteError("FBOCompositor2: Exception during recv in 'receiverJob'\n");
            }
        } catch (...) {
            vislib::sys::Log::DefaultLog.WriteError("FBOCompositor2: Exception during recv in 'receiverJob'\n");
        }

        /*{
            std::lock_guard<std::mutex> recv_guard{this->buffer_recv_guard_};
            char* buf_ptr = buf.data();
            std::copy(buf_ptr, buf_ptr + sizeof(fbo_msg_header_t), reinterpret_cast<char*>(&*this->fbo_msg_recv_));
            buf_ptr += sizeof(fbo_msg_header_t);
            size_t fbo_depth_size;
            size_t fbo_col_size = fbo_depth_size = static_cast<size_t>(this->fbo_msg_recv_->updated_area.volume());
            fbo_col_size *= static_cast<size_t>(col_buf_el_size_);
            fbo_depth_size *= static_cast<size_t>(depth_buf_el_size_);
            this->color_buf_recv_->resize(fbo_col_size);
            std::copy(buf_ptr, buf_ptr + fbo_col_size, this->color_buf_recv_->begin());
            buf_ptr += fbo_col_size;
            this->depth_buf_recv_->resize(fbo_depth_size);
            std::copy(buf_ptr, buf_ptr + fbo_depth_size, this->depth_buf_recv_->begin());
        }*/

        // this->swapBuffers();

        fbo_msg_header_t header;
        char* buf_ptr = buf.data();
        std::copy(buf_ptr, buf_ptr + sizeof(fbo_msg_header_t), reinterpret_cast<char*>(&header));
        buf_ptr += sizeof(fbo_msg_header_t);
        size_t fbo_depth_size;
        size_t fbo_col_size = fbo_depth_size = static_cast<size_t>(header.updated_area.volume());
        fbo_col_size *= static_cast<size_t>(col_buf_el_size_);
        fbo_depth_size *= static_cast<size_t>(depth_buf_el_size_);
        std::vector<char> col_buf(fbo_col_size);
        std::copy(buf_ptr, buf_ptr + fbo_col_size, col_buf.begin());
        buf_ptr += fbo_col_size;
        std::vector<char> depth_buf(fbo_depth_size);
        std::copy(buf_ptr, buf_ptr + fbo_depth_size, depth_buf.begin());

        auto const msg = fbo_msg{std::move(header), std::move(col_buf), std::move(depth_buf)};

        while (true) {
            try {
                fbo_msg_promise.set_value(msg);
                break;
            } catch (std::future_error const& e) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
}


void megamol::pbs::FBOCompositor2::collectorJob(std::vector<FBOCommFabric>&& comms, std::future<bool>&& close) {
    // initialize threads
    std::vector<std::thread> jobs;
    std::vector<std::future<fbo_msg_t>> fbo_msg_futures;
    std::vector<std::promise<bool>> recv_close_sig;
    for (auto& comm : comms) {
        std::promise<bool> close_sig;
        auto close_sig_fut = close_sig.get_future();
        recv_close_sig.emplace_back(std::move(close_sig));
        std::promise<fbo_msg_t> fbo_msg_promise;
        fbo_msg_futures.emplace_back(fbo_msg_promise.get_future());
        jobs.emplace_back(
            &FBOCompositor2::receiverJob, this, std::ref(comm), std::move(fbo_msg_promise), std::move(close_sig_fut));
    }

    // collector loop
    std::vector<bool> fbo_gate(jobs.size());
    while (!close.valid()) {
        std::fill(fbo_gate.begin(), fbo_gate.end(), false);
        while (!std::all_of(fbo_gate.begin(), fbo_gate.end(), [](bool const& a) { return a; })) {
            for (size_t i = 0; i < fbo_gate.size(); ++i) {
                if (!fbo_gate[i]) {
                    auto const status = fbo_msg_futures[i].wait_for(std::chrono::milliseconds(1));
                    if (status == std::future_status::ready) {
                        fbo_gate[i] = true;
                    }
                }
            }
        }

        {
            std::lock_guard<std::mutex> fbo_recv_guard(this->buffer_recv_guard_);
            for (size_t i = 0; i < fbo_msg_futures.size(); ++i) {
                (*this->fbo_msg_recv_)[i] = fbo_msg_futures[i].get();
            }
        }
        this->swapBuffers();
    }

    // deinitialization
    for (auto& sig : recv_close_sig) {
        sig.set_value(true);
    }
    for (auto& job : jobs) {
        job.join();
    }
}


void megamol::pbs::FBOCompositor2::initTextures(size_t n, GLsizei width, GLsizei heigth) {
    glActiveTexture(GL_TEXTURE0);

    GLint oldBind = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &oldBind);

    this->color_textures_.resize(n);
    glGenTextures(this->color_textures_.size(), this->color_textures_.data());
    for (auto& tex : this->color_textures_) {
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, heigth, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }
    this->depth_textures_.resize(n);
    glGenTextures(this->depth_textures_.size(), this->depth_textures_.data());
    for (auto& tex : this->depth_textures_) {
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, heigth, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }

    glBindTexture(GL_TEXTURE_2D, oldBind);
}


void megamol::pbs::FBOCompositor2::resize(size_t n, GLsizei width, GLsizei height) {
    // this->fbo_ = megamol::core::utility::gl::FramebufferObject{width, height};

    glDeleteTextures(this->color_textures_.size(), this->color_textures_.data());
    glDeleteTextures(this->depth_textures_.size(), this->depth_textures_.data());

    this->initTextures(n, width, height);
}


std::vector<std::string> megamol::pbs::FBOCompositor2::getAddresses(std::string const& str) const noexcept {
    std::vector<std::string> ret;

    std::string token;
    std::istringstream strs(str);

    while (std::getline(strs, token, ';')) {
        if (!token.empty()) {
            ret.push_back(token);
        }
    }

    return ret;
}


std::vector<megamol::pbs::FBOCommFabric> megamol::pbs::FBOCompositor2::connectComms(
    std::vector<std::string> const& addr) const {
    std::vector<FBOCommFabric> ret;

    for (auto const& el : addr) {
        FBOCommFabric comm(std::make_unique<ZMQCommFabric>(zmq::socket_type::req));
        if (comm.Connect(el)) {
            ret.push_back(std::move(comm));
        } else {
            vislib::sys::Log::DefaultLog.WriteWarn(
                "FBOCompositor2: Could not connect socket to address %s\n", el.c_str());
        }
    }

    return ret;
}


std::vector<megamol::pbs::FBOCommFabric> megamol::pbs::FBOCompositor2::connectComms(
    std::vector<std::string>&& addr) const {
    return this->connectComms(addr);
}
