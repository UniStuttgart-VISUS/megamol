#include "stdafx.h"
#include "FBOCompositor2.h"

#include <fstream>
#include <sstream>

#include "mmcore/param/StringParam.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/sys/Log.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/CoreInstance.h"


megamol::pbs::FBOCompositor2::FBOCompositor2()
    : addressesSlot_{"addresses", "Put all addresses of FBOTransmitter2s separated by a ';'"}
    , close_future_{close_promise_.get_future()}
    , fbo_msg_write_{new std::vector<fbo_msg_t>}
    , fbo_msg_recv_{new std::vector<fbo_msg_t>}
    , data_has_changed_{false}
    , col_buf_el_size_{4}
    , depth_buf_el_size_{4}
    , connected_{false} {
    addressesSlot_ << new megamol::core::param::StringParam("tcp://127.0.0.1:34242");
    this->MakeSlotAvailable(&addressesSlot_);
}


megamol::pbs::FBOCompositor2::~FBOCompositor2() { this->Release(); }


bool megamol::pbs::FBOCompositor2::create() {
    glGenVertexArrays(1, &this->vao);
    glBindVertexArray(this->vao);
    glGenBuffers(1, &this->vbo);
    glBindBuffer(GL_ARRAY_BUFFER, this->vbo);
    float buffer[] = {-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0};
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 12, buffer, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, nullptr);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // create shader
    auto vert_shader = glCreateShader(GL_VERTEX_SHADER);
    auto frag_shader = glCreateShader(GL_FRAGMENT_SHADER);

    auto path_to_vert = core::utility::ResourceWrapper::getFileName(
        this->GetCoreInstance()->Configuration(), "pbscompositor.vert.glsl");
    auto path_to_frag = core::utility::ResourceWrapper::getFileName(
        this->GetCoreInstance()->Configuration(), "pbscompositor.frag.glsl");

    std::ifstream shader_file(W2A(path_to_vert.PeekBuffer()));
    std::string shader_string =
        std::string(std::istreambuf_iterator<char>(shader_file), std::istreambuf_iterator<char>());
    shader_file.close();

    auto shader_cstring = shader_string.c_str();
    GLint string_size = shader_string.size();
    glShaderSource(vert_shader, 1, &shader_cstring, &string_size);

    shader_file.open(W2A(path_to_frag.PeekBuffer()));
    shader_string = std::string(std::istreambuf_iterator<char>(shader_file), std::istreambuf_iterator<char>());
    shader_file.close();

    shader_cstring = shader_string.c_str();
    string_size = shader_string.size();
    glShaderSource(frag_shader, 1, &shader_cstring, &string_size);

    glCompileShader(vert_shader);
    if (!printShaderInfoLog(vert_shader)) {
        return false;
    }
    glCompileShader(frag_shader);
    if (!printShaderInfoLog(frag_shader)) {
        return false;
    }

    this->shader = glCreateProgram();
    glAttachShader(this->shader, vert_shader);
    glAttachShader(this->shader, frag_shader);
    glLinkProgram(this->shader);

    if (!printProgramInfoLog(this->shader)) {
        return false;
    }

    glDeleteShader(vert_shader);
    glDeleteShader(frag_shader);

    return true;
}


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


bool megamol::pbs::FBOCompositor2::GetExtents(megamol::core::Call& call) {
    auto cr = dynamic_cast<megamol::core::view::CallRender3D*>(&call);
    if (cr == nullptr) return false;

    cr->SetTimeFramesCount(1);

    auto& out_bbox = cr->AccessBoundingBoxes();

    std::lock_guard<std::mutex> write_guard(this->buffer_write_guard_);
    if (!this->fbo_msg_write_->empty()) {
        float bbox[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (auto& el : *this->fbo_msg_write_) {
            //bbox.unite(el.fbo_msg_header.os_bbox);
            for (int i = 0; i < 3; ++i) {
                bbox[i] = fmin(bbox[i], el.fbo_msg_header.os_bbox[i]);
            }
            for (int i = 3; i < 6; ++i) {
                bbox[i] = fmax(bbox[i], el.fbo_msg_header.os_bbox[i]);
            }
        }
        out_bbox.SetObjectSpaceBBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
        out_bbox.SetObjectSpaceClipBox(out_bbox.ObjectSpaceBBox());

        float scaling = out_bbox.ObjectSpaceBBox().LongestEdge();
        if (scaling > 0.0000001) {
            scaling = 10.0f / scaling;
        } else {
            scaling = 1.0f;
        }
        out_bbox.MakeScaledWorld(scaling);
        // out_bbox.MakeScaledWorld(1.0f);
    } else {
        out_bbox.SetObjectSpaceBBox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        out_bbox.SetObjectSpaceClipBox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    }

    return true;
}


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
        //close_future_ = close_promise_.get_future();

        auto const addresses =
            std::string{T2A(this->addressesSlot_.Param<megamol::core::param::StringParam>()->Value())};

        auto comms = this->connectComms(this->getAddresses(addresses));
        this->collector_thread_ =
            std::thread{&FBOCompositor2::collectorJob, this, std::move(comms)};

        connected_ = true;
    }

    // if data changed check is size has changed
    // if no directly upload
    // it yes resize textures and upload afterward
    if (data_has_changed_.load()) {
        std::lock_guard<std::mutex> write_guard(this->buffer_write_guard_);

        auto const width = (*this->fbo_msg_write_)[0].fbo_msg_header.screen_area[2] -
                           (*this->fbo_msg_write_)[0].fbo_msg_header.screen_area[0];
        auto const height = (*this->fbo_msg_write_)[0].fbo_msg_header.screen_area[3] -
                            (*this->fbo_msg_write_)[0].fbo_msg_header.screen_area[1];

        if (this->width_ != width || this->height_ != height) {
            this->width_ = width;
            this->height_ = height;
            this->resize(this->color_textures_.size(), this->width_, this->height_);
        }

        for (size_t i = 0; i < this->color_textures_.size(); ++i) {
            auto const& fbo = (*this->fbo_msg_write_)[i];
            glBindTexture(GL_TEXTURE_2D, this->color_textures_[i]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, fbo.color_buf.data());
            glBindTexture(GL_TEXTURE_2D, this->depth_textures_[i]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT, fbo.depth_buf.data());
        }

        data_has_changed_.store(false);
    }


    // constantly render current texture set
    // this is the apex of suck and must die
    GLfloat modelViewMatrix_column[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    GLfloat projMatrix_column[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);
    GLfloat light_pos[4];
    glGetLightfv(GL_LIGHT0, GL_POSITION, light_pos);
    // end suck

    glUseProgram(this->shader);

    glUniformMatrix4fv(glGetUniformLocation(this->shader, "modelview"), 1, GL_FALSE, modelViewMatrix_column);
    glUniformMatrix4fv(glGetUniformLocation(this->shader, "project"), 1, GL_FALSE, projMatrix_column);

    for (size_t i = 0; i < this->color_textures_.size(); ++i) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, this->color_textures_[i]);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, this->depth_textures_[i]);

        // render command
        glUniform1i(glGetUniformLocation(this->shader, "color"), 0);
        glUniform1i(glGetUniformLocation(this->shader, "depth"), 1);

        glBindVertexArray(this->vao);
        glDrawArrays(GL_QUADS, 0, 4);
    }

    glUseProgram(0);

    return true;
}


void megamol::pbs::FBOCompositor2::receiverJob(
    FBOCommFabric& comm, core::utility::sys::FutureReset<fbo_msg_t>* fbo_msg_future, std::future<bool>&& close) {
    while (true) {
        auto const status = close.wait_for(std::chrono::milliseconds(1));
        if (status == std::future_status::ready) break;

        // send a request for data
        std::vector<char> buf{'r', 'e', 'q'};
        try {
#if _DEBUG
            vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Sending request\n");
#endif
            if (!comm.Send(buf, send_type::SEND)) {
                vislib::sys::Log::DefaultLog.WriteError("FBOCompositor2: Exception during send in 'receiverJob'\n");
            }
#if _DEBUG
            else {
                vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Request sent\n");
            }
#endif
        } catch (...) {
            vislib::sys::Log::DefaultLog.WriteError("FBOCompositor2: Exception during send in 'receiverJob'\n");
        }

        // receive requested frame info
        try {
#if _DEBUG
            vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Waiting for answer\n");
#endif
            if (!comm.Recv(buf, recv_type::RECV)) {
                vislib::sys::Log::DefaultLog.WriteError("FBOCompositor2: Exception during recv in 'receiverJob'\n");
            }
#if _DEBUG
            else {
                vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Answer received\n");
            }
#endif
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
        auto vol =
            (header.updated_area[2] - header.updated_area[0]) * (header.updated_area[3] - header.updated_area[1]);
        size_t fbo_col_size = fbo_depth_size = static_cast<size_t>(vol);
        fbo_col_size *= static_cast<size_t>(col_buf_el_size_);
        fbo_depth_size *= static_cast<size_t>(depth_buf_el_size_);
        std::vector<char> col_buf(fbo_col_size);
        std::copy(buf_ptr, buf_ptr + fbo_col_size, col_buf.begin());
        buf_ptr += fbo_col_size;
        std::vector<char> depth_buf(fbo_depth_size);
        std::copy(buf_ptr, buf_ptr + fbo_depth_size, depth_buf.begin());

        auto const msg = fbo_msg{std::move(header), std::move(col_buf), std::move(depth_buf)};

        /*while (promise_atomic_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }*/

        while (true) {
            try {
                fbo_msg_future->SetPromise(msg);
                break;
            } catch (std::future_error const& e) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        /*while (true) {
            try {
                fbo_msg_promise->set_value(msg);
                break;
            } catch (std::future_error const& e) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }*/

        //promise_exchange_guard_.unlock();
        //promise_exchange_.notify_one();
    }
    vislib::sys::Log::DefaultLog.WriteWarn("FBOCompositor2: Closing receiverJob\n");
}


void megamol::pbs::FBOCompositor2::collectorJob(std::vector<FBOCommFabric>&& comms) {
    // initialize threads
    std::vector<std::thread> jobs;
    std::vector<core::utility::sys::FutureReset<fbo_msg_t>> fbo_msg_futures(comms.size());
    std::vector<std::promise<bool>> recv_close_sig;
    size_t i = 0;
    for (auto& comm : comms) {
        std::promise<bool> close_sig;
        auto close_sig_fut = close_sig.get_future();
        recv_close_sig.emplace_back(std::move(close_sig));
        //fbo_msg_futures.emplace_back();
        jobs.emplace_back(
            &FBOCompositor2::receiverJob, this, std::ref(comm), fbo_msg_futures[i].GetPtr(), std::move(close_sig_fut));
        i += 1;
    }

    {
        std::scoped_lock<std::mutex, std::mutex> guard(buffer_recv_guard_, buffer_write_guard_);
        this->fbo_msg_write_.reset(new std::vector<fbo_msg_t>);
        this->fbo_msg_write_->resize(jobs.size());
        this->fbo_msg_recv_.reset(new std::vector<fbo_msg_t>);
        this->fbo_msg_recv_->resize(jobs.size());
        this->width_ = 1;
        this->height_ = 1;
        this->initTextures(jobs.size(), this->width_, this->height_);
    }

    // collector loop
    std::vector<bool> fbo_gate(jobs.size());
    while (true) {
        auto const status = close_future_.wait_for(std::chrono::milliseconds(1));
        if (status == std::future_status::ready) break;
        std::fill(fbo_gate.begin(), fbo_gate.end(), false);
        while (!std::all_of(fbo_gate.begin(), fbo_gate.end(), [](bool const& a) { return a; })) {
            //promise_exchange_.notify_all();
            for (size_t i = 0; i < fbo_gate.size(); ++i) {
                if (!fbo_gate[i]) {
                    auto const status = fbo_msg_futures[i].WaitFor(std::chrono::milliseconds(1));
                    if (status == std::future_status::ready) {
                        fbo_gate[i] = true;
                    }
                }
            }
        }

        {
            std::lock_guard<std::mutex> fbo_recv_guard(this->buffer_recv_guard_);

            for (size_t i = 0; i < fbo_msg_futures.size(); ++i) {
                (*this->fbo_msg_recv_)[i] = fbo_msg_futures[i].GetAndReset();
                /*fbo_msg_promises[i] = std::promise<fbo_msg_t>();
                fbo_msg_futures[i] = fbo_msg_promises[i].get_future();*/
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
    vislib::sys::Log::DefaultLog.WriteWarn("FBOCompositor2: Closing collectorJob\n");
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


bool megamol::pbs::FBOCompositor2::printShaderInfoLog(GLuint shader) const {
    int infoLogLen = 0;
    int charsWritten = 0;
    GLchar* infoLog;

    GLint compileStatus;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compileStatus);
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLen);

    if (compileStatus == GL_FALSE && infoLogLen > 1) {
        infoLog = new GLchar[infoLogLen];
        glGetShaderInfoLog(shader, infoLogLen, &charsWritten, infoLog);
        fprintf(stderr, "InfoLog : %s\n", infoLog);
        delete[] infoLog;
        return false;
    }
    return (compileStatus == GL_TRUE);
}


bool megamol::pbs::FBOCompositor2::printProgramInfoLog(GLuint shaderProg) const {
    int infoLogLen = 0;
    int charsWritten = 0;
    GLchar* infoLog;

    GLint linkStatus;
    glGetProgramiv(shaderProg, GL_INFO_LOG_LENGTH, &infoLogLen);
    glGetProgramiv(shaderProg, GL_LINK_STATUS, &linkStatus);

    if (linkStatus == GL_FALSE && infoLogLen > 1) {
        infoLog = new GLchar[infoLogLen];
        glGetProgramInfoLog(shaderProg, infoLogLen, &charsWritten, infoLog);
        fprintf(stderr, "\nProgramInfoLog :\n\n%s\n", infoLog);
        delete[] infoLog;
        return false;
    }
    return (linkStatus == GL_TRUE);
}
