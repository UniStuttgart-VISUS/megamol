#include "stdafx.h"
#include "FBOCompositor2.h"

#include <fstream>
#include <sstream>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ResourceWrapper.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/Camera_2.h"
#include "vislib/sys/Log.h"

#include "snappy.h"

#include <exception>
#include "vislib/Exception.h"

//#define _DEBUG 1
//#define VERBOSE 1

megamol::remote::FBOCompositor2::FBOCompositor2()
    : provide_img_slot_{"getImg", "Provides received images"}
    , commSelectSlot_{"communicator", "Select the communicator to use"}
    // addressesSlot_{"addresses", "Put all addresses of FBOTransmitter2s separated by a ';'"}
    , targetBandwidthSlot_{"targetBandwidth", "The targeted bandwidth for the compositor to use in MB"}
    , numRendernodesSlot_{"NumRenderNodes", "Set the expected number of rendernodes"}
    , handshakePortSlot_{"handshakePort", "Port for ZMQ handshake"}
    , startSlot_{"start", "Start listening for connections"}
    , restartSlot_{"restart", "Restart compositor to wait for incoming connections"}
    , renderOnlyRequestedFramesSlot_{"only_requested_frames",
          "Required to be set for cinematic rendering. If true, rendering is skipped until frame for requested camera "
          "and time is received."}
    , close_future_{close_promise_.get_future()}
    , fbo_msg_write_{new std::vector<fbo_msg_t>}
    , fbo_msg_recv_{new std::vector<fbo_msg_t>}
    , data_has_changed_{false}
    , col_buf_el_size_{4}
    , depth_buf_el_size_{4}
    , width_{0}
    , height_{0}
    , frame_times_{0.0f}
    , camera_params_{0.0f}
    , connected_{false}
    , registerComm_{std::make_unique<ZMQCommFabric>(zmq::socket_type::rep)}
    , isRegistered_{false} {
    provide_img_slot_.SetCallback(megamol::image_calls::Image2DCall::ClassName(),
        megamol::image_calls::Image2DCall::FunctionName(0), &FBOCompositor2::getImageCallback);
    this->MakeSlotAvailable(&provide_img_slot_);
    // addressesSlot_ << new megamol::core::param::StringParam("tcp://127.0.0.1:34242");
    // this->MakeSlotAvailable(&addressesSlot_);
    handshakePortSlot_ << new megamol::core::param::IntParam(42000);
    this->MakeSlotAvailable(&handshakePortSlot_);
    auto ep = new megamol::core::param::EnumParam(FBOCommFabric::ZMQ_COMM);
    ep->SetTypePair(FBOCommFabric::ZMQ_COMM, "ZMQ");
    ep->SetTypePair(FBOCommFabric::MPI_COMM, "MPI");
    commSelectSlot_ << ep;
    this->MakeSlotAvailable(&commSelectSlot_);
    targetBandwidthSlot_ << new megamol::core::param::IntParam(100, 1, std::numeric_limits<int>::max());
    this->MakeSlotAvailable(&targetBandwidthSlot_);
    numRendernodesSlot_ << new megamol::core::param::IntParam(1, 1, std::numeric_limits<int>::max());
    this->MakeSlotAvailable(&numRendernodesSlot_);
    startSlot_ << new megamol::core::param::ButtonParam(core::view::Key::KEY_F10);
    startSlot_.SetUpdateCallback(&FBOCompositor2::startCallback);
    this->MakeSlotAvailable(&startSlot_);

    renderOnlyRequestedFramesSlot_ << new megamol::core::param::BoolParam(false);
    this->MakeSlotAvailable(&renderOnlyRequestedFramesSlot_);
}


megamol::remote::FBOCompositor2::~FBOCompositor2() { this->Release(); }


bool megamol::remote::FBOCompositor2::create() {
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
        this->GetCoreInstance()->Configuration(), "compositor.vert.glsl");
    auto path_to_frag = core::utility::ResourceWrapper::getFileName(
        this->GetCoreInstance()->Configuration(), "compositor.frag.glsl");

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


void megamol::remote::FBOCompositor2::release() { shutdownThreads(); }


bool megamol::remote::FBOCompositor2::GetExtents(megamol::core::view::CallRender3D_2& call) {
    auto& out_bbox = call.AccessBoundingBoxes();

#if _DEBUG && VERBOSE
    vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Entering mutex GetExtent\n");
#endif

    std::lock_guard<std::mutex> write_guard(this->buffer_write_guard_);

#if _DEBUG && VERBOSE
    vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Leaving mutex GetExtent\n");
#endif

    if (!this->fbo_msg_write_->empty()) {

        auto& vec = (*this->fbo_msg_write_);

        float bbox[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        memcpy(bbox, vec[0].fbo_msg_header.os_bbox, 6 * sizeof(float));

        for (size_t bidx = 1; bidx < this->fbo_msg_write_->size(); ++bidx) {
            for (int i = 0; i < 3; ++i) {
                bbox[i] = fmin(bbox[i], vec[bidx].fbo_msg_header.os_bbox[i]);
            }
            for (int i = 3; i < 6; ++i) {
                bbox[i] = fmax(bbox[i], vec[bidx].fbo_msg_header.os_bbox[i]);
            }
        }

        // for (auto& el : *this->fbo_msg_write_) {
        //    // bbox.unite(el.fbo_msg_header.os_bbox);
        //    for (int i = 0; i < 3; ++i) {
        //        bbox[i] = fmin(bbox[i], el.fbo_msg_header.os_bbox[i]);
        //    }
        //    for (int i = 3; i < 6; ++i) {
        //        bbox[i] = fmax(bbox[i], el.fbo_msg_header.os_bbox[i]);
        //    }
        //}
        out_bbox.SetBoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
        out_bbox.SetClipBox(out_bbox.BoundingBox());

        float timeFramesCount = vec[0].fbo_msg_header.frame_times[1];
        for (size_t bidx = 1; bidx < this->fbo_msg_write_->size(); ++bidx) {
            timeFramesCount = fmin(timeFramesCount, vec[bidx].fbo_msg_header.frame_times[1]);
        }
        call.SetTimeFramesCount(static_cast<unsigned int>((timeFramesCount > 0.0f) ? (timeFramesCount) : (1.0f)));
    } else {
        call.SetTimeFramesCount(1);

        out_bbox.SetBoundingBox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        out_bbox.SetClipBox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    }

    return true;
}


bool megamol::remote::FBOCompositor2::Render(megamol::core::view::CallRender3D_2& call) {
    // initThreads();

    auto req_time = call.Time();
    core::view::Camera_2 cam;
    call.GetCamera(cam);
    core::view::Camera_2::snapshot_type cam_snap;
    core::view::Camera_2::matrix_type view, proj;
    cam.calc_matrices(cam_snap, view, proj, core::thecam::snapshot_content::all);
    auto req_cam_pos = cam_snap.position;
    auto req_cam_up = cam_snap.up_vector;
    auto req_cam_view = cam_snap.view_vector;
    auto only_req_frame = this->renderOnlyRequestedFramesSlot_.Param<megamol::core::param::BoolParam>()->Value();

    // if data changed check if size has changed
    // if no, directly upload
    // if yes, resize textures and upload afterward
    if (data_has_changed_.load()) {
#if _DEBUG && VERBOSE
        vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Entering mutex Render\n");
#endif
        std::lock_guard<std::mutex> write_guard(this->buffer_write_guard_);

#if _DEBUG && VERBOSE
        vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Leaving mutex Render\n");
#endif
        if (only_req_frame) {
            for (int i = 0; i < 2; ++i) {
                this->frame_times_[i] = (*this->fbo_msg_write_)[0].fbo_msg_header.frame_times[i];
            }
            for (int i = 0; i < 9; ++i) {
                this->camera_params_[i] = (*this->fbo_msg_write_)[0].fbo_msg_header.cam_params[i];
            }
        }

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

    if (only_req_frame) {
        float min = 0.00001f; // == 0 does not work (?)
        // Aborting rendering if requested frame has not been received yet
        if ((std::fabs(req_time - this->frame_times_[0]) >= min) ||
            (std::fabs(req_cam_pos.x() - this->camera_params_[0]) >= min) ||
            (std::fabs(req_cam_pos.y() - this->camera_params_[1]) >= min) ||
            (std::fabs(req_cam_pos.z() - this->camera_params_[2]) >= min) ||
            (std::fabs(req_cam_up.x() - this->camera_params_[3]) >= min) ||
            (std::fabs(req_cam_up.y() - this->camera_params_[4]) >= min) ||
            (std::fabs(req_cam_up.z() - this->camera_params_[5]) >= min) || 
            (std::fabs(req_cam_view.x() - this->camera_params_[6]) >= min) ||
            (std::fabs(req_cam_view.y() - this->camera_params_[7]) >= min) ||
            (std::fabs(req_cam_view.z() - this->camera_params_[8]) >= min)) {
            // Resetting FBO in cr3d (to nullptr). This is detected by CinemativView to skip not requested frames while
            // rendering.
            call.ResetOutputBuffer();
            return false;
        }
    }

    // constantly render current texture set
    glEnable(GL_DEPTH_TEST);

    glUseProgram(this->shader);

    glUniformMatrix4fv(glGetUniformLocation(this->shader, "modelview"), 1, GL_FALSE, glm::value_ptr(glm::mat4(view)));
    glUniformMatrix4fv(glGetUniformLocation(this->shader, "project"), 1, GL_FALSE, glm::value_ptr(glm::mat4(proj)));

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

    glActiveTexture(GL_TEXTURE0);
    glBindVertexArray(0);
    glUseProgram(0);

    glDisable(GL_DEPTH_TEST);

    return true;
}


bool megamol::remote::FBOCompositor2::getImageCallback(megamol::core::Call& c) {
    // initThreads();

    auto imgc = dynamic_cast<megamol::image_calls::Image2DCall*>(&c);
    if (imgc == nullptr) return false;

    if (data_has_changed_.load()) {
        std::lock_guard<std::mutex> write_guard(this->buffer_write_guard_);


        this->width_ = (*this->fbo_msg_write_)[0].fbo_msg_header.screen_area[2] -
                       (*this->fbo_msg_write_)[0].fbo_msg_header.screen_area[0];
        this->height_ = (*this->fbo_msg_write_)[0].fbo_msg_header.screen_area[3] -
                        (*this->fbo_msg_write_)[0].fbo_msg_header.screen_area[1];

        // TODO For now, we only provide FBO 0
        auto const& fbo = (*this->fbo_msg_write_)[0];

        RGBAtoRGB(fbo.color_buf, this->img_data_);

        ++hash_;

        data_has_changed_.store(false);
    }

    // img_data_ptr_ = std::make_shared<unsigned char[]>(this->img_data_.data());
    imgc->SetData(megamol::image_calls::Image2DCall::RAW, megamol::image_calls::Image2DCall::RGB, width_, height_,
        this->img_data_.size(), this->img_data_.data());

    imgc->SetDataHash(hash_);

    return true;
}

bool megamol::remote::FBOCompositor2::startCallback(megamol::core::param::ParamSlot& p) {
    shutdownThreads();
    this->initThreadsThread_ = std::thread{&FBOCompositor2::initThreads, this};
    return true;
}


void megamol::remote::FBOCompositor2::RGBAtoRGB(std::vector<char> const& rgba, std::vector<unsigned char>& rgb) {
    auto const num_pixels = rgba.size() / 4;
    rgb.resize(num_pixels * 3);

    for (size_t pidx = 0; pidx < num_pixels; ++pidx) {
        rgb[pidx * 3] = rgba[pidx * 4];
        rgb[pidx * 3 + 1] = rgba[pidx * 4 + 1];
        rgb[pidx * 3 + 2] = rgba[pidx * 4 + 2];
    }
}


bool megamol::remote::FBOCompositor2::initThreads() {
    if (!register_done_) {
        auto const bind_str = std::string("tcp://*:") +
                              std::to_string(this->handshakePortSlot_.Param<megamol::core::param::IntParam>()->Value());
        registerComm_.Bind(bind_str);

        registerThread_ = std::thread{&FBOCompositor2::registerJob, this, std::ref(addresses_)};
        registerThread_.join();
        register_done_ = true;
    }

    if (!connected_ && isRegistered_.load()) {
        // close_future_ = close_promise_.get_future();

#if _DEBUG
        vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Starting collector thread\n");
#endif
        /*auto const addresses =
            std::string{T2A(this->addressesSlot_.Param<megamol::core::param::StringParam>()->Value())};*/

        // auto comms = this->connectComms(this->getAddresses(addresses));
        auto comms = this->connectComms(addresses_);
        this->collector_thread_ = std::thread{&FBOCompositor2::collectorJob, this, std::move(comms)};

        connected_ = true;
    }

    return true;
}


void megamol::remote::FBOCompositor2::receiverJob(
    FBOCommFabric& comm, core::utility::sys::FutureReset<fbo_msg_t>* fbo_msg_future, std::future<bool>&& close) {
    try {
        while (!shutdown_) {
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
                /*if (!comm.Recv(buf, recv_type::RECV)) {
                    vislib::sys::Log::DefaultLog.WriteError("FBOCompositor2: Exception during recv in 'receiverJob'\n");
                }*/
                // std::future_status status;
                while (!comm.Recv(buf, recv_type::RECV) && !shutdown_) {
                    // status = close.wait_for(std::chrono::milliseconds(1));
                    // if (status == std::future_status::ready) break;
#if _DEBUG
                    vislib::sys::Log::DefaultLog.WriteWarn(
                        "FBOCompositor2: Recv failed in 'receiverJob', trying again\n");
#endif
                }
                if (shutdown_) break;
                /*#if _DEBUG
                                else {
                                    vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Answer received\n");
                                }
                #endif*/
            } catch (...) {
                vislib::sys::Log::DefaultLog.WriteError("FBOCompositor2: Exception during recv in 'receiverJob'\n");
            }

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

            if (header.depth_buf_size <= 1 || header.color_buf_size <= 1) {
#if _DEBUG
                vislib::sys::Log::DefaultLog.WriteWarn(
                    "FBOCompositor2: Bad size for alloc color/depth; col_buf size: %d; col_comp_buf size: %d; "
                    "depth_buf size: %d; depth_comp_buf size: %d;\n",
                    fbo_col_size, header.color_buf_size, fbo_depth_size, header.depth_buf_size);
#endif
                continue;
            }

            std::vector<char> col_buf(fbo_col_size);
            std::vector<char> col_comp_buf(header.color_buf_size);
            std::copy(buf_ptr, buf_ptr + header.color_buf_size, col_comp_buf.begin());
            buf_ptr += header.color_buf_size;
            std::vector<char> depth_buf(fbo_depth_size);
            std::vector<char> depth_comp_buf(header.depth_buf_size);
            std::copy(buf_ptr, buf_ptr + header.depth_buf_size, depth_comp_buf.begin());

            // snappy uncompress
            snappy::RawUncompress(col_comp_buf.data(), col_comp_buf.size(), col_buf.data());
            snappy::RawUncompress(depth_comp_buf.data(), depth_comp_buf.size(), depth_buf.data());

            /*std::vector<char> col_buf(fbo_col_size);
            std::copy(buf_ptr, buf_ptr + fbo_col_size, col_buf.begin());
            buf_ptr += fbo_col_size;
            std::vector<char> depth_buf(fbo_depth_size);
            std::copy(buf_ptr, buf_ptr + fbo_depth_size, depth_buf.begin());*/

#ifdef _DEBUG
            vislib::sys::Log::DefaultLog.WriteInfo(
                "FBOCompositor2: Got message with col_buf size %d and depth_buf size %d\n", col_buf.size(),
                depth_buf.size());
#endif

            auto const msg = fbo_msg{std::move(header), std::move(col_buf), std::move(depth_buf)};

            while (!shutdown_) {
                try {
                    fbo_msg_future->SetPromise(msg);
                    break;
                } catch (std::future_error const& e) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }

#if 0
        {
#    if _DEBUG
    vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Entering mutex receiverJob Heartbeat\n");
#    endif
            std::shared_lock<std::shared_mutex> heartbeat_guard(heartbeat_lock_);

#    if _DEBUG
    vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Leaving mutex receiverJob Heartbeat\n");
#    endif
            heartbeat_.wait(heartbeat_guard);
        }
#endif
        }
        vislib::sys::Log::DefaultLog.WriteWarn("FBOCompositor2: Closing receiverJob\n");
    } catch (std::exception& e) {
        vislib::sys::Log::DefaultLog.WriteError("FBOCompositor2: ReceiverJob died: %s\n", e.what());
    } catch (vislib::Exception& e) {
        vislib::sys::Log::DefaultLog.WriteError("FBOCompositor2: ReceiverJob died: %s\n", e.GetMsgA());
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("FBOCompositor2: ReceiverJob died\n");
    }
}


void megamol::remote::FBOCompositor2::collectorJob(std::vector<FBOCommFabric>&& comms) {
    try {
        auto const num_jobs = comms.size();
        // initialize threads
        std::vector<std::thread> jobs;
        std::vector<core::utility::sys::FutureReset<fbo_msg_t>> fbo_msg_futures(num_jobs);
        std::vector<std::promise<bool>> recv_close_sig;
        size_t i = 0;
        for (auto& comm : comms) {
            std::promise<bool> close_sig;
            auto close_sig_fut = close_sig.get_future();
            recv_close_sig.emplace_back(std::move(close_sig));
            // fbo_msg_futures.emplace_back();
            jobs.emplace_back(&FBOCompositor2::receiverJob, this, std::ref(comm), fbo_msg_futures[i].GetPtr(),
                std::move(close_sig_fut));
            i += 1;
        }

        {

#if _DEBUG && VERBOSE
            vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Entering mutex collectorJob\n");
#endif

            std::scoped_lock<std::mutex, std::mutex> guard(buffer_recv_guard_, buffer_write_guard_);

#if _DEBUG && VERBOSE
            vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Leaving mutex collectorJob\n");
#endif
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
        while (!shutdown_) {
            /*auto const status = close_future_.wait_for(std::chrono::milliseconds(1));
            if (status == std::future_status::ready) break;*/

            auto const start = std::chrono::high_resolution_clock::now();

            std::fill(fbo_gate.begin(), fbo_gate.end(), false);
            while (!std::all_of(fbo_gate.begin(), fbo_gate.end(), [](bool const& a) { return a; }) && !shutdown_) {
                // promise_exchange_.notify_all();
                for (size_t i = 0; i < fbo_gate.size(); ++i) {
                    if (!fbo_gate[i]) {
                        auto const status = fbo_msg_futures[i].WaitFor(std::chrono::milliseconds(1));
                        if (status == std::future_status::ready) {
                            fbo_gate[i] = true;
                        }
                    }
                }
            }

            if (shutdown_) break;

            {

#if _DEBUG && VERBOSE
                vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Entering mutex collectorJob collect\n");
#endif

                std::lock_guard<std::mutex> fbo_recv_guard(this->buffer_recv_guard_);

#if _DEBUG && VERBOSE
                vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Leaving mutex colectorJob collect\n");
#endif

#ifdef _DEBUG
                vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Got all messages ... comitting\n");
#endif

                for (size_t i = 0; i < fbo_msg_futures.size(); ++i) {
                    (*this->fbo_msg_recv_)[i] = fbo_msg_futures[i].GetAndReset();
                }
            }


#if 0
        auto const end = std::chrono::high_resolution_clock::now();

        // calculate heartbeat timer
        auto recv_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        auto const msg_size =
            (sizeof(fbo_msg_header_t) + width_ * height_ * (col_buf_el_size_ + depth_buf_el_size_)) * num_jobs;
        auto const bandwidth = msg_size * 1000.0f / recv_duration.count();

        float const target_bandwidth =
            this->targetBandwidthSlot_.Param<megamol::core::param::IntParam>()->Value() * 1000 * 1000;
        float const target_fps = target_bandwidth / msg_size;

#    if _DEBUG
        vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Bandwidth %f\n", bandwidth);
        vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Target FPS %f\n", target_fps);
#    endif
        {
#    if _DEBUG
    vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Entering mutex collector Job heartbeat\n");
#    endif
            std::unique_lock<std::shared_mutex> heartbeat_guard(heartbeat_lock_);

#    if _DEBUG
    vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Leaving mutex collectorJob heartbeat\n");
#    endif

            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<size_t>(1000.0f / target_fps)));
            heartbeat_.notify_all();
        }
#endif

            this->swapBuffers();
        }

        // deinitialization
        vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Sending close signals\n");
        for (auto& sig : recv_close_sig) {
            sig.set_value(true);
        }
        for (auto& job : jobs) {
            job.join();
        }
        vislib::sys::Log::DefaultLog.WriteWarn("FBOCompositor2: Closing collectorJob\n");
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("FBOCompositor2: CollectorJob\n");
    }
}


void megamol::remote::FBOCompositor2::registerJob(std::vector<std::string>& addresses) {
    try {
        int const numNodes = this->numRendernodesSlot_.Param<megamol::core::param::IntParam>()->Value();

        std::vector<char> buf;

#if _DEBUG
        vislib::sys::Log::DefaultLog.WriteInfo(
            "FBOCompositor2: Starting client registration of %d clients\n", numNodes);
#endif

        do {
            if (shutdown_) break;

            try {
#if _DEBUG
                vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Receiving client address\n");
#endif
                while (!registerComm_.Recv(buf) && !shutdown_) {
#if _DEBUG
                    vislib::sys::Log::DefaultLog.WriteWarn(
                        "FBOCompositor2: Recv failed on 'registerComm', trying again\n");
#endif
                }
#if _DEBUG
                if (!shutdown_) {
                    vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Received client address\n");
                }
#endif
            } catch (zmq::error_t const& e) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "FBOCompositor2: Failed to recv on register socket %s\n", e.what());
            }
            std::string str{buf.begin(), buf.end()};

            if (shutdown_) break;
#if _DEBUG
            vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Received address: %s\n", str.c_str());
#endif
            addresses.push_back(str);

            try {
#if _DEBUG
                vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Sending client ack\n");
#endif
                registerComm_.Send(buf);
#if _DEBUG
                vislib::sys::Log::DefaultLog.WriteInfo("FBOCompositor2: Sent client ack\n");
#endif
            } catch (zmq::error_t const& e) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "FBOCompositor2: Failed to send  on register socket %s\n", e.what());
            }
        } while (addresses.size() < numNodes);
        if (!this->shutdown_) {
            this->isRegistered_.store(true);
        }
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("FBOCompositor2: RegisterJob died\n");
    }
}


void megamol::remote::FBOCompositor2::initTextures(size_t n, GLsizei width, GLsizei heigth) {
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


void megamol::remote::FBOCompositor2::resize(size_t n, GLsizei width, GLsizei height) {
    // this->fbo_ = megamol::core::utility::gl::FramebufferObject{width, height};

    glDeleteTextures(this->color_textures_.size(), this->color_textures_.data());
    glDeleteTextures(this->depth_textures_.size(), this->depth_textures_.data());

    this->initTextures(n, width, height);
}


std::vector<std::string> megamol::remote::FBOCompositor2::getAddresses(std::string const& str) const noexcept {
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


std::vector<megamol::remote::FBOCommFabric> megamol::remote::FBOCompositor2::connectComms(
    std::vector<std::string> const& addr) const {
    std::vector<FBOCommFabric> ret;

    auto const comm_type =
        static_cast<FBOCommFabric::commtype>(this->commSelectSlot_.Param<megamol::core::param::EnumParam>()->Value());

    for (auto const& el : addr) {
        std::unique_ptr<AbstractCommFabric> pimpl;
        switch (comm_type) {
        case FBOCommFabric::MPI_COMM: {
            int const rank = atoi(el.c_str());
            pimpl.reset(new MPICommFabric{rank, rank});
        } break;
        case FBOCommFabric::ZMQ_COMM:
        default:
            pimpl.reset(new ZMQCommFabric{zmq::socket_type::req});
        }

        // FBOCommFabric comm(std::make_unique<ZMQCommFabric>(zmq::socket_type::req));
        FBOCommFabric comm{std::move(pimpl)};
        if (comm.Connect(el)) {
            ret.push_back(std::move(comm));
        } else {
            vislib::sys::Log::DefaultLog.WriteWarn(
                "FBOCompositor2: Could not connect socket to address %s\n", el.c_str());
        }
    }

    return ret;
}


std::vector<megamol::remote::FBOCommFabric> megamol::remote::FBOCompositor2::connectComms(
    std::vector<std::string>&& addr) const {
    return this->connectComms(addr);
}


bool megamol::remote::FBOCompositor2::printShaderInfoLog(GLuint shader) const {
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


bool megamol::remote::FBOCompositor2::printProgramInfoLog(GLuint shaderProg) const {
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


bool megamol::remote::FBOCompositor2::shutdownThreads() {
    // close_promise_.set_value(true);
    shutdown_ = true;
    if (collector_thread_.joinable()) collector_thread_.join();
    if (this->initThreadsThread_.joinable()) this->initThreadsThread_.join();

    connected_ = false;
    isRegistered_.store(false);
    shutdown_ = false;
    register_done_ = false;

    return true;
}
