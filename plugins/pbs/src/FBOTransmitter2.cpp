#include "stdafx.h"
#include "FBOTransmitter2.h"

#include <array>

#include "glad/glad.h"

#include "snappy.h"

#include "vislib/sys/Log.h"

#include "mmcore/CallerSlot.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/view/CallRender3D.h"
#include "mmcore/cluster/mpi/MpiCall.h"
#include "vislib/sys/SystemInformation.h"
#include "vislib/Trace.h"

#ifdef __unix__
#include <limits.h>
#include <unistd.h>
#endif


megamol::pbs::FBOTransmitter2::FBOTransmitter2()
    : address_slot_{"address", "The address the transmitter should connect to"}
    , commSelectSlot_{"communicator", "Select the communicator to use"}
    , view_name_slot_{"view", "The name of the view instance to be used"}
    , trigger_button_slot_{"trigger", "Triggers transmission"}
    , target_machine_slot_ {
    "targetMachine", "Name of the target machine"
}
    , callRequestMpi("requestMpi", "Requests initialisation of MPI and the communicator for the view.")
#ifdef WITH_MPI
, toggle_aggregate_slot_{"aggregate", "Toggle whether to aggregate and composite FBOs prior to transmission"},
    aggregate_{false}
#endif // WITH_MPI
, frame_id_{0}, thread_stop_{false}, fbo_msg_read_{new fbo_msg_header_t}, fbo_msg_send_{new fbo_msg_header_t},
    color_buf_read_{new std::vector<char>}, depth_buf_read_{new std::vector<char>},
    color_buf_send_{new std::vector<char>}, depth_buf_send_{new std::vector<char>}, col_buf_el_size_{4},
    depth_buf_el_size_{4}, connected_{false} {
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
    this->target_machine_slot_ << new megamol::core::param::StringParam{"127.0.0.1"};
    this->MakeSlotAvailable(&this->target_machine_slot_);
    this->callRequestMpi.SetCompatibleCall<core::cluster::mpi::MpiCallDescription>();
    this->MakeSlotAvailable(&this->callRequestMpi);
#ifdef WITH_MPI
    toggle_aggregate_slot_ << new megamol::core::param::BoolParam{false};
    this->MakeSlotAvailable(&toggle_aggregate_slot_);
#endif // WITH_MPI
}


megamol::pbs::FBOTransmitter2::~FBOTransmitter2() { this->Release(); }


bool megamol::pbs::FBOTransmitter2::create() {
#if _DEBUG
    vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter2: Creating ...\n");
#endif
    return true;
}


void megamol::pbs::FBOTransmitter2::release() {
    this->thread_stop_ = true;

    this->transmitter_thread_.join();

#ifdef WITH_MPI
    if (useMpi) {
        icetDestroyMPICommunicator(icet_comm_);
        icetDestroyContext(icet_ctx_);
    }
#endif // WITH_MPI
}


void megamol::pbs::FBOTransmitter2::AfterRender(megamol::core::view::AbstractView* view) {
    if (!connected_) {
#ifdef _DEBUG
        vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter2: Connecting ...\n");
#endif
#ifdef WITH_MPI

        useMpi = initMPI();
        aggregate_ = this->toggle_aggregate_slot_.Param<megamol::core::param::BoolParam>()->Value();
        if (aggregate_ && !useMpi) {
            vislib::sys::Log::DefaultLog.WriteError("Cannot aggregate without MPI!");
            this->toggle_aggregate_slot_.Param<megamol::core::param::BoolParam>()->SetValue(false);
        }

        if ((aggregate_ && mpiRank == 0) || !aggregate_) {
#ifdef _DEBUG
            vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter2: Connecting rank %d\n", mpiRank);
#endif
#endif // WITH_MPI
            auto const address =
                std::string(T2A(this->address_slot_.Param<megamol::core::param::StringParam>()->Value()));
            auto const target =
                std::string(T2A(this->target_machine_slot_.Param<megamol::core::param::StringParam>()->Value()));

            FBOCommFabric registerComm = FBOCommFabric{std::make_unique<ZMQCommFabric>(zmq::socket_type::req)};
            std::string const registerAddress = std::string("tcp://") + target + std::string(":42000");
            printf("FBOTransmitter2: registerAddress: %s", registerAddress.c_str());
            registerComm.Connect(registerAddress);

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
            // auto const address =
            // std::string(T2A(this->address_slot_.Param<megamol::core::param::StringParam>()->Value()));
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

#ifdef WITH_MPI
        }
#endif // WITH_MPI
        connected_ = true;
#ifdef WITH_MPI
        // aggregate_ = this->toggle_aggregate_slot_.Param<megamol::core::param::BoolParam>()->Value();
        // get viewport of current render context
        if (aggregate_) {
#if _DEBUG
            vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter2: Initializing IceT at rank %d\n", mpiRank);
#endif
            //MPI_Comm *heinz = (MPI_Comm *)malloc(sizeof(MPI_Comm));
            //MPI_Comm_dup(MPI_COMM_WORLD, heinz);
            icet_comm_ = icetCreateMPICommunicator(MPI_COMM_WORLD);
            icet_ctx_ = icetCreateContext(icet_comm_);
            icetStrategy(ICET_STRATEGY_SEQUENTIAL);
            icetSingleImageStrategy(ICET_SINGLE_IMAGE_STRATEGY_AUTOMATIC);
            icetCompositeMode(ICET_COMPOSITE_MODE_Z_BUFFER);
            icetSetColorFormat(ICET_IMAGE_COLOR_RGBA_UBYTE);
            icetSetDepthFormat(ICET_IMAGE_DEPTH_FLOAT);
            icetDisable(ICET_COMPOSITE_ONE_BUFFER);

            GLint viewport[4];
            glGetIntegerv(GL_VIEWPORT, viewport);

            auto const width = viewport[2] - viewport[0];
            auto const height = viewport[3] - viewport[1];

            icetResetTiles();
            icetAddTile(
                viewport[0], viewport[1], width, height, 0); //< might not be necessary due to IceT's OpenGL layer
#if _DEBUG
            vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter2: Initialized IceT at rank %d\n", mpiRank);
#endif
        }
#endif // WITH_MPI
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

#ifdef WITH_MPI
    IceTUByte* icet_col_buf = reinterpret_cast<IceTUByte*>(col_buf.data());
    IceTFloat* icet_depth_buf = reinterpret_cast<IceTFloat*>(depth_buf.data());

    if (aggregate_) {
#if _DEBUG
        vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter2: Simple IceT commit at rank %d\n", mpiRank);
#endif
        std::array<IceTFloat, 4> backgroundColor = { 0, 0, 0, 0 };
        auto const icet_comp_image = icetCompositeImage(col_buf.data(), depth_buf.data(), nullptr, nullptr, nullptr, backgroundColor.data());
        icet_col_buf = icetImageGetColorub(icet_comp_image);
        icet_depth_buf = icetImageGetDepthf(icet_comp_image);
    }

    if ((aggregate_ && mpiRank == 0) || !aggregate_) {
#endif // WITH_MPI
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
            //std::copy(col_buf.begin(), col_buf.end(), this->color_buf_read_->begin());
            memcpy(this->color_buf_read_->data(), icet_col_buf, width * height * col_buf_el_size_);
            this->depth_buf_read_->resize(depth_buf.size());
            //std::copy(depth_buf.begin(), depth_buf.end(), this->depth_buf_read_->begin());
            memcpy(this->depth_buf_read_->data(), icet_depth_buf, width * height * depth_buf_el_size_);

            this->fbo_msg_read_->frame_id = this->frame_id_.fetch_add(1);
        }

        this->swapBuffers();
#ifdef WITH_MPI
    } 
#endif // WITH_MPI
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

//#ifdef WITH_MPI
//            IceTUByte* icet_col_buf = nullptr;
//            IceTFloat* icet_depth_buf = nullptr;
//            if (aggregate_) {
//#if _DEBUG
//                vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter2: Complex IceT commit at rank %d\n", rank_);
//#endif
//                std::array<IceTFloat, 4> backgroundColor = {0, 0, 0, 0};
//                auto const icet_comp_image = icetCompositeImage(this->color_buf_send_->data(),
//                    this->depth_buf_send_->data(), nullptr, nullptr, nullptr, backgroundColor.data());
//#if _DEBUG
//                vislib::sys::Log::DefaultLog.WriteInfo("FBOTransmitter2: Recieved IceT image at rank %d\n", rank_);
//#endif
//                icet_col_buf = icetImageGetColorub(icet_comp_image);
//                icet_depth_buf = icetImageGetDepthf(icet_comp_image);
//            }
//#endif

            // snappy compression
            std::vector<char> col_comp_buf(snappy::MaxCompressedLength(this->color_buf_send_->size()));
            std::vector<char> depth_comp_buf(snappy::MaxCompressedLength(this->depth_buf_send_->size()));
            size_t col_comp_size = 0;
            size_t depth_comp_size = 0;

            //if (aggregate_) {
            //    snappy::RawCompress(reinterpret_cast<char*>(icet_col_buf), this->color_buf_send_->size(),
            //        col_comp_buf.data(), &col_comp_size);
            //    snappy::RawCompress(reinterpret_cast<char*>(icet_depth_buf), this->depth_buf_send_->size(),
            //        depth_comp_buf.data(), &depth_comp_size);
            //} else {
                snappy::RawCompress(
                    this->color_buf_send_->data(), this->color_buf_send_->size(), col_comp_buf.data(), &col_comp_size);
                snappy::RawCompress(this->depth_buf_send_->data(), this->depth_buf_send_->size(), depth_comp_buf.data(),
                    &depth_comp_size);
            //}

            fbo_msg_send_->color_buf_size = col_comp_size;
            fbo_msg_send_->depth_buf_size = depth_comp_size;
            // compose message from header, color_buf, and depth_buf
            buf.resize(sizeof(fbo_msg_header_t) + col_comp_size + depth_comp_size);
            std::copy(reinterpret_cast<char*>(&(*fbo_msg_send_)),
                reinterpret_cast<char*>(&(*fbo_msg_send_)) + sizeof(fbo_msg_header_t), buf.data());
            /*std::copy(
                this->color_buf_send_->begin(), this->color_buf_send_->end(), buf.data() +
            sizeof(fbo_msg_header_t)); std::copy(this->depth_buf_send_->begin(), this->depth_buf_send_->end(),
                buf.data() + sizeof(fbo_msg_header_t) + this->color_buf_send_->size());*/
            std::copy(col_comp_buf.data(), col_comp_buf.data() + col_comp_size, buf.data() + sizeof(fbo_msg_header_t));
            std::copy(depth_comp_buf.data(), depth_comp_buf.data() + depth_comp_size,
                buf.data() + sizeof(fbo_msg_header_t) + col_comp_size);

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


bool megamol::pbs::FBOTransmitter2::initMPI() {
    bool retval = false;
#ifdef WITH_MPI
    if (this->mpi_comm_ == MPI_COMM_NULL) {
        VLTRACE(vislib::Trace::LEVEL_INFO, "FBOTransmitter2: Need to initialize MPI\n");
        auto c = this->callRequestMpi.CallAs<core::cluster::mpi::MpiCall>();
        if (c != nullptr) {
            /* New method: let MpiProvider do all the stuff. */
            if ((*c)(core::cluster::mpi::MpiCall::IDX_PROVIDE_MPI)) {
                vislib::sys::Log::DefaultLog.WriteInfo("Got MPI communicator.");
                this->mpi_comm_ = c->GetComm();
            } else {
                vislib::sys::Log::DefaultLog.WriteError(_T("Could not ")
                    _T("retrieve MPI communicator for the MPI-based view ")
                    _T("from the registered provider module."));
            }
        }

        if (this->mpi_comm_ != MPI_COMM_NULL) {
            vislib::sys::Log::DefaultLog.WriteInfo(_T("MPI is ready, ")
                _T("retrieving communicator properties ..."));
            ::MPI_Comm_rank(this->mpi_comm_, &this->mpiRank);
            ::MPI_Comm_size(this->mpi_comm_, &this->mpiSize);
            vislib::sys::Log::DefaultLog.WriteInfo(_T("This view on %hs is %d ")
                _T("of %d."),
                vislib::sys::SystemInformation::ComputerNameA().PeekBuffer(),
                this->mpiRank, this->mpiSize);
        } /* end if (this->comm != MPI_COMM_NULL) */
        VLTRACE(vislib::Trace::LEVEL_INFO, "FBOTransmitter2: MPI initialized: %s (%i)\n", this->mpi_comm_ != MPI_COMM_NULL ? "true" : "false", mpi_comm_);
    } /* end if (this->comm == MPI_COMM_NULL) */

      /* Determine success of the whole operation. */
    retval = (this->mpi_comm_ != MPI_COMM_NULL);
#endif /* WITH_MPI */
    return retval;
}