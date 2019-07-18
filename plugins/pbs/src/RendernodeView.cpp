#include "stdafx.h"
#include "RendernodeView.h"

#include <array>
#include <chrono>

#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"


#include "mmcore/cluster/SyncDataSourcesCall.h"
#include "mmcore/cluster/mpi/MpiCall.h"
#include "vislib/RawStorageSerialiser.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/SystemInformation.h"

//#define RV_DEBUG_OUTPUT = 1
#define CINEMA = 1

#ifdef CINEMA
#    include <iomanip>
#    include <sstream>
#    include "PNGWriter.h"
#    include "mmcore/view/CallRender3D.h"
#    include "vislib/graphics/CameraParamsStore.h"
#    include "vislib/sys/Environment.h"
static std::vector<vislib::graphics::CameraParamsStore> _cinemaCams;
static unsigned int _index = 0;
static unsigned int _loopID = 0;
#endif

megamol::pbs::RendernodeView::RendernodeView()
    : request_mpi_slot_("requestMPI", "Requests initialization of MPI and the communicator for the view.")
    , sync_data_slot_("syncData", "Requests synchronization of data sources in the MPI world.")
    , BCastRankSlot_("BCastRank", "Set which MPI rank is the broadcast master")
    , address_slot_("address", "Address of headnode in ZMQ syntax (e.g. \"tcp://127.0.0.1:33333\")")
    , recv_comm_(std::make_unique<ZMQCommFabric>(zmq::socket_type::req))
    , run_threads(false)
#ifdef WITH_MPI
    , comm_(MPI_COMM_NULL)
#else
    , comm_(0x04000000)
#endif
    , rank_(-1)
    , bcast_rank_(0)
    , comm_size_(0) {
    request_mpi_slot_.SetCompatibleCall<core::cluster::mpi::MpiCallDescription>();
    this->MakeSlotAvailable(&request_mpi_slot_);

    sync_data_slot_.SetCompatibleCall<core::cluster::SyncDataSourcesCallDescription>();
    this->MakeSlotAvailable(&sync_data_slot_);

    BCastRankSlot_ << new core::param::IntParam(-1, -1);
    BCastRankSlot_.SetUpdateCallback(&RendernodeView::onBCastRankChanged);
    this->MakeSlotAvailable(&BCastRankSlot_);

    address_slot_ << new core::param::StringParam("tcp://127.0.0.1:66566");
    address_slot_.SetUpdateCallback(&RendernodeView::onAddressChanged);
    this->MakeSlotAvailable(&address_slot_);

    data_has_changed_.store(false);
}


megamol::pbs::RendernodeView::~RendernodeView() { this->Release(); }


void megamol::pbs::RendernodeView::release(void) { shutdown_threads(); }


bool megamol::pbs::RendernodeView::create(void) { return true; }


bool megamol::pbs::RendernodeView::process_msgs(Message_t const& msgs) {
    auto ibegin = msgs.cbegin();
    auto const iend = msgs.cend();

    while (ibegin < iend) {
        auto const type = static_cast<MessageType>(*ibegin);
        auto size = 0;
        switch (type) {
        case MessageType::PRJ_FILE_MSG: {
            auto const call = this->getCallRenderView();
            if (call != nullptr) {
                this->disconnectOutgoingRenderCall();
                this->GetCoreInstance()->CleanupModuleGraph();
            }
        }
        case MessageType::PARAM_UPD_MSG: {

            if (std::distance(ibegin, iend) > MessageHeaderSize) {
                std::copy(ibegin + MessageTypeSize, ibegin + MessageHeaderSize, reinterpret_cast<char*>(&size));
            }
            Message_t msg;
            if (std::distance(ibegin, iend) >= MessageHeaderSize + size) {
                msg.resize(size);
                std::copy(ibegin + MessageHeaderSize, ibegin + MessageHeaderSize + size, msg.begin());
            }

            std::string mg(msg.begin(), msg.end());
            std::string result;
            auto const success = this->GetCoreInstance()->GetLuaState()->RunString(mg, result);
            if (!success) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "RendernodeView: Could not queue project file: %s", result.c_str());
            }
        } break;
        case MessageType::CAM_UPD_MSG: {

            if (std::distance(ibegin, iend) > MessageHeaderSize) {
                std::copy(ibegin + MessageTypeSize, ibegin + MessageHeaderSize, reinterpret_cast<char*>(&size));
            }
            Message_t msg;
            if (std::distance(ibegin, iend) >= MessageHeaderSize + size) {
                msg.resize(size);
                std::copy(ibegin + MessageHeaderSize, ibegin + MessageHeaderSize + size, msg.begin());
            }
            vislib::RawStorageSerialiser ser(reinterpret_cast<unsigned char*>(msg.data()), msg.size());
            auto view = this->getConnectedView();
            if (view != nullptr) {
                view->DeserialiseCamera(ser);
            } else {
                vislib::sys::Log::DefaultLog.WriteError("RendernodeView: Cannot update camera. No view connected.");
            }
        } break;
        case MessageType::HEAD_DISC_MSG:
        case MessageType::NULL_MSG:
            break;
        default:
            vislib::sys::Log::DefaultLog.WriteWarn("RendernodeView: Unknown msg type.");
        }
        ibegin += size + MessageHeaderSize;
    }

    return true;
}


void megamol::pbs::RendernodeView::Render(const mmcRenderViewContext& context) {
#ifdef WITH_MPI
    this->initMPI();
    // 0 time, 1 instanceTime
    std::array<double, 2> timestamps = {0.0, 0.0};

    // if broadcastmaster, start listening thread
    // auto isBCastMaster = isBCastMasterSlot_.Param<core::param::BoolParam>->Value();
    // auto BCastRank = BCastRankSlot_.Param<core::param::IntParam>->Value();
    auto const BCastMaster = isBCastMaster();
    if (!threads_initialized_ && BCastMaster) {
        init_threads();
    }

    // if listening thread announces new param, broadcast them
    Message_t msg;
    uint64_t msg_size = 0;
    if (BCastMaster) {
        timestamps[0] = context.Time;
        timestamps[1] = context.InstanceTime;
        if (data_has_changed_.load()) {
            std::lock_guard<std::mutex> guard(recv_msgs_mtx_);
            msg = recv_msgs_;
            recv_msgs_.clear();
            data_has_changed_.store(false);
        } else {
            msg = prepare_null_msg();
        }
        msg_size = msg.size();
    }
#    ifdef RV_DEBUG_OUTPUT
    vislib::sys::Log::DefaultLog.WriteInfo("RendernodeView: Starting broadcast.");
#    endif
    MPI_Bcast(timestamps.data(), 2, MPI_DOUBLE, bcast_rank_, this->comm_);
    MPI_Bcast(&msg_size, 1, MPI_UINT64_T, bcast_rank_, this->comm_);
    msg.resize(msg_size);
    MPI_Bcast(msg.data(), msg_size, MPI_UNSIGNED_CHAR, bcast_rank_, this->comm_);
#    ifdef RV_DEBUG_OUTPUT
    vislib::sys::Log::DefaultLog.WriteInfo("RendernodeView: Finished broadcast.");
#    endif

    // handle new param from broadcast
    if (!process_msgs(msg)) {
        vislib::sys::Log::DefaultLog.WriteError(
            "RendernodeView: Error occured during processing of broadcasted messages.");
    }

    // initialize rendering
    int allFnameDirty = 0;
    auto ss = this->sync_data_slot_.CallAs<core::cluster::SyncDataSourcesCall>();
    if (ss != nullptr) {
        if (!(*ss)(0)) { // check for dirty filenamesslot
            vislib::sys::Log::DefaultLog.WriteError("RendernodeView: SyncData GetDirty callback failed.");
            return;
        }
        int fnameDirty = ss->getFilenameDirty();
        MPI_Allreduce(&fnameDirty, &allFnameDirty, 1, MPI_INT, MPI_LAND, this->comm_);
#    ifdef RV_DEBUG_OUTPUT
        vislib::sys::Log::DefaultLog.WriteInfo("RendernodeView: allFnameDirty: %d", allFnameDirty);
#    endif

        if (allFnameDirty) {
            if (!(*ss)(1)) { // finally set the filename in the data source
                vislib::sys::Log::DefaultLog.WriteError("RendernodeView: SyncData SetFilename callback failed.");
                return;
            }
            ss->resetFilenameDirty();
        }
#    ifdef RV_DEBUG_OUTPUT
        if (!allFnameDirty && fnameDirty) {
            vislib::sys::Log::DefaultLog.WriteInfo("RendernodeView: Waiting for data in MPI world to be ready.");
        }
#    endif
    } else {
#    ifdef RV_DEBUG_OUTPUT
        vislib::sys::Log::DefaultLog.WriteInfo("RendernodeView: No sync object connected.");
#    endif
    }
    // check whether rendering is possible in current state
    auto crv = this->getCallRenderView();
    if (crv != nullptr) {
        crv->ResetAll();
        crv->SetTime(static_cast<float>(timestamps[0]));
        crv->SetInstanceTime(timestamps[1]);
        crv->SetGpuAffinity(context.GpuAffinity);
        crv->SetProjection(this->getProjType(), this->getEye());

        if (this->hasTile()) {
            crv->SetTile(this->getVirtWidth(), this->getVirtHeight(), this->getTileX(), this->getTileY(),
                this->getTileW(), this->getTileH());
        }


#    ifdef CINEMA
        unsigned int numLoops = 5;


        // All ranks calculate the camera positions 
        if (allFnameDirty) {
            // we need to call the render callback to get the correct bbox
            if (!crv->operator()(core::view::CallRenderView::CALL_RENDER)) {
                vislib::sys::Log::DefaultLog.WriteError("RendernodeView: Failed to call render on dependend view.");
            }
            _index = 0;
            auto view = this->getConnectedView();
            core::CallerSlot* crSlot = dynamic_cast<core::CallerSlot*>(view->FindSlot("rendering"));
            if (crSlot == nullptr) return;
            core::view::CallRender3D* cr = crSlot->CallAs<core::view::CallRender3D>();
            if (cr == nullptr) return;

            // auto box = cr->GetBoundingBoxes().ObjectSpaceBBox();
            auto box = cr->GetBoundingBoxes().WorldSpaceBBox();
            std::array<float, 3> dims = {box.Width(), box.Height(), box.Depth()};
            unsigned int max_dim = std::distance(dims.begin(), std::max_element(dims.begin(), dims.end()));

            unsigned int num_sections = 16;
            unsigned int num_angles = 16;
            float length_step_size = box.LongestEdge() / (num_sections - 3); // includes end of cylinder
            float angle_step_size = 2 * 3.14159265358979f / num_angles;
            vislib::math::Point<float, 3> la;
            vislib::math::Point<float, 3> pos;

            std::function<std::array<float, 3>(float, float)> parametrization;
            float radius;
            std::array<float, 3> start;
            std::array<float, 3> direction;
            //= []() { print_num(42); }
            if (max_dim == 0) { // x
                radius = std::sqrt(std::pow(box.Height(), 2) + std::pow(box.Depth(), 2)) / 2;
                start = {box.GetLeft(), box.GetBottom() + box.Height() / 2, box.GetFront() - box.Depth() / 2};
                direction = {1, 0, 0};

                parametrization = [](float r, float angle) {
                    return std::array<float, 3>{0, r * cos(angle), r * sin(angle)};
                };

            } else if (max_dim == 1) { // y
                radius = std::sqrt(std::pow(box.Width(), 2) + std::pow(box.Depth(), 2)) / 2;
                start = {box.GetLeft() + box.Width() / 2, box.GetBottom(), box.GetFront() - box.Depth() / 2};
                direction = {0, 1, 0};

                parametrization = [](float r, float angle) {
                    return std::array<float, 3>{r * cos(angle), 0, r * sin(angle)};
                };
            } else { // z
                radius = std::sqrt(std::pow(box.Height(), 2) + std::pow(box.Width(), 2)) / 2;
                start = {box.GetLeft() + box.Width() / 2, box.GetBottom() + box.Height() / 2, box.GetFront()};
                direction = {0, 0, 1};

                parametrization = [](float r, float angle) {
                    return std::array<float, 3>{r * cos(angle), r * sin(angle), 0};
                };
            }

            _cinemaCams.resize(num_sections * num_angles);

            float radius_offset_cylinder = 5.0f;
            float radius_offset_spheres = 2.0f;

            // start sphere
            for (unsigned int j = 0; j < num_angles; j++) {
                for (unsigned int n = 0; n < 3; n++) {

                    pos[n] = start[n] + parametrization((radius + radius_offset_spheres), angle_step_size * j)[n] +
                             -(radius + radius_offset_spheres) * direction[n];

                    la[n] = start[n];
                }
                _cinemaCams[j].SetView(pos, la, {direction[0], direction[1], direction[2]});
            }


            // middle part
            for (unsigned int i = 0; i < num_sections - 2; i++) {
                for (unsigned int j = 0; j < num_angles; j++) {
                    for (unsigned int n = 0; n < 3; n++) {

                        pos[n] = start[n] + parametrization(radius + radius_offset_cylinder, angle_step_size * j)[n] +
                                 length_step_size * i * direction[n];

                        la[n] = start[n] + length_step_size * i * direction[n];
                    }
                    _cinemaCams[(i + 1) * num_angles + j].SetView(pos, la, {direction[0], direction[1], direction[2]});
                }
            }

            // end sphere
            for (unsigned int j = 0; j < num_angles; j++) {
                for (unsigned int n = 0; n < 3; n++) {

                    pos[n] = start[n] + parametrization((radius + radius_offset_spheres), angle_step_size * j)[n] +
                             (radius + radius_offset_spheres + box.LongestEdge()) * direction[n];

                    la[n] = start[n] + box.LongestEdge() * direction[n];
                }
                _cinemaCams[(num_sections - 1) * (num_angles) + j].SetView(
                    pos, la, {direction[0], direction[1], direction[2]});
            }

        } else if (!_cinemaCams.empty()) {
            auto view = this->getConnectedView();
            core::CallerSlot* crSlot = dynamic_cast<core::CallerSlot*>(view->FindSlot("rendering"));
            if (crSlot == nullptr) return;
            core::view::CallRender3D* cr = crSlot->CallAs<core::view::CallRender3D>();
            if (cr == nullptr) return;

            // std::this_thread::sleep_for(std::chrono::milliseconds(100));

            if (_index  < _cinemaCams.size()) {
                cr->SetCameraView(
                    _cinemaCams[_index].Position(), _cinemaCams[_index].LookAt(), _cinemaCams[_index].Up());
            }
        }
#    endif


        crv->SetOutputBuffer(GL_BACK, this->getViewportWidth(), this->getViewportHeight());

        if (!crv->operator()(core::view::CallRenderView::CALL_RENDER)) {
            vislib::sys::Log::DefaultLog.WriteError("RendernodeView: Failed to call render on dependend view.");
        }


#    ifdef CINEMA
        if (!_cinemaCams.empty()) {
            std::stringstream _path;
            std::stringstream _filename;
            if (_index >= 0 && this->rank_ == bcast_rank_) {


                _filename << _loopID << "_" << std::setfill('0') << std::setw(3) << _index << ".png";

#        ifndef _WIN32
                _path << "/dev/shm/";
                // get job number
                std::string jobID = std::string(vislib::sys::Environment::GetVariable("SLURM_JOB_ID"));
                if (jobID.empty()) jobID = "test";
                _path << jobID << "/";
#        endif


                // read FBO
                std::vector<char> col_buf(crv->ViewportWidth() * crv->ViewportHeight() * 3);
                glReadPixels(
                    0, 0, crv->ViewportWidth(), crv->ViewportHeight(), GL_RGB, GL_UNSIGNED_BYTE, col_buf.data());

                try {
                    PNGWriter png_writer;
                    png_writer.setup((_path.str() + _filename.str()).c_str());
                    png_writer.set_buffer(
                        reinterpret_cast<BYTE*>(col_buf.data()), crv->ViewportWidth(), crv->ViewportHeight(), 3);
                    png_writer.render2file();
                    png_writer.finish();
                } catch (...) {
                    vislib::sys::Log::DefaultLog.WriteError("RendernodeView: Exception while writing PNG\n");
                }
            }


            if (_index >= _cinemaCams.size() - 1) {
                _index = 0;
                _loopID++;
            } else {
                _index++;
            }

            if (_loopID + 1 >= numLoops) {
                vislib::sys::Log::DefaultLog.WriteInfo("RendernodeView: All screenshots taken. Shutting down.");
#        ifndef _WIN32
                if (this->rank_ == bcast_rank_) {
                    const std::string scratch = std::string(vislib::sys::Environment::GetVariable("SCRATCH")) + "/";
                    std::stringstream move_command;
                    move_command << "mv " << _path.str() << " " << scratch;
                    ::system(move_command.str().c_str());
                }
#        endif
                this->GetCoreInstance()->Shutdown();
                return;
            }
        }

#    endif // CINEMA


        glFinish();
    } else {
#    ifdef RV_DEBUG_OUTPUT
        vislib::sys::Log::DefaultLog.WriteWarn("RendernodeView: crv_ is nullptr.\n");
#    endif
    }

    // sync barrier
    MPI_Barrier(this->comm_);
#endif
}


void megamol::pbs::RendernodeView::recv_loop() {
    using namespace std::chrono_literals;
    vislib::sys::Log::DefaultLog.WriteInfo("RendernodeView: Starting recv_loop.");
    try {
        while (run_threads) {
            Message_t buf = {'r', 'e', 'q'};
            auto start = std::chrono::high_resolution_clock::now();
            if (!recv_comm_.Send(buf, send_type::SEND)) {
#ifdef RV_DEBUG_OUTPUT
                vislib::sys::Log::DefaultLog.WriteWarn("RendernodeView: Failed to send request.");
#endif
            }
            // vislib::sys::Log::DefaultLog.WriteInfo(
            //     "RendernodeView: MSG Send took %d ms", (std::chrono::duration_cast<std::chrono::milliseconds>(
            //                                                 std::chrono::high_resolution_clock::now() - start))
            //                                                .count());

            start = std::chrono::high_resolution_clock::now();
            while (!recv_comm_.Recv(buf, recv_type::RECV) && run_threads) {
#ifdef RV_DEBUG_OUTPUT
                vislib::sys::Log::DefaultLog.WriteWarn("RendernodeView: Failed to recv message.");
#endif
            }
            // vislib::sys::Log::DefaultLog.WriteInfo(
            //     "RendernodeView: MSG Recv took %d ms", (std::chrono::duration_cast<std::chrono::milliseconds>(
            //                                                 std::chrono::high_resolution_clock::now() - start))
            //                                                .count());
            if (!run_threads) break;

#ifdef RV_DEBUG_OUTPUT
            vislib::sys::Log::DefaultLog.WriteInfo("RendernodeView: Starting data copy in recv loop.");
#endif
            {
                std::lock_guard<std::mutex> guard(recv_msgs_mtx_);
                recv_msgs_.insert(recv_msgs_.end(), buf.begin(), buf.end());
            }
#ifdef RV_DEBUG_OUTPUT
            vislib::sys::Log::DefaultLog.WriteInfo("RendernodeView: Finished data copy in recv loop.");
#endif

            data_has_changed_.store(true);

            std::this_thread::sleep_for(1000ms / 60);
        }
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError("RendernodeView: Error during communication.");
    }

    vislib::sys::Log::DefaultLog.WriteInfo("RendernodeView: Exiting recv_loop.");
}


bool megamol::pbs::RendernodeView::shutdown_threads() {
    run_threads = false;
    if (receiver_thread_.joinable()) {
        receiver_thread_.join();
    }
    threads_initialized_ = false;
    return true;
}


bool megamol::pbs::RendernodeView::init_threads() {
    shutdown_threads();
    this->recv_comm_ = FBOCommFabric(std::make_unique<ZMQCommFabric>(zmq::socket_type::req));
    auto const address = std::string(this->address_slot_.Param<core::param::StringParam>()->Value());
    this->recv_comm_.Connect(address);
    run_threads = true;
    receiver_thread_ = std::thread(&RendernodeView::recv_loop, this);
    threads_initialized_ = true;
    return true;
}


bool megamol::pbs::RendernodeView::initMPI() {
    bool retval = false;

#ifdef WITH_MPI
    if (this->comm_ == MPI_COMM_NULL) {
        auto const c = this->request_mpi_slot_.CallAs<core::cluster::mpi::MpiCall>();
        if (c != nullptr) {
            /* New method: let MpiProvider do all the stuff. */
            if ((*c)(core::cluster::mpi::MpiCall::IDX_PROVIDE_MPI)) {
                vislib::sys::Log::DefaultLog.WriteInfo("RendernodeView: Got MPI communicator.");
                this->comm_ = c->GetComm();
            } else {
                vislib::sys::Log::DefaultLog.WriteError("RendernodeView: Could not retrieve MPI communicator for the "
                                                        "MPI-based view from the registered provider module.");
            }

        } else {
            vislib::sys::Log::DefaultLog.WriteError(
                "RendernodeView: MPI cannot be initialized lazily. Please initialize MPI before using this module.");
        } /* end if (c != nullptr) */

        if (this->comm_ != MPI_COMM_NULL) {
            vislib::sys::Log::DefaultLog.WriteInfo(
                "RendernodeView: MPI is ready, retrieving communicator properties ...");
            MPI_Comm_rank(this->comm_, &this->rank_);
            MPI_Comm_size(this->comm_, &this->comm_size_);
            vislib::sys::Log::DefaultLog.WriteInfo("RendernodeView on %hs is %d of %d.",
                vislib::sys::SystemInformation::ComputerNameA().PeekBuffer(), this->rank_, this->comm_size_);
        } /* end if (this->comm != MPI_COMM_NULL) */
    }     /* end if (this->comm == MPI_COMM_NULL) */

    /* Determine success of the whole operation. */
    retval = (this->comm_ != MPI_COMM_NULL);
#endif /* WITH_MPI */

    // TODO: Register data types as necessary

    return retval;
}
