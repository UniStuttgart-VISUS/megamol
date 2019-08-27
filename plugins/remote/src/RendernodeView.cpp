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


megamol::remote::RendernodeView::RendernodeView()
    : request_mpi_slot_("requestMPI", "Requests initialization of MPI and the communicator for the view.")
    , sync_data_slot_("syncData", "Requests synchronization of data sources in the MPI world.")
    , BCastRankSlot_("BCastRank", "Set which MPI rank is the broadcast master")
    , address_slot_("address", "Address of headnode in ZMQ syntax (e.g. \"tcp://127.0.0.1:33333\")")
    , recv_comm_(std::make_unique<ZMQCommFabric>(zmq::socket_type::pull))
    , run_threads(false)
#ifdef WITH_MPI
    , comm_(MPI_COMM_NULL)
#else
    , comm_(0x04000000)
#endif
    , rank_(-1)
    , bcast_rank_(-2)
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


megamol::remote::RendernodeView::~RendernodeView() { this->Release(); }


void megamol::remote::RendernodeView::release(void) { shutdown_threads(); }


bool megamol::remote::RendernodeView::create(void) { return true; }


bool megamol::remote::RendernodeView::process_msgs(Message_t const& msgs) {
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

            /*if (std::distance(ibegin, iend) > MessageHeaderSize) {
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
            }*/
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


void megamol::remote::RendernodeView::Render(const mmcRenderViewContext& context) {
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
    auto ss = this->sync_data_slot_.CallAs<core::cluster::SyncDataSourcesCall>();
    if (ss != nullptr) {
        if (!(*ss)(0)) { // check for dirty filenamesslot
            vislib::sys::Log::DefaultLog.WriteError("RendernodeView: SyncData GetDirty callback failed.");
            return;
        }
        int fnameDirty = ss->getFilenameDirty();
        int allFnameDirty = 0;
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

        crv->SetOutputBuffer(GL_BACK, this->getViewportWidth(), this->getViewportHeight());

        if (!crv->operator()(core::view::CallRenderView::CALL_RENDER)) {
            vislib::sys::Log::DefaultLog.WriteError("RendernodeView: Failed to call render on dependend view.");
        }

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


bool megamol::remote::RendernodeView::OnRenderView(core::Call& call) {
    auto crv = dynamic_cast<core::view::CallRenderView*>(&call);
    if (crv == nullptr) return false;
    auto overrideCall = dynamic_cast<core::view::AbstractCallRender*>(&call);

    float time = crv->Time();
    if (time < 0.0f) time = this->DefaultTime(crv->InstanceTime());
    mmcRenderViewContext context;
    ::ZeroMemory(&context, sizeof(context));
    context.Time = time;
    context.InstanceTime = crv->InstanceTime();

    this->Render(context);

	return true;
}


void megamol::remote::RendernodeView::recv_loop() {
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


bool megamol::remote::RendernodeView::shutdown_threads() {
    run_threads = false;
    if (receiver_thread_.joinable()) {
        receiver_thread_.join();
    }
    threads_initialized_ = false;
    return true;
}


bool megamol::remote::RendernodeView::init_threads() {
    shutdown_threads();
    this->recv_comm_ = FBOCommFabric(std::make_unique<ZMQCommFabric>(zmq::socket_type::req));
    auto const address = std::string(this->address_slot_.Param<core::param::StringParam>()->Value());
    this->recv_comm_.Connect(address);
    run_threads = true;
    receiver_thread_ = std::thread(&RendernodeView::recv_loop, this);
    threads_initialized_ = true;
    return true;
}


bool megamol::remote::RendernodeView::initMPI() {
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
