#include "stdafx.h"
#include "adiosWriter.h"
#include <chrono>
#include "mmcore/cluster/mpi/MpiCall.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/EnumParam.h"
#include "vislib/Trace.h"
#include "vislib/sys/CmdLineProvider.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/SystemInformation.h"

namespace megamol {
namespace adios {

adiosWriter::adiosWriter(void)
    : core::AbstractDataWriter()
    , callRequestMpi("requestMpi", "Requests initialisation of MPI and the communicator for the view.")
    , filename("filename", "The path to the ADIOS-based file to load.")
    , getData("getdata", "Slot to request data from this data source.")
    , outputPatternSlot("outputPattern","Sets an file IO pattern.")
    , encodingSlot("encoding","Specifiy encoding")
    , io(nullptr) {

    this->filename.SetParameter(new core::param::FilePathParam(""));
    this->MakeSlotAvailable(&this->filename);

    this->getData.SetCompatibleCall<CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->getData);

    auto opat = new core::param::EnumParam(0);
    opat->SetTypePair(0, "PerNode");
    opat->SetTypePair(1, "Parallel");
    this->outputPatternSlot << opat;
    this->MakeSlotAvailable(&this->outputPatternSlot);

    auto encEnum = new core::param::EnumParam(0);
    encEnum->SetTypePair(0, "None");
    this->encodingSlot << encEnum;
    this->MakeSlotAvailable(&this->encodingSlot);

    this->callRequestMpi.SetCompatibleCall<core::cluster::mpi::MpiCallDescription>();
    this->MakeSlotAvailable(&this->callRequestMpi);
}

adiosWriter::~adiosWriter(void) {

    if (writer) {
        writer.Close();
    }
    vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Writer Closed");
    this->Release();
}

/*
 * adiosWriter::create
 */
bool adiosWriter::create(void) {
#ifdef WITH_MPI
    MpiInitialized = this->initMPI();
    try {
        vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Initializing with MPI");
        if (MpiInitialized) {
            adiosInst = adios2::ADIOS(this->mpi_comm_, adios2::DebugON);
        } else {
            adiosInst = adios2::ADIOS(adios2::DebugON);
        }
#else
    try {
        vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Initializing without MPI");
        adiosInst = adios2::ADIOS();
#endif

        vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Declaring IO");
        io = std::make_shared<adios2::IO>(adiosInst.DeclareIO("Output"));
        vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Setting Engine");


        io->SetEngine("BP3");

        } catch (std::invalid_argument& e) {
#ifdef WITH_MPI
        vislib::sys::Log::DefaultLog.WriteError(
            "[adiosWriter] Invalid argument exception, STOPPING PROGRAM from rank %d", this->mpiRank);
#else
        vislib::sys::Log::DefaultLog.WriteError("[adiosWriter] Invalid argument exception, STOPPING PROGRAM");
#endif
        vislib::sys::Log::DefaultLog.WriteError(e.what());
    } catch (std::ios_base::failure& e) {
#ifdef WITH_MPI
        vislib::sys::Log::DefaultLog.WriteError(
            "[adiosWriter] IO System base failure exception, STOPPING PROGRAM from rank %d", this->mpiRank);
#else
        vislib::sys::Log::DefaultLog.WriteError("[adiosWriter] IO System base failure exception, STOPPING PROGRAM");
#endif
        vislib::sys::Log::DefaultLog.WriteError(e.what());
    } catch (std::exception& e) {
#ifdef WITH_MPI
        vislib::sys::Log::DefaultLog.WriteError("[adiosWriter] Exception, STOPPING PROGRAM from rank %d", this->mpiRank);
#else
        vislib::sys::Log::DefaultLog.WriteError("[adiosWriter] Exception, STOPPING PROGRAM");
#endif
        vislib::sys::Log::DefaultLog.WriteError(e.what());
    }

    return true;
}

/*
 * adiosWriter::release
 */
void adiosWriter::release(void) { /* empty */
}

/*
 * adiosWriter::getCapabilities
 */
bool adiosWriter::getCapabilities(core::DataWriterCtrlCall& call) { return true; }

/*
 * adiosWriter::initMPI
 */
bool adiosWriter::initMPI() {
    bool retval = false;
#ifdef WITH_MPI
    if (this->mpi_comm_ == MPI_COMM_NULL) {
        VLTRACE(vislib::Trace::LEVEL_INFO, "[adiosWriter] Need to initialize MPI\n");
        auto c = this->callRequestMpi.CallAs<core::cluster::mpi::MpiCall>();
        if (c != nullptr) {
            /* New method: let MpiProvider do all the stuff. */
            if ((*c)(core::cluster::mpi::MpiCall::IDX_PROVIDE_MPI)) {
                vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Got MPI communicator.");
                this->mpi_comm_ = c->GetComm();
            } else {
                vislib::sys::Log::DefaultLog.WriteError(_T("[adiosWriter] Could not ")
                                                        _T("retrieve MPI communicator for the MPI-based view ")
                                                        _T("from the registered provider module."));
            }
        } else {
            int initializedBefore = 0;
            MPI_Initialized(&initializedBefore);
            if (!initializedBefore) {
                this->mpi_comm_ = MPI_COMM_WORLD;
                vislib::sys::CmdLineProviderA cmdLine(this->getCommandLine());
                int argc = cmdLine.ArgC();
                char** argv = cmdLine.ArgV();
                ::MPI_Init(&argc, &argv);
            }
        }

        if (this->mpi_comm_ != MPI_COMM_NULL) {
            vislib::sys::Log::DefaultLog.WriteInfo(_T("[adiosWriter] MPI is ready, ")
                                                   _T("retrieving communicator properties ..."));
            ::MPI_Comm_rank(this->mpi_comm_, &this->mpiRank);
            ::MPI_Comm_size(this->mpi_comm_, &this->mpiSize);
            vislib::sys::Log::DefaultLog.WriteInfo(_T("[adiosWriter] on %hs is %d ")
                                                   _T("of %d."),
                vislib::sys::SystemInformation::ComputerNameA().PeekBuffer(), this->mpiRank, this->mpiSize);
        } /* end if (this->comm != MPI_COMM_NULL) */
        VLTRACE(vislib::Trace::LEVEL_INFO, "[adiosWriter] MPI initialized: %s (%i)\n",
            this->mpi_comm_ != MPI_COMM_NULL ? "true" : "false", mpi_comm_);
    } /* end if (this->comm == MPI_COMM_NULL) */

    /* Determine success of the whole operation. */
    retval = (this->mpi_comm_ != MPI_COMM_NULL);
#endif /* WITH_MPI */
    return retval;
}

/*
 * adiosWriter::run
 */
bool adiosWriter::run() {

    // get data
    CallADIOSData* cad = this->getData.CallAs<CallADIOSData>();
    if (cad == nullptr) return false;


    if (!(*cad)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("[adiosWriter] Error during GetHeader");
        return false;
    }


    const auto frameCount = cad->getFrameCount();
    for (auto i = 0; i < frameCount; i++) { // for each frame

        vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Starting frame %d", i);

            cad->setFrameIDtoLoad(i);

            auto avaiVars = cad->getAvailableVars();

            if (!(*cad)(0)) {
                vislib::sys::Log::DefaultLog.WriteError("[adiosWriter] Error during GetData");
                return false;
            }

        try {
            if (!this->writer) {
                const std::string fname = std::string(T2A(this->filename.Param<core::param::FilePathParam>()->Value()));
                vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Opening File %s", fname.c_str());
                writer = io->Open(fname, adios2::Mode::Write);
            }

            vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] BeginStep");
            writer.BeginStep();

            io->RemoveAllVariables();
            for (auto var : avaiVars) {

                std::vector<size_t> globalDim;
                std::vector<size_t> offsets;
                std::vector<size_t> localDim;

                const size_t num = cad->getData(var)->size();

                if (this->outputPatternSlot.Param<core::param::EnumParam>()->Value() == 1 && !cad->getData(var)->singleValue) {
                    std::vector<size_t> shape;
                    if (!cad->getData(var)->shape.empty())
                        shape = cad->getData(var)->shape;
                    else {
                        shape = {cad->getData(var)->size()};
                    }
                    localDim = shape;
                    globalDim = localDim;
#ifdef WITH_MPI                  
                    offsets.resize(shape.size());
                    // offsets
                    auto mpierror = MPI_Scan(localDim.data(), offsets.data(), 1, MPI_UINT64_T, MPI_SUM, this->mpi_comm_);
                    if (mpierror != MPI_SUCCESS)
                        vislib::sys::Log::DefaultLog.WriteError("[adiosWriter] MPI_Allreduce of offsets failed.");
                    offsets[0] -= localDim[0];
                    // global dim
                    mpierror = MPI_Allreduce(localDim.data(), globalDim.data(), 1, MPI_UINT64_T, MPI_SUM, this->mpi_comm_);
                    if (mpierror != MPI_SUCCESS)
                        vislib::sys::Log::DefaultLog.WriteError("[adiosWriter] MPI_Allreduce of offsets failed.");
                    //vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter]");
#else
                    globalDim = shape;
                    offsets = std::vector<size_t>(shape.size(), 0);
#endif
                } else {
                    globalDim = {static_cast<size_t>(num)};
                    offsets = {static_cast<size_t>(0)};
                    localDim = {static_cast<size_t>(num)};
                }

                if (cad->getData(var)->getType() == "float") {

                    std::vector<float>& values = dynamic_cast<FloatContainer*>(cad->getData(var).get())->getVec();

                    vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Defining Variables");
                    adios2::Variable<float> adiosVar =
                        io->DefineVariable<float>(var, globalDim, offsets, localDim, false);

                    vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Putting Variables");
                    if (adiosVar) writer.Put<float>(adiosVar, values.data());
                } else if (cad->getData(var)->getType() == "double") {

                    std::vector<double>& values = dynamic_cast<DoubleContainer*>(cad->getData(var).get())->getVec();

                    vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Defining Variables");
                    adios2::Variable<double> adiosVar =
                        io->DefineVariable<double>(var, globalDim, offsets, localDim, false);

                    vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Putting Variables");
                    if (adiosVar) writer.Put<double>(adiosVar, values.data());
                } else if (cad->getData(var)->getType() == "int32_t") {

                    std::vector<int32_t>& values = dynamic_cast<Int32Container*>(cad->getData(var).get())->getVec();

                    vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Defining Variables");
                    adios2::Variable<int> adiosVar = io->DefineVariable<int>(var, globalDim, offsets, localDim, false);

                    vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Putting Variables");
                    if (adiosVar) writer.Put<int32_t>(adiosVar, values.data());
                } else if (cad->getData(var)->getType() == "uint64_t") {

                    std::vector<uint64_t>& values =
                        dynamic_cast<UInt64Container*>(cad->getData(var).get())->getVec();

                    vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Defining Variables");
                    adios2::Variable<uint64_t> adiosVar =
                        io->DefineVariable<uint64_t>(var, globalDim, offsets, localDim, false);

                    vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Putting Variables");
                    if (adiosVar) writer.Put<uint64_t>(adiosVar, values.data());
                } else if (cad->getData(var)->getType() == "unsigned char") {

                    std::vector<unsigned char>& values =
                        dynamic_cast<UCharContainer*>(cad->getData(var).get())->getVec();

                    vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Defining Variables");
                    adios2::Variable<unsigned char> adiosVar =
                        io->DefineVariable<unsigned char>(var, globalDim, offsets, localDim, false);

                    vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Putting Variables");
                    if (adiosVar) writer.Put<unsigned char>(adiosVar, values.data());
                } else if (cad->getData(var)->getType() == "uint32_t") {

                    std::vector<unsigned int>& values =
                        dynamic_cast<UInt32Container*>(cad->getData(var).get())->getVec();

                    vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Defining Variables");
                    adios2::Variable<unsigned int> adiosVar =
                        io->DefineVariable<unsigned int>(var, globalDim, offsets, localDim, false);

                    vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Putting Variables");
                    if (adiosVar) writer.Put<unsigned int>(adiosVar, values.data());
                }
                vislib::sys::Log::DefaultLog.WriteInfo(
                    "[adiosWriter] Trying to write - var: %s size: %d", var.c_str(), num);
            }

            vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] EndStep");
            const auto t1 = std::chrono::high_resolution_clock::now();
            writer.EndStep();
            const auto t2 = std::chrono::high_resolution_clock::now();
            const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Time spent for writing frame: %d ms", duration);

        } catch (std::invalid_argument& e) {
#ifdef WITH_MPI
            vislib::sys::Log::DefaultLog.WriteError(
                "[adiosWriter] Invalid argument exception, STOPPING PROGRAM from rank %d", this->mpiRank);
#else
            vislib::sys::Log::DefaultLog.WriteError("[adiosWriter] Invalid argument exception, STOPPING PROGRAM");
#endif
            vislib::sys::Log::DefaultLog.WriteError(e.what());
        } catch (std::ios_base::failure& e) {
#ifdef WITH_MPI
            vislib::sys::Log::DefaultLog.WriteError(
                "[adiosWriter] IO System base failure exception, STOPPING PROGRAM from rank %d", this->mpiRank);
#else
            vislib::sys::Log::DefaultLog.WriteError("[adiosWriter] IO System base failure exception, STOPPING PROGRAM");
#endif
            vislib::sys::Log::DefaultLog.WriteError(e.what());
        } catch (std::exception& e) {
#ifdef WITH_MPI
            vislib::sys::Log::DefaultLog.WriteError("[adiosWriter] Exception, STOPPING PROGRAM from rank %d", this->mpiRank);
#else
            vislib::sys::Log::DefaultLog.WriteError("[adiosWriter] Exception, STOPPING PROGRAM");
#endif
            vislib::sys::Log::DefaultLog.WriteError(e.what());
        }

    } // end for each frame

    return true;
}


vislib::StringA adiosWriter::getCommandLine(void) {
    vislib::StringA retval;

#ifdef WIN32
    retval = ::GetCommandLineA();
#else  /* _WIN32 */
    char* arg = nullptr;
    size_t size = 0;

    auto fp = ::fopen("/proc/self/cmdline", "rb");
    if (fp != nullptr) {
        while (::getdelim(&arg, &size, 0, fp) != -1) {
            retval.Append(arg, size);
            retval.Append(" ");
        }
        ::free(arg);
        ::fclose(fp);
    }
#endif /* _WIN32 */

    vislib::sys::Log::DefaultLog.WriteInfo("[adiosWriter] Command line used for MPI "
                                           "initialisation is \"%s\".",
        retval.PeekBuffer());
    return retval;
}

} // namespace adios
} // namespace megamol
