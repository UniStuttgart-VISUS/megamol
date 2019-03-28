#include "stdafx.h"
#include "adiosWriter.h"
#include "mmcore/cluster/mpi/MpiCall.h"
#include "mmcore/param/FilePathParam.h"
#include "vislib/Trace.h"
#include "vislib/sys/CmdLineProvider.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/SystemInformation.h"
#include <chrono>

namespace megamol {
namespace adios {

adiosWriter::adiosWriter(void) : core::AbstractDataWriter()
    , callRequestMpi("requestMpi", "Requests initialisation of MPI and the communicator for the view.")
    , filename("filename", "The path to the ADIOS-based file to load.")
    , getData("getdata", "Slot to request data from this data source.")
    , io(nullptr) {

    this->filename.SetParameter(new core::param::FilePathParam(""));
    this->MakeSlotAvailable(&this->filename);

    this->getData.SetCompatibleCall<CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->getData);

    this->callRequestMpi.SetCompatibleCall<core::cluster::mpi::MpiCallDescription>();
    this->MakeSlotAvailable(&this->callRequestMpi);
}

adiosWriter::~adiosWriter(void) {

    if (writer) {
       writer.Close();
    }
    vislib::sys::Log::DefaultLog.WriteInfo("Writer Closed");
    this->Release();
}

/*
 * adiosWriter::create
 */
bool adiosWriter::create(void) {
    MpiInitialized = this->initMPI();
    try {
        vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: Initializing");
        if (MpiInitialized) {
            adiosInst = adios2::ADIOS(this->mpi_comm_, adios2::DebugON);
        } else {
            adiosInst = adios2::ADIOS(adios2::DebugON);
        }

        vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: Declaring IO");
        io = std::make_shared<adios2::IO>(adiosInst.DeclareIO("Output"));
        vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: Setting Engine");


        io->SetEngine("BPFile");

    } catch (std::invalid_argument& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Invalid argument exception, STOPPING PROGRAM from rank %d", this->mpiRank);
        vislib::sys::Log::DefaultLog.WriteError(e.what());
    } catch (std::ios_base::failure& e) {
        vislib::sys::Log::DefaultLog.WriteError(
            "IO System base failure exception, STOPPING PROGRAM from rank %d", this->mpiRank);
        vislib::sys::Log::DefaultLog.WriteError(e.what());
    } catch (std::exception& e) {
        vislib::sys::Log::DefaultLog.WriteError("Exception, STOPPING PROGRAM from rank %d", this->mpiRank);
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
        VLTRACE(vislib::Trace::LEVEL_INFO, "adiosWriter: Need to initialize MPI\n");
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
            vislib::sys::Log::DefaultLog.WriteInfo(_T("MPI is ready, ")
                                                   _T("retrieving communicator properties ..."));
            ::MPI_Comm_rank(this->mpi_comm_, &this->mpiRank);
            ::MPI_Comm_size(this->mpi_comm_, &this->mpiSize);
            vislib::sys::Log::DefaultLog.WriteInfo(_T("This view on %hs is %d ")
                                                   _T("of %d."),
                vislib::sys::SystemInformation::ComputerNameA().PeekBuffer(), this->mpiRank, this->mpiSize);
        } /* end if (this->comm != MPI_COMM_NULL) */
        VLTRACE(vislib::Trace::LEVEL_INFO, "adiosWriter: MPI initialized: %s (%i)\n",
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
    
    //get data
    CallADIOSData* cad = this->getData.CallAs<CallADIOSData>();
    if (cad == nullptr) return false;


    if (!(*cad)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("ADIOS2writer: Error during GetHeader");
        return false;
    }


    const auto frameCount = cad->getFrameCount();
    for (auto i = 0; i < frameCount; i++) { // for each frame

        vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: Starting frame %d", i);

        try {
        
        cad->setFrameIDtoLoad(i);

        auto avaiVars = cad->getAvailableVars();

        for (auto var : avaiVars) {
            cad->inquire(var);
        }

        if (!(*cad)(0)) {
            vislib::sys::Log::DefaultLog.WriteError("ADIOS2writer: Error during GetData");
            return false;
        }

        if (!this->writer) {
            const std::string fname = std::string(T2A(this->filename.Param<core::param::FilePathParam>()->Value()));
            vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2: Opening File %s", fname.c_str());
            writer = io->Open(fname, adios2::Mode::Write);
        }

        vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: BeginStep");
        writer.BeginStep();

        std::vector<std::shared_ptr<float*>> fCollector;
        std::vector<std::shared_ptr<int>> iCollector;



        io->RemoveAllVariables();
        for (auto var : avaiVars) {

            const size_t num = cad->getData(var)->size();
            if (cad->getData(var)->getType() == "float") {

                std::vector<float>& values = dynamic_cast<FloatContainer*>(cad->getData(var).get())->getVec();

                vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: Defining Variables");
                adios2::Variable<float> adiosVar =
                    io->DefineVariable<float>(var, {static_cast<size_t>(this->mpiSize * num)},
                        {static_cast<size_t>(this->mpiRank * num)}, {static_cast<size_t>(num)}, false);
                adiosVar.SetShape({this->mpiSize * num});

                vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: Putting Variables");
                if (adiosVar) writer.Put<float>(adiosVar, values.data());
            } else if (cad->getData(var)->getType() == "double") {

                std::vector<double>& values = dynamic_cast<DoubleContainer*>(cad->getData(var).get())->getVec();

                vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: Defining Variables");
                adios2::Variable<double> adiosVar =
                    io->DefineVariable<double>(var, {static_cast<size_t>(this->mpiSize * num)},
                        {static_cast<size_t>(this->mpiRank * num)}, {static_cast<size_t>(num)});
                adiosVar.SetShape({this->mpiSize * num});

                vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: Putting Variables");
                if (adiosVar) writer.Put<double>(adiosVar, values.data());
            } else if (cad->getData(var)->getType() == "int") {

                std::vector<int>& values = dynamic_cast<IntContainer*>(cad->getData(var).get())->getVec();

                vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: Defining Variables");
                adios2::Variable<int> adiosVar =
                    io->DefineVariable<int>(var, {static_cast<size_t>(this->mpiSize * num)},
                        {static_cast<size_t>(this->mpiRank * num)}, {static_cast<size_t>(num)});
                adiosVar.SetShape({this->mpiSize * num});

                vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: Putting Variables");
                if (adiosVar) writer.Put<int>(adiosVar, values.data());
            } else if (cad->getData(var)->getType() == "unsigned long long int") {

                std::vector<unsigned long long int>& values =
                    dynamic_cast<UInt64Container*>(cad->getData(var).get())->getVec();

                vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: Defining Variables");
                adios2::Variable<unsigned long long int> adiosVar =
                    io->DefineVariable<unsigned long long int>(var, {static_cast<size_t>(this->mpiSize * num)},
                        {static_cast<size_t>(this->mpiRank * num)}, {static_cast<size_t>(num)});
                adiosVar.SetShape({this->mpiSize * num});

                vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: Putting Variables");
                if (adiosVar) writer.Put<unsigned long long int>(adiosVar, values.data());
            } else if (cad->getData(var)->getType() == "unsigned char") {

                std::vector<unsigned char>& values =
                    dynamic_cast<UCharContainer*>(cad->getData(var).get())->getVec();

                vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: Defining Variables");
                adios2::Variable<unsigned char> adiosVar =
                    io->DefineVariable<unsigned char>(var, {static_cast<size_t>(this->mpiSize * num)},
                        {static_cast<size_t>(this->mpiRank * num)}, {static_cast<size_t>(num)});
                adiosVar.SetShape({this->mpiSize * num});

                vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: Putting Variables");
                if (adiosVar) writer.Put<unsigned char>(adiosVar, values.data());
            } else if (cad->getData(var)->getType() == "unsigned int") {

                std::vector<unsigned int>& values = dynamic_cast<UInt32Container*>(cad->getData(var).get())->getVec();

                vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: Defining Variables");
                adios2::Variable<unsigned int> adiosVar =
                    io->DefineVariable<unsigned int>(var, {static_cast<size_t>(this->mpiSize * num)},
                        {static_cast<size_t>(this->mpiRank * num)}, {static_cast<size_t>(num)});
                adiosVar.SetShape({this->mpiSize * num});

                vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: Putting Variables");
                if (adiosVar) writer.Put<unsigned int>(adiosVar, values.data());
            }
            vislib::sys::Log::DefaultLog.WriteInfo(
                "ADIOS2writer: Trying to write - var: %s size: %d", var.c_str(), num);
        }

        vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: EndStep");
        auto t1 = std::chrono::high_resolution_clock::now();
        writer.EndStep();
        auto t2 = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: Time spent for writing frame: %d us", duration);

        } catch (std::invalid_argument& e) {
            vislib::sys::Log::DefaultLog.WriteError(
                "Invalid argument exception, STOPPING PROGRAM from rank %d", this->mpiRank);
            vislib::sys::Log::DefaultLog.WriteError(e.what());
        } catch (std::ios_base::failure& e) {
            vislib::sys::Log::DefaultLog.WriteError(
                "IO System base failure exception, STOPPING PROGRAM from rank %d", this->mpiRank);
            vislib::sys::Log::DefaultLog.WriteError(e.what());
        } catch (std::exception& e) {
            vislib::sys::Log::DefaultLog.WriteError("Exception, STOPPING PROGRAM from rank %d", this->mpiRank);
            vislib::sys::Log::DefaultLog.WriteError(e.what());
        }

    } // end for each frame

    return true;
}


vislib::StringA adiosWriter::getCommandLine(void) {
    vislib::StringA retval;

#ifdef WIN32
    retval = ::GetCommandLineA();
#else /* _WIN32 */
    char *arg = nullptr;
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

    vislib::sys::Log::DefaultLog.WriteInfo("Command line used for MPI "
        "initialisation is \"%s\".", retval.PeekBuffer());
    return retval;
}

} // namespace adios
} // namespace megamol
