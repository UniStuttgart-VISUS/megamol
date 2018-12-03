#include "stdafx.h"
#include "adiosWriter.h"
#include "mmcore/cluster/mpi/MpiCall.h"
#include "mmcore/param/FilePathParam.h"
#include "vislib/Trace.h"
#include "vislib/sys/CmdLineProvider.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/SystemInformation.h"

namespace megamol {
namespace adios {

adiosWriter::adiosWriter(void)
    : core::AbstractDataWriter()
    , filename("filename", "The path to the ADIOS-based file to load.")
    , getData("getdata", "Slot to request data from this data source.")
    , callRequestMpi("requestMpi", "Requests initialisation of MPI and the communicator for the view.")
    , io(nullptr)
    , writer() {

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
                vislib::sys::CmdLineProviderA cmdLine(::GetCommandLineA());
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
        vislib::sys::Log::DefaultLog.WriteError("ADIOStoMultiParticle: Error during GetHeader");
        return false;
    }

    cad->inquire("x");
    cad->inquire("y");
    cad->inquire("z");
    cad->inquire("box");

    if (!(*cad)(0)) {
        vislib::sys::Log::DefaultLog.WriteError("ADIOStoMultiParticle: Error during GetData");
        return false;
    }

    auto X = cad->getData("x")->GetAsFloat();
    auto Y = cad->getData("y")->GetAsFloat();
    auto Z = cad->getData("z")->GetAsFloat();
    auto box = cad->getData("box")->GetAsFloat();

    size_t num_particles = X.size();


    adios2::Variable<float> varX;
    adios2::Variable<float> varY;
    adios2::Variable<float> varZ;
    adios2::Variable<float> varBox;

    vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: Defining Variables");
    varX = io->DefineVariable<float>("x", 
        {static_cast<unsigned long long>(this->mpiSize * num_particles)},
        {static_cast<unsigned long long>(this->mpiRank * num_particles)}, {static_cast<unsigned long long>(num_particles)});
    varY = io->DefineVariable<float>("y", 
        {static_cast<unsigned long long>(this->mpiSize * num_particles)},
        {static_cast<unsigned long long>(this->mpiRank * num_particles)},
        {static_cast<unsigned long long>(num_particles)});
    varZ = io->DefineVariable<float>("z", 
        {static_cast<unsigned long long>(this->mpiSize * num_particles)},
        {static_cast<unsigned long long>(this->mpiRank * num_particles)},
        {static_cast<unsigned long long>(num_particles)});
    varBox = io->DefineVariable<float>("box",
        {static_cast<unsigned long long>(box.size())}, {0}, {static_cast<unsigned long long>(box.size())});


    const std::string fname = std::string(T2A(this->filename.Param<core::param::FilePathParam>()->Value()));
    vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2: Opening File %s", fname.c_str());
    writer = io->Open(fname, adios2::Mode::Write);


    vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: BeginStep");
    writer.BeginStep();

    vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: Putting Variables");
    if (varX) writer.Put<float>(varX, X.data());
    if (varY) writer.Put<float>(varY, Y.data());
    if (varZ) writer.Put<float>(varZ, Z.data());
    if (varBox) writer.Put<float>(varBox, box.data());


    vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2writer: EndStep");
    writer.EndStep();

    return true;
}


} // namespace adios
} // namespace megamol
