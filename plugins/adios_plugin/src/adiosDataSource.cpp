#include "stdafx.h"
#include "adiosDataSource.h"
#include "mmcore/cluster/mpi/MpiCall.h"
#include "mmcore/param/FilePathParam.h"
#include "vislib/Trace.h"
#include "vislib/sys/CmdLineProvider.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/SystemInformation.h"
#include <algorithm>

namespace megamol {
namespace adios {

adiosDataSource::adiosDataSource(void)
    : core::Module()
    , callRequestMpi("requestMpi", "Requests initialization of MPI and the communicator for the view.")
    , getData("getdata", "Slot to request data from this data source.")
    , data_hash(0)
    , filename("filename", "The path to the ADIOS-based file to load.")
    , frameCount(0)
    , loadedFrameID(-1)
    , io(nullptr) {

    this->filename.SetParameter(new core::param::FilePathParam(""));
    this->filename.SetUpdateCallback(&adiosDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);


    this->getData.SetCallback("CallADIOSData", "GetData", &adiosDataSource::getDataCallback);
    this->getData.SetCallback("CallADIOSData", "GetHeader", &adiosDataSource::getHeaderCallback);
    this->MakeSlotAvailable(&this->getData);

    this->callRequestMpi.SetCompatibleCall<core::cluster::mpi::MpiCallDescription>();
    this->MakeSlotAvailable(&this->callRequestMpi);
}


/**
 * adiosDataSource::~adiosDataSource(void)
 */
adiosDataSource::~adiosDataSource() { this->Release(); }


/*
 * adiosDataSource::create
 */
bool adiosDataSource::create() {
    MpiInitialized = this->initMPI();
    vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2: Initializing");
    if (MpiInitialized) {
        adiosInst = adios2::ADIOS(this->mpi_comm_, adios2::DebugON);
    } else {
        adiosInst = adios2::ADIOS(adios2::DebugON);
    }

    vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2: Declaring IO");
    io = std::make_shared<adios2::IO>(adiosInst.DeclareIO("ReadBP"));

    return true;
}


/*
 * adiosDataSource::release
 */
void adiosDataSource::release() { /* empty */
}


/*
 * adiosDataSource::getDataCallback
 */
bool adiosDataSource::getDataCallback(core::Call& caller) {
    CallADIOSData* cad = dynamic_cast<CallADIOSData*>(&caller);
    if (cad == nullptr) return false;

    if (dataHashChanged || loadedFrameID != cad->getFrameIDtoLoad()) {

        try {
            const std::string fname = std::string(T2A(this->filename.Param<core::param::FilePathParam>()->Value()));
            if (this->reader) {
                this->reader.Close();
                this->io->RemoveAllVariables();
            }
            this->reader = io->Open(fname, adios2::Mode::Read);

            vislib::sys::Log::DefaultLog.WriteInfo(
                "ADIOS2datasource: Stepping to frame number: %d", cad->getFrameIDtoLoad());
            if (cad->getFrameIDtoLoad() != 0) {
                for (auto i = 0; i < cad->getFrameIDtoLoad(); i++) {
                    reader.BeginStep();
                    reader.EndStep();
                }
            }

            vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2: Beginning step");
            const adios2::StepStatus status = reader.BeginStep();
            if (status != adios2::StepStatus::OK) {
                vislib::sys::Log::DefaultLog.WriteError("ADIOS2 ERROR: BeginStep returned an error.");
                return false;
            }


            auto varsToInquire = cad->getVarsToInquire();
            if (varsToInquire.empty()) {
                vislib::sys::Log::DefaultLog.WriteError("adiosDataSource: varsToInquire is empty.");
                return false;
            }

            for (auto toInq : varsToInquire) {
                for (auto var : variables) {
					if (var.first == toInq) {
						size_t num = 1;
						bool singleValue = true;
						if (var.second["SingleValue"] != std::string("true")) {
							//num = std::stoi(var.second["Shape"]);
							singleValue = false;
						}
						if (var.second["Type"] == "float") {

                            auto fc = std::make_shared<FloatContainer>(FloatContainer());
                            std::vector<float>& tmp_vec = fc->getVec();
                            adios2::Variable<float> advar = io->InquireVariable<float>(var.first);
                            auto info = reader.BlocksInfo(advar, cad->getFrameIDtoLoad());
                            num = info[0].Count[0];
                            tmp_vec.resize(num);
						    if (this->MpiInitialized && !singleValue) {
                                advar.SetSelection({{num * this->mpiRank}, {num}});
                            }
                            
                            reader.Get<float>(advar, tmp_vec);
                            dataMap[var.first] = std::move(fc);

                        } else if (var.second["Type"] == "double") {

                            auto fc = std::make_shared<DoubleContainer>(DoubleContainer());
                            std::vector<double>& tmp_vec = fc->getVec();

                            adios2::Variable<double> advar = io->InquireVariable<double>(var.first);
                            auto info = reader.BlocksInfo(advar, cad->getFrameIDtoLoad());
                            num = info[0].Count[0];
                            tmp_vec.resize(num);
                            if (this->MpiInitialized && !singleValue) {
                                advar.SetSelection({{num * this->mpiRank}, {num}});
                            }

                            reader.Get<double>(advar, tmp_vec);
                            dataMap[var.first] = std::move(fc);

                        } else if (var.second["Type"] == "int") {

                            auto fc = std::make_shared<IntContainer>(IntContainer());
                            std::vector<int>& tmp_vec = fc->getVec();

                            adios2::Variable<int> advar = io->InquireVariable<int>(var.first);
                            auto info = reader.BlocksInfo(advar, cad->getFrameIDtoLoad());
                            num = info[0].Count[0];
                            tmp_vec.resize(num);
                            if (this->MpiInitialized && !singleValue) {
                                advar.SetSelection({{num * this->mpiRank}, {num}});
                            }

                            reader.Get<int>(advar, tmp_vec);
                            dataMap[var.first] = std::move(fc);
                        } else if (var.second["Type"] == "unsigned long long int") {
                            auto fc = std::make_shared<UInt64Container>(UInt64Container());
                            std::vector<unsigned long long int>& tmp_vec = fc->getVec();

                            adios2::Variable<unsigned long long int> advar =
                                io->InquireVariable<unsigned long long int>(var.first);
                            auto info = reader.BlocksInfo(advar, cad->getFrameIDtoLoad());
                            num = info[0].Count[0];
                            tmp_vec.resize(num);
                            if (this->MpiInitialized && !singleValue) {
                                advar.SetSelection({{num * this->mpiRank}, {num}});
                            }

                            reader.Get<unsigned long long int>(advar, tmp_vec);
                            dataMap[var.first] = std::move(fc);
                        } else if (var.second["Type"] == "unsigned char") {
                            auto fc = std::make_shared<UCharContainer>(UCharContainer());
                            std::vector<unsigned char>& tmp_vec = fc->getVec();

                            adios2::Variable<unsigned char> advar = io->InquireVariable<unsigned char>(var.first);
                            auto info = reader.BlocksInfo(advar, cad->getFrameIDtoLoad());
                            num = info[0].Count[0];
                            tmp_vec.resize(num);
                            if (this->MpiInitialized && !singleValue) {
                                advar.SetSelection({{num * this->mpiRank}, {num}});
                            }

                            reader.Get<unsigned char>(advar, tmp_vec);
                            dataMap[var.first] = std::move(fc);
                        } else if (var.second["Type"] == "unsigned int") {
                            auto fc = std::make_shared<UInt32Container>(UInt32Container());
                            std::vector<unsigned int>& tmp_vec = fc->getVec();

                            adios2::Variable<unsigned int> advar = io->InquireVariable<unsigned int>(var.first);
                            auto info = reader.BlocksInfo(advar, cad->getFrameIDtoLoad());
                            num = info[0].Count[0];
                            tmp_vec.resize(num);
                            if (this->MpiInitialized && !singleValue) {
                                advar.SetSelection({{num * this->mpiRank}, {num}});
                            }

                            reader.Get<unsigned int>(advar, tmp_vec);
                            dataMap[var.first] = std::move(fc);
                        }
                    }
                }
            }

            reader.EndStep();
            loadedFrameID = cad->getFrameIDtoLoad();
            // here data is loaded
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

        cad->setData(std::make_shared<adiosDataMap>(dataMap));
        cad->setDataHash(this->data_hash);
		this->dataHashChanged = false;
    }
    return true;
}


/*
 * adiosDataSource::filenameChanged
 */
bool adiosDataSource::filenameChanged(core::param::ParamSlot& slot) {
    using vislib::sys::Log;
    this->data_hash++;
	this->dataHashChanged = true;
    this->frameCount = 1;

    return true;
}


bool adiosDataSource::getHeaderCallback(core::Call& caller) {
    CallADIOSData* cad = dynamic_cast<CallADIOSData*>(&caller);
    if (cad == nullptr) return false;

    if (dataHashChanged || loadedFrameID != cad->getFrameIDtoLoad()) {

        try {
            vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2: Setting Engine");
            // io.SetEngine("InSituMPI");
            io->SetEngine("bpfile");
            io->SetParameter("verbose", "5");
            const std::string fname = std::string(T2A(this->filename.Param<core::param::FilePathParam>()->Value()));

            vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2: Opening File %s", fname.c_str());

            if (this->reader) {
                this->reader.Close();
                this->io->RemoveAllVariables();
                }
            this->reader = io->Open(fname, adios2::Mode::Read);

            // vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2: Reading available attributes");
            // auto availAttrib = io->AvailableAttributes();
            // vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2: Number of attributes %d", availAttrib.size());

            this->variables = io->AvailableVariables();
            vislib::sys::Log::DefaultLog.WriteInfo("ADIOS2: Number of variables %d", variables.size());

            std::vector<std::string> availVars;
            availVars.reserve(variables.size());

            std::vector<std::size_t> timesteps;
            for (auto var : variables) {
                availVars.push_back(var.first);
                vislib::sys::Log::DefaultLog.WriteInfo("%s", var.first.c_str());
                // get timesteps
                timesteps.push_back(std::stoi(var.second["AvailableStepsCount"]));
            }

            cad->setAvailableVars(availVars);
            // Check of all variables have same timestep count
            std::sort(timesteps.begin(), timesteps.end());
            auto last = std::unique(timesteps.begin(), timesteps.end());
            timesteps.erase(last, timesteps.end());

            if (timesteps.size() != 1) {
                vislib::sys::Log::DefaultLog.WriteWarn(
                    "Detected variables with different count of time steps - Using lowest");
                cad->setFrameCount(*std::min_element(timesteps.begin(), timesteps.end()));
            } else {
                cad->setFrameCount(timesteps[0]);
            }
			cad->setDataHash(this->data_hash);
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
    }
    return true;
}

bool adiosDataSource::initMPI() {
#ifdef WITH_MPI
    if (this->mpi_comm_ == MPI_COMM_NULL) {
        VLTRACE(vislib::Trace::LEVEL_INFO, "adiosDataSource: Need to initialize MPI\n");
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
        VLTRACE(vislib::Trace::LEVEL_INFO, "adiosDataSource: MPI initialized: %s (%i)\n",
            this->mpi_comm_ != MPI_COMM_NULL ? "true" : "false", mpi_comm_);
    } /* end if (this->comm == MPI_COMM_NULL) */

    /* Determine success of the whole operation. */
    const bool retval = (this->mpi_comm_ != MPI_COMM_NULL);
#endif /* WITH_MPI */
    return retval;
}

vislib::StringA adiosDataSource::getCommandLine(void) {
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
