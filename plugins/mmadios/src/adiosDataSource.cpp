#include "adiosDataSource.h"
#include "mmcore/cluster/mpi/MpiCall.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/utility/log/Log.h"
#include "stdafx.h"
#include "vislib/StringConverter.h"
#include "vislib/Trace.h"
#include "vislib/sys/CmdLineProvider.h"
#include "vislib/sys/SystemInformation.h"
#include <algorithm>
#include <numeric>


namespace megamol {
namespace adios {

adiosDataSource::adiosDataSource()
        : callRequestMpi("requestMpi", "Requests initialization of MPI and the communicator for the view.")
        , getData("getdata", "Slot to request data from this data source.")
        , filenameSlot("filename", "The path to the ADIOS-based file to load.") {

    this->filenameSlot.SetParameter(new core::param::FilePathParam("", core::param::FilePathParam::Flag_Any));
    this->filenameSlot.SetUpdateCallback(&adiosDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filenameSlot);


    this->getData.SetCallback("CallADIOSData", "GetData", &adiosDataSource::getDataCallback);
    this->getData.SetCallback("CallADIOSData", "GetHeader", &adiosDataSource::getHeaderCallback);
    this->MakeSlotAvailable(&this->getData);

    this->callRequestMpi.SetCompatibleCall<core::cluster::mpi::MpiCallDescription>();
    this->MakeSlotAvailable(&this->callRequestMpi);
}


/**
 * adiosDataSource::~adiosDataSource(void)
 */
adiosDataSource::~adiosDataSource() {
    this->Release();
}


/*
 * adiosDataSource::create
 */
bool adiosDataSource::create() {
    try {
#ifdef WITH_MPI
        MpiInitialized = this->initMPI();
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[adiosDataSource] Initializing with MPI");
        if (MpiInitialized) {
            adiosInst = std::make_shared<adios2::ADIOS>(adios2::ADIOS(this->mpi_comm_));
        } else {
            adiosInst = std::make_shared<adios2::ADIOS>(adios2::ADIOS());
        }
#else
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[adiosDataSource] Initializing without MPI");
        adiosInst = std::make_shared<adios2::ADIOS>(adios2::ADIOS());
#endif

        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[adiosDataSource] Declaring IO");
        this->io = std::make_shared<adios2::IO>(adiosInst->DeclareIO("Input"));

    } catch (std::invalid_argument& e) {
#ifdef WITH_MPI
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[adiosDataSource] Invalid argument exception, STOPPING PROGRAM from rank %d", this->mpiRank);
#else
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[adiosDataSource] Invalid argument exception, STOPPING PROGRAM");
#endif
        megamol::core::utility::log::Log::DefaultLog.WriteError(e.what());
    } catch (std::ios_base::failure& e) {
#ifdef WITH_MPI
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[adiosDataSource] IO System base failure exception, STOPPING PROGRAM from rank %d", this->mpiRank);
#else
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[adiosDataSource] IO System base failure exception, STOPPING PROGRAM");
#endif
        megamol::core::utility::log::Log::DefaultLog.WriteError(e.what());
    } catch (std::exception& e) {
#ifdef WITH_MPI
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[adiosDataSource] Exception, STOPPING PROGRAM from rank %d", this->mpiRank);
#else
        megamol::core::utility::log::Log::DefaultLog.WriteError("[adiosDataSource] Exception, STOPPING PROGRAM");
#endif
        megamol::core::utility::log::Log::DefaultLog.WriteError(e.what());
    }

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
    if (cad == nullptr)
        return false;

    if (!this->dataMap.empty()) {
        auto inqV = cad->getVarsToInquire();
        for (auto var : inqV) {
            this->inquireChanged = this->inquireChanged || this->dataMap.find(var) == this->dataMap.end();
        }
        auto inqA = cad->getAttributesToInquire();
        for (auto attr : inqA) {
            this->inquireChanged = this->inquireChanged || this->dataMap.find(attr) == this->dataMap.end();
        }
    } else {
        if (!cad->getVarsToInquire().empty() || !cad->getAttributesToInquire().empty()) {
            this->inquireChanged = true;
        }
    }

    if (dataHashChanged || inquireChanged || loadedFrameID != cad->getFrameIDtoLoad()) {

        try {
            auto fname = this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string();
#ifdef _WIN32
            std::replace(fname.begin(), fname.end(), '/', '\\');
#endif
            if (!this->reader) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[adiosDataSource] Header callback not called yet.");
                return false;
            }


            //megamol::core::utility::log::Log::DefaultLog.WriteInfo("[adiosDataSource] Beginning step");
            //const adios2::StepStatus status = reader->BeginStep();
            //if (status != adios2::StepStatus::OK) {
            //    megamol::core::utility::log::Log::DefaultLog.WriteError("[adiosDataSource] BeginStep returned an error.");
            //    return false;
            //}


            auto toInquire = cad->getVarsToInquire();
            auto attrsToInquire = cad->getAttributesToInquire();
            toInquire.insert(toInquire.end(), attrsToInquire.begin(), attrsToInquire.end());
            if (toInquire.empty()) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[adiosDataSource] Nothing inquired ... exiting");
                return false;
            }

            auto const frameIDtoLoad = cad->getFrameIDtoLoad();
            std::vector<adios2Params> content = variables;
            content.insert(content.end(), attributes.begin(), attributes.end());

            for (auto toInq : toInquire) {
                for (auto var : content) {
                    if (var.name == toInq) {
                        std::vector<size_t> shape(1);
                        bool singleValue = true;
                        if (var.params["SingleValue"] != std::string("true")) {
                            // num = std::stoi(var.second["Shape"]);
                            singleValue = false;
                        }
                        if (var.params["Type"] == "float") {
                            auto fc = std::make_shared<FloatContainer>(FloatContainer());
                            inquireRead<float>(fc, var, frameIDtoLoad, singleValue);
                        } else if (var.params["Type"] == "double") {
                            auto fc = std::make_shared<DoubleContainer>(DoubleContainer());
                            inquireRead<double>(fc, var, frameIDtoLoad, singleValue);
                        } else if (var.params["Type"] == "int32_t") {
                            auto fc = std::make_shared<Int32Container>(Int32Container());
                            inquireRead<int32_t>(fc, var, frameIDtoLoad, singleValue);
                        } else if (var.params["Type"] == "int8_t" || var.params["Type"] == "char") {
                            auto fc = std::make_shared<CharContainer>(CharContainer());
                            inquireRead<char>(fc, var, frameIDtoLoad, singleValue);
                        } else if (var.params["Type"] == "uint64_t") {
                            auto fc = std::make_shared<UInt64Container>(UInt64Container());
                            inquireRead<uint64_t>(fc, var, frameIDtoLoad, singleValue);
                        } else if ((var.params["Type"] == "unsigned char") || (var.params["Type"] == "uint8_t")) {
                            auto fc = std::make_shared<UCharContainer>(UCharContainer());
                            inquireRead<unsigned char>(fc, var, frameIDtoLoad, singleValue);
                        } else if (var.params["Type"] == "uint32_t") {
                            auto fc = std::make_shared<UInt32Container>(UInt32Container());
                            inquireRead<uint32_t>(fc, var, frameIDtoLoad, singleValue);
                        } else if (var.params["Type"] == "string") {
                            auto fc = std::make_shared<StringContainer>(StringContainer());
                            inquireRead<std::string>(fc, var, frameIDtoLoad, singleValue);
                        }
                    }
                }
            }
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("[adiosDataSource] PerformGets");
            const auto t1 = std::chrono::high_resolution_clock::now();
            reader->PerformGets();
            const auto t2 = std::chrono::high_resolution_clock::now();
            const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[adiosDataSource] Time spent for reading frame: %d ms", duration);

            loadedFrameID = cad->getFrameIDtoLoad();
            // here data is loaded
        } catch (std::invalid_argument& e) {
#ifdef WITH_MPI
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[adiosDataSource] Invalid argument exception, STOPPING PROGRAM from rank %d", this->mpiRank);
#else
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[adiosDataSource] Invalid argument exception, STOPPING PROGRAM");
#endif
            megamol::core::utility::log::Log::DefaultLog.WriteError(e.what());
        } catch (std::ios_base::failure& e) {
#ifdef WITH_MPI
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[adiosDataSource] IO System base failure exception, STOPPING PROGRAM from rank %d", this->mpiRank);
#else
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[adiosDataSource] IO System base failure exception, STOPPING PROGRAM");
#endif
            megamol::core::utility::log::Log::DefaultLog.WriteError(e.what());
        } catch (std::exception& e) {
#ifdef WITH_MPI
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[adiosDataSource] Exception, STOPPING PROGRAM from rank %d", this->mpiRank);
#else
            megamol::core::utility::log::Log::DefaultLog.WriteError("[adiosDataSource] Exception, STOPPING PROGRAM");
#endif
            megamol::core::utility::log::Log::DefaultLog.WriteError(e.what());
        }


        this->dataHashChanged = false;
        this->inquireChanged = false;
    }
    cad->setData(std::make_shared<adiosDataMap>(dataMap));
    cad->setDataHash(this->data_hash);
    return true;
}


/*
 * adiosDataSource::filenameChanged
 */
bool adiosDataSource::filenameChanged(core::param::ParamSlot& slot) {
    this->dataHashChanged = true;
    this->frameCount = 1;

    return true;
}


bool adiosDataSource::getHeaderCallback(core::Call& caller) {
    CallADIOSData* cad = dynamic_cast<CallADIOSData*>(&caller);
    if (cad == nullptr)
        return false;

    if (dataHashChanged || loadedFrameID != cad->getFrameIDtoLoad()) {
        if (loadedFrameID != cad->getFrameIDtoLoad())
            this->dataMap.clear();

        try {
            megamol::core::utility::log::Log::DefaultLog.WriteInfo("[adiosDataSource] Setting Engine");
            // io.SetEngine("InSituMPI");
            io->SetEngine("bpfile");
            // io->SetEngine("BP3"); this is for v2.4.0
            // adiosInst->AtIO("Input").SetParameters({{"verbose", "4"}});
            io->SetParameter("verbose", "5");
            auto fname = this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string();
#ifdef _WIN32
            std::replace(fname.begin(), fname.end(), '/', '\\');
#endif

            megamol::core::utility::log::Log::DefaultLog.WriteInfo("[adiosDataSource] Opening File %s", fname.c_str());

            if (this->reader && dataHashChanged) {
                this->reader->Close();
                io->RemoveAllVariables();
                io->RemoveAllAttributes();
                this->reader = std::make_shared<adios2::Engine>(io->Open(fname, adios2::Mode::Read));
            } else if (!this->reader) {
                this->reader = std::make_shared<adios2::Engine>(io->Open(fname, adios2::Mode::Read));
            }


            // megamol::core::utility::log::Log::DefaultLog.WriteInfo("ADIOS2: Reading available attributes");
            // auto availAttrib =io->AvailableAttributes();
            // megamol::core::utility::log::Log::DefaultLog.WriteInfo("ADIOS2: Number of attributes %d", availAttrib.size());

            auto tmp_variables = io->AvailableVariables();
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[adiosDataSource] Number of variables %d", tmp_variables.size());
            auto tmp_attributes = io->AvailableAttributes();
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                "[adiosDataSource] Number of attributes %d", tmp_attributes.size());

            availVars.clear();
            availVars.reserve(tmp_variables.size());
            variables.clear();
            variables.reserve(tmp_variables.size());
            availAttribs.clear();
            availAttribs.reserve(tmp_attributes.size());
            attributes.clear();
            attributes.reserve(tmp_attributes.size());
            timesteps.clear();
            for (auto var : tmp_variables) {
                adios2Params tmp_param;
                tmp_param.name = var.first;
                tmp_param.params = var.second;
                variables.emplace_back(tmp_param);
                availVars.emplace_back(var.first);
                megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                    "[adiosDataSource]: Available Variable %s", var.first.c_str());
                // get timesteps
                timesteps.push_back(std::stoi(var.second["AvailableStepsCount"]));
            }

            for (auto atr : tmp_attributes) {
                adios2Params tmp_param;
                tmp_param.name = atr.first;
                tmp_param.params = atr.second;
                tmp_param.isAttribute = true;
                attributes.emplace_back(tmp_param);
                availAttribs.emplace_back(atr.first);
                megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                    "[adiosDataSource]: Available Attribute %s", atr.first.c_str());
            }

            // Check of all variables have same timestep count
            std::sort(timesteps.begin(), timesteps.end());
            auto last = std::unique(timesteps.begin(), timesteps.end());
            timesteps.erase(last, timesteps.end());

            this->data_hash++;

        } catch (std::invalid_argument& e) {
#ifdef WITH_MPI
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[adiosDataSource] Invalid argument exception, STOPPING PROGRAM from rank %d", this->mpiRank);
#else
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[adiosDataSource] Invalid argument exception, STOPPING PROGRAM");
#endif
            megamol::core::utility::log::Log::DefaultLog.WriteError(e.what());
        } catch (std::ios_base::failure& e) {
#ifdef WITH_MPI
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[adiosDataSource] IO System base failure exception, STOPPING PROGRAM from rank %d", this->mpiRank);
#else
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[adiosDataSource] IO System base failure exception, STOPPING PROGRAM");
#endif
            megamol::core::utility::log::Log::DefaultLog.WriteError(e.what());
        } catch (std::exception& e) {
#ifdef WITH_MPI
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[adiosDataSource] Exception, STOPPING PROGRAM from rank %d", this->mpiRank);
#else
            megamol::core::utility::log::Log::DefaultLog.WriteError("[adiosDataSource] Exception, STOPPING PROGRAM");
#endif
            megamol::core::utility::log::Log::DefaultLog.WriteError(e.what());
        }
    }

    cad->setAvailableVars(availVars);
    cad->setAvailableAttributes(availAttribs);
    if (timesteps.size() != 1) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[adiosDataSource] Detected variables with different count of time steps - Using lowest");
        cad->setFrameCount(*std::min_element(timesteps.begin(), timesteps.end()));
    } else {
        cad->setFrameCount(timesteps[0]);
    }

    cad->setDataHash(this->data_hash);
    dataHashChanged = false;
    loadedFrameID = cad->getFrameIDtoLoad();

    return true;
}

bool adiosDataSource::initMPI() {
#ifdef WITH_MPI
    if (this->mpi_comm_ == MPI_COMM_NULL) {
        auto c = this->callRequestMpi.CallAs<core::cluster::mpi::MpiCall>();
        if (c != nullptr) {
            /* New method: let MpiProvider do all the stuff. */
            if ((*c)(core::cluster::mpi::MpiCall::IDX_PROVIDE_MPI)) {
                megamol::core::utility::log::Log::DefaultLog.WriteInfo("[adiosDataSource] Got MPI communicator.");
                this->mpi_comm_ = c->GetComm();
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    _T("[adiosDataSource] Could not ")
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
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(_T("[adiosDataSource] MPI is ready, ")
                                                                   _T("retrieving communicator properties ..."));
            ::MPI_Comm_rank(this->mpi_comm_, &this->mpiRank);
            ::MPI_Comm_size(this->mpi_comm_, &this->mpiSize);
            megamol::core::utility::log::Log::DefaultLog.WriteInfo(_T("[adiosDataSource] on %hs is %d ")
                                                                   _T("of %d."),
                vislib::sys::SystemInformation::ComputerNameA().PeekBuffer(), this->mpiRank, this->mpiSize);
        } /* end if (this->comm != MPI_COMM_NULL) */
        VLTRACE(vislib::Trace::LEVEL_INFO, "[adiosDataSource] MPI initialized: %s (%i)\n",
            this->mpi_comm_ != MPI_COMM_NULL ? "true" : "false", mpi_comm_);
    } /* end if (this->comm == MPI_COMM_NULL) */

    /* Determine success of the whole operation. */
    const bool retval = (this->mpi_comm_ != MPI_COMM_NULL);
    return retval;
#else
    return false;
#endif /* WITH_MPI */
}

vislib::StringA adiosDataSource::getCommandLine(void) {
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

    megamol::core::utility::log::Log::DefaultLog.WriteInfo("[adiosDataSource] Command line used for MPI "
                                                           "initialisation is \"%s\".",
        retval.PeekBuffer());
    return retval;
}

} // namespace adios
} // namespace megamol
