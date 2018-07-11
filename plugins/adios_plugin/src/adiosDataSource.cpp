#include "stdafx.h"
#include "adiosDataSource.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "vislib/sys/Log.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/cluster/mpi/MpiCall.h"
#include "vislib/Trace.h"

#include <adios_plugin/adios2.h>


namespace megamol {
namespace adios {

adiosDataSource::adiosDataSource(void) : 
  core::view::AnimDataModule(),
  filename("filename", "The path to the ADIOS-based file to load."),
  getData("getdata", "Slot to request data from this data source."),
  callRequestMpi("requestMpi", "Requests initialisation of MPI and the communicator for the view."),
  bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f),
  data_hash(0)
 {

    this->filename.SetParameter(new core::param::FilePathParam(""));
    this->filename.SetUpdateCallback(&adiosDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);


    this->getData.SetCallback("MultiParticleDataCall", "GetData", &adiosDataSource::getDataCallback);
    this->getData.SetCallback("MultiParticleDataCall", "GetExtent", &adiosDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getData);

	this->callRequestMpi.SetCompatibleCall<core::cluster::mpi::MpiCallDescription>();
	this->MakeSlotAvailable(&this->callRequestMpi);

    this->setFrameCount(1);

}

  adiosDataSource::~adiosDataSource(void) {
    this->Release();
  }

/*
 * adiosDataSource::create
 */
bool adiosDataSource::create(void) {
    return true;
}


/*
 * adiosDDataSource::release
 */
void adiosDataSource::release(void) {
    /* empty */
}


/*
 * adiosDataSource::getDataCallback
 */
bool adiosDataSource::getDataCallback(core::Call& caller) {
    core::moldyn::MultiParticleDataCall *c2 = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&caller);
    if (c2 == NULL) return false;

    if (c2 != NULL) {
        // TODO: adios load data

		if (!this->adiosRead()) {
			vislib::sys::Log::DefaultLog.WriteError("Error while reading with ADIOS");
			return false;
		}


        c2->SetFrameID(0);
        c2->SetDataHash(this->data_hash);
		c2->SetParticleListCount(1);
		c2->AccessParticles(0).SetGlobalRadius(1.0f);
		c2->AccessParticles(0).SetCount(this->particleCount);

		std::vector<float> mix;
		mix.resize(this->particleCount*3);

		for (int i = 0; i < this->particleCount; i++) {
			mix[3 * i + 0] = X[i];
			mix[3 * i + 1] = Y[i];
			mix[3 * i + 2] = Z[i];
		}

		c2->AccessParticles(0).SetVertexData(megamol::core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR, mix.data(), 3*sizeof(float));
		c2->AccessParticles(0).SetColourData(megamol::core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE, nullptr);
    }

    return true;
}

/*
 * adiosDataSource::getExtentCallback
 */
bool adiosDataSource::getExtentCallback(core::Call& caller) {
	core::moldyn::MultiParticleDataCall *c2 = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&caller);

	this->getDataCallback(caller);

    if (c2 != NULL) {
        c2->SetFrameCount(this->FrameCount());
        c2->AccessBoundingBoxes().Clear();
        c2->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
        c2->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox);
        c2->SetDataHash(this->data_hash);
        return true;
    }

    return false;
}

void adiosDataSource::setData(core::Call & c) {
	// TODO: set data into call

}

/*
* adiosDataSource::filenameChanged
*/
bool adiosDataSource::filenameChanged(core::param::ParamSlot& slot) {
	using vislib::sys::Log;
	this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
	this->data_hash++;

	// TODO: (re)initialize adios
	
	adios(this->mpi_comm_, adios2::DebugON);
	io = adios.DeclareIO("dummy");

	io.SetEngine("InSituMPI");
	io.SetParameter("verbose", "5");
	reader = io.Open(this->filename.Param<core::param::FilePathParam>()->Value(), adios2::Mode::Read);

	this->setFrameCount(1);


	return true;
}


bool adiosDataSource::initMPI() {
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
												   vislib::sys::SystemInformation::ComputerNameA().PeekBuffer(), this->mpiRank, this->mpiSize);
		} /* end if (this->comm != MPI_COMM_NULL) */
		VLTRACE(vislib::Trace::LEVEL_INFO, "FBOTransmitter2: MPI initialized: %s (%i)\n",
				this->mpi_comm_ != MPI_COMM_NULL ? "true" : "false", mpi_comm_);
	} /* end if (this->comm == MPI_COMM_NULL) */

	  /* Determine success of the whole operation. */
	retval = (this->mpi_comm_ != MPI_COMM_NULL);
#endif /* WITH_MPI */
	return retval;
}

bool adiosDataSource::adiosRead() {

	adios2::StepStatus status = reader.BeginStep(adios2::StepMode::NextAvailable, 0.0f);
	if (status != adios2::StepStatus::OK) {
		vislib::sys::Log::DefaultLog.WriteError("ADIOS2 ERROR: BeginStep returned an error.");
		return false;
	}

	vBox = io.InquireVariable<float>("box");
	vX = io.InquireVariable<float>("x");
	vY = io.InquireVariable<float>("y");
	vZ = io.InquireVariable<float>("z");

	if (!vBox || !vX || !vY || !vZ) {
		vislib::sys::Log::DefaultLog.WriteError("Error: One or alle variables not found. Unable to proceed.");
		return false;
	}

	if ((vX.Shape()[0] != vY.Shape()[0]) && (vX.Shape()[0] != vZ.Shape()[0])) {
		vislib::sys::Log::DefaultLog.WriteError("Error: Position lists are of different shape.");
		return false;
	}

	this->particleCount = vX.Shape()[0];

	box.resize(vBox.Shape()[0]);
	X.resize(vX.Shape()[0]);
	Y.resize(vY.Shape()[0]);
	Z.resize(vZ.Shape()[0]);

	reader.Get<float>(vBox, box.data());
	reader.Get<float>(vX, X.data());
	reader.Get<float>(vY, Y.data());
	reader.Get<float>(vZ, Z.data());

	reader.EndStep();
	this->data_hash++;

	return true;
}

} /* end namespace megamol */
} /* end namespace adios */
