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

    Frame *f = NULL;
    if (c2 != NULL) {
        // TODO: adios load data
        if (f == NULL) return false;

        c2->SetFrameID(f->FrameNumber());
        c2->SetDataHash(this->data_hash);
        this->setData(*c2);
    }

    return true;
}

/*
 * adiosDataSource::getExtentCallback
 */
bool adiosDataSource::getExtentCallback(core::Call& caller) {
	core::moldyn::MultiParticleDataCall *c2 = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&caller);

	// TODO: adios get extends

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
	
	adios(this->mpi_comm_, true);
	io = adios.DeclareIO("dummy");

	io.SetEngine("InSituMPI");
	io.SetParameter("verbose", "5");
	rStream = io.Open(this->filename.Param<core::param::FilePathParam>()->Value(), adios2::Mode::Read);

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




	return true;
}

} /* end namespace megamol */
} /* end namespace adios */
