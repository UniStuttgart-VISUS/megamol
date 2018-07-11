#pragma once

#include "mmcore/view/AnimDataModule.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "vislib/math/Cuboid.h"
#include "mmcore/CallerSlot.h"
#ifdef WITH_MPI
#include <mpi.h>
#endif


namespace megamol {
namespace adios {

class adiosDataSource : public core::view::AnimDataModule {
public:
	/**
	 * Answer the name of this module.
	 *
	 * @return The name of this module.
	 */
	static const char *ClassName(void) {
		return "adiosDataSource";
	}

	/**
	 * Answer a human readable description of this module.
	 *
	 * @return A human readable description of this module.
	 */
	static const char *Description(void) {
		return "Data source module for ADIOS-based IO.";
	}

	/**
	 * Answers whether this module is available on the current system.
	 *
	 * @return 'true' if the module is available, 'false' otherwise.
	 */
	static bool IsAvailable(void) {
		return true;
	}

	/** Ctor. */
	adiosDataSource(void);

	/** Dtor. */
	virtual ~adiosDataSource(void);

	bool create(void);

protected:

	void release(void);

	/**
	 * Gets the data from the source.
	 *
	 * @param caller The calling call.
	 *
	 * @return 'true' on success, 'false' on failure.
	 */
	bool getDataCallback(core::Call& caller);

	/**
	 * Gets the data from the source.
	 *
	 * @param caller The calling call.
	 *
	 * @return 'true' on success, 'false' on failure.
	 */
	bool getExtentCallback(core::Call& caller);


private:

	/** slot for MPIprovider */
	core::CallerSlot callRequestMpi;
	bool initMPI();
	bool adiosRead();
	bool initADIOS();

#ifdef WITH_MPI
	MPI_Comm mpi_comm_ = MPI_COMM_NULL;
	bool useMpi = false;
	int mpiRank = -1, mpiSize = -1;
#endif

	void setData(core::Call& c);

	bool adiosDataSource::filenameChanged(core::param::ParamSlot& slot);

	/** The slot for requesting data */
	core::CalleeSlot getData;

	/** The frame index table */
	std::vector<UINT64> frameIdx;

	/** Data file load id counter */
	size_t data_hash;

	/** The data set bounding box */
	vislib::math::Cuboid<float> bbox;

	/** The file name */
	core::param::ParamSlot filename;

	/** The data set bounding box */
	vislib::math::Cuboid<float> bbox;


	adios2::ADIOS adiosInst;
	adios2::IO io;
	adios2::Engine reader;

	std::vector<float> X;
	std::vector<float> Y;
	std::vector<float> Z;
	std::vector<float> box;
	adios2::Variable<float> vX;
	adios2::Variable<float> vY;
	adios2::Variable<float> vZ;
	adios2::Variable<float> vBox;

	int step = 0;

	int particleCount = 0;

};


} /* end namespace adios */
} /* end namespace megamol */
