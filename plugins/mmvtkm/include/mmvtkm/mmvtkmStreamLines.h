#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "mesh/MeshCalls.h"

#pragma once

using namespace megamol::mesh;

namespace megamol {
namespace mmvtkm {

typedef vtkm::Vec<float, 3> Vec3f;

class mmvtkmStreamLines : public core::Module {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() { return "vtkmStreamLines"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Creates streamlines for a vtkm tetraeder mesh and converts the streamlines "
               "into megamols CallMesh";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() { return true; }

    /**
     * Ctor
     */
    mmvtkmStreamLines();

    /**
     * Dtor
     */
    virtual ~mmvtkmStreamLines();


protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create();

    /**
     * Implementation of 'Release'.
     */
    void release();

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getDataCallback(core::Call& caller);

    /**
     * Gets the meta data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool getMetaDataCallback(core::Call& caller);


private:
    /** Callback function for to update the version after changing data */
    bool dataChanged(core::param::ParamSlot& slot);

	/** Callback functions for the seed bounds */
    bool lowerBoundChanged(core::param::ParamSlot& slot);
    bool upperBoundChanged(core::param::ParamSlot& slot);

	/** Check whether the lower bound is actually lower than the upper bound */
	bool seedBoundCheck();

    /** Gets converted vtk streamline data as megamol mesh */
    core::CalleeSlot meshCalleeSlot_;

    /** Callerslot from which the vtk data is coming from */
    core::CallerSlot vtkCallerSlot_;

    /** Paramslot to specify the field name of streamline vector field */
    core::param::ParamSlot fieldName_;

	/** Paramslot to specify the seeds for the streamline */
    core::param::ParamSlot numStreamlineSeed_;

	/** Paramslot to specify the seeds for the streamline */
    core::param::ParamSlot lowerStreamlineSeedBound_;

	/** Paramslot to specify the seeds for the streamline */
    core::param::ParamSlot upperStreamlineSeedBound_;

	/** Paramslot to specify the step size of the streamline */
    core::param::ParamSlot streamlineStepSize_;

	/** Paramslot to specify the number of steps of the streamline */
    core::param::ParamSlot numStreamlineSteps_;

	/** Used for conversion seed bounding param type to vtkm vec type */
    Vec3f lowerSeedBound_;
    Vec3f upperSeedBound_;

	/** Used for data version control, same as 'hash_data' */
	uint32_t old_version_;
    uint32_t new_version_;

	/** Used for mesh data call */
    std::shared_ptr<MeshDataAccessCollection> mesh_data_access_;
    core::Spatial3DMetaData meta_data_;
};

} // end namespace mmvtkm
} // end namespace megamol