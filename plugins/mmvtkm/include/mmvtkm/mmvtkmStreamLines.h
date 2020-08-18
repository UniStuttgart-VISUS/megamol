#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

#include "mesh/MeshCalls.h"

#pragma once

using namespace megamol::mesh;

namespace megamol {
namespace mmvtkm {


typedef vislib::math::Point<float, 3> visPoint3f;
typedef vislib::math::Vector<float, 3> visVec3f;
//typedef vtkm::Vec<float, 3> visVec3f;
typedef vislib::math::Plane<float> visPlanef;


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
    enum planeMode { NORMAL, PARAMETER };
    
	struct Triangle {
        visPoint3f a;
        visPoint3f b;
        visPoint3f c;

		visVec3f v1;
		visVec3f v2;

		float area = 0.f;

		// used for sampling within surrounding polygon
		// weight = triangle_area / polygon_area
        float weight = 0.f;

		Triangle(visPoint3f rhs_a, visPoint3f rhs_b, visPoint3f rhs_c) : a(rhs_a), b(rhs_b), c(rhs_c) 
		{
            this->v1 = this->b - this->a;
            this->v2 = this->c - this->a;

            float ab = (this->a - this->b).Length();
            float ac = (this->a - this->c).Length();
            float theta = acos(ac / ab);

            this->area = 0.5f * ab * ac * sin(theta);
		}
	};

	/** Calculates intersectin points of sampling plane and bounding box */
    std::vector<visPoint3f> calcPlaneBboxIntersectionPoints(const visPlanef& sample_plane, const vtkm::Bounds& bounds);

	/** Decomposes a (convex) polygon into triangles */
    std::vector<Triangle> decomposePolygon(const std::vector<visPoint3f>& polygon);

	/** Checks whether a point is outside the given boundaries */
    bool isOutsideCube(const visPoint3f& p, const vtkm::Bounds& bounds);

    /** Callback functions to update the version after changing data */
    bool dataChanged(core::param::ParamSlot& slot);
    bool setConfiguration(core::param::ParamSlot& slot);
    bool planeModeChanged(core::param::ParamSlot& slot);

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
    core::param::ParamSlot streamlineFieldName_;

    /** Paramslot to specify the seeds for the streamline */
    core::param::ParamSlot numStreamlineSeed_;

    /** Paramslot to specify the step size of the streamline */
    core::param::ParamSlot streamlineStepSize_;

    /** Paramslot to specify the number of steps of the streamline */
    core::param::ParamSlot numStreamlineSteps_;

    /** Paramslot to specify the seeds for the streamline */
    core::param::ParamSlot lowerStreamlineSeedBound_;

    /** Paramslot to specify the seeds for the streamline */
    core::param::ParamSlot upperStreamlineSeedBound_;

    /** Paramslot for plane mode */
    core::param::ParamSlot seedPlaneMode_;

	/** Paramslot for plane color */
    //core::param::ParamSlot seedPlaneColor_;

    /** Paramslot for lower left corner of seed plane */
    core::param::ParamSlot lowerLeftSeedPoint_;

    /** Paramslot for lower left corner of seed plane */
    core::param::ParamSlot upperRightSeedPoint_;

    /** Paramslot for normal of seed plane */
    core::param::ParamSlot seedPlaneNormal_;

	/** Paramslot for point on seed plane */
    core::param::ParamSlot seedPlanePoint_;

	/** Paramslot for seed plane distance */
    //core::param::ParamSlot seedPlaneDistance_;

    /** Paramslot to apply changes to streamline configurations */
    core::param::ParamSlot applyChanges_;

    /** Used for conversion seed bounding param type to vtkm vec type */
    visVec3f lowerSeedBound_;
    visVec3f upperSeedBound_;

    /** Used for data version control, same as 'hash_data' */
    uint32_t old_version_;
    uint32_t new_version_;

    /** Used for mesh data call */
    std::shared_ptr<MeshDataAccessCollection> mesh_data_access_;
    core::Spatial3DMetaData meta_data_;

    /** Pointers to streamline data */
    std::vector<std::vector<float>> streamline_data_;
    std::vector<std::vector<float>> streamline_color_;
    std::vector<std::vector<unsigned int>> streamline_indices_;

    /** Temporary data storage for streamline parameter changes */
    vtkm::Id tmp_num_seeds_;
    vtkm::Id tmp_num_steps_;
    vtkm::FloatDefault tmp_step_size_;
    vislib::TString tmp_active_field_;
    visPoint3f tmp_lower_left_seed_point_;
    visPoint3f tmp_upper_right_seed_point_;
    visVec3f tmp_seed_plane_normal_;
    visPoint3f tmp_seed_plane_point_;
    float tmp_seed_plane_distance_;

    std::vector<visPoint3f> seed_plane_;
    std::vector<visVec3f> seed_plane_colors_;
    std::vector<float> seed_plane_color_;
    std::vector<unsigned int> plane_idx_;

    /** used for plane mode */
    int plane_mode_;

	/** some colors for testing */
    visPoint3f red = visPoint3f(0.5f, 0.f, 0.f);
    visPoint3f green = visPoint3f(0.f, 0.5f, 0.f);
    visPoint3f blue = visPoint3f(0.f, 0.f, 0.5);
};

} // end namespace mmvtkm
} // end namespace megamol