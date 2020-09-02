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
        glm::vec3 a;
        glm::vec3 b;
        glm::vec3 c;
		glm::vec3 o;

		glm::vec3 v1;
		glm::vec3 v2;

		float area = 0.f;

		// used for sampling within surrounding polygon
		// weight = triangle_area / polygon_area
        float weight = 0.f;

		Triangle(const glm::vec3& rhsA, const glm::vec3& rhsB, const glm::vec3& rhsC) : a(rhsA), b(rhsB), c(rhsC) 
		{
            float ab = glm::length(rhsB - rhsA);
            float ac = glm::length(rhsC - rhsA);
            float bc = glm::length(rhsC - rhsB);
            float max = std::max(ab, std::max(ac, bc));

			// get hypothenuse in v1 and adjacent in v2
            if (ab == max) {
                this->v1 = rhsB - rhsA;
                this->v2 = rhsC - rhsA;
                this->o = rhsA;
            } else if (ac == max) {
                this->v1 = rhsC - rhsA;
                this->v2 = rhsB - rhsA;
                this->o = rhsA;
            } else {
                this->v1 = rhsC - rhsB;
                this->v2 = rhsA - rhsB;
                this->o = rhsB;
            }

			float v1_length = glm::length(v1);
            float v2_length = glm::length(v2);
            
			// alternatively: 
			// float cos_theta = v1.Dot(v2) / (v1_length * v2_length);
			// float sin_theta = sqrt(1.f - cos_theta * cos_theta)
            float arg = glm::dot(v1, v2) / (v1_length * v2_length); // angle via dot product
            float theta = acos(trunc(arg * 1000000.f) / 1000000.f);		// avoids -nan in late decimal positions
            this->area = 0.5f * v1_length * v2_length * sin(theta);
		}
	};

	/** Callback functions to update the version after changing data */
    bool dataChanged(core::param::ParamSlot& slot);
    bool setConfiguration(core::param::ParamSlot& slot);
    bool planeModeChanged(core::param::ParamSlot& slot);


	/** Calculates intersectin points of sampling plane and bounding box */
    std::vector<glm::vec3> calcPlaneBboxIntersectionPoints(const visPlanef& samplePlane, const vtkm::Bounds& bounds);

	/** Checks whether a point is outside the given boundaries */
    bool isOutsideCube(const glm::vec3& p, const vtkm::Bounds& bounds);

	/** Orders vertices that form a polygon, so that the order can be used for a triangle fan */
    void orderPolygonVertices(std::vector<glm::vec3>& vertices);

	/** Decomposes a (convex) polygon into triangles */
    std::vector<Triangle> decomposePolygon(const std::vector<glm::vec3>& polygon);

	/** Checks whether a point is inside a given triangle */
    bool isInsideTri(const glm::vec3& p, const Triangle& tri);

	/** Creates and adds MeshDataAccessCollection to the mesh datacall */
    bool createAndAddMeshDataToCall(std::vector<glm::vec3>& lineData, std::vector<glm::vec3>& lineColor,
        std::vector<unsigned int>& lineIdcs,
        int numPoints, int numIndices,
        MeshDataAccessCollection::PrimitiveType linePt = MeshDataAccessCollection::PrimitiveType::TRIANGLES);

	/** Adds the MeshDataAccessCollection to the mesh datacall */
	bool addMeshDataToCall(const std::vector<MeshDataAccessCollection::VertexAttribute>& va, const MeshDataAccessCollection::IndexData& id,
        MeshDataAccessCollection::PrimitiveType pt);


    /** Gets converted vtk streamline data as megamol mesh */
    core::CalleeSlot meshCalleeSlot_;

    /** Callerslot from which the vtk data is coming from */
    core::CallerSlot vtkCallerSlot_;

    /** Paramslot to specify the field name of streamline vector field */
    core::param::ParamSlot psStreamlineFieldName_;

    /** Paramslot to specify the seeds for the streamline */
    core::param::ParamSlot psNumStreamlineSeed_;

    /** Paramslot to specify the step size of the streamline */
    core::param::ParamSlot psStreamlineStepSize_;

    /** Paramslot to specify the number of steps of the streamline */
    core::param::ParamSlot psNumStreamlineSteps_;

    /** Paramslot to specify the seeds for the streamline */
    core::param::ParamSlot psLowerStreamlineSeedBound_;

    /** Paramslot to specify the seeds for the streamline */
    core::param::ParamSlot psUpperStreamlineSeedBound_;

    /** Paramslot for plane mode */
    core::param::ParamSlot psSeedPlaneMode_;

    /** Paramslot for lower left corner of seed plane */
    core::param::ParamSlot psLowerLeftSeedPoint_;

    /** Paramslot for lower left corner of seed plane */
    core::param::ParamSlot psUpperRightSeedPoint_;

    /** Paramslot for normal of seed plane */
    core::param::ParamSlot psSeedPlaneNormal_;

	/** Paramslot for point on seed plane */
    core::param::ParamSlot psSeedPlanePoint_;

	/** Paramslot for seed plane distance */
    //core::param::ParamSlot psSeedPlaneDistance_;

	/** Paramslot for seed plane color */
    core::param::ParamSlot psSeedPlaneColor_;

    /** Paramslot to apply changes to streamline configurations */
    core::param::ParamSlot applyChanges_;


    /** Used for data version control, same as 'hash_data' */
    uint32_t oldVersion_;
    uint32_t newVersion_;

    /** Used for mesh data call */
    std::shared_ptr<MeshDataAccessCollection> meshDataAccess_;
    core::Spatial3DMetaData metaData_;

    /** Pointers to streamline data */
    std::vector<std::vector<glm::vec3>> streamlineData_;
    std::vector<std::vector<glm::vec3>> streamlineColor_;
    std::vector<std::vector<unsigned int>> streamlineIndices_;

    /** Data storage for streamline parameters */
    vtkm::Id numSeeds_;
    vtkm::Id numSteps_;
    vtkm::FloatDefault stepSize_;
    vislib::TString activeField_;
    glm::vec3 lowerLeftSeedPoint_;
    glm::vec3 upperRightSeedPoint_;
    glm::vec3 seedPlaneNormal_;
    glm::vec3 seedPlanePoint_;
    glm::vec3 seedPlaneColor_;
    float seedPlaneDistance_;

    std::vector<glm::vec3> seedPlane_;
    std::vector<glm::vec3> seedPlaneColorVec_;
    std::vector<unsigned int> seedPlaneIdcs_;

    /** used for plane mode */
    int planeMode_;

	/** some colors for testing */
    glm::vec3 red_ = glm::vec3(0.5f, 0.f, 0.f);
    glm::vec3 green_ = glm::vec3(0.f, 0.5f, 0.f);
    glm::vec3 blue_ = glm::vec3(0.f, 0.f, 0.5f);
};

} // end namespace mmvtkm
} // end namespace megamol