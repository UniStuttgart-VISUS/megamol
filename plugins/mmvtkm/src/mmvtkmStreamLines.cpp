#include "vtkm/filter/Streamline.h"
#include "vtkm/io/writer/VTKDataSetWriter.h"

#include "mmvtkm/mmvtkmDataCall.h"
#include "mmvtkm/mmvtkmStreamLines.h"

#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/Vector3fParam.h"

#include "vislib/math/Matrix4.h"


using namespace megamol;
using namespace megamol::mmvtkm;


// TODO order function so they have same order as in mmvtkmstreamlines.h


mmvtkmStreamLines::mmvtkmStreamLines()
    : core::Module()
    , meshCalleeSlot_("meshCalleeSlot", "Requests streamline mesh data from vtk data")
    , vtkCallerSlot_("vtkCallerSlot", "Requests vtk data for streamlines")
    , psStreamlineFieldName_("fieldName", "Specifies the field name of the streamline vector field")
    , psNumStreamlineSeed_("numSeeds", "Specifies the number of seeds for the streamlines")
    , psStreamlineStepSize_("stepSize", "Specifies the step size for the streamlines")
    , psNumStreamlineSteps_("numSteps", "Specifies the number of steps for the streamlines")
    , psLowerStreamlineSeedBound_("lowerSeedBound", "Specifies the lower streamline seed bound")
    , psUpperStreamlineSeedBound_("upperSeedBound", "Specifies the upper streamline seed bound")
    , psSeedPlaneMode_("planeMode", "Specifies the representation of the seed plane")
    , psLowerLeftSeedPoint_("lowerPlanePoint", "Specifies the lower left point of the seed plane")
    , psUpperRightSeedPoint_("upperPlanePoint", "Specifies the upper right point of the seed plane")
    , psSeedPlaneNormal_("planeNormal", "Specifies the normal of the seed plane")
    , psSeedPlanePoint_("planePoint", "Specifies a point on the seed plane")
    //, seedPlaneDistance_("planeDistance", "Specifies the distance of the seed plane to the origin")
    , psSeedPlaneColor_("planeColor", "Specifies the color of the seed plane")
    , applyChanges_("apply", "Press to apply changes for streamline configuration")
    , numSeeds_(100)
    , numSteps_(1000)
    , stepSize_(0.1f)
    , activeField_("hs1")
    , lowerLeftSeedPoint_(-50.f, -50.f, 0)
    , upperRightSeedPoint_(50.f, 50.f, 100.f)
    , seedPlaneNormal_(1.f, 0.f, 0.f)
    , seedPlanePoint_(0.f, 0.f, 0.f)
    , seedPlaneDistance_(0.f)
    , seedPlaneColor_({0.5f, 0.f, 0.f})
    , oldVersion_(0)
    , newVersion_(1)
    , planeMode_(0) {
    this->meshCalleeSlot_.SetCallback(mesh::CallMesh::ClassName(),
        mesh::CallMesh::FunctionName(0), // used to be mesh::CallMesh
        &mmvtkmStreamLines::getDataCallback);
    this->meshCalleeSlot_.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &mmvtkmStreamLines::getMetaDataCallback);
    this->MakeSlotAvailable(&this->meshCalleeSlot_);

    this->vtkCallerSlot_.SetCompatibleCall<mmvtkmDataCallDescription>();
    this->MakeSlotAvailable(&this->vtkCallerSlot_);

    // TODO: instead of hardcoding fieldnames,
    // maybe also read all field names and show them as dropdown menu in megamol
    this->psStreamlineFieldName_.SetParameter(new core::param::StringParam(activeField_));
    this->psStreamlineFieldName_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->psStreamlineFieldName_);

    this->psNumStreamlineSeed_.SetParameter(new core::param::IntParam(numSeeds_, 0));
    this->psNumStreamlineSeed_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->psNumStreamlineSeed_);

    this->psStreamlineStepSize_.SetParameter(new core::param::FloatParam(stepSize_, 0.f));
    this->psStreamlineStepSize_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->psStreamlineStepSize_);

    this->psNumStreamlineSteps_.SetParameter(new core::param::IntParam(numSteps_, 0));
    this->psNumStreamlineSteps_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->psNumStreamlineSteps_);

    this->psLowerStreamlineSeedBound_.SetParameter(new core::param::Vector3fParam({0, 0, 0}));
    this->psLowerStreamlineSeedBound_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->psLowerStreamlineSeedBound_.Param<core::param::Vector3fParam>()->SetGUIReadOnly(true);
    this->MakeSlotAvailable(&this->psLowerStreamlineSeedBound_);

    this->psUpperStreamlineSeedBound_.SetParameter(new core::param::Vector3fParam({1, 1, 1}));
    this->psUpperStreamlineSeedBound_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->psUpperStreamlineSeedBound_.Param<core::param::Vector3fParam>()->SetGUIReadOnly(true);
    this->MakeSlotAvailable(&this->psUpperStreamlineSeedBound_);

    this->psSeedPlaneMode_.SetParameter(new core::param::EnumParam(planeMode_));
    this->psSeedPlaneMode_.Param<core::param::EnumParam>()->SetTypePair(0, "normal");
    this->psSeedPlaneMode_.Param<core::param::EnumParam>()->SetTypePair(1, "parameter");
    this->psSeedPlaneMode_.SetUpdateCallback(&mmvtkmStreamLines::planeModeChanged);
    this->MakeSlotAvailable(&this->psSeedPlaneMode_);

    this->psLowerLeftSeedPoint_.SetParameter(new core::param::Vector3fParam({-50.f, -50.f, 0}));
    this->psLowerLeftSeedPoint_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->psLowerLeftSeedPoint_);
    this->psLowerLeftSeedPoint_.Param<core::param::Vector3fParam>()->SetGUIVisible(false);

    this->psUpperRightSeedPoint_.SetParameter(new core::param::Vector3fParam({50.f, 50.f, 100.f}));
    this->psUpperRightSeedPoint_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->psUpperRightSeedPoint_);
    this->psUpperRightSeedPoint_.Param<core::param::Vector3fParam>()->SetGUIVisible(false);

    this->psSeedPlaneNormal_.SetParameter(new core::param::Vector3fParam({1.f, 0.f, 0.f}));
    this->psSeedPlaneNormal_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->psSeedPlaneNormal_);
    this->psSeedPlaneNormal_.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Rotation3D_Direction);

    this->psSeedPlanePoint_.SetParameter(new core::param::Vector3fParam({0.f, 0.f, 0.f}));
    this->psSeedPlanePoint_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->psSeedPlanePoint_);

    // this->psSeedPlaneDistance_.SetParameter(new core::param::FloatParam(psSeedPlaneDistance_));
    // this->psSeedPlaneDistance_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    // this->MakeSlotAvailable(&this->psSeedPlaneDistance_);

    this->psSeedPlaneColor_.SetParameter(new core::param::Vector3fParam({red_.x, red_.y, red_.z}, {0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}));
    this->psSeedPlanePoint_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->psSeedPlaneColor_);

    this->applyChanges_.SetParameter(new core::param::ButtonParam());
    this->applyChanges_.SetUpdateCallback(&mmvtkmStreamLines::setConfiguration);
    this->MakeSlotAvailable(&this->applyChanges_);
}


mmvtkmStreamLines::~mmvtkmStreamLines() { this->Release(); }


void mmvtkmStreamLines::release() {}


bool mmvtkmStreamLines::create() {
    this->meshDataAccess_ = std::make_shared<MeshDataAccessCollection>();
    return true;
}


bool mmvtkmStreamLines::dataChanged(core::param::ParamSlot& slot) {
    return true;
}


bool mmvtkmStreamLines::setConfiguration(core::param::ParamSlot& slot) {
    activeField_ = this->psStreamlineFieldName_.Param<core::param::StringParam>()->Value();
    numSeeds_ = this->psNumStreamlineSeed_.Param<core::param::IntParam>()->Value();
    stepSize_ = this->psStreamlineStepSize_.Param<core::param::FloatParam>()->Value();
    numSteps_ = this->psNumStreamlineSteps_.Param<core::param::IntParam>()->Value();
    visVec3f ll = this->psLowerLeftSeedPoint_.Param<core::param::Vector3fParam>()->Value();
    lowerLeftSeedPoint_ = {ll.GetX(), ll.GetY(), ll.GetZ()};
    visVec3f ur = this->psUpperRightSeedPoint_.Param<core::param::Vector3fParam>()->Value();
    upperRightSeedPoint_ = {ur.GetX(), ur.GetY(), ur.GetZ()};
    visVec3f normal = this->psSeedPlaneNormal_.Param<core::param::Vector3fParam>()->Value();
    seedPlaneNormal_ = {normal.GetX(), normal.GetY(), normal.GetZ()};
    visVec3f point = this->psSeedPlanePoint_.Param<core::param::Vector3fParam>()->Value();
    seedPlanePoint_ = {point.GetX(), point.GetY(), point.GetZ()};
    // psSeedPlaneDistance_ = this->seedPlaneDistance_.Param<core::param::FloatParam>()->Value();
    visVec3f tmpColor = psSeedPlaneColor_.Param<core::param::Vector3fParam>()->Value();
    seedPlaneColor_ = glm::vec3(tmpColor.GetX(), tmpColor.GetY(), tmpColor.GetZ());


    this->newVersion_++;

    return true;
}


bool mmvtkmStreamLines::planeModeChanged(core::param::ParamSlot& slot) {
    if (slot.Param<core::param::EnumParam>()->Value() == NORMAL) {
        planeMode_ = 0;
        this->psUpperRightSeedPoint_.Param<core::param::Vector3fParam>()->SetGUIVisible(false);
        this->psLowerLeftSeedPoint_.Param<core::param::Vector3fParam>()->SetGUIVisible(false);
        this->psSeedPlaneNormal_.Param<core::param::Vector3fParam>()->SetGUIVisible(true);
        this->psSeedPlanePoint_.Param<core::param::Vector3fParam>()->SetGUIVisible(true);
        // this->psSeedPlaneDistance_.Param<core::param::FloatParam>()->SetGUIVisible(true);
    } else if (slot.Param<core::param::EnumParam>()->Value() == PARAMETER) {
        planeMode_ = 1;
        this->psUpperRightSeedPoint_.Param<core::param::Vector3fParam>()->SetGUIVisible(true);
        this->psLowerLeftSeedPoint_.Param<core::param::Vector3fParam>()->SetGUIVisible(true);
        this->psSeedPlaneNormal_.Param<core::param::Vector3fParam>()->SetGUIVisible(false);
        this->psSeedPlanePoint_.Param<core::param::Vector3fParam>()->SetGUIVisible(false);
        // this->psSeedPlaneDistance_.Param<core::param::FloatParam>()->SetGUIVisible(false);
    } else {
    }

    return true;
}


std::vector<glm::vec3> mmvtkmStreamLines::calcPlaneBboxIntersectionPoints(
    const visPlanef& samplePlane, const vtkm::Bounds& bounds) {
    std::vector<glm::vec3> intersectionPoints;
    seedPlaneColorVec_.clear();
    seedPlaneIdcs_.clear();

    vislib::math::Cuboid<float> bbox(
        bounds.X.Min, bounds.Y.Min, bounds.Z.Min, bounds.X.Max, bounds.Y.Max, bounds.Z.Max);


    // order of point pairs:
    // front left - front top - front right - front bottom
    // back left - back top - back right - back bottom
    // mid bottom left - mid bottom right - mid top left - mid top right
    std::vector<vislib::Pair<visPoint3f, visPoint3f>> itOut(12);
    bbox.GetLineSegments(itOut.data());


    // intersect every line against the sample plane
    int cnt = 0;
    for (const auto& line : itOut) {
        visPoint3f ip;
        int numIntersections = samplePlane.Intersect(ip, line.GetFirst(), line.GetSecond());

        // if line lies within the plane, then ip = line.GetFirst()
        // so no need for further checks
        if (numIntersections != 0) {
            // check if ip is outside of cube
            // if so, then adjust ip to be on the edge of cube
            if (isOutsideCube(glm::vec3(ip[0], ip[1], ip[2]), bounds)) {
                continue;
            }

            intersectionPoints.push_back(glm::vec3(ip[0], ip[1], ip[2]));
            seedPlaneColorVec_.push_back(seedPlaneColor_);
            seedPlaneIdcs_.push_back(cnt++);
        }
    }

    orderPolygonVertices(intersectionPoints);

    return intersectionPoints;
}


bool mmvtkmStreamLines::isOutsideCube(const glm::vec3& p, const vtkm::Bounds& bounds) {
    bool left = p.x < bounds.X.Min;
    bool right = p.x > bounds.X.Max;
    bool x_out = left || right;

    bool bottom = p.y < bounds.Y.Min;
    bool top = p.y > bounds.Y.Max;
    bool y_out = bottom || top;

    bool back = p.z < bounds.Z.Min;
    bool front = p.z > bounds.Z.Max;
    bool z_out = back || front;

    bool is_out = x_out || y_out || z_out;


    return is_out;
}


void mmvtkmStreamLines::orderPolygonVertices(
    std::vector<glm::vec3>& vertices) { 

    if (vertices.size() < 2) {
        core::utility::log::Log::DefaultLog.WriteError("Number of vertices is too low: %i. Needs to be at least 2.", vertices.size());
	}

	float numVertices = (float)vertices.size();
    std::vector<std::pair<float, glm::vec3>> angles;
	glm::vec3 center = vertices[0] + 0.5f * (vertices[1] - vertices[0]);


    visVec3f tmpNormal = psSeedPlaneNormal_.Param<core::param::Vector3fParam>()->Value();
    glm::vec3 nPlane = glm::normalize(glm::vec3(tmpNormal[0], tmpNormal[1], tmpNormal[2]));

	glm::vec3 xAxis(1.f, 0.f, 0.f);
	glm::vec3 yAxis(0.f, 1.f, 0.f);
	glm::vec3 zAxis(0.f, 0.f, 1.f);


	// construct vector for calculating angle between x- and y-axis
    glm::vec3 projectedXYDir = glm::normalize(glm::vec3(nPlane.x, nPlane.y, 0.f));
    float angleAroundZ = acos(glm::dot(projectedXYDir, yAxis));
    // if x points in neg dir, then rotation needs to occur to the right
    if (nPlane.x < 0.f) angleAroundZ = -angleAroundZ;
    glm::mat4 rotateAroundZ = glm::rotate(glm::mat4(1.f), angleAroundZ, zAxis);
    glm::vec3 rotatedNormalAroundZ = rotateAroundZ * glm::vec4(nPlane, 1.f);


	// construct vector for calculating angle between y- and z-axis
    glm::vec3 projectedYZDir = glm::normalize(glm::vec3(0.f, rotatedNormalAroundZ.y, rotatedNormalAroundZ.z));
    float angleAroundX = acos(glm::dot(projectedYZDir, yAxis));
	// if z points in pos dir, then rotation needs to occur to the right
    if (rotatedNormalAroundZ.z > 0.f) angleAroundX = -angleAroundX;
    glm::mat4 rotateAroundX = glm::rotate(glm::mat4(1.f), angleAroundX, xAxis);
    

	// one rotation matrix
    glm::mat4 rotToXZ = rotateAroundX * rotateAroundZ;
    // glm::mat4 invRotToXZ = glm::inverse(rotToXZ);


	// calc angles and sort by this angle (ascending order)
	for (const auto& v : vertices) {
		// transform current vertex into xz-plane
        glm::vec3 rotatedV = rotToXZ * glm::vec4(v, 1.f);
        glm::vec3 rotatedCenter = rotToXZ * glm::vec4(center, 1.f);
        glm::vec3 edgePoint = glm::vec4(rotatedV.x, rotatedCenter.y, rotatedCenter.z, 1.f);
        glm::vec3 opp = rotatedV - edgePoint;
        glm::vec3 adj = edgePoint - rotatedCenter;

		float oppLength = opp.z > 0.f ? glm::length(opp) : -glm::length(opp);
        float adjLength = adj.x > 0.f ? glm::length(adj) : -glm::length(adj);

		// can be done better without using atan2f
		// instead just compare x and z values
        float angle = atan2f(oppLength, adjLength);

		angles.push_back(std::pair<float, glm::vec3>(angle, v));
	}


	std::sort(angles.begin(), angles.end(), 
		[](const std::pair<float, glm::vec3>& a, const std::pair<float, glm::vec3>& b) -> bool
	{
            return a.first < b.first;
	}
	);
	

	for (int i = 0; i < (int)numVertices; ++i) {
        vertices[i] = angles[i].second;
	} 
}


std::vector<mmvtkmStreamLines::Triangle> mmvtkmStreamLines::decomposePolygon(const std::vector<glm::vec3>& polygon) {
    std::vector<Triangle> triangles;

    int numVertices = polygon.size();

    // decompose like a triangle fan
    // this requires the vertices to be in a triangle fan compatible order
    // but since the polygon is previously build that way, this should be fine
    glm::vec3 fix = polygon[0];
    glm::vec3 last = polygon[1];

    float polygonArea = 0.f;

    for (int i = 2; i < numVertices; ++i) {
        glm::vec3 next = polygon[i];
        Triangle tri(fix, last, next);
        polygonArea += tri.area;

        triangles.push_back(tri);

        last = next;
    }


    // calc area of triangles and weight triangles according to their area
    // for sampling with normal distribution within entire polygon
    for (auto& tri : triangles) {
        tri.weight = tri.area / polygonArea;
    }


    return triangles;
}


bool mmvtkmStreamLines::isInsideTri(const glm::vec3& p, const Triangle& tri) {
    // calc barycentric coordinates
    glm::vec3 v3 = p - tri.o;

    float d00 = glm::dot(tri.v1, tri.v1);
    float d01 = glm::dot(tri.v1, tri.v2);
    float d11 = glm::dot(tri.v2, tri.v2);
    float d20 = glm::dot(v3, tri.v1);
    float d21 = glm::dot(v3, tri.v2);
    float denom = d00 * d11 - d01 * d01;
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;

    return v > 0.f && w > 0.f && u > 0.f;

    // alternativ approach: check if point is on correct side of the line bc (if a is origin)
    // http://alienryderflex.com/point_left_of_ray/
    // might be more efficient if transformation can be done efficiently
}


bool mmvtkmStreamLines::createAndAddMeshDataToCall(std::vector<glm::vec3>& lineData, std::vector<glm::vec3>& lineColor,
    std::vector<unsigned int>& lineIdcs, int numPoints, int numIndices,
    MeshDataAccessCollection::PrimitiveType linePt) {

	if (lineData.size() == 0 || lineColor.size() == 0 || lineIdcs.size() == 0) {
        //core::utility::log::Log::DefaultLog.WriteError("In %s at line %d. LineData, color, " \
		"or index vector is empty.", __FILE__, __LINE__);
        return false;
	}


    MeshDataAccessCollection::VertexAttribute va;
    va.data = reinterpret_cast<uint8_t*>(lineData.data());
    va.byte_size = 3 * numPoints * sizeof(float);
    va.component_cnt = 3;
    va.component_type = MeshDataAccessCollection::ValueType::FLOAT;
    va.stride = 0;
    va.offset = 0;
    va.semantic = MeshDataAccessCollection::AttributeSemanticType::POSITION;

    MeshDataAccessCollection::VertexAttribute vcolor;
    vcolor.data = reinterpret_cast<uint8_t*>(lineColor.data());
    vcolor.byte_size = 3 * numPoints * sizeof(float);
    vcolor.component_cnt = 3;
    vcolor.component_type = MeshDataAccessCollection::ValueType::FLOAT;      
    vcolor.stride = 0;                                                       
    vcolor.offset = 0;                                                       
    vcolor.semantic = MeshDataAccessCollection::AttributeSemanticType::COLOR;

    MeshDataAccessCollection::IndexData idxData;
    idxData.data = reinterpret_cast<uint8_t*>(lineIdcs.data());              
    idxData.byte_size = numIndices * sizeof(unsigned int);                   
    idxData.type = MeshDataAccessCollection::ValueType::UNSIGNED_INT;


	if (!addMeshDataToCall({va, vcolor}, idxData, linePt)) {
        return false;
	}

	return true;
}


bool mmvtkmStreamLines::addMeshDataToCall(const std::vector<MeshDataAccessCollection::VertexAttribute>& va,
    const MeshDataAccessCollection::IndexData& id, MeshDataAccessCollection::PrimitiveType pt) {

    if (va.size() == 0 || id.data == nullptr) {
        core::utility::log::Log::DefaultLog.WriteError(
            "In %s at line %d. LineData, color, or index vector is empty.", __FILE__, __LINE__);
        return false;
    }

    this->meshDataAccess_->addMesh(va, id, pt);

    return true;
}


bool mmvtkmStreamLines::getDataCallback(core::Call& caller) {
    mmvtkm::mmvtkmDataCall* rhsVtkmDc = this->vtkCallerSlot_.CallAs<mmvtkm::mmvtkmDataCall>();
    if (rhsVtkmDc == nullptr) {
        core::utility::log::Log::DefaultLog.WriteError("rhsVtkmDc is nullptr. In %s at line %d", __FILE__, __LINE__);
        return false;
    }

    if (!(*rhsVtkmDc)(0)) {
        return false;
    }


    bool localUpdate = this->oldVersion_ < this->newVersion_;
    bool vtkmUpdate = rhsVtkmDc->HasUpdate();
    // update data only when we have new data
    if (vtkmUpdate || localUpdate) {

        if (vtkmUpdate) {
            if (!(*rhsVtkmDc)(1)) {
                return false;
            }

            ++this->newVersion_;
        }


        mesh::CallMesh* lhsMeshDc = dynamic_cast<mesh::CallMesh*>(&caller);
        if (lhsMeshDc == nullptr) {
            core::utility::log::Log::DefaultLog.WriteError("lhsMeshDc is nullptr. In %s at line %d", __FILE__, __LINE__);
            return false;
        }


        vtkm::cont::DataSet* vtkmMesh = rhsVtkmDc->GetDataSet();

        // for non-temporal data (steady flow) it holds that streamlines = streaklines = pathlines
        // therefore we can calculate the pathlines via the streamline filter
        vtkm::filter::Streamline vtkmStreamlines;

        // specify the seeds
        vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>> seedArray;
        std::vector<vtkm::Vec<vtkm::FloatDefault, 3>> seeds;


        vtkm::Bounds bounds = rhsVtkmDc->GetBounds();
        this->psLowerStreamlineSeedBound_.Param<core::param::Vector3fParam>()->SetValue(
            {(float)bounds.X.Min, (float)bounds.Y.Min, (float)bounds.Z.Min});
        this->psUpperStreamlineSeedBound_.Param<core::param::Vector3fParam>()->SetValue(
            {(float)bounds.X.Max, (float)bounds.Y.Max, (float)bounds.Z.Max});


        glm::vec3 n, dir1, dir2;
        glm::vec3 o, ll, ur, lr, ul;
        visPlanef samplePlane;

        // calc boundary points of plane
        if (planeMode_ == 0) {
            n = glm::normalize(seedPlaneNormal_);
            o = seedPlanePoint_;
            // d = seedPlaneDistance_;

            samplePlane = visPlanef(visPoint3f(o.x, o.y, o.z), visVec3f(n.x, n.y, n.z));

        } else if (planeMode_ == 1) {
            ll = lowerLeftSeedPoint_;
            ur = upperRightSeedPoint_;
            lr = {ur[0], ll[1], ll[2]};
            ul = {ll[0], ur[1], ur[2]};

            dir1 = glm::normalize(lr - ll);
            dir2 = glm::normalize(ul - ll);

            samplePlane =
                visPlanef(visPoint3f(ll.x, ll.y, ll.z), visPoint3f(lr.x, lr.y, lr.z), visPoint3f(ul.x, ul.y, ul.z));
        }

        seedPlane_ = calcPlaneBboxIntersectionPoints(samplePlane, bounds);


        // decompose polygon into triangles
        std::vector<Triangle> planeTriangles = decomposePolygon(seedPlane_);


        // sample points in triangles and combine each triangles' samples
        for (auto tri : planeTriangles) {
            unsigned int num_tri_seeds = (unsigned int)floor(numSeeds_ * tri.weight);

            for (int i = 0; i < num_tri_seeds; ++i) {
				// minor TODO: watch out for bad randomness
                float s = (float)rand() / (float)RAND_MAX;
                float t = (float)rand() / (float)RAND_MAX;
                glm::vec3 p = tri.o + s * tri.v1 + t * tri.v2;


                if (!isInsideTri(p, tri)) {
					// reject via --i;
					// alternativ: transform into other triangle (no rejection needed)
                    --i;
                    continue;
				}

				// can't just simply push back p because seedArray needs vtkm::Vec structure
                seeds.push_back({p[0], p[1], p[2]});
            }
        }


        seedArray = vtkm::cont::make_ArrayHandle(seeds);

        std::string activeField = static_cast<std::string>(activeField_);
        vtkmStreamlines.SetActiveField(activeField);
        vtkmStreamlines.SetStepSize(stepSize_);
        vtkmStreamlines.SetNumberOfSteps(numSteps_);
        vtkmStreamlines.SetSeeds(seedArray);

        core::utility::log::Log::DefaultLog.WriteInfo(
            "NumSeeds: %i. StepSize: %f. NumSteps: %i.", numSeeds_, stepSize_, numSteps_);


		// calc streamlines
        vtkm::cont::DataSet output = vtkmStreamlines.Execute(*vtkmMesh);
        vtkm::io::writer::VTKDataSetWriter writer("streamlines.vtk");
        writer.WriteDataSet(output);


        // get polylines
        vtkm::cont::DynamicCellSet polylineSet = output.GetCellSet(0);
        vtkm::cont::CellSet* polylineSetBase = polylineSet.GetCellSetBase();
        int numPolylines = polylineSetBase->GetNumberOfCells();


        // number of points used to create the polylines (may differ for each polyline)
        std::vector<vtkm::IdComponent> numPointsInPolyline;
        for (int i = 0; i < numPolylines; ++i) {
            numPointsInPolyline.emplace_back(polylineSetBase->GetNumberOfPointsInCell(i));
        }


        // get the indices for the points of the polylines
        std::vector<std::vector<vtkm::Id>> polylinePointIds(numPolylines);
        for (int i = 0; i < numPolylines; ++i) {

            int numPoints = numPointsInPolyline[i];
            std::vector<vtkm::Id> pointIds(numPoints);

            polylineSetBase->GetCellPointIds(i, pointIds.data());

            polylinePointIds[i] = pointIds;
        }


        // there most probably will only be one coordinate system which name isn't specifically set
        // so this should be sufficient
        vtkm::cont::CoordinateSystem coordData = output.GetCoordinateSystem(0);
        vtkm::cont::ArrayHandleVirtualCoordinates coordDataVirtual =
            vtkm::cont::make_ArrayHandleVirtual(coordData.GetData());
        vtkm::ArrayPortalRef<vtkm::Vec<vtkm::FloatDefault, 3>> coords = coordDataVirtual.GetPortalConstControl();


        // clear and resize streamline data
        streamlineData_.clear();
        streamlineData_.resize(numPolylines);
        streamlineColor_.clear();
        streamlineColor_.resize(numPolylines);
        streamlineIndices_.clear();
        streamlineIndices_.resize(numPolylines);

        // in case new streamlines should be calculated
        this->meshDataAccess_->accessMesh().clear();

        // build polylines for megamol mesh
        for (int i = 0; i < numPolylines; ++i) {
            int numPoints = numPointsInPolyline[i];

            // calc data
            streamlineData_[i].clear();
            streamlineData_[i].resize(numPoints);
            streamlineColor_[i].clear();
            streamlineColor_[i].resize(numPoints);

            for (int j = 0; j < numPoints; ++j) {
                vtkm::Vec<vtkm::FloatDefault, 3> crnt = coords.Get(polylinePointIds[i][j]);
                streamlineData_[i][j] = glm::vec3(crnt[0], crnt[1], crnt[2]);

                streamlineColor_[i][j] = glm::vec3((float)j / (float)numPoints * 0.9f + 0.1f);
            }


            // calc indices
            int numLineSegments = numPoints - 1;
            int numIndices = 2 * numLineSegments;

            streamlineIndices_[i].clear();
            streamlineIndices_[i].resize(numIndices);

            for (int j = 0; j < numLineSegments; ++j) {
                int idx = 2 * j;
                streamlineIndices_[i][idx + 0] = j;
                streamlineIndices_[i][idx + 1] = j + 1;
            }

			// adds the mdacs of the streamlines to the call here
            createAndAddMeshDataToCall(streamlineData_[i], streamlineColor_[i], streamlineIndices_[i], numPoints,
                numIndices, MeshDataAccessCollection::PrimitiveType::LINE_STRIP);
        }

		// adds the mdac for the seed plane
		createAndAddMeshDataToCall(seedPlane_, seedPlaneColorVec_, seedPlaneIdcs_, seedPlane_.size(),
            seedPlaneIdcs_.size(), MeshDataAccessCollection::PrimitiveType::TRIANGLE_FAN);


        std::array<float, 6> bbox;
        bbox[0] = bounds.X.Min;
        bbox[1] = bounds.Y.Min;
        bbox[2] = bounds.Z.Min;
        bbox[3] = bounds.X.Max;
        bbox[4] = bounds.Y.Max;
        bbox[5] = bounds.Z.Max;

        this->metaData_.m_bboxs.SetBoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
        this->metaData_.m_bboxs.SetClipBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);

        lhsMeshDc->setMetaData(metaData_);
        lhsMeshDc->setData(meshDataAccess_, this->newVersion_);


        this->oldVersion_ = this->newVersion_;

        return true;
    }

    return true;
}


bool mmvtkmStreamLines::getMetaDataCallback(core::Call& caller) {
    mmvtkm::mmvtkmDataCall* rhsVtkmDc = this->vtkCallerSlot_.CallAs<mmvtkm::mmvtkmDataCall>();
    if (rhsVtkmDc == nullptr) {
        core::utility::log::Log::DefaultLog.WriteError("rhsVtkmDc is nullptr. In %s at line %d", __FILE__, __LINE__);
        return false;
    }

    mesh::CallMesh* lhsMeshDc = dynamic_cast<mesh::CallMesh*>(&caller);
    if (lhsMeshDc == nullptr) {
        core::utility::log::Log::DefaultLog.WriteError("lhsMeshDc is nullptr. In %s at line %d", __FILE__, __LINE__);
        return false;
    }

    if (!(*rhsVtkmDc)(1)) {
        return false;
    }

    // only set it once
    auto md = lhsMeshDc->getMetaData();
    md.m_frame_cnt = 1;
    lhsMeshDc->setMetaData(md);

    return true;
}