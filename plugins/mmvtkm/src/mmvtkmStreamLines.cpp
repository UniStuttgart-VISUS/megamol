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


using namespace megamol;
using namespace megamol::mmvtkm;


mmvtkmStreamLines::mmvtkmStreamLines()
    : core::Module()
    , meshCalleeSlot_("meshCalleeSlot", "Requests streamline mesh data from vtk data")
    , vtkCallerSlot_("vtkCallerSlot", "Requests vtk data for streamlines")
    , streamlineFieldName_("fieldName", "Specifies the field name of the streamline vector field")
    , numStreamlineSeed_("numSeeds", "Specifies the number of seeds for the streamlines")
    , streamlineStepSize_("stepSize", "Specifies the step size for the streamlines")
    , numStreamlineSteps_("numSteps", "Specifies the number of steps for the streamlines")
    , lowerStreamlineSeedBound_("lowerSeedBound", "Specifies the lower streamline seed bound")
    , upperStreamlineSeedBound_("upperSeedBound", "Specifies the upper streamline seed bound")
    , seedPlaneMode_("planeMode", "Specifies the representation of the seed plane")
    , lowerLeftSeedPoint_("lowerPlanePoint", "Specifies the lower left point of the seed plane")
    , upperRightSeedPoint_("upperPlanePoint", "Specifies the upper right point of the seed plane")
    , seedPlaneNormal_("planeNormal", "Specifies the normal of the seed plane")
    , seedPlanePoint_("planePoint", "Specifies a point on the seed plane")
    //, seedPlaneDistance_("planeDistance", "Specifies the distance of the seed plane to the origin")
    , applyChanges_("apply", "Press to apply changes for streamline configuration")
    , lowerSeedBound_(0, 0, 0)
    , upperSeedBound_(1, 1, 1)
    , tmp_num_seeds_(100)
    , tmp_num_steps_(1000)
    , tmp_step_size_(0.1f)
    , tmp_active_field_("hs1")
    , tmp_lower_left_seed_point_(-50.f, -50.f, 0)
    , tmp_upper_right_seed_point_(50.f, 50.f, 100.f)
    , tmp_seed_plane_normal_(1.f, 0.f, 0.f)
    , tmp_seed_plane_point_(0.f, 0.f, 0.f)
    , tmp_seed_plane_distance_(0.f)
    , seed_plane_color_({0.5f, 0.f, 0.f})
    , old_version_(0)
    , new_version_(1)
    , plane_mode_(0) {
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
    this->streamlineFieldName_.SetParameter(new core::param::StringParam(tmp_active_field_));
    this->streamlineFieldName_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->streamlineFieldName_);

    this->numStreamlineSeed_.SetParameter(new core::param::IntParam(tmp_num_seeds_, 0));
    this->numStreamlineSeed_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->numStreamlineSeed_);

    this->streamlineStepSize_.SetParameter(new core::param::FloatParam(tmp_step_size_, 0.f));
    this->streamlineStepSize_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->streamlineStepSize_);

    this->numStreamlineSteps_.SetParameter(new core::param::IntParam(tmp_num_steps_, 0));
    this->numStreamlineSteps_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->numStreamlineSteps_);

    this->lowerStreamlineSeedBound_.SetParameter(new core::param::Vector3fParam({0, 0, 0}));
    this->lowerStreamlineSeedBound_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->lowerStreamlineSeedBound_.Param<core::param::Vector3fParam>()->SetGUIReadOnly(true);
    this->MakeSlotAvailable(&this->lowerStreamlineSeedBound_);

    this->upperStreamlineSeedBound_.SetParameter(new core::param::Vector3fParam({1, 1, 1}));
    this->upperStreamlineSeedBound_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->upperStreamlineSeedBound_.Param<core::param::Vector3fParam>()->SetGUIReadOnly(true);
    this->MakeSlotAvailable(&this->upperStreamlineSeedBound_);

    this->seedPlaneMode_.SetParameter(new core::param::EnumParam(plane_mode_));
    this->seedPlaneMode_.Param<core::param::EnumParam>()->SetTypePair(0, "normal");
    this->seedPlaneMode_.Param<core::param::EnumParam>()->SetTypePair(1, "parameter");
    this->seedPlaneMode_.SetUpdateCallback(&mmvtkmStreamLines::planeModeChanged);
    this->MakeSlotAvailable(&this->seedPlaneMode_);

    // TODO add seedPlaneColor_ Paramslot

    this->lowerLeftSeedPoint_.SetParameter(new core::param::Vector3fParam({-50.f, -50.f, 0}));
    this->lowerLeftSeedPoint_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->lowerLeftSeedPoint_);

    this->upperRightSeedPoint_.SetParameter(new core::param::Vector3fParam({50.f, 50.f, 100.f}));
    this->upperRightSeedPoint_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->upperRightSeedPoint_);

    this->seedPlaneNormal_.SetParameter(new core::param::Vector3fParam({1.f, 0.f, 0.f}));
    this->seedPlaneNormal_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    //this->seedPlaneNormal_.Parameter()->SetGUIPresentation(core::param::AbstractParamPresentation::Presentation::)
    this->MakeSlotAvailable(&this->seedPlaneNormal_);

    this->seedPlanePoint_.SetParameter(new core::param::Vector3fParam({0.f, 0.f, 0.f}));
    this->seedPlanePoint_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->seedPlanePoint_);

    // this->seedPlaneDistance_.SetParameter(new core::param::FloatParam(tmp_seed_plane_distance_));
    // this->seedPlaneDistance_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    // this->MakeSlotAvailable(&this->seedPlaneDistance_);

    this->applyChanges_.SetParameter(new core::param::ButtonParam());
    this->applyChanges_.SetUpdateCallback(&mmvtkmStreamLines::setConfiguration);
    this->MakeSlotAvailable(&this->applyChanges_);
}


mmvtkmStreamLines::~mmvtkmStreamLines() { this->Release(); }


void mmvtkmStreamLines::release() {}


bool mmvtkmStreamLines::create() {
    this->mesh_data_access_ = std::make_shared<MeshDataAccessCollection>();
    return true;
}


bool mmvtkmStreamLines::dataChanged(core::param::ParamSlot& slot) {
    vislib::sys::Log::DefaultLog.WriteInfo("changed");
    // this->new_version_++;

    return true;
}


bool mmvtkmStreamLines::planeModeChanged(core::param::ParamSlot& slot) {
    if (slot.Param<core::param::EnumParam>()->Value() == NORMAL) {
        plane_mode_ = 0;
        this->upperRightSeedPoint_.Param<core::param::Vector3fParam>()->SetGUIVisible(false);
        this->lowerLeftSeedPoint_.Param<core::param::Vector3fParam>()->SetGUIVisible(false);
        this->seedPlaneNormal_.Param<core::param::Vector3fParam>()->SetGUIVisible(true);
        this->seedPlanePoint_.Param<core::param::Vector3fParam>()->SetGUIVisible(true);
        // this->seedPlaneDistance_.Param<core::param::FloatParam>()->SetGUIVisible(true);
    } else if (slot.Param<core::param::EnumParam>()->Value() == PARAMETER) {
        plane_mode_ = 1;
        this->upperRightSeedPoint_.Param<core::param::Vector3fParam>()->SetGUIVisible(true);
        this->lowerLeftSeedPoint_.Param<core::param::Vector3fParam>()->SetGUIVisible(true);
        this->seedPlaneNormal_.Param<core::param::Vector3fParam>()->SetGUIVisible(false);
        this->seedPlanePoint_.Param<core::param::Vector3fParam>()->SetGUIVisible(false);
        // this->seedPlaneDistance_.Param<core::param::FloatParam>()->SetGUIVisible(false);
    } else {
    }

    return true;
}


bool mmvtkmStreamLines::setConfiguration(core::param::ParamSlot& slot) {
    tmp_active_field_ = this->streamlineFieldName_.Param<core::param::StringParam>()->Value();
    tmp_num_seeds_ = this->numStreamlineSeed_.Param<core::param::IntParam>()->Value();
    tmp_step_size_ = this->streamlineStepSize_.Param<core::param::FloatParam>()->Value();
    tmp_num_steps_ = this->numStreamlineSteps_.Param<core::param::IntParam>()->Value();
    vislib::math::Vector<float, 3U> ll = this->lowerLeftSeedPoint_.Param<core::param::Vector3fParam>()->Value();
    tmp_lower_left_seed_point_ = {ll.GetX(), ll.GetY(), ll.GetZ()};
    vislib::math::Vector<float, 3U> ur = this->upperRightSeedPoint_.Param<core::param::Vector3fParam>()->Value();
    tmp_upper_right_seed_point_ = {ur.GetX(), ur.GetY(), ur.GetZ()};
    vislib::math::Vector<float, 3U> normal = this->seedPlaneNormal_.Param<core::param::Vector3fParam>()->Value();
    tmp_seed_plane_normal_ = {normal.GetX(), normal.GetY(), normal.GetZ()};
    vislib::math::Vector<float, 3U> point = this->seedPlanePoint_.Param<core::param::Vector3fParam>()->Value();
    tmp_seed_plane_point_ = {point.GetX(), point.GetY(), point.GetZ()};
    // tmp_seed_plane_distance_ = this->seedPlaneDistance_.Param<core::param::FloatParam>()->Value();

    this->new_version_++;

    vislib::sys::Log::DefaultLog.WriteInfo("Configuration set");
    return true;
}


bool mmvtkmStreamLines::lowerBoundChanged(core::param::ParamSlot& slot) {
    core::param::Vector3fParam lower = this->lowerStreamlineSeedBound_.Param<core::param::Vector3fParam>()->Value();

    float lowerX = lower.Value().GetX();
    float lowerY = lower.Value().GetY();
    float lowerZ = lower.Value().GetZ();

    lowerSeedBound_ = visVec3f(lowerX, lowerY, lowerZ);

    this->new_version_++;

    return true;
}

bool mmvtkmStreamLines::upperBoundChanged(core::param::ParamSlot& slot) {
    core::param::Vector3fParam upper = this->upperStreamlineSeedBound_.Param<core::param::Vector3fParam>()->Value();

    float upperX = upper.Value().GetX();
    float upperY = upper.Value().GetY();
    float upperZ = upper.Value().GetZ();

    upperSeedBound_ = visVec3f(upperX, upperY, upperZ);

    this->new_version_++;

    return true;
}

bool mmvtkmStreamLines::seedBoundCheck() {
    core::param::Vector3fParam lower = this->lowerStreamlineSeedBound_.Param<core::param::Vector3fParam>()->Value();
    core::param::Vector3fParam upper = this->upperStreamlineSeedBound_.Param<core::param::Vector3fParam>()->Value();

    float lowerX = lower.Value().GetX();
    float lowerY = lower.Value().GetY();
    float lowerZ = lower.Value().GetZ();
    float upperX = upper.Value().GetX();
    float upperY = upper.Value().GetY();
    float upperZ = upper.Value().GetZ();

    bool compareX = lowerX > upperX;
    bool compareY = lowerY > upperY;
    bool compareZ = lowerZ > upperZ;

    if (compareX || compareY || compareZ) {
        vislib::sys::Log::DefaultLog.WriteError("lower bound is higher than upper bound");
        return false;
    }

    return true;
}


bool mmvtkmStreamLines::isOutsideCube(const visPoint3f& p, const vtkm::Bounds& bounds) {
    bool left = p.GetX() < bounds.X.Min;
    bool right = p.GetX() > bounds.X.Max;
    bool x_out = left || right;

    bool bottom = p.GetY() < bounds.Y.Min;
    bool top = p.GetY() > bounds.Y.Max;
    bool y_out = bottom || top;

    bool back = p.GetZ() < bounds.Z.Min;
    bool front = p.GetZ() > bounds.Z.Max;
    bool z_out = back || front;

    bool is_out = x_out || y_out || z_out;


    if (is_out) {
        // currently it appears to be enough to just check if the point is outside
        /*if (x_out) {
            if (left) {
            

            } else {
            

            }
        }
        else if (y_out) {
            if (top) {

            } else {
                

            }
        }
        else
        {
            if (bottom) {

            } else {
                

            }
        }*/

        return true;
    }

    return false;
}


std::vector<visPoint3f> mmvtkmStreamLines::calcPlaneBboxIntersectionPoints(
    const visPlanef& sample_plane, const vtkm::Bounds& bounds) {
    std::vector<visPoint3f> intersection_points;
    seed_plane_colors_.clear();
    plane_idx_.clear();

    vislib::math::Cuboid<float> bbox(
        bounds.X.Min, bounds.Y.Min, bounds.Z.Min, bounds.X.Max, bounds.Y.Max, bounds.Z.Max);

    // order of point pairs:
    // front left - front top - front right - front bottom
    // back left - back top - back right - back bottom
    // bottom left - bottom right - top left - top right
    std::vector<vislib::Pair<visPoint3f, visPoint3f>> it_out(12);
    bbox.GetLineSegments(it_out.data());

    // re-order points for interesctions, so that intersection points
    // are already in order for triangle fan
    std::vector<vislib::Pair<visPoint3f, visPoint3f>> fan_ordered_lines(12);
    // front face
    fan_ordered_lines[0] = it_out[1]; // front top
    fan_ordered_lines[1] = it_out[0]; // front left
    fan_ordered_lines[2] = it_out[3]; // front bottom
    fan_ordered_lines[3] = it_out[2]; // front right

    // right face
    fan_ordered_lines[4] = it_out[9];  // mid bottom right
    fan_ordered_lines[5] = it_out[6];  // back right
    fan_ordered_lines[6] = it_out[11]; // mid top right

    // back face
    fan_ordered_lines[7] = it_out[7]; // back bottom
    fan_ordered_lines[8] = it_out[4]; // back left
    fan_ordered_lines[9] = it_out[5]; // back top

    // left face
    fan_ordered_lines[10] = it_out[8];  // mid bottom left
    fan_ordered_lines[11] = it_out[10]; // mid top left


    // intersect every line against the sample plane
    int cnt = 0;
    for (auto line : fan_ordered_lines) {
        visPoint3f ip;
        int num_intersections = sample_plane.Intersect(ip, line.GetFirst(), line.GetSecond());

        // if line lies within the plane, then ip = line.GetFirst()
        // so no need for further checks
        if (num_intersections != 0) {
            // check if ip is outside of cube
            // if so, then adjust ip to be on the edge of cube
            if (isOutsideCube(ip, bounds)) {
                continue;
            }

            intersection_points.push_back(ip);
            seed_plane_colors_.push_back(red);
            plane_idx_.push_back(cnt++);
        }
    }

    return intersection_points;
}


std::vector<mmvtkmStreamLines::Triangle> mmvtkmStreamLines::decomposePolygon(const std::vector<visPoint3f>& polygon) {
    std::vector<Triangle> triangles;

    int num_vertices = polygon.size();

    // decompose like a triangle fan
    // this requires the vertices to be in a triangle fan compatible order
    // but since the polygon is previously build that way, this should be fine
    visPoint3f fix = polygon[0];
    visPoint3f last = polygon[1];

    float polygon_area = 0.f;

    for (int i = 2; i < num_vertices; ++i) {
        visPoint3f next = polygon[i];
        Triangle tri(fix, last, next);
        polygon_area += tri.area;

        triangles.push_back(tri);

        last = next;
    }

    // calc area of triangles and weight triangles according to their area
    // for sampling with normal distribution within entire polygon
    for (auto tri : triangles) {
        tri.weight = tri.area / polygon_area;
    }


    return triangles;
}


// TODO: check for 64 bit ids (ifdef)

bool mmvtkmStreamLines::getDataCallback(core::Call& caller) {
    mmvtkm::mmvtkmDataCall* vtkm_dc = this->vtkCallerSlot_.CallAs<mmvtkm::mmvtkmDataCall>();
    if (vtkm_dc == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("vtkm_dc is nullptr. In %s at line %d", __FILE__, __LINE__);
        return false;
    }

    if (!(*vtkm_dc)(0)) {
        return false;
    }


    bool local_update = this->old_version_ < this->new_version_;
    bool vtkm_update = vtkm_dc->HasUpdate();
    // update data only when we have new data
    if (vtkm_update || local_update) {

        if (vtkm_update) {
            if (!(*vtkm_dc)(1)) {
                return false;
            }

            ++this->new_version_;
        }


        mesh::CallMesh* mesh_dc = dynamic_cast<mesh::CallMesh*>(&caller);
        if (mesh_dc == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError("mesh_dc is nullptr. In %s at line %d", __FILE__, __LINE__);
            return false;
        }

        vtkm::cont::DataSet* vtkm_mesh = vtkm_dc->GetDataSet();

        // for non-temporal data (steady flow) it holds that streamlines = streaklines = pathlines
        // therefore we can calculate the pathlines via the streamline filter
        vtkm::filter::Streamline vtkm_streamlines;

        // specify the seeds
        vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>> seedArray;


        std::vector<vtkm::Vec<vtkm::FloatDefault, 3>> seeds;
        vtkm::Id numSeeds = tmp_num_seeds_;
        vtkm::Bounds bounds = vtkm_dc->GetBounds();
        this->lowerStreamlineSeedBound_.Param<core::param::Vector3fParam>()->SetValue(
            {(float)bounds.X.Min, (float)bounds.Y.Min, (float)bounds.Z.Min});
        this->upperStreamlineSeedBound_.Param<core::param::Vector3fParam>()->SetValue(
            {(float)bounds.X.Max, (float)bounds.Y.Max, (float)bounds.Z.Max});


        visVec3f n, dir1, dir2;
        visPoint3f o, ll, ur, lr, ul;
        float d;
        visPlanef sample_plane;


        // calc 4 boundary points of plane
        if (plane_mode_ == 0) {
            tmp_seed_plane_normal_.Normalise();
            n = tmp_seed_plane_normal_;
            o = tmp_seed_plane_point_;
            // d = tmp_seed_plane_distance_;

            sample_plane = visPlanef(o, n);

            // ll = {d / n[0], 0.f, 0.f};
            // dir1 = {-n[1] / n[0], 1.f, 0.f};
            // dir2 = {-n[2] / n[0], 0.f, 1.f};
        } else if (plane_mode_ == 1) {
            ll = tmp_lower_left_seed_point_;
            ur = tmp_upper_right_seed_point_;
            lr = {ur[0], ll[1], ll[2]};
            ul = {ll[0], ur[1], ur[2]};

            dir1 = lr - ll;
            dir1.Normalise();

            dir2 = ul - ll;
            dir2.Normalise();

            sample_plane = visPlanef(ll, lr, ul);
        }

        seed_plane_ = calcPlaneBboxIntersectionPoints(sample_plane, bounds);


        // decompose polygon into triangles
        std::vector<Triangle> plane_triangles = decomposePolygon(seed_plane_);


        // sample points in triangles and combine each triangles' samples
        for (auto tri : plane_triangles) {
            unsigned int num_tri_seeds = (unsigned int)floor(numSeeds * tri.weight);

            for (int i = 0; i < num_tri_seeds; ++i) {
                float s = (float)rand() / (float)RAND_MAX;
                float t = (float)rand() / (float)RAND_MAX;
                visPoint3f p = tri.a + s * tri.v1 + t * tri.v2;


				// TODO: check if its inside triangle
                // if not: either transform back or reject and re-sample


				// can't just simply push back p because seedArray needs vtkm::Vec structure
                seeds.push_back({p[0], p[1], p[2]});
            }
            seeds.push_back({1, 1, 1});
            seeds.push_back({-20, 20, 40});
            seeds.push_back({20, -20, 70});
        }


        seedArray = vtkm::cont::make_ArrayHandle(seeds);

        std::string activeField = static_cast<std::string>(tmp_active_field_);
        vtkm_streamlines.SetActiveField(activeField);
        vtkm_streamlines.SetStepSize(tmp_step_size_);
        vtkm_streamlines.SetNumberOfSteps(tmp_num_steps_);
        vtkm_streamlines.SetSeeds(seedArray);

        vislib::sys::Log::DefaultLog.WriteInfo(
            "NumSeeds: %i. StepSize: %f. NumSteps: %i.", numSeeds, tmp_step_size_, tmp_num_steps_);


		// calc streamlines
        vtkm::cont::DataSet output = vtkm_streamlines.Execute(*vtkm_mesh);
        vtkm::io::writer::VTKDataSetWriter writer("streamlines.vtk");
        writer.WriteDataSet(output);


        // get polylines
        vtkm::cont::DynamicCellSet polylineSet = output.GetCellSet(0);
        vtkm::cont::CellSet* polylineSetBase = polylineSet.GetCellSetBase();
        int numPolylines = polylineSetBase->GetNumberOfCells();

        // number of points used to create the polylines (may be different for each polyline)
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
        streamline_data_.clear();
        streamline_data_.resize(numPolylines);
        streamline_color_.clear();
        streamline_color_.resize(numPolylines);
        streamline_indices_.clear();
        streamline_indices_.resize(numPolylines);

        // in case new streamlines should be calculated
        this->mesh_data_access_->accessMesh().clear();

        // build polylines for megamol mesh
        for (int i = 0; i < numPolylines; ++i) {
            int numPoints = numPointsInPolyline[i];

            // calc data
            streamline_data_[i].clear();
            streamline_data_[i].resize(3 * numPoints);
            streamline_color_[i].clear();
            streamline_color_[i].resize(3 * numPoints);

            for (int j = 0; j < numPoints; ++j) {
                vtkm::Vec<vtkm::FloatDefault, 3> crnt = coords.Get(polylinePointIds[i][j]);
                streamline_data_[i][3 * j + 0] = crnt[0];
                streamline_data_[i][3 * j + 1] = crnt[1];
                streamline_data_[i][3 * j + 2] = crnt[2];

                streamline_color_[i][3 * j + 0] = (float)j / (float)numPoints * 0.9f + 0.1f;
                streamline_color_[i][3 * j + 1] = (float)j / (float)numPoints * 0.9f + 0.1f;
                streamline_color_[i][3 * j + 2] = (float)j / (float)numPoints * 0.9f + 0.1f;
            }


            // calc indices
            int numLineSegments = numPoints - 1;
            int numIndices = 2 * numLineSegments;

            streamline_indices_[i].clear();
            streamline_indices_[i].resize(numIndices);

            for (int j = 0; j < numLineSegments; ++j) {
                int idx = 2 * j;
                streamline_indices_[i][idx + 0] = j;
                streamline_indices_[i][idx + 1] = j + 1;
            }


            // build meshdata for corresponding call
            MeshDataAccessCollection::VertexAttribute va;
            va.data = reinterpret_cast<uint8_t*>(streamline_data_[i].data()); // uint8_t* data;
            va.byte_size = 3 * numPoints * sizeof(float);                     // size_t byte_size;       Buffergröße
            va.component_cnt = 3; // unsigned int component_cnt;    3 für vec4  4 für vec4
            va.component_type = MeshDataAccessCollection::ValueType::FLOAT;          // ValueType component_type;
            va.stride = 0;                                                           // size_t stride;
            va.offset = 0;                                                           // size_t offset;
            va.semantic = MeshDataAccessCollection::AttributeSemanticType::POSITION; // AttributeSemanticType semantic

            MeshDataAccessCollection::VertexAttribute vcolor;
            vcolor.data = reinterpret_cast<uint8_t*>(streamline_color_[i].data()); // uint8_t* data;
            vcolor.byte_size = 3 * numPoints * sizeof(float); // size_t byte_size;       Buffergröße
            vcolor.component_cnt = 3;                         // unsigned int component_cnt;    3 für vec4  4 für vec4
            vcolor.component_type = MeshDataAccessCollection::ValueType::FLOAT;       // ValueType component_type;
            vcolor.stride = 0;                                                        // size_t stride;
            vcolor.offset = 0;                                                        // size_t offset;
            vcolor.semantic = MeshDataAccessCollection::AttributeSemanticType::COLOR; // AttributeSemanticType semantic

            MeshDataAccessCollection::IndexData idxData;
            idxData.data = reinterpret_cast<uint8_t*>(streamline_indices_[i].data()); // uint8_t* data;
            idxData.byte_size = numIndices * sizeof(unsigned int);                    // size_t byte_size;
            idxData.type = MeshDataAccessCollection::ValueType::UNSIGNED_INT;         // ValueType type

            MeshDataAccessCollection::PrimitiveType pt = MeshDataAccessCollection::PrimitiveType::LINES;
            this->mesh_data_access_->addMesh({va, vcolor}, idxData, pt);
        }

        // replace with triangle fan without indices
        // build meshdata for corresponding call
        MeshDataAccessCollection::VertexAttribute va;
        va.data = reinterpret_cast<uint8_t*>(seed_plane_.data()); // uint8_t* data;
        va.byte_size = 3 * seed_plane_.size() * sizeof(float);    // size_t byte_size;       Buffergröße
        va.component_cnt = 3; // unsigned int component_cnt;    3 für vec4  4 für vec4
        va.component_type = MeshDataAccessCollection::ValueType::FLOAT;          // ValueType component_type;
        va.stride = 0;                                                           // size_t stride;
        va.offset = 0;                                                           // size_t offset;
        va.semantic = MeshDataAccessCollection::AttributeSemanticType::POSITION; // AttributeSemanticType semantic

        MeshDataAccessCollection::VertexAttribute vcolor;
        vcolor.data = reinterpret_cast<uint8_t*>(seed_plane_colors_.data()); // uint8_t* data;
        vcolor.byte_size = 3 * seed_plane_colors_.size() * sizeof(float);    // size_t byte_size;       Buffergröße
        vcolor.component_cnt = 3; // unsigned int component_cnt;    3 für vec4  4 für vec4
        vcolor.component_type = MeshDataAccessCollection::ValueType::FLOAT;       // ValueType component_type;
        vcolor.stride = 0;                                                        // size_t stride;
        vcolor.offset = 0;                                                        // size_t offset;
        vcolor.semantic = MeshDataAccessCollection::AttributeSemanticType::COLOR; // AttributeSemanticType semantic

        MeshDataAccessCollection::IndexData idxData;
        idxData.data = reinterpret_cast<uint8_t*>(plane_idx_.data());     // uint8_t* data;
        idxData.byte_size = plane_idx_.size() * sizeof(unsigned int);     // size_t byte_size;
        idxData.type = MeshDataAccessCollection::ValueType::UNSIGNED_INT; // ValueType type

        MeshDataAccessCollection::PrimitiveType pt = MeshDataAccessCollection::PrimitiveType::TRIANGLE_FAN;
        this->mesh_data_access_->addMesh({va, vcolor}, idxData, pt);


        std::array<float, 6> bbox;
        bbox[0] = bounds.X.Min;
        bbox[1] = bounds.Y.Min;
        bbox[2] = bounds.Z.Min;
        bbox[3] = bounds.X.Max;
        bbox[4] = bounds.Y.Max;
        bbox[5] = bounds.Z.Max;

        this->meta_data_.m_bboxs.SetBoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
        this->meta_data_.m_bboxs.SetClipBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);

        mesh_dc->setMetaData(meta_data_);
        mesh_dc->setData(mesh_data_access_, this->new_version_);


        this->old_version_ = this->new_version_;


        vislib::sys::Log::DefaultLog.WriteInfo("Streamlines done.");
        return true;
    }

    return true;
}


bool mmvtkmStreamLines::getMetaDataCallback(core::Call& caller) {
    mmvtkm::mmvtkmDataCall* vtkm_dc = this->vtkCallerSlot_.CallAs<mmvtkm::mmvtkmDataCall>();
    if (vtkm_dc == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("vtkm_dc is nullptr. In %s at line %d", __FILE__, __LINE__);
        return false;
    }

    mesh::CallMesh* mesh_dc = dynamic_cast<mesh::CallMesh*>(&caller);
    if (mesh_dc == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("mesh_dc is nullptr. In %s at line %d", __FILE__, __LINE__);
        return false;
    }

    if (!(*vtkm_dc)(1)) {
        return false;
    }

    // only set it once
    auto md = mesh_dc->getMetaData();
    md.m_frame_cnt = 1;
    mesh_dc->setMetaData(md);

    return true;
}