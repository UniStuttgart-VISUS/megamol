#include "vtkm/filter/Streamline.h"
#include "vtkm/io/writer/VTKDataSetWriter.h"

#include "mmvtkm/mmvtkmDataCall.h"
#include "mmvtkm/mmvtkmStreamLines.h"

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
    , fieldName_("fieldName", "Specifies the field name of the streamline vector field")
    , numStreamlineSeed_("numSeeds", "Specifies the number of seeds for the streamlines")
    , lowerStreamlineSeedBound_("lowerSeedBound", "Specifies the lower streamline seed bound")
    , upperStreamlineSeedBound_("upperSeedBound", "Specifies the upper streamline seed bound")
    , streamlineStepSize_("stepSize", "Specifies the step size for the streamlines")
    , numStreamlineSteps_("numSteps", "Specifies the number of steps for the streamlines")
    , lowerSeedBound_(0, 0, 0)
    , upperSeedBound_(1, 1, 1)
    , old_version_(0)
    , new_version_(1) {
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
    this->fieldName_.SetParameter(new core::param::StringParam("hs1"));
    this->fieldName_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->fieldName_);

    this->numStreamlineSeed_.SetParameter(new core::param::IntParam(2, 0));
    this->numStreamlineSeed_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->numStreamlineSeed_);

    // 0 <= lower bound <= 1
    core::param::Vector3fParam lowerMin = core::param::Vector3fParam({0, 0, 0});
    core::param::Vector3fParam lowerMax = core::param::Vector3fParam({1, 1, 1});
    this->lowerStreamlineSeedBound_.SetParameter(new core::param::Vector3fParam({0, 0, 0}, lowerMin, lowerMax));
    this->lowerStreamlineSeedBound_.SetUpdateCallback(&mmvtkmStreamLines::lowerBoundChanged);
    this->MakeSlotAvailable(&this->lowerStreamlineSeedBound_);

    // lower bound <= upper bound <= 1
    core::param::Vector3fParam upperMin = lowerMax;
    core::param::Vector3fParam upperMax = core::param::Vector3fParam({1, 1, 1});
    this->upperStreamlineSeedBound_.SetParameter(new core::param::Vector3fParam({1, 1, 1}, upperMin, upperMax));
    this->upperStreamlineSeedBound_.SetUpdateCallback(&mmvtkmStreamLines::upperBoundChanged);
    this->MakeSlotAvailable(&this->upperStreamlineSeedBound_);

    this->streamlineStepSize_.SetParameter(new core::param::FloatParam(0.1f, 0.f));
    this->streamlineStepSize_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->streamlineStepSize_);

    this->numStreamlineSteps_.SetParameter(new core::param::IntParam(100, 0));
    this->numStreamlineSteps_.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->numStreamlineSteps_);
}


mmvtkmStreamLines::~mmvtkmStreamLines() { this->Release(); }


void mmvtkmStreamLines::release() {}


bool mmvtkmStreamLines::create() { 
	this->mesh_data_access_ = std::make_shared<MeshDataAccessCollection>();
    return true;
}


bool mmvtkmStreamLines::dataChanged(core::param::ParamSlot& slot) {
    this->new_version_++;

    return true;
}

bool mmvtkmStreamLines::lowerBoundChanged(core::param::ParamSlot& slot) {
    core::param::Vector3fParam lower = this->lowerStreamlineSeedBound_.Param<core::param::Vector3fParam>()->Value();

    float lowerX = lower.Value().GetX();
    float lowerY = lower.Value().GetY();
    float lowerZ = lower.Value().GetZ();

    lowerSeedBound_ = Vec3f(lowerX, lowerY, lowerZ);

    this->new_version_++;

    return true;
}

bool mmvtkmStreamLines::upperBoundChanged(core::param::ParamSlot& slot) {
    core::param::Vector3fParam upper = this->upperStreamlineSeedBound_.Param<core::param::Vector3fParam>()->Value();

    float upperX = upper.Value().GetX();
    float upperY = upper.Value().GetY();
    float upperZ = upper.Value().GetZ();

    upperSeedBound_ = Vec3f(upperX, upperY, upperZ);

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

    // update data only when we have a new data
    if (vtkm_dc->HasUpdate() || this->old_version_ < this->new_version_) {
        ++this->new_version_;

		if (!(*vtkm_dc)(1)) {
            return false;
		}

		vtkm::Bounds bounds = vtkm_dc->GetBounds();
        core::param::Vector3fParam min({(float)bounds.X.Min, (float)bounds.Y.Min, (float)bounds.Z.Min});
        core::param::Vector3fParam max({(float)bounds.X.Max, (float)bounds.Y.Max, (float)bounds.Z.Max});
		// det jet nisch
		// mach det anders
        //this->lowerStreamlineSeedBound.SetParameter(new core::param::Vector3fParam(min, min, max));
        //this->upperStreamlineSeedBound.SetParameter(new core::param::Vector3fParam(max, min, max));

        mesh::CallMesh* mesh_dc = dynamic_cast<mesh::CallMesh*>(&caller);
        if (mesh_dc == nullptr) {
            vislib::sys::Log::DefaultLog.WriteError("mesh_dc is nullptr. In %s at line %d", __FILE__, __LINE__);
            return false;
        }

        vtkm::cont::DataSet* vtkm_mesh = vtkm_dc->GetDataSet();

        // for non-temporal data (steady flow) it holds that streamlines = streaklines = pathlines
        // therefore we can calculate the pathlines via the streamline filter
        vtkm::filter::Streamline streamlines;

        // specify the seeds
        vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>> seedArray;
        // TODO: currently the seeds have to be manually set
        // maybe cool to have some random generation of seeds
        // (maybe along a line or within a plane, cf. custom source in paraview)

        std::vector<vtkm::Vec<vtkm::FloatDefault, 3>> seeds;

        vtkm::Id numSeeds = this->numStreamlineSeed_.Param<core::param::IntParam>()->Value();

        for (int i = 0; i < numSeeds; i++) {
            vtkm::Vec<vtkm::FloatDefault, 3> p;
            vtkm::FloatDefault rx = (vtkm::FloatDefault)rand() / (vtkm::FloatDefault)RAND_MAX;
            vtkm::FloatDefault ry = (vtkm::FloatDefault)rand() / (vtkm::FloatDefault)RAND_MAX;
            vtkm::FloatDefault rz = (vtkm::FloatDefault)rand() / (vtkm::FloatDefault)RAND_MAX;
            p[0] = static_cast<vtkm::FloatDefault>(bounds.X.Min + rx * bounds.X.Length());
            p[1] = static_cast<vtkm::FloatDefault>(bounds.Y.Min + ry * bounds.Y.Length());
            p[2] = static_cast<vtkm::FloatDefault>(bounds.Z.Min + rz * bounds.Z.Length());
            seeds.push_back(p);
        }

        seedArray = vtkm::cont::make_ArrayHandle(seeds);

        std::string activeField =
            "hs1"; // static_cast<std::string>(this->fieldName.Param<core::param::StringParam>()->ValueString());
        streamlines.SetActiveField(activeField);
        streamlines.SetStepSize(0.1f);     // this->streamlineStepSize.Param<core::param::FloatParam>()->Value());
        streamlines.SetNumberOfSteps(100); // this->numStreamlineSteps.Param<core::param::IntParam>()->Value());
        streamlines.SetSeeds(seedArray);
        
        // get parallel computing
        // atm just single core --> slow af
  //      vtkm::cont::DataSet output = streamlines.Execute(*vtkm_mesh);
  //      vtkm::io::writer::VTKDataSetWriter writer("streamlines.vtk");
  //      writer.WriteDataSet(output);

		//
  //      // get polylines
  //      vtkm::cont::DynamicCellSet polylineSet = output.GetCellSet(0);
  //      vtkm::cont::CellSet* polylineSetBase = polylineSet.GetCellSetBase();
  //      int numPolylines = polylineSetBase->GetNumberOfCells();
  //      
  //      // number of points used to create the polylines (may be different for each polyline)
  //      std::vector<vtkm::IdComponent> numPointsInPolyline;
  //      for (int i = 0; i < numPolylines; ++i) {
  //          numPointsInPolyline.emplace_back(polylineSetBase->GetNumberOfPointsInCell(i));
  //      }

  //      // get the indices for the points of the polylines
  //      std::vector<std::vector<vtkm::Id>> polylinePointIds(numPolylines);
  //      for (int i = 0; i < numPolylines; ++i) {

  //          int numPoints = numPointsInPolyline[i];
  //          std::vector<vtkm::Id> pointIds(numPoints);

  //          polylineSetBase->GetCellPointIds(i, pointIds.data());

  //          polylinePointIds[i] = pointIds;
  //      }

  //      
  //      // there most probably will only be one coordinate system which name isn't specifically set
  //      // so this should be sufficient
  //      vtkm::cont::CoordinateSystem coordData = output.GetCoordinateSystem(0);
  //      vtkm::cont::ArrayHandleVirtualCoordinates coordDataVirtual =
  //          vtkm::cont::make_ArrayHandleVirtual(coordData.GetData());
  //      vtkm::ArrayPortalRef<vtkm::Vec<vtkm::FloatDefault, 3>> coords = coordDataVirtual.GetPortalConstControl();


		//// build polylines for megamol mesh
  //      for (int i = 0; i < numPolylines; ++i) {
  //          int numPoints = numPointsInPolyline[i];
  //          std::vector<float> points(3 * numPoints);

  //          for (int j = 0; j < numPoints; ++j) {
  //              vtkm::Vec<vtkm::FloatDefault, 3> crnt = coords.Get(polylinePointIds[i][j]);
  //              points[3 * j + 0] = crnt[0];
  //              points[3 * j + 1] = crnt[1];
  //              points[3 * j + 2] = crnt[2];
  //          }

  //          // calc indices
  //          int numLineSegments = numPoints - 1;
  //          int numIndices = 2 * numLineSegments;
  //          std::vector<int> indices(numIndices);
  //          for (int j = 0; j < numLineSegments; ++j) {
  //              int idx = 2 * j;
  //              indices[idx + 0] = (int)polylinePointIds[i][j + 0];
  //              indices[idx + 1] = (int)polylinePointIds[i][j + 1];
  //          }

  //          MeshDataAccessCollection::VertexAttribute va;
  //          va.data = reinterpret_cast<uint8_t*>(points.data());	// uint8_t* data;
  //          va.byte_size = 3 * numPoints * sizeof(float);			// size_t byte_size;       Buffergröße
  //          va.component_cnt = 3;									// unsigned int component_cnt;    3 für vec4  4 für vec4
  //          va.component_type = MeshDataAccessCollection::ValueType::FLOAT; // ValueType component_type;
  //          va.stride = 0;											// size_t stride;
  //          va.offset = 0;											// size_t offset;
  //          va.semantic = MeshDataAccessCollection::AttributeSemanticType::POSITION; // AttributeSemanticType semantic

  //          MeshDataAccessCollection::IndexData idxData;
  //          idxData.data = reinterpret_cast<uint8_t*>(indices.data());	// uint8_t* data;
  //          idxData.byte_size = numIndices * sizeof(int);               // size_t byte_size;
  //          idxData.type = MeshDataAccessCollection::ValueType::INT;    // ValueType type

  //          MeshDataAccessCollection::PrimitiveType pt = MeshDataAccessCollection::PrimitiveType::LINES;
  //          //this->mesh_data_access_->addMesh({va}, idxData, pt);
  //      }

		std::vector<float> test = {-1.f, -1.f, 0.5f, 1.f, -1.f, 0.5f, 0.f, 1.f, 0.5f};
		MeshDataAccessCollection::VertexAttribute va2;
        va2.data = reinterpret_cast<uint8_t*>(test.data());							// uint8_t* data;
        va2.byte_size =
            3 * 3 *
            MeshDataAccessCollection::getByteSize(MeshDataAccessCollection::FLOAT); // size_t byte_size; Buffergröße
        va2.component_cnt = 3;														// unsigned int component_cnt;    3 für vec4  4 für vec4
        va2.component_type = MeshDataAccessCollection::ValueType::FLOAT;			// ValueType component_type;
        va2.stride = 0;																// size_t stride;
        va2.offset = 0;																// size_t offset;
        va2.semantic = MeshDataAccessCollection::AttributeSemanticType::POSITION;	// AttributeSemanticType semantic
		
        MeshDataAccessCollection::IndexData idxData2;
        std::vector<int> ti = {0, 1, 2};
        idxData2.data = reinterpret_cast<uint8_t*>(ti.data());
        idxData2.byte_size = 3 * MeshDataAccessCollection::getByteSize(MeshDataAccessCollection::INT);
        idxData2.type = MeshDataAccessCollection::ValueType::INT;

        MeshDataAccessCollection::PrimitiveType pt2 = MeshDataAccessCollection::PrimitiveType::TRIANGLES;
        this->mesh_data_access_->addMesh({ va2 }, idxData2, pt2);

		std::array<float, 6> bbox;
        bbox[0] = -1.f;
        bbox[1] = -1.f;
        bbox[2] = 0.f;
        bbox[3] = 1.f;
        bbox[4] = 1.f;
        bbox[5] = 1.f;

		this->meta_data_.m_bboxs.SetBoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
		this->meta_data_.m_bboxs.SetClipBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);

		mesh_dc->setMetaData(meta_data_);
		mesh_dc->setData(mesh_data_access_, this->new_version_);


        this->old_version_ = this->new_version_;


        vislib::sys::Log::DefaultLog.WriteInfo("Streamlines done.");
        return true;
    }

    return false;
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

	/*vtkm::Bounds b = vtkm_dc->GetBounds();
    core::param::Vector3fParam min({(float)b.X.Min, (float)b.Y.Min, (float)b.Z.Min});
    core::param::Vector3fParam max({(float)b.X.Max, (float)b.Y.Max, (float)b.Z.Max});
    this->lowerStreamlineSeedBound_.SetParameter(new core::param::Vector3fParam(min, min, max));
    this->upperStreamlineSeedBound_.SetParameter(new core::param::Vector3fParam(max, min, max));*/

    // only set it once
    if (this->old_version_ == 0) {
        auto md = mesh_dc->getMetaData();
        md.m_frame_cnt = 1;
        mesh_dc->setMetaData(md);
        //++this->old_version_;
    }

    return true;
}