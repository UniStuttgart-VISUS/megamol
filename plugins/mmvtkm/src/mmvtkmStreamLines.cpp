#include "vtkm/filter/Streamline.h"
#include "vtkm/io/writer/VTKDataSetWriter.h"

#include "mesh/MeshCalls.h"

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
    , meshCalleeSlot("meshCalleeSlot", "Requests streamline mesh data from vtk data")
    , vtkCallerSlot("vtkCallerSlot", "Requests vtk data for streamlines")
    , fieldName("fieldName", "Specifies the field name of the streamline vector field")
    , numStreamlineSeed("numSeeds", "Specifies the number of seeds for the streamlines")
    , lowerStreamlineSeedBound("lowerSeedBound", "Specifies the lower streamline seed bound")
    , upperStreamlineSeedBound("upperSeedBound", "Specifies the upper streamline seed bound")
    , streamlineStepSize("stepSize", "Specifies the step size for the streamlines")
    , numStreamlineSteps("numSteps", "Specifies the number of steps for the streamlines")
    , lowerSeedBound(0, 0, 0)
    , upperSeedBound(1, 1, 1)
    , old_version(0)
    , new_version(0) {
    this->meshCalleeSlot.SetCallback(mesh::CallGPUMeshData::ClassName(),
        mesh::CallGPUMeshData::FunctionName(0), // used to be mesh::CallGPUMeshData
        &mmvtkmStreamLines::getDataCallback);
    this->meshCalleeSlot.SetCallback(
        mesh::CallGPUMeshData::ClassName(), mesh::CallGPUMeshData::FunctionName(1), &mmvtkmStreamLines::getMetaDataCallback);
    this->MakeSlotAvailable(&this->meshCalleeSlot);

    this->vtkCallerSlot.SetCompatibleCall<mmvtkmDataCallDescription>();
    this->MakeSlotAvailable(&this->vtkCallerSlot);

    // TODO: instead of hardcoding fieldnames,
    // maybe also read all field names and show them as dropdown menu in megamol
    this->fieldName.SetParameter(new core::param::StringParam("hs1"));
    this->fieldName.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->fieldName);

    this->numStreamlineSeed.SetParameter(new core::param::IntParam(2, 0));
    this->numStreamlineSeed.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->numStreamlineSeed);

    // 0 <= lower bound <= 1
    core::param::Vector3fParam lowerMin = core::param::Vector3fParam({0, 0, 0});
    core::param::Vector3fParam lowerMax = core::param::Vector3fParam({1, 1, 1});
    this->lowerStreamlineSeedBound.SetParameter(new core::param::Vector3fParam({0, 0, 0}, lowerMin, lowerMax));
    this->lowerStreamlineSeedBound.SetUpdateCallback(&mmvtkmStreamLines::lowerBoundChanged);
    this->MakeSlotAvailable(&this->lowerStreamlineSeedBound);

    // lower bound <= upper bound <= 1
    core::param::Vector3fParam upperMin = lowerMax;
    core::param::Vector3fParam upperMax = core::param::Vector3fParam({1, 1, 1});
    this->upperStreamlineSeedBound.SetParameter(new core::param::Vector3fParam({1, 1, 1}, upperMin, upperMax));
    this->upperStreamlineSeedBound.SetUpdateCallback(&mmvtkmStreamLines::upperBoundChanged);
    this->MakeSlotAvailable(&this->upperStreamlineSeedBound);

    this->streamlineStepSize.SetParameter(new core::param::FloatParam(0.1f, 0.f));
    this->streamlineStepSize.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->streamlineStepSize);

    this->numStreamlineSteps.SetParameter(new core::param::IntParam(100, 0));
    this->numStreamlineSteps.SetUpdateCallback(&mmvtkmStreamLines::dataChanged);
    this->MakeSlotAvailable(&this->numStreamlineSteps);
}


mmvtkmStreamLines::~mmvtkmStreamLines() { this->Release(); }


void mmvtkmStreamLines::release() {}


bool mmvtkmStreamLines::create() { return true; }


bool mmvtkmStreamLines::dataChanged(core::param::ParamSlot& slot) {
    this->new_version++;

    return true;
}

bool mmvtkmStreamLines::lowerBoundChanged(core::param::ParamSlot& slot) {
    core::param::Vector3fParam lower = this->lowerStreamlineSeedBound.Param<core::param::Vector3fParam>()->Value();

    float lowerX = lower.Value().GetX();
    float lowerY = lower.Value().GetY();
    float lowerZ = lower.Value().GetZ();

    lowerSeedBound = Vec3f(lowerX, lowerY, lowerZ);

    this->new_version++;

    return true;
}

bool mmvtkmStreamLines::upperBoundChanged(core::param::ParamSlot& slot) {
    core::param::Vector3fParam upper = this->upperStreamlineSeedBound.Param<core::param::Vector3fParam>()->Value();

    float upperX = upper.Value().GetX();
    float upperY = upper.Value().GetY();
    float upperZ = upper.Value().GetZ();

    upperSeedBound = Vec3f(upperX, upperY, upperZ);

    this->new_version++;

    return true;
}


bool mmvtkmStreamLines::seedBoundCheck() {
    core::param::Vector3fParam lower = this->lowerStreamlineSeedBound.Param<core::param::Vector3fParam>()->Value();
    core::param::Vector3fParam upper = this->upperStreamlineSeedBound.Param<core::param::Vector3fParam>()->Value();

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

// TODO: ifdef id64 components check for vtkm

bool mmvtkmStreamLines::getDataCallback(core::Call& caller) {
    mmvtkm::mmvtkmDataCall* vtkm_dc = this->vtkCallerSlot.CallAs<mmvtkm::mmvtkmDataCall>();
    if (vtkm_dc == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("vtkm_dc is nullptr. In %s at line %d", __FILE__, __LINE__);
        return false;
    }

    if (!(*vtkm_dc)(0)) {
        return false;
    }

    // update data only when we have a new data
    if (vtkm_dc->HasUpdate() || this->old_version < this->new_version) {

		if (!(*vtkm_dc)(1)) {
            return false;
		}

		vtkm::Bounds b = vtkm_dc->GetBounds();
        core::param::Vector3fParam min({(float)b.X.Min, (float)b.Y.Min, (float)b.Z.Min});
        core::param::Vector3fParam max({(float)b.X.Max, (float)b.Y.Max, (float)b.Z.Max});
		// det jet nisch
		// mach det anders
        //this->lowerStreamlineSeedBound.SetParameter(new core::param::Vector3fParam(min, min, max));
        //this->upperStreamlineSeedBound.SetParameter(new core::param::Vector3fParam(max, min, max));

        mesh::CallGPUMeshData* mesh_dc = dynamic_cast<mesh::CallGPUMeshData*>(&caller);
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

        vtkm::Id numSeeds = this->numStreamlineSeed.Param<core::param::IntParam>()->Value();

        vtkm::Bounds bounds(lowerSeedBound, upperSeedBound);

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
        vtkm::cont::DataSet output = streamlines.Execute(*vtkm_mesh);
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
            //vtkm::Id* pointIds = new vtkm::Id(numPoints);
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


		using GMC = megamol::mesh::GPUMeshCollection;
        std::shared_ptr<GMC> gmc = std::make_shared<GMC>();

		// one mesh with numPolylines attributes?
		// or numPolylines meshes with one attribute each
		for (int i = 0; i < numPolylines; ++i) {
            int numPoints = numPointsInPolyline[i];
            std::vector<float> points(3 * numPoints);

            for (int j = 0; j < numPoints; ++j) {
                vtkm::Vec<vtkm::FloatDefault, 3> crnt = coords.Get(polylinePointIds[i][j]);
                points[3 * j + 0] = crnt[0];
                points[3 * j + 1] = crnt[1];
                points[3 * j + 2] = crnt[2];
            }

			// calc indices
            int numLineSegments = numPoints - 1;
            int numIndices = 2 * numLineSegments;
            std::vector<int> indices(numIndices);
            for (int j = 0; j < numLineSegments; ++j) {
                int idx = 2 * j;
                indices[idx + 0] = (int)polylinePointIds[i][idx + 0];
                indices[idx + 1] = (int)polylinePointIds[i][idx + 1];
			}

            glowl::VertexLayout vertex_descriptor;
            glowl::VertexLayout::Attribute attrib(sizeof(float) * 3 * numPoints, GL_FLOAT, false, 0);
            vertex_descriptor.attributes.push_back(attrib);
            vertex_descriptor.byte_size = sizeof(float);
            std::vector<std::pair<uint8_t*, uint8_t*>> vertex_buffers = {
                {(uint8_t*)points.data(), (uint8_t*)points.data() + (uint8_t)sizeof(float)}};
            std::pair<uint8_t*, uint8_t*> index_buffer = {
                (uint8_t*)indices.data(), (uint8_t*)indices.data() + (uint8_t)sizeof(int)};
            GLenum index_type = GL_INT;
            GLenum usage = GL_STATIC_DRAW;
            GLenum primitive_type = GL_LINES;
            bool store_seperate = false;
			
            gmc->addMesh(vertex_descriptor, vertex_buffers, index_buffer, index_type, usage, primitive_type);
        }

		mesh_dc->setData(gmc, this->new_version);


        this->new_version++;


        this->old_version = this->new_version;

        return true;
    }

    return false;
}


bool mmvtkmStreamLines::getMetaDataCallback(core::Call& caller) {
    mmvtkm::mmvtkmDataCall* vtkm_dc = this->vtkCallerSlot.CallAs<mmvtkm::mmvtkmDataCall>();
    if (vtkm_dc == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("vtkm_dc is nullptr. In %s at line %d", __FILE__, __LINE__);
        return false;
    }

    mesh::CallGPUMeshData* mesh_dc = dynamic_cast<mesh::CallGPUMeshData*>(&caller);
    if (mesh_dc == nullptr) {
        vislib::sys::Log::DefaultLog.WriteError("mesh_dc is nullptr. In %s at line %d", __FILE__, __LINE__);
        return false;
    }

	if (!(*vtkm_dc)(1)) {
        return false;
	}

	vtkm::Bounds b = vtkm_dc->GetBounds();
    core::param::Vector3fParam min({(float)b.X.Min, (float)b.Y.Min, (float)b.Z.Min});
    core::param::Vector3fParam max({(float)b.X.Max, (float)b.Y.Max, (float)b.Z.Max});
    this->lowerStreamlineSeedBound.SetParameter(new core::param::Vector3fParam(min, min, max));
    this->upperStreamlineSeedBound.SetParameter(new core::param::Vector3fParam(max, min, max));

    // only set it once
    if (this->old_version == 0) {
        auto md = mesh_dc->getMetaData();
        md.m_frame_cnt = 1;
        mesh_dc->setMetaData(md);
        ++this->old_version;
    }

    return true;
}