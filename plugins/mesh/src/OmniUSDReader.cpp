#include <iostream>
#include <iomanip>

#include "OmniUsdReader.h"
#define TBB_USE_DEBUG 0
#include <iostream>
#include <iomanip>
#include "OmniClient.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usd/primRange.h"
#include "pxr/usd/usdGeom/metrics.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/base/vt/api.h"

#include "mmcore/param/FilePathParam.h"

using namespace pxr;

megamol::mesh::OmniUsdReader::OmniUsdReader()
        : AbstractMeshDataSource()
        , m_filename_slot("Omniverse URL to USD stage", "URL to omniverseserver") {
    this->m_filename_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->m_filename_slot);
}

megamol::mesh::OmniUsdReader::~OmniUsdReader() {}

static void OmniClientConnectionStatusCallbackImpl(
    void* userData, const char* url, OmniClientConnectionStatus status) noexcept {
    std::cout << "Connection Status: " << omniClientGetConnectionStatusString(status) << " [" << url << "]"
              << std::endl;
    if (status == eOmniClientConnectionStatus_ConnectError) {
        std::cout << "[ERROR] Failed connection, exiting." << std::endl;
        exit(-1);
    }
}

// Startup Omniverse
static bool startOmniverse() {
    // Register a function to be called whenever the library wants to print something to a log
    omniClientSetLogCallback(
        [](char const* threadName, char const* component, OmniClientLogLevel level, char const* message) {
            std::cout << "[" << omniClientGetLogLevelString(level) << "] " << message << std::endl;
        });

    // The default log level is "Info", set it to "Debug" to see all messages
    omniClientSetLogLevel(eOmniClientLogLevel_Info);

    // Initialize the library and pass it the version constant defined in OmniClient.h
    // This allows the library to verify it was built with a compatible version. It will
    // return false if there is a version mismatch.
    if (!omniClientInitialize(kOmniClientVersion)) {
        std::cout << "Not the right version" << std::endl;
        return false;
    }

    omniClientRegisterConnectionStatusCallback(nullptr, OmniClientConnectionStatusCallbackImpl);
    return true;
}



bool megamol::mesh::OmniUsdReader::create(void) {
    AbstractMeshDataSource::create();

    std::string stageUrl = "omniverse://10.1.241.198/Users/test/helloworld.usd";
    startOmniverse();
    
    UsdStageRefPtr stage = UsdStage::Open(stageUrl);
    if (!stage) {
        std::cout << "Failed to open stage" << std::endl;
    }

    // Print the up-axis
    std::cout << "Stage up-axis: " << UsdGeomGetStageUpAxis(stage) << std::endl;

    // Print the stage's linear units, or "meters per unit"
    std::cout << "Meters per unit: " << std::setprecision(5) << UsdGeomGetStageMetersPerUnit(stage) << std::endl;

    // Traverse the stage and return the first UsdGeomMesh we find
    auto range = stage->Traverse();
    for (const auto& node : range) {
        std::cout << "\n" << node.GetPath() << std::endl;
        std::cout << node.GetTypeName() << std::endl;

        if (node.GetTypeName() == "Mesh") {
            UsdGeomMesh geom = UsdGeomMesh(node.GetPrim());
            UsdAttribute normalsAttr = geom.GetNormalsAttr();
            UsdAttribute pointsAttr = geom.GetPointsAttr();
            UsdAttribute faceVertexCountsAttr = geom.GetFaceVertexCountsAttr();
            UsdAttribute cornerIndicesAttr = geom.GetCornerIndicesAttr();
            VtArray<GfVec3f> vtNormals;
            VtArray<GfVec3f> vtPoints;
            VtArray<int> vtFaceVertexCounts;
            VtArray<int> vtCornerIndices;
            normalsAttr.Get(&vtNormals);
            pointsAttr.Get(&vtPoints);
            faceVertexCountsAttr.Get(&vtFaceVertexCounts);
            cornerIndicesAttr.Get(&vtCornerIndices);
            //std::cout << "faceCount: " << geom.GetFaceCount() << std::endl;
            std::cout << "normalSize: " << vtNormals.size() << std::endl;
            std::cout << "pointsSize: " << vtPoints.size() << std::endl;
            std::cout << "faceVertexCounts: " << vtFaceVertexCounts << std::endl;
            std::cout << "cornerIndices: " << vtCornerIndices << std::endl;


            std::vector<UsdGeomPrimvar> primvars = geom.GetPrimvars();
            for (const auto& primvar : primvars) {
                VtValue primvarString;
                primvar.Get(&primvarString);
                //std::cout << "primvar: " << primvar.GetName() << ", " << primvarString << std::endl;
            }

            auto properties = node.GetPropertyNames();
            for (const auto& property : properties) {
                std::cout << property << std::endl;
            }

        } else {
            auto properties = node.GetPropertyNames();
            for (const auto& property : properties) {
                std::cout << property << std::endl;
            }
        }
    }

    stage.Reset();

    omniClientShutdown();
    return true;
}

bool megamol::mesh::OmniUsdReader::getMeshDataCallback(core::Call& caller) {

    return true;
}

bool megamol::mesh::OmniUsdReader::getMeshMetaDataCallback(core::Call& caller) {
    return AbstractMeshDataSource::getMeshMetaDataCallback(caller);
}

void megamol::mesh::OmniUsdReader::release() {}

