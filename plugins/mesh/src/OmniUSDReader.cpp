#include <iostream>
#include <iomanip>

#include "OmniUsdReader.h"
#include <iostream>
#include <iomanip>
#include "OmniClient.h"
#include "pxr/usd/usd/primRange.h"
#include "pxr/usd/usdGeom/metrics.h"
#include "pxr/usd/usdGeom/mesh.h"
#include "pxr/base/vt/api.h"
#include "pxr/base/gf/vec3f.h"

#include "mmcore/param/StringParam.h"

using namespace pxr;

megamol::mesh::OmniUsdReader::OmniUsdReader()
        : AbstractMeshDataSource()
        , m_filename_slot("Omniverse URL to USD stage", "URL to omniverseserver") {
    this->m_filename_slot << new core::param::StringParam("");
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
    return true;
}

bool megamol::mesh::OmniUsdReader::getMeshDataCallback(core::Call& caller) {
    CallMesh* lhs_mesh_call = dynamic_cast<CallMesh*>(&caller);
    if (lhs_mesh_call == NULL) {
        return false;
    }

    //Check if filename slot(string param) has been changed
    if (this->m_filename_slot.IsDirty()) {
        //Reset dirty flag and update version id
        m_filename_slot.ResetDirty();
        ++m_version;

        // right now taking test scene and omitting filepath parameter for testing purposes
        std::string stageUrl = "omniverse://10.1.241.198/Users/test/helloworld.usd";
        startOmniverse();

        //open the stage and return false if stage could not be opened
        m_usd_stage = UsdStage::Open(stageUrl);
       
        if (!m_usd_stage) {
            std::cout << "Failed to open stage" << std::endl;
            return false;
        }

        //Traverse the stage and save prims in a UsdRange range (iterable)
        auto range = m_usd_stage->Traverse();
        
        //Initialize bounding box for stage
        std::array<float, 6> bbox;
        bbox[0] = std::numeric_limits<float>::max();
        bbox[1] = std::numeric_limits<float>::max();
        bbox[2] = std::numeric_limits<float>::max();
        bbox[3] = -std::numeric_limits<float>::max();
        bbox[4] = -std::numeric_limits<float>::max();
        bbox[5] = -std::numeric_limits<float>::max();
      
        std::vector<UsdGeomMesh> meshes;
        // Traverse stage and save out meshes
        for (const auto& prim : range) {
            std::cout << "\n" << prim.GetPath() << std::endl;
            std::cout << prim.GetTypeName() << std::endl;


            // Check if prim is a mesh
            if (prim.IsA<UsdGeomMesh>()) {
                UsdGeomMesh geom = UsdGeomMesh(prim.GetPrim());
                meshes.emplace_back(geom);
            }
        }

        this->m_positions.clear();

        //shape count
        int s_cnt = meshes.size();
        m_indices.resize(s_cnt);
        m_positions.resize(s_cnt);

        for (int current_m = 0; current_m < s_cnt; current_m ++) {
            UsdGeomMesh geom = meshes[current_m];
            //Perform UsdAttribute requests on the mesh and print them 
            UsdAttribute normalsAttr = geom.GetNormalsAttr();
            UsdAttribute pointsAttr = geom.GetPointsAttr();
            UsdAttribute faceVertexCountsAttr = geom.GetFaceVertexCountsAttr();
            UsdAttribute faceVertexIndicesAttr = geom.GetFaceVertexIndicesAttr();
            UsdGeomXformable::XformQuery xformquery = UsdGeomXformable::XformQuery(geom);
            GfMatrix4d xformOps;
            xformquery.GetLocalTransformation(&xformOps, 0.0);
            VtArray<GfVec3f> vtNormals;
            VtArray<GfVec3f> vtPoints;
            VtArray<int> vtFaceVertexCounts;
            VtArray<int> vtFaceVertexIndices;
            normalsAttr.Get(&vtNormals);
            pointsAttr.Get(&vtPoints);
            faceVertexCountsAttr.Get(&vtFaceVertexCounts);
            faceVertexIndicesAttr.Get(&vtFaceVertexIndices);

            // vertex count for hole mesh
            uint64_t point_size = vtPoints.size();
       
            
            uint64_t indices_size = 0;
            for (int i = 0; i < vtFaceVertexCounts.size(); i++) {
                if (vtFaceVertexCounts[i] == 3) {
                    indices_size+=3;
                } else if (vtFaceVertexCounts[i] == 4) {
                    indices_size += 6;
                }
            }

            //reserve space for vertex information
            m_indices[current_m].resize(indices_size);
            m_positions[current_m].reserve(point_size * 3);
            std::cout << "reserving positions of: " << current_m << std::endl;

            //index offset for facevertexindices
            size_t index_offset = 0;
            size_t usd_index_offset = 0;
            //Loop over triangled faces in the mesh
            for (size_t f = 0; f < vtFaceVertexCounts.size(); f++) {
                // Check if vertex count is 3 for the face
                if (vtFaceVertexCounts[f] == 3) {
                    // Loop over vertices in the face.
                    for (size_t v = 0; v < vtFaceVertexCounts[f]; v++) {
                                  
                        //update m_indices
                        m_indices[current_m][index_offset + v] = vtFaceVertexIndices[usd_index_offset + v];
                    }
                    index_offset += 3;
                    usd_index_offset += 3;
                    
                } else if (vtFaceVertexCounts[f] == 4) {
                    // Loop over vertices in the face.
                    
                    //update m_indices
                    m_indices[current_m][index_offset] = vtFaceVertexIndices[usd_index_offset];
                    m_indices[current_m][index_offset + 1] = vtFaceVertexIndices[usd_index_offset + 1];
                    m_indices[current_m][index_offset + 2] = vtFaceVertexIndices[usd_index_offset + 2];
                    m_indices[current_m][index_offset + 3] = vtFaceVertexIndices[usd_index_offset + 0];
                    m_indices[current_m][index_offset + 4] = vtFaceVertexIndices[usd_index_offset + 2];
                    m_indices[current_m][index_offset+ 5] = vtFaceVertexIndices[usd_index_offset + 3];
                    
                    index_offset += 6;
                    usd_index_offset += 4;
                    
                } else {
                    std::cout << "N-GON FACE" << std::endl;
                }
            }
            //update m_positions
            for (int p = 0; p < point_size; p++) {
                vtPoints[p][0] += xformOps[3][0];
                vtPoints[p][1] += xformOps[3][1];
                vtPoints[p][2] += xformOps[3][2];
                float vx = vtPoints[p][0];
                float vy = vtPoints[p][1];
                float vz = vtPoints[p][2];

                //Update bounding box
                bbox[0] = std::min(bbox[0], vx);
                bbox[1] = std::min(bbox[1], vy);
                bbox[2] = std::min(bbox[2], vz);
                bbox[3] = std::max(bbox[3], vx);
                bbox[4] = std::max(bbox[4], vy);
                bbox[5] = std::max(bbox[5], vz);
                auto current_position_ptr = &(vtPoints[p][0]);
                m_positions[current_m].insert(
                    m_positions[current_m].end(), current_position_ptr, current_position_ptr + 3);
            }

            const auto pos_ptr = m_positions[current_m].data();
            std::vector<MeshDataAccessCollection::VertexAttribute> mesh_attributes;
            // create vertexattribute for positions
            mesh_attributes.emplace_back(MeshDataAccessCollection::VertexAttribute{reinterpret_cast<uint8_t*>(pos_ptr),
                3 * vtPoints.size() * MeshDataAccessCollection::getByteSize(MeshDataAccessCollection::FLOAT), 3,
                MeshDataAccessCollection::FLOAT, 12, 0, MeshDataAccessCollection::AttributeSemanticType::POSITION});

            // create MeshDataAccessCollection indexdata
            MeshDataAccessCollection::IndexData mesh_indices;
            mesh_indices.data = reinterpret_cast<uint8_t*>(m_indices[current_m].data());
            mesh_indices.byte_size =
                m_indices[current_m].size() * MeshDataAccessCollection::getByteSize(MeshDataAccessCollection::UNSIGNED_INT);
            mesh_indices.type = MeshDataAccessCollection::UNSIGNED_INT;

            std::string identifier = "mesh" + current_m;
            m_mesh_access_collection.first->addMesh(
                identifier, mesh_attributes, mesh_indices, MeshDataAccessCollection::PrimitiveType::TRIANGLES);
            m_mesh_access_collection.second.push_back(identifier);
        }
        m_meta_data.m_bboxs.SetBoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
        m_meta_data.m_bboxs.SetClipBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);
        m_meta_data.m_frame_cnt = 1;
    }
    if (lhs_mesh_call->version() < m_version) {
        lhs_mesh_call->setMetaData(m_meta_data);
        lhs_mesh_call->setData(m_mesh_access_collection.first, m_version);
    }

    return true;
}

bool megamol::mesh::OmniUsdReader::getMeshMetaDataCallback(core::Call& caller) {
    return AbstractMeshDataSource::getMeshMetaDataCallback(caller);
}

void megamol::mesh::OmniUsdReader::release() {
    m_usd_stage.Reset();
    omniClientShutdown();
}

