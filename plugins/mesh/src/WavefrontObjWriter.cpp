#include <fstream>

#include "WavefrontObjWriter.h"

#include "mmcore/param/FilePathParam.h"

megamol::mesh::WavefrontObjWriter::WavefrontObjWriter()
        : core::AbstractDataWriter()
        , _version(0)
        , _meta_data()
        , _filename_slot("filename", "The name of the obj file to load")
        , _rhs_mesh_slot("getMesh", "") {

    _rhs_mesh_slot.SetCompatibleCall<CallMeshDescription>();
    this->MakeSlotAvailable(&_rhs_mesh_slot);

    this->_filename_slot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->_filename_slot);
}

megamol::mesh::WavefrontObjWriter::~WavefrontObjWriter() {}

bool megamol::mesh::WavefrontObjWriter::create(void) {
    return true;
}

bool megamol::mesh::WavefrontObjWriter::run() {

    auto rhs_mesh_call = _rhs_mesh_slot.CallAs<CallMesh>();

    auto meta_data = rhs_mesh_call->getMetaData();
    meta_data.m_frame_ID = 0;
    rhs_mesh_call->setMetaData(meta_data);

    if (!(*rhs_mesh_call)(1))
        return false;
    if (!(*rhs_mesh_call)(0))
        return false;

    auto mac = rhs_mesh_call->getData();

    std::string fname = _filename_slot.Param<core::param::FilePathParam>()->Value().string();

    auto meshes = mac->accessMeshes();
    auto num_meshes = meshes.size();
    std::vector<ObjMesh> objMeshes(num_meshes);
    auto objIter = objMeshes.begin();
    int affix = 0;
    for (auto& mesh : meshes) {
        // attributes (vertices, normals, texcoords, colors)
        for (auto& attribute : mesh.second.attributes) {
            if (attribute.semantic == MeshDataAccessCollection::AttributeSemanticType::POSITION) {
                auto num_vertices = attribute.byte_size / attribute.stride;
                auto floats = reinterpret_cast<float*>(attribute.data);
                if (objIter->vertices.empty())
                    objIter->vertices.resize(num_vertices);
                for (int i = 0; i < num_vertices; ++i) {
                    objIter->vertices[i].position.x = floats[attribute.component_cnt * i + 0];
                    objIter->vertices[i].position.y = floats[attribute.component_cnt * i + 1];
                    objIter->vertices[i].position.z = floats[attribute.component_cnt * i + 2];
                }
            } else if (attribute.semantic == MeshDataAccessCollection::AttributeSemanticType::NORMAL) {
                auto num_normals = attribute.byte_size / attribute.stride;
                auto floats = reinterpret_cast<float*>(attribute.data);
                if (objIter->vertices.empty())
                    objIter->vertices.resize(num_normals);
                for (int i = 0; i < num_normals; ++i) {
                    objIter->vertices[i].normal.x = floats[attribute.component_cnt * i + 0];
                    objIter->vertices[i].normal.y = floats[attribute.component_cnt * i + 1];
                    objIter->vertices[i].normal.z = floats[attribute.component_cnt * i + 2];
                }
            } else if (attribute.semantic == MeshDataAccessCollection::AttributeSemanticType::TEXCOORD) {
                auto num_texcoords = attribute.byte_size / attribute.stride;
                auto floats = reinterpret_cast<float*>(attribute.data);
                if (objIter->vertices.empty())
                    objIter->vertices.resize(num_texcoords);
                for (int i = 0; i < num_texcoords; ++i) {
                    objIter->vertices[i].tex_coord.x = floats[attribute.component_cnt * i + 0];
                    objIter->vertices[i].tex_coord.y = floats[attribute.component_cnt * i + 1];
                }
            }
        }
        // indices
        auto num_indices =
            mesh.second.indices.byte_size / mesh::MeshDataAccessCollection::getByteSize(mesh.second.indices.type);
        auto uints = reinterpret_cast<unsigned*>(mesh.second.indices.data);
        objIter->indices.reserve(num_indices);
        objIter->indices.insert(objIter->indices.begin(), uints, uints + num_indices);

        if (meshes.size() > 1) {
            WriteMesh(fname + std::to_string(affix), *objIter);
        } else {
            WriteMesh(fname, *objIter);
        }

        objIter = std::next(objIter);
        ++affix;
    }


    return true;
}

bool megamol::mesh::WavefrontObjWriter::getCapabilities(core::DataWriterCtrlCall& call) {
    return true;
}

void megamol::mesh::WavefrontObjWriter::release() {}

void megamol::mesh::WavefrontObjWriter::WriteMesh(const std::string& fname, const ObjMesh& mesh) {
    //code from https://github.com/thinks/obj-io
    const auto vtx_iend = std::end(mesh.vertices);

    // Mappers have two responsibilities:
    // (1) - Iterating over a certain attribute of the mesh (e.g. positions).
    // (2) - Translating from users types to OBJ types (e.g. Vec3 ->
    //       Position<float, 3>)

    // Positions.
    auto pos_vtx_iter = std::begin(mesh.vertices);
    auto pos_mapper = [&pos_vtx_iter, vtx_iend]() {
        using ObjPositionType = thinks::ObjPosition<float, 3>;

        if (pos_vtx_iter == vtx_iend) {
            // End indicates that no further calls should be made to this mapper,
            // in this case because the captured iterator has reached the end
            // of the vector.
            return thinks::ObjEnd<ObjPositionType>();
        }

        // Map indicates that additional positions may be available after this one.
        const auto pos = (*pos_vtx_iter++).position;
        return thinks::ObjMap(ObjPositionType(pos.x, pos.y, pos.z));
    };

    // Faces.
    auto idx_iter = std::begin(mesh.indices);
    const auto idx_iend = std::end(mesh.indices);
    auto face_mapper = [&idx_iter, idx_iend]() {
        using ObjIndexType = thinks::ObjIndex<uint16_t>;
        using ObjFaceType = thinks::ObjTriangleFace<ObjIndexType>;

        // Check that there are 3 more indices (trailing indices handled below).
        if (std::distance(idx_iter, idx_iend) < 3) {
            return thinks::ObjEnd<ObjFaceType>();
        }

        // Create a face from the mesh indices.
        const auto idx0 = ObjIndexType(*idx_iter++);
        const auto idx1 = ObjIndexType(*idx_iter++);
        const auto idx2 = ObjIndexType(*idx_iter++);
        return thinks::ObjMap(ObjFaceType(idx0, idx1, idx2));
    };

    // Texture coordinates [optional].
    auto tex_vtx_iter = std::begin(mesh.vertices);
    auto tex_mapper = [&tex_vtx_iter, vtx_iend]() {
        using ObjTexCoordType = thinks::ObjTexCoord<float, 2>;

        if (tex_vtx_iter == vtx_iend) {
            return thinks::ObjEnd<ObjTexCoordType>();
        }

        const auto tex = (*tex_vtx_iter++).tex_coord;
        return thinks::ObjMap(ObjTexCoordType(tex.x, tex.y));
    };

    // Normals [optional].
    auto nml_vtx_iter = std::begin(mesh.vertices);
    auto nml_mapper = [&nml_vtx_iter, vtx_iend]() {
        using ObjNormalType = thinks::ObjNormal<float>;

        if (nml_vtx_iter == vtx_iend) {
            return thinks::ObjEnd<ObjNormalType>();
        }

        const auto nml = (*nml_vtx_iter++).normal;
        return thinks::ObjMap(ObjNormalType(nml.x, nml.y, nml.z));
    };

    // Open the OBJ file and pass in the mappers, which will be called
    // internally to write the contents of the mesh to the file.
    auto ofs = std::ofstream(fname);
    assert(ofs);
    const auto result = thinks::WriteObj(ofs, pos_mapper, face_mapper, tex_mapper, nml_mapper);
    ofs.close();

    // Some sanity checks.
    assert(result.position_count == mesh.vertices.size() && "bad position count");
    assert(result.tex_coord_count == mesh.vertices.size() && "bad position count");
    assert(result.normal_count == mesh.vertices.size() && "bad normal count");
    assert(result.face_count == mesh.indices.size() / 3 && "bad face count");
    assert(idx_iter == idx_iend && "trailing indices");
}
