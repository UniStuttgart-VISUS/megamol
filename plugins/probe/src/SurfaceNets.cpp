/*
 * SurfaceNets.cpp
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "SurfaceNets.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "geometry_calls/VolumetricDataCall.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"

namespace megamol {
namespace probe {


SurfaceNets::SurfaceNets()
        : Module()
        , _getDataCall("getData", "")
        , _deployMeshCall("deployMesh", "")
        , _deployNormalsCall("deployNormals", "")
        , _isoSlot("IsoValue", "")
        , _faceTypeSlot("FaceType", "") {

    this->_isoSlot << new core::param::FloatParam(1.0f);
    this->_isoSlot.SetUpdateCallback(&SurfaceNets::isoChanged);
    this->MakeSlotAvailable(&this->_isoSlot);

    core::param::EnumParam* ep = new core::param::EnumParam(0);
    ep->SetTypePair(0, "Trianges");
    ep->SetTypePair(1, "Quads");
    this->_faceTypeSlot << ep;
    this->MakeSlotAvailable(&this->_faceTypeSlot);

    this->_deployMeshCall.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(0), &SurfaceNets::getData);
    this->_deployMeshCall.SetCallback(
        mesh::CallMesh::ClassName(), mesh::CallMesh::FunctionName(1), &SurfaceNets::getMetaData);
    this->MakeSlotAvailable(&this->_deployMeshCall);

    this->_deployNormalsCall.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(0), &SurfaceNets::getNormalData);
    this->_deployNormalsCall.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(1), &SurfaceNets::getNormalMetaData);
    this->MakeSlotAvailable(&this->_deployNormalsCall);

    this->_getDataCall.SetCompatibleCall<geocalls::VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->_getDataCall);
}

SurfaceNets::~SurfaceNets() {
    this->Release();
}

bool SurfaceNets::create() {
    return true;
}

void SurfaceNets::release() {}

bool SurfaceNets::InterfaceIsDirty() {
    return this->_isoSlot.IsDirty();
}


void SurfaceNets::calculateSurfaceNets() {

    _bboxs.Clear();

    _vertices.clear();
    _normals.clear();
    _faces.clear();
    _triangles.clear();

    std::array<std::array<uint32_t, 3>, 8> cube_offsets;
    cube_offsets[0] = {0, 0, 0};
    cube_offsets[1] = {1, 0, 0};
    cube_offsets[2] = {0, 0, 1};
    cube_offsets[3] = {1, 0, 1};
    cube_offsets[4] = {0, 1, 0};
    cube_offsets[5] = {1, 1, 0};
    cube_offsets[6] = {0, 1, 1};
    cube_offsets[7] = {1, 1, 1};

    const std::array<uint32_t, 24> edge_vertex_offsets = {// 0
        0, 1,
        // 1
        0, 2,
        // 2
        0, 4,
        // 3
        1, 3,
        // 4
        4, 5,
        // 5
        5, 7,
        // 6
        7, 6,
        // 7
        6, 4,
        // 8
        3, 2,
        // 9
        1, 5,
        // 10
        3, 7,
        // 11
        2, 6};

    float const iso_value = this->_isoSlot.Param<core::param::FloatParam>()->Value();

    std::vector<uint32_t> voxel_lookup(_dims[0] * _dims[1] * _dims[2]);
    std::vector<uint32_t> voxel_filled;
    voxel_filled.reserve(voxel_lookup.size());

    std::vector<uint32_t> voxel_edge_crossings(_dims[0] * _dims[1] * _dims[2]);

    auto dims = _dims;

    auto const offset_now = [dims](
                                uint32_t x, uint32_t y, uint32_t z) { return dims[0] * dims[1] * z + dims[0] * y + x; };

    for (uint32_t z = 0; z < _dims[2] - 1; z++) {
        for (uint32_t y = 0; y < _dims[1] - 1; y++) {
            for (uint32_t x = 0; x < _dims[0] - 1; x++) {

                std::array<float, 8> sample_value;
                sample_value[0] =
                    _data[offset_now(x + cube_offsets[0][0], y + cube_offsets[0][1], z + cube_offsets[0][2])];
                sample_value[1] =
                    _data[offset_now(x + cube_offsets[1][0], y + cube_offsets[1][1], z + cube_offsets[1][2])];
                sample_value[2] =
                    _data[offset_now(x + cube_offsets[2][0], y + cube_offsets[2][1], z + cube_offsets[2][2])];
                sample_value[3] =
                    _data[offset_now(x + cube_offsets[3][0], y + cube_offsets[3][1], z + cube_offsets[3][2])];
                sample_value[4] =
                    _data[offset_now(x + cube_offsets[4][0], y + cube_offsets[4][1], z + cube_offsets[4][2])];
                sample_value[5] =
                    _data[offset_now(x + cube_offsets[5][0], y + cube_offsets[5][1], z + cube_offsets[5][2])];
                sample_value[6] =
                    _data[offset_now(x + cube_offsets[6][0], y + cube_offsets[6][1], z + cube_offsets[6][2])];
                sample_value[7] =
                    _data[offset_now(x + cube_offsets[7][0], y + cube_offsets[7][1], z + cube_offsets[7][2])];

                uint32_t edge_crossings = 0;

                std::array<float, 3> center_of_mass = {0.0f, 0.0f, 0.0f};
                float normalization = 0.0f;

                // Compute edge crossings and center of mass
                for (int i = 0; i < 12; ++i) {
                    uint32_t const idx_0 = edge_vertex_offsets[i * 2 + 0];
                    uint32_t const idx_1 = edge_vertex_offsets[i * 2 + 1];

                    auto const v_0 = sample_value[idx_0];
                    auto const v_1 = sample_value[idx_1];

                    auto edge_crossing = uint32_t(!((v_0 > iso_value) == (v_1 > iso_value)));
                    edge_crossings |= (edge_crossing << i);


                    if (edge_crossing == 1) {

                        float d = ((iso_value - v_0) / (v_1 - v_0));
                        std::array<float, 3> mix;
                        mix[0] = static_cast<float>(cube_offsets[idx_0][0]) * (1.0f - d) +
                                 static_cast<float>(cube_offsets[idx_1][0]) * d;
                        mix[1] = static_cast<float>(cube_offsets[idx_0][1]) * (1.0f - d) +
                                 static_cast<float>(cube_offsets[idx_1][1]) * d;
                        mix[2] = static_cast<float>(cube_offsets[idx_0][2]) * (1.0f - d) +
                                 static_cast<float>(cube_offsets[idx_1][2]) * d;
                        std::array<float, 3> intersect_pos;
                        intersect_pos[0] = static_cast<float>(x) + mix[0];
                        intersect_pos[1] = static_cast<float>(y) + mix[1];
                        intersect_pos[2] = static_cast<float>(z) + mix[2];
                        center_of_mass[0] += intersect_pos[0];
                        center_of_mass[1] += intersect_pos[1];
                        center_of_mass[2] += intersect_pos[2];
                        normalization += 1.0f;
                    }
                } // for i < 12

                voxel_edge_crossings[offset_now(x, y, z)] = edge_crossings;

                if (normalization > 0.0f) {

                    center_of_mass[0] /= normalization;
                    center_of_mass[1] /= normalization;
                    center_of_mass[2] /= normalization;

                    std::array<float, 4> position;
                    position[0] = ((center_of_mass[0] / _dims[0]) * _dims[0] * _spacing[0]) + _volume_origin[0];
                    position[1] = ((center_of_mass[1] / _dims[1]) * _dims[1] * _spacing[1]) + _volume_origin[1];
                    position[2] = ((center_of_mass[2] / _dims[2]) * _dims[2] * _spacing[2]) + _volume_origin[2];
                    position[3] = 1.0f;
                    _vertices.push_back(position);

                    float eps = 0.005;
                    vislib::math::Cuboid<float> point_box(position[0] - eps, position[1] - eps, position[2] - eps,
                        position[0] + eps, position[1] + eps, position[2] + eps);

                    auto bbox = _bboxs.BoundingBox();
                    auto cbox = _bboxs.ClipBox();
                    bbox.Union(point_box);
                    cbox.Union(point_box);
                    _bboxs.SetBoundingBox(bbox);
                    _bboxs.SetClipBox(cbox);

                    voxel_lookup[offset_now(x, y, z)] = _vertices.size() - 1;
                    voxel_filled.push_back(offset_now(x, y, z));

                    std::array<float, 3> normal;
                    normal[0] = _data[offset_now(x >= _dims[0] - 1 ? x : x + 1, y, z)] -
                                _data[offset_now(x < 1 ? x : x - 1, y, z)];
                    normal[1] = _data[offset_now(x, y >= _dims[1] - 1 ? y : y + 1, z)] -
                                _data[offset_now(x, y < 1 ? y : y - 1, z)];
                    normal[2] = _data[offset_now(x, y, z >= _dims[2] - 1 ? z : z + 1)] -
                                _data[offset_now(x, y, z < 1 ? z : z - 1)];
                    if (normal[0] <= 1e-6 && normal[1] <= 1e-6 && normal[2] <= 1e-6) {
                        normal[0] = _data[offset_now(x >= _dims[0] - 2 ? x : x + 2, y, z)] -
                                    _data[offset_now(x < 2 ? x : x - 2, y, z)];
                        normal[1] = _data[offset_now(x, y >= _dims[1] - 2 ? y : y + 2, z)] -
                                    _data[offset_now(x, y < 2 ? y : y - 2, z)];
                        normal[2] = _data[offset_now(x, y, z >= _dims[2] - 2 ? z : z + 2)] -
                                    _data[offset_now(x, y, z < 2 ? z : z - 2)];
                    }
                    auto const normal_length =
                        std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
                    normal[0] /= (normal_length < 0.00000001) ? 1.0 : normal_length;
                    normal[1] /= (normal_length < 0.00000001) ? 1.0 : normal_length;
                    normal[2] /= (normal_length < 0.00000001) ? 1.0 : normal_length;
                    _normals.push_back(normal);
                }
            } // for x
        }     // for y
    }         // for z
    voxel_filled.shrink_to_fit();

    auto const coordinateFromLinearIndex = [](uint32_t idx, uint32_t max_x, uint32_t max_y) {
        std::array<uint32_t, 3> coords;
        coords[0] = idx % (max_x);
        idx /= (max_x);
        coords[1] = idx % (max_y);
        idx /= (max_y);
        coords[2] = idx;
        return coords;
    };


    for (auto id : voxel_filled) {
        for (uint32_t i = 0; i < 3; ++i) {
            auto const edge_crossing = 1 & (voxel_edge_crossings[id] >> i);
            if (edge_crossing == 1) {
                auto coords = coordinateFromLinearIndex(id, _dims[0], _dims[1]);
                if (coords[0] > 0 && coords[1] > 0 && coords[2] > 0) {
                    std::array<uint32_t, 4> indices;
                    std::array<uint32_t, 3> triangle1;
                    std::array<uint32_t, 3> triangle2;
                    if (i == 0) {
                        indices[0] = voxel_lookup[offset_now(coords[0], coords[1] - 1, coords[2])];
                        indices[1] = voxel_lookup[offset_now(coords[0], coords[1] - 1, coords[2] - 1)];
                        indices[2] = voxel_lookup[offset_now(coords[0], coords[1], coords[2] - 1)];
                        indices[3] = voxel_lookup[offset_now(coords[0], coords[1], coords[2])]; // or just id
                    } else if (i == 1) {
                        indices[0] = voxel_lookup[offset_now(coords[0] - 1, coords[1] - 1, coords[2])];
                        indices[1] = voxel_lookup[offset_now(coords[0], coords[1] - 1, coords[2])];
                        indices[2] = voxel_lookup[offset_now(coords[0], coords[1], coords[2])];
                        indices[3] = voxel_lookup[offset_now(coords[0] - 1, coords[1], coords[2])];
                    } else if (i == 2) {
                        indices[0] = voxel_lookup[offset_now(coords[0] - 1, coords[1], coords[2])];
                        indices[1] = voxel_lookup[offset_now(coords[0], coords[1], coords[2])];
                        indices[2] = voxel_lookup[offset_now(coords[0], coords[1], coords[2] - 1)];
                        indices[3] = voxel_lookup[offset_now(coords[0] - 1, coords[1], coords[2] - 1)];
                    }
                    triangle1[0] = indices[0];
                    triangle1[1] = indices[1];
                    triangle1[2] = indices[2];
                    triangle2[0] = indices[0];
                    triangle2[1] = indices[2];
                    triangle2[2] = indices[3];

                    // hack normals
                    auto tangent = _vertices[indices[2]];
                    auto bitangent = _vertices[indices[1]];

                    tangent[0] -= _vertices[indices[0]][0];
                    tangent[1] -= _vertices[indices[0]][1];
                    tangent[2] -= _vertices[indices[0]][2];
                    auto t_length =
                        std::sqrt(tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2]);
                    tangent[0] /= t_length;
                    tangent[1] /= t_length;
                    tangent[2] /= t_length;

                    bitangent[0] -= _vertices[indices[0]][0];
                    bitangent[1] -= _vertices[indices[0]][1];
                    bitangent[2] -= _vertices[indices[0]][2];
                    auto bt_length = std::sqrt(
                        bitangent[0] * bitangent[0] + bitangent[1] * bitangent[1] + bitangent[2] * bitangent[2]);
                    bitangent[0] /= bt_length;
                    bitangent[1] /= bt_length;
                    bitangent[2] /= bt_length;

                    std::array<float, 3> normal;
                    normal[0] = tangent[1] * bitangent[2] - tangent[2] * bitangent[1];
                    normal[1] = tangent[2] * bitangent[0] - tangent[0] * bitangent[2];
                    normal[2] = tangent[0] * bitangent[1] - tangent[1] * bitangent[0];
                    auto n_length = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
                    normal[0] /= n_length;
                    normal[1] /= n_length;
                    normal[2] /= n_length;

                    auto myDot = [](std::array<float, 3> const& v0, std::array<float, 3> const& v1) -> float {
                        return (v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2]);
                    };

                    _normals[indices[0]] = myDot(_normals[indices[0]], normal) > 0.0
                                               ? normal
                                               : std::array<float, 3>{-normal[0], -normal[1], -normal[2]};
                    _normals[indices[1]] = myDot(_normals[indices[1]], normal) > 0.0
                                               ? normal
                                               : std::array<float, 3>{-normal[0], -normal[1], -normal[2]};
                    _normals[indices[2]] = myDot(_normals[indices[2]], normal) > 0.0
                                               ? normal
                                               : std::array<float, 3>{-normal[0], -normal[1], -normal[2]};
                    _normals[indices[3]] = myDot(_normals[indices[3]], normal) > 0.0
                                               ? normal
                                               : std::array<float, 3>{-normal[0], -normal[1], -normal[2]};

                    _faces.emplace_back(indices);
                    _triangles.emplace_back(triangle1);
                    _triangles.emplace_back(triangle2);
                }
            }
        } // for i < 3
    }     // for voxel_filled
}

bool SurfaceNets::getData(core::Call& call) {

    bool something_changed = _recalc;

    auto cm = dynamic_cast<mesh::CallMesh*>(&call);
    if (cm == nullptr)
        return false;

    auto cd = this->_getDataCall.CallAs<geocalls::VolumetricDataCall>();
    if (cd == nullptr)
        return false;

    if (!(*cd)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS))
        return false;
    if (!(*cd)(geocalls::VolumetricDataCall::IDX_GET_METADATA))
        return false;
    if (!(*cd)(geocalls::VolumetricDataCall::IDX_GET_DATA))
        return false;

    // get data from volumetric call
    if (cd->DataHash() != _old_datahash) {
        something_changed = true;
        auto mesh_meta_data = cm->getMetaData();
        //mesh_meta_data.m_bboxs = cd->AccessBoundingBoxes();
        mesh_meta_data.m_frame_cnt = cd->GetAvailableFrames();
        cm->setMetaData(mesh_meta_data);


        auto meta_data = cd->GetMetadata();
        _dims[0] = meta_data->Resolution[0];
        _dims[1] = meta_data->Resolution[1];
        _dims[2] = meta_data->Resolution[2];
        _volume_origin[0] = meta_data->Origin[0];
        _volume_origin[1] = meta_data->Origin[1];
        _volume_origin[2] = meta_data->Origin[2];
        _spacing[0] = meta_data->SliceDists[0][0];
        _spacing[1] = meta_data->SliceDists[1][0];
        _spacing[2] = meta_data->SliceDists[2][0];
        if (cd->GetScalarType() == geocalls::FLOATING_POINT) {
            _data = static_cast<float*>(cd->GetData());
        } else if (cd->GetScalarType() == geocalls::UNSIGNED_INTEGER) {
            _converted_data.reserve(_dims[0] * _dims[1] * _dims[2]);
            auto c_data = static_cast<unsigned char*>(cd->GetData());
            for (uint32_t z = 0; z < _dims[2]; ++z) {
                for (uint32_t y = 0; y < _dims[1]; ++y) {
                    for (uint32_t x = 0; x < _dims[0]; ++x) {
                        uint64_t idx = z * _dims[0] * _dims[1] + y * _dims[0] + x;
                        _converted_data.emplace_back(c_data[idx]);
                    }
                }
            }
            _data = _converted_data.data();
        }
    }

    if (something_changed && _data) {
        this->calculateSurfaceNets();

        _mesh_attribs.resize(2);
        _mesh_attribs[0].component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
        _mesh_attribs[0].byte_size = _vertices.size() * sizeof(std::array<float, 4>);
        _mesh_attribs[0].component_cnt = 4;
        _mesh_attribs[0].stride = sizeof(std::array<float, 4>);
        _mesh_attribs[0].offset = 0;
        _mesh_attribs[0].data = reinterpret_cast<uint8_t*>(_vertices.data());
        _mesh_attribs[0].semantic = mesh::MeshDataAccessCollection::POSITION;

        _mesh_attribs[1].component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
        _mesh_attribs[1].byte_size = _normals.size() * sizeof(std::array<float, 3>);
        _mesh_attribs[1].component_cnt = 3;
        _mesh_attribs[1].stride = sizeof(std::array<float, 3>);
        _mesh_attribs[1].offset = 0;
        _mesh_attribs[1].data = reinterpret_cast<uint8_t*>(_normals.data());
        _mesh_attribs[1].semantic = mesh::MeshDataAccessCollection::NORMAL;

        if (this->_faceTypeSlot.Param<core::param::EnumParam>()->Value() == 0) {
            _mesh_indices.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
            _mesh_indices.byte_size = _triangles.size() * sizeof(std::array<uint32_t, 3>);
            _mesh_indices.data = reinterpret_cast<uint8_t*>(_triangles.data());
            _mesh_type = mesh::MeshDataAccessCollection::PrimitiveType::TRIANGLES;
        } else {
            _mesh_indices.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
            _mesh_indices.byte_size = _faces.size() * sizeof(std::array<uint32_t, 4>);
            _mesh_indices.data = reinterpret_cast<uint8_t*>(_faces.data());
            _mesh_type = mesh::MeshDataAccessCollection::PrimitiveType::QUADS;
        }
        ++_version;
    }

    // put data in mesh
    mesh::MeshDataAccessCollection mesh;

    std::string identifier = std::string(FullName()) + "_mesh";
    mesh.addMesh(identifier, _mesh_attribs, _mesh_indices, _mesh_type);
    cm->setData(std::make_shared<mesh::MeshDataAccessCollection>(std::move(mesh)), _version);
    _old_datahash = cd->DataHash();
    _recalc = false;

    auto meta_data = cm->getMetaData();
    meta_data.m_bboxs = _bboxs;
    cm->setMetaData(meta_data);

    return true;
}

bool SurfaceNets::getMetaData(core::Call& call) {

    auto cm = dynamic_cast<mesh::CallMesh*>(&call);
    if (cm == nullptr)
        return false;

    auto cd = this->_getDataCall.CallAs<geocalls::VolumetricDataCall>();
    if (cd == nullptr)
        return false;

    auto meta_data = cm->getMetaData();

    // get metadata for volumetric call
    cd->SetFrameID(meta_data.m_frame_ID);
    if (!(*cd)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS))
        return false;
    if (!(*cd)(geocalls::VolumetricDataCall::IDX_GET_METADATA))
        return false;

    //if (cd->DataHash() == _old_datahash && !_recalc)
    //    return true;

    // put metadata in mesh call
    //meta_data.m_bboxs = cd->AccessBoundingBoxes();
    meta_data.m_bboxs = _bboxs;
    meta_data.m_frame_cnt = cd->GetAvailableFrames();
    cm->setMetaData(meta_data);

    return true;
}

bool SurfaceNets::isoChanged(core::param::ParamSlot& p) {

    _recalc = true;
    _vertices.clear();
    _normals.clear();
    _faces.clear();
    _triangles.clear();

    return true;
}


bool SurfaceNets::getNormalData(core::Call& call) {
    bool something_changed = _recalc;

    auto mpd = dynamic_cast<geocalls::MultiParticleDataCall*>(&call);
    if (mpd == nullptr)
        return false;

    auto cd = this->_getDataCall.CallAs<geocalls::VolumetricDataCall>();
    if (cd == nullptr)
        return false;

    if (!(*cd)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS))
        return false;
    if (!(*cd)(geocalls::VolumetricDataCall::IDX_GET_METADATA))
        return false;
    if (!(*cd)(geocalls::VolumetricDataCall::IDX_GET_DATA))
        return false;

    if (cd->DataHash() != _old_datahash) {
        something_changed = true;
    }

    _dims[0] = cd->GetResolution(0);
    _dims[1] = cd->GetResolution(1);
    _dims[2] = cd->GetResolution(2);
    auto meta_data = cd->GetMetadata();
    _volume_origin[0] = meta_data->Origin[0];
    _volume_origin[1] = meta_data->Origin[1];
    _volume_origin[2] = meta_data->Origin[2];
    _spacing[0] = meta_data->SliceDists[0][0];
    _spacing[1] = meta_data->SliceDists[1][0];
    _spacing[2] = meta_data->SliceDists[2][0];
    if (cd->GetScalarType() != geocalls::FLOATING_POINT)
        return false;
    _data = reinterpret_cast<float*>(cd->GetData());

    if (something_changed || _recalc) {
        this->calculateSurfaceNets();
    }

    mpd->SetParticleListCount(1);
    mpd->AccessParticles(0).SetGlobalRadius(cd->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge() * 1e-3);
    mpd->AccessParticles(0).SetCount(_vertices.size());
    mpd->AccessParticles(0).SetVertexData(
        geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ, _vertices.data(), sizeof(std::array<float, 4>));
    mpd->AccessParticles(0).SetDirData(
        geocalls::MultiParticleDataCall::Particles::DIRDATA_FLOAT_XYZ, _normals.data(), sizeof(std::array<float, 3>));

    _old_datahash = cd->DataHash();
    _recalc = false;

    return true;
}

bool SurfaceNets::getNormalMetaData(core::Call& call) {
    auto mpd = dynamic_cast<geocalls::MultiParticleDataCall*>(&call);
    if (mpd == nullptr)
        return false;

    auto cd = this->_getDataCall.CallAs<geocalls::VolumetricDataCall>();
    if (cd == nullptr)
        return false;

    if (cd->DataHash() == _old_datahash && !_recalc)
        return true;

    // get metadata from adios
    if (!(*cd)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS))
        return false;

    mpd->AccessBoundingBoxes() = cd->AccessBoundingBoxes();
    mpd->SetFrameCount(cd->FrameCount());
    cd->SetFrameID(mpd->FrameID());
    mpd->SetDataHash(mpd->DataHash() + 1);


    return true;
}


} // namespace probe
} // namespace megamol
