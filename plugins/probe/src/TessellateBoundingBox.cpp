/*
 * TessellateBoundingBox.h
 * Copyright (C) 2020 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "glm/glm.hpp"

#include "TessellateBoundingBox.h"
#include "mesh/Utility.h"
#include "mmadios/CallADIOSData.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

megamol::probe::TessellateBoundingBox::TessellateBoundingBox(void)
        : AbstractMeshDataSource()
        , _bounding_box_rhs_slot("adios", "Connect to rendering call to access bounding box")
        , _x_subdiv_slot("x_subdiv_param", "")
        , _y_subdiv_slot("y_subdiv_param", "")
        , _z_subdiv_slot("z_subdiv_param", "")
        , _padding_slot("padding_param", "") {

    this->_bounding_box_rhs_slot.SetCompatibleCall<adios::CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->_bounding_box_rhs_slot);

    this->_x_subdiv_slot << new core::param::IntParam(16);
    this->MakeSlotAvailable(&this->_x_subdiv_slot);
    this->_y_subdiv_slot << new core::param::IntParam(16);
    this->MakeSlotAvailable(&this->_y_subdiv_slot);
    this->_z_subdiv_slot << new core::param::IntParam(16);
    this->MakeSlotAvailable(&this->_z_subdiv_slot);

    this->_padding_slot << new core::param::FloatParam(0.01f);
    this->MakeSlotAvailable(&this->_padding_slot);
}

megamol::probe::TessellateBoundingBox::~TessellateBoundingBox(void) {
    this->Release();
}

bool megamol::probe::TessellateBoundingBox::create() {
    AbstractMeshDataSource::create();
    return true;
}

void megamol::probe::TessellateBoundingBox::release() {}

bool megamol::probe::TessellateBoundingBox::InterfaceIsDirty() {
    return _x_subdiv_slot.IsDirty() || _y_subdiv_slot.IsDirty() || _z_subdiv_slot.IsDirty() || _padding_slot.IsDirty();
}

bool megamol::probe::TessellateBoundingBox::getMeshMetaDataCallback(core::Call& call) {

    auto cm = dynamic_cast<mesh::CallMesh*>(&call);
    if (cm == nullptr) {
        return false;
    }
    auto meta_data = cm->getMetaData();

    adios::CallADIOSData* bboxc = this->_bounding_box_rhs_slot.CallAs<adios::CallADIOSData>();
    if (bboxc != nullptr) {
        bboxc->setFrameIDtoLoad(meta_data.m_frame_ID);
        if (!(*bboxc)(1)) {
            return false;
        }
        meta_data.m_frame_cnt = bboxc->getFrameCount();
    }

    meta_data.m_bboxs = _bboxs;
    cm->setMetaData(meta_data);

    return true;
}

bool megamol::probe::TessellateBoundingBox::getMeshDataCallback(core::Call& call) {

    mesh::CallMesh* lhs_mesh_call = dynamic_cast<mesh::CallMesh*>(&call);
    mesh::CallMesh* rhs_mesh_call = m_mesh_rhs_slot.CallAs<mesh::CallMesh>();

    if (lhs_mesh_call == NULL) {
        return false;
    }

    syncMeshAccessCollection(lhs_mesh_call, rhs_mesh_call);

    // if there is a mesh connection to the right, pass on the mesh collection
    if (rhs_mesh_call != NULL) {
        if (!(*rhs_mesh_call)(0)) {
            return false;
        }
        if (rhs_mesh_call->hasUpdate()) {
            ++_version;
            rhs_mesh_call->getData();
        }
    }

    adios::CallADIOSData* bboxc = this->_bounding_box_rhs_slot.CallAs<adios::CallADIOSData>();
    if (bboxc != nullptr) {

        auto something_has_changed = (bboxc->getDataHash() != _old_datahash || InterfaceIsDirty());

        if (something_has_changed) {
            ++_version;
            _old_datahash = bboxc->getDataHash();

            _x_subdiv_slot.ResetDirty();
            _y_subdiv_slot.ResetDirty();
            _z_subdiv_slot.ResetDirty();
            _padding_slot.ResetDirty();

            auto meta_data = lhs_mesh_call->getMetaData();
            //bboxc->setFrameIDtoLoad(meta_data.m_frame_ID);
            //if (!(*bboxc)(1)) {
            //    return false;
            //}

            auto x_var_str = "x";
            auto y_var_str = "y";
            auto z_var_str = "z";

            std::vector<std::string> toInq;
            toInq.emplace_back(x_var_str);
            toInq.emplace_back(y_var_str);
            toInq.emplace_back(z_var_str);

            // get data from adios
            for (auto var : toInq) {
                if (!bboxc->inquireVar(var)) {
                    return false;
                }
            }

            if (!(*bboxc)(0)) {
                return false;
            }

            // clear mesh data
            for (int i = 0; i < 6; ++i) {
                _vertices[i].clear();
                _normals[i].clear();
                _faces[i].clear();
                _probe_index[i].clear();
            }

            clearMeshAccessCollection();

            // compute mesh call specific update
            std::array<float, 6> bbox;
            bbox[0] = std::numeric_limits<float>::max(); // min x
            bbox[1] = std::numeric_limits<float>::max(); // min y
            bbox[2] = std::numeric_limits<float>::max(); // min z
            bbox[3] = std::numeric_limits<float>::min(); // max x
            bbox[4] = std::numeric_limits<float>::min(); // max y
            bbox[5] = std::numeric_limits<float>::min(); // max z

            if (bboxc->getData(x_var_str)->getType() == "double" && bboxc->getData(y_var_str)->getType() == "double" &&
                bboxc->getData(z_var_str)->getType() == "double") {
                std::vector<double> data_x = bboxc->getData(x_var_str)->GetAsDouble();
                std::vector<double> data_y = bboxc->getData(y_var_str)->GetAsDouble();
                std::vector<double> data_z = bboxc->getData(z_var_str)->GetAsDouble();

                for (size_t i = 0; i < data_x.size(); ++i) {
                    bbox[0] = std::min(static_cast<float>(data_x[i]), bbox[0]);
                    bbox[1] = std::min(static_cast<float>(data_y[i]), bbox[1]);
                    bbox[2] = std::min(static_cast<float>(data_z[i]), bbox[2]);

                    bbox[3] = std::max(static_cast<float>(data_x[i]), bbox[3]);
                    bbox[4] = std::max(static_cast<float>(data_y[i]), bbox[4]);
                    bbox[5] = std::max(static_cast<float>(data_z[i]), bbox[5]);
                }

            } else if (bboxc->getData(x_var_str)->getType() == "float" &&
                       bboxc->getData(y_var_str)->getType() == "float" &&
                       bboxc->getData(z_var_str)->getType() == "float") {
                std::vector<float> data_x = bboxc->getData(x_var_str)->GetAsFloat();
                std::vector<float> data_y = bboxc->getData(y_var_str)->GetAsFloat();
                std::vector<float> data_z = bboxc->getData(z_var_str)->GetAsFloat();

                for (size_t i = 0; i < data_x.size(); ++i) {
                    bbox[0] = std::min(data_x[i], bbox[0]);
                    bbox[1] = std::min(data_y[i], bbox[1]);
                    bbox[2] = std::min(data_z[i], bbox[2]);

                    bbox[3] = std::max(data_x[i], bbox[3]);
                    bbox[4] = std::max(data_y[i], bbox[4]);
                    bbox[5] = std::max(data_z[i], bbox[5]);
                }
            }

            // apply additional padding to bounding box
            bbox[0] -= _padding_slot.Param<megamol::core::param::FloatParam>()->Value();
            bbox[1] -= _padding_slot.Param<megamol::core::param::FloatParam>()->Value();
            bbox[2] -= _padding_slot.Param<megamol::core::param::FloatParam>()->Value();

            bbox[3] += _padding_slot.Param<megamol::core::param::FloatParam>()->Value();
            bbox[4] += _padding_slot.Param<megamol::core::param::FloatParam>()->Value();
            bbox[5] += _padding_slot.Param<megamol::core::param::FloatParam>()->Value();

            _bboxs.SetBoundingBox(bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);

            // get bounding box corners
            glm::vec3 lbb = glm::vec3(bbox[0], bbox[1], bbox[2]);
            glm::vec3 rbb = glm::vec3(bbox[3], bbox[1], bbox[2]);
            glm::vec3 rbf = glm::vec3(bbox[3], bbox[1], bbox[5]);
            glm::vec3 lbf = glm::vec3(bbox[0], bbox[1], bbox[5]);
            glm::vec3 ltb = glm::vec3(bbox[0], bbox[4], bbox[2]);
            glm::vec3 rtb = glm::vec3(bbox[3], bbox[4], bbox[2]);
            glm::vec3 rtf = glm::vec3(bbox[3], bbox[4], bbox[5]);
            glm::vec3 ltf = glm::vec3(bbox[0], bbox[4], bbox[5]);

            // tessellate each face
            using VertexPositions = std::vector<std::array<float, 3>>;
            using VertexNormals = std::vector<std::array<float, 3>>;
            using QuadIndices = std::vector<std::array<uint32_t, 4>>;

            auto x_subdivs = _x_subdiv_slot.Param<megamol::core::param::IntParam>()->Value();
            auto y_subdivs = _y_subdiv_slot.Param<megamol::core::param::IntParam>()->Value();
            auto z_subdivs = _z_subdiv_slot.Param<megamol::core::param::IntParam>()->Value();

            std::tuple<VertexPositions, VertexNormals, QuadIndices> face_xy_back =
                mesh::utility::tessellateFace(lbb, rbb, rtb, ltb, x_subdivs, y_subdivs);
            std::tuple<VertexPositions, VertexNormals, QuadIndices> face_xy_front =
                mesh::utility::tessellateFace(rbf, lbf, ltf, rtf, x_subdivs, y_subdivs);
            std::tuple<VertexPositions, VertexNormals, QuadIndices> face_zy_right =
                mesh::utility::tessellateFace(rbb, rbf, rtf, rtb, z_subdivs, y_subdivs);
            std::tuple<VertexPositions, VertexNormals, QuadIndices> face_zy_left =
                mesh::utility::tessellateFace(lbf, lbb, ltb, ltf, z_subdivs, y_subdivs);
            std::tuple<VertexPositions, VertexNormals, QuadIndices> face_xz_top =
                mesh::utility::tessellateFace(ltf, ltb, rtb, rtf, z_subdivs, x_subdivs);
            std::tuple<VertexPositions, VertexNormals, QuadIndices> face_xz_bottom =
                mesh::utility::tessellateFace(lbb, lbf, rbf, rbb, z_subdivs, x_subdivs);

            // copy to persistent storage
            _vertices[0] = std::get<0>(face_xy_back);
            _normals[0] = std::get<1>(face_xy_back);
            _faces[0] = std::get<2>(face_xy_back);
            _vertices[1] = std::get<0>(face_xy_front);
            _normals[1] = std::get<1>(face_xy_front);
            _faces[1] = std::get<2>(face_xy_front);
            _vertices[2] = std::get<0>(face_zy_right);
            _normals[2] = std::get<1>(face_zy_right);
            _faces[2] = std::get<2>(face_zy_right);
            _vertices[3] = std::get<0>(face_zy_left);
            _normals[3] = std::get<1>(face_zy_left);
            _faces[3] = std::get<2>(face_zy_left);
            _vertices[4] = std::get<0>(face_xz_top);
            _normals[4] = std::get<1>(face_xz_top);
            _faces[4] = std::get<2>(face_xz_top);
            _vertices[5] = std::get<0>(face_xz_bottom);
            _normals[5] = std::get<1>(face_xz_bottom);
            _faces[5] = std::get<2>(face_xz_bottom);

            // build accessor for this modules mesh data
            for (int i = 0; i < 6; ++i) {
                std::vector<mesh::MeshDataAccessCollection::VertexAttribute> mesh_attribs;
                mesh::MeshDataAccessCollection::IndexData mesh_indices;
                mesh::MeshDataAccessCollection::PrimitiveType mesh_type;

                mesh_attribs.resize(3);
                mesh_attribs[0].component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
                mesh_attribs[0].byte_size = _vertices[i].size() * sizeof(std::array<float, 3>);
                mesh_attribs[0].component_cnt = 3;
                mesh_attribs[0].stride = sizeof(std::array<float, 3>);
                mesh_attribs[0].offset = 0;
                mesh_attribs[0].data = reinterpret_cast<uint8_t*>(_vertices[i].data());
                mesh_attribs[0].semantic = mesh::MeshDataAccessCollection::POSITION;

                mesh_attribs[1].component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
                mesh_attribs[1].byte_size = _normals[i].size() * sizeof(std::array<float, 3>);
                mesh_attribs[1].component_cnt = 3;
                mesh_attribs[1].stride = sizeof(std::array<float, 3>);
                mesh_attribs[1].offset = 0;
                mesh_attribs[1].data = reinterpret_cast<uint8_t*>(_normals[i].data());
                mesh_attribs[1].semantic = mesh::MeshDataAccessCollection::NORMAL;

                _probe_index[i].resize(_vertices[i].size(),
                    std::numeric_limits<int>::
                        max()); // allocate memory for probe ID now, but the acutal data will be written later (by probe placement)
                mesh_attribs[2].component_type = mesh::MeshDataAccessCollection::ValueType::INT;
                mesh_attribs[2].byte_size = _probe_index[i].size() * sizeof(int);
                mesh_attribs[2].component_cnt = 1;
                mesh_attribs[2].stride = sizeof(int);
                mesh_attribs[2].offset = 0;
                mesh_attribs[2].data = reinterpret_cast<uint8_t*>(_probe_index[i].data());
                mesh_attribs[2].semantic = mesh::MeshDataAccessCollection::AttributeSemanticType::ID;

                mesh_indices.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
                mesh_indices.byte_size = _faces[i].size() * sizeof(std::array<uint32_t, 4>);
                mesh_indices.data = reinterpret_cast<uint8_t*>(_faces[i].data());
                mesh_type = mesh::MeshDataAccessCollection::PrimitiveType::QUADS;

                std::string identifier = std::string(FullName()) + "_mesh_" + std::to_string(i);
                m_mesh_access_collection.first->addMesh(identifier, mesh_attribs, mesh_indices, mesh_type);
                m_mesh_access_collection.second.push_back(identifier);
            }

            meta_data.m_bboxs = _bboxs;
            lhs_mesh_call->setMetaData(meta_data);
        }
    }

    lhs_mesh_call->setData(m_mesh_access_collection.first, _version);

    return true;
}
