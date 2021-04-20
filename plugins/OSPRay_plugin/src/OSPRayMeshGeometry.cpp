/*
 * OSPRayMeshGeometry.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "OSPRayMeshGeometry.h"
#include <functional>
#include "geometry_calls/CallTriMeshData.h"
#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/Call.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/utility/log/Log.h"


using namespace megamol::ospray;


OSPRayMeshGeometry::OSPRayMeshGeometry(void)
        : AbstractOSPRayStructure()
        , _getTrimeshDataSlot("getTrimeshData", "Connects to the data source")
        , _getMeshDataSlot("getMeshData", "Connects to the data source") {

    this->_getTrimeshDataSlot.SetCompatibleCall<geocalls::CallTriMeshDataDescription>();
    this->MakeSlotAvailable(&this->_getTrimeshDataSlot);

    this->_getMeshDataSlot.SetCompatibleCall<mesh::CallMeshDescription>();
    this->MakeSlotAvailable(&this->_getMeshDataSlot);
}


bool OSPRayMeshGeometry::readData(megamol::core::Call& call) {

    // fill material container
    this->processMaterial();

    // fill transformation container
    this->processTransformation();

    //fill clipping plane container
    this->processClippingPlane();

    // read Data, calculate  shape parameters, fill data vectors
    CallOSPRayStructure* os = dynamic_cast<CallOSPRayStructure*>(&call);

    mesh::CallMesh* cm = this->_getMeshDataSlot.CallAs<mesh::CallMesh>();

    auto fcw = writeFlagsSlot.CallAs<core::FlagCallWrite_CPU>();
    auto fcr = readFlagsSlot.CallAs<core::FlagCallRead_CPU>();

    if (cm != nullptr) {
        auto meta_data = cm->getMetaData();
        this->structureContainer.dataChanged = false;
        if (os->getTime() > meta_data.m_frame_cnt) {
            meta_data.m_frame_ID = meta_data.m_frame_cnt - 1;
        } else {
            meta_data.m_frame_ID = os->getTime();
        }
        cm->setMetaData(meta_data);
        if (!(*cm)(1))
            return false;
        if (!(*cm)(0))
            return false;
        meta_data = cm->getMetaData();
        if (cm->hasUpdate() || this->time != os->getTime() || this->InterfaceIsDirty()) {
            this->time = os->getTime();
            this->structureContainer.dataChanged = true;
            this->extendContainer.boundingBox = std::make_shared<megamol::core::BoundingBoxes_2>(meta_data.m_bboxs);
            this->extendContainer.timeFramesCount = meta_data.m_frame_cnt;
            this->extendContainer.isValid = true;
            meshStructure mesh_str;
            mesh_str.mesh = cm->getData();
            this->structureContainer.structure = mesh_str;

            _mesh_prefix_count.clear();
            auto const& meshes = mesh_str.mesh->accessMeshes();
            _mesh_prefix_count.resize(meshes.size());
            auto counter = 0u;
            for (auto const& entry : meshes) {
                auto c_count = 0;
                switch (entry.second.primitive_type) {
                case mesh::MeshDataAccessCollection::PrimitiveType::TRIANGLES: {
                    c_count = 3;
                } break;
                case mesh::MeshDataAccessCollection::PrimitiveType::QUADS: {
                    c_count = 4;
                } break;
                }
                if (c_count == 0) {
                    _mesh_prefix_count[counter] = counter == 0 ? 0 : _mesh_prefix_count[counter - 1];
                    ++counter;
                    break;
                }
                auto const c_bs = mesh::MeshDataAccessCollection::getByteSize(entry.second.indices.type);
                auto const num_el = entry.second.indices.byte_size / (c_bs * c_count);
                _mesh_prefix_count[counter] = counter == 0 ? num_el : _mesh_prefix_count[counter - 1] + num_el;
                ++counter;
            }
            if (fcw != nullptr && fcr != nullptr && !_mesh_prefix_count.empty()) {
                if ((*fcr)(core::FlagCallWrite_CPU::CallGetData)) {
                    auto data = fcr->getData();
                    auto version = fcr->version();
                    data->validateFlagCount(_mesh_prefix_count.back());
                    /*fcw->setData(data, version + 1);
                    (*fcw)(core::FlagCallWrite_CPU::CallGetData);*/
                }
            }
        }
        if (fcw != nullptr && fcr != nullptr && !_mesh_prefix_count.empty()) {
            auto const idx = os->getPickResult();
            if (std::get<0>(idx) != -1 && std::get<0>(idx) < _mesh_prefix_count.size()) {
                if ((*fcr)(core::FlagCallWrite_CPU::CallGetData)) {
                    auto data = fcr->getData();
                    auto version = fcr->version();

                    auto const base_idx = std::get<0>(idx) == 0 ? 0 : _mesh_prefix_count[std::get<0>(idx) - 1];
                    auto const a_idx = base_idx + std::get<1>(idx);
                    /*core::utility::log::Log::DefaultLog.WriteInfo(
                        "[OSPRayMeshGeometry] Got prim id %d, setting id %d", std::get<1>(idx), a_idx);*/

                    if (a_idx < _mesh_prefix_count.back()) {
                        auto const cur_sel = data->flags->operator[](a_idx);
                        data->flags->operator[](a_idx) = cur_sel == core::FlagStorage::ENABLED
                                                             ? core::FlagStorage::SELECTED
                                                             : core::FlagStorage::ENABLED;
                        fcw->setData(data, version + 1);
                        (*fcw)(core::FlagCallWrite_CPU::CallGetData);
                        os->setPickResult(-1, -1);
                    }
                }
            }
        }
    } else {

        geocalls::CallTriMeshData* cd = this->_getTrimeshDataSlot.CallAs<geocalls::CallTriMeshData>();

        this->structureContainer.dataChanged = false;
        if (cd == NULL)
            return false;
        if (os->getTime() > cd->FrameCount()) {
            cd->SetFrameID(cd->FrameCount() - 1, true); // isTimeForced flag set to true
        } else {
            cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
        }
        if (this->datahash != cd->DataHash() || this->time != os->getTime() || this->InterfaceIsDirty()) {
            this->datahash = cd->DataHash();
            this->time = os->getTime();
            this->structureContainer.dataChanged = true;
        } else {
            return true;
        }

        if (!(*cd)(1))
            return false;
        if (!(*cd)(0))
            return false;

        meshStructure ms;
        ms.mesh = std::make_shared<mesh::MeshDataAccessCollection>();



        unsigned int triangleCount = 0;
        unsigned int vertexCount = 0;

        const unsigned int objectCount = cd->Count();


        for (unsigned int i = 0; i < objectCount; i++) {

            std::vector<mesh::MeshDataAccessCollection::VertexAttribute> attrib;
            mesh::MeshDataAccessCollection::IndexData index;


            const geocalls::CallTriMeshData::Mesh& obj = cd->Objects()[i];
            triangleCount = obj.GetTriCount();
            vertexCount = obj.GetVertexCount();


            // check vertex data type
            switch (obj.GetVertexDataType()) {
            case geocalls::CallTriMeshData::Mesh::DT_FLOAT:
                attrib.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute{
                    const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(obj.GetVertexPointerFloat())),
                    3 * vertexCount *
                        mesh::MeshDataAccessCollection::getByteSize(mesh::MeshDataAccessCollection::FLOAT),
                    3, mesh::MeshDataAccessCollection::FLOAT, 0, 0,
                    mesh::MeshDataAccessCollection::AttributeSemanticType::POSITION});
                break;
            // case geocalls::CallTriMeshData::Mesh::DT_DOUBLE:
            default:
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[OSPRayMeshGeometry] Vertex: No other data types than FLOAT are supported.");
                return false;
            }

            // check normal pointer
            if (obj.HasNormalPointer()) {
                switch (obj.GetNormalDataType()) {
                case geocalls::CallTriMeshData::Mesh::DT_FLOAT:
                    attrib.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute{
                        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(obj.GetNormalPointerFloat())),
                        3 * vertexCount *
                            mesh::MeshDataAccessCollection::getByteSize(mesh::MeshDataAccessCollection::FLOAT),
                        3, mesh::MeshDataAccessCollection::FLOAT, sizeof(float) * 3, 0,
                        mesh::MeshDataAccessCollection::AttributeSemanticType::NORMAL});
                    break;
                default:
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[OSPRayMeshGeometry] Normals: No other data types than FLOAT are supported.");
                    return false;
                }
            }

            // check colorpointer and convert to rgba
            if (obj.HasColourPointer()) {
                _color.clear();
                switch (obj.GetColourDataType()) {
                case geocalls::CallTriMeshData::Mesh::DT_BYTE:
                    _color.reserve(vertexCount * 4);
                    for (unsigned int i = 0; i < 3 * obj.GetVertexCount(); i++) {
                        _color.push_back((float) obj.GetColourPointerByte()[i] / 255.0f);
                        if ((i + 1) % 3 == 0) {
                            _color.push_back(1.0f);
                        }
                    }
                    break;
                case geocalls::CallTriMeshData::Mesh::DT_FLOAT:
                    // TODO: not tested
                    _color.reserve(vertexCount * 4);
                    for (unsigned int i = 0; i < 3 * obj.GetVertexCount(); i++) {
                        _color.push_back(obj.GetColourPointerFloat()[i]);
                        if ((i + 1) % 3 == 0) {
                            _color.push_back(1.0f);
                        }
                    }
                    break;
                default:
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[OSPRayMeshGeometry] Color: No other data types than BYTE or FLOAT are supported.");
                    return false;
                }

                attrib.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute{
                    const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(_color.data())),
                    4 * vertexCount *
                        mesh::MeshDataAccessCollection::getByteSize(mesh::MeshDataAccessCollection::FLOAT),
                    4, mesh::MeshDataAccessCollection::FLOAT, 4 * sizeof(float), 0,
                    mesh::MeshDataAccessCollection::AttributeSemanticType::COLOR});
            }


            // check texture array
            if (obj.HasTextureCoordinatePointer()) {
                switch (obj.GetTextureCoordinateDataType()) {
                case geocalls::CallTriMeshData::Mesh::DT_FLOAT:
                    attrib.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute{
                        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(obj.GetTextureCoordinatePointerFloat())),
                        2 * vertexCount *
                            mesh::MeshDataAccessCollection::getByteSize(mesh::MeshDataAccessCollection::FLOAT),
                        2, mesh::MeshDataAccessCollection::FLOAT, 2 * sizeof(float), 0,
                        mesh::MeshDataAccessCollection::AttributeSemanticType::TEXCOORD});
                    break;
                default:
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[OSPRayMeshGeometry] TextureCoordinate: No other data "
                        "types than BYTE or FLOAT are supported.");
                    return false;
                }
            }

            // check index pointer
            if (obj.HasTriIndexPointer()) {
                switch (obj.GetTriDataType()) {
                    // case trisoup::CallTriMeshData::Mesh::DT_BYTE:
                    //    break;
                    // case trisoup::CallTriMeshData::Mesh::DT_UINT16:
                    //    break;
                case geocalls::CallTriMeshData::Mesh::DT_UINT32:
                    index.data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(obj.GetTriIndexPointerUInt32()));
                    index.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
                    index.byte_size = 3 * sizeof(uint32_t) * (triangleCount - 1);
                    break;

                default:
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[OSPRayMeshGeometry] Index: No other data types than BYTE or FLOAT are supported.");
                    return false;
                }
            }
            std::string identifier = std::string(FullName()) + "_object_" + std::to_string(i);
            ms.mesh->addMesh(identifier, attrib, index);

        } // end for

        structureContainer.structure = ms;
    }


    // Write stuff into the structureContainer
    this->structureContainer.type = structureTypeEnum::GEOMETRY;
    this->structureContainer.geometryType = geometryTypeEnum::MESH;
    // this->structureContainer.vertexData = std::make_shared<std::vector<float>>(std::move(vertexD));
    // this->structureContainer.colorData = std::make_shared<std::vector<float>>(std::move(colorD));
    // this->structureContainer.normalData = std::make_shared<std::vector<float>>(std::move(normalD));
    // this->structureContainer.texData = std::make_shared<std::vector<float>>(std::move(texD));
    // this->structureContainer.indexData = std::make_shared<std::vector<uint32_t>>(std::move(indexD));
    // this->structureContainer.vertexCount = vertexCount;
    // this->structureContainer.triangleCount = triangleCount;

    return true;
}


OSPRayMeshGeometry::~OSPRayMeshGeometry() {
    this->Release();
}

bool OSPRayMeshGeometry::create() {
    return true;
}

void OSPRayMeshGeometry::release() {}

/*
ospray::OSPRaySphereGeometry::InterfaceIsDirty()
*/
bool OSPRayMeshGeometry::InterfaceIsDirty() {
    return false;
}


bool OSPRayMeshGeometry::getExtends(megamol::core::Call& call) {
    CallOSPRayStructure* os = dynamic_cast<CallOSPRayStructure*>(&call);

    mesh::CallMesh* cm = this->_getMeshDataSlot.CallAs<mesh::CallMesh>();

    if (cm != nullptr) {

        if (!(*cm)(1))
            return false;
        auto meta_data = cm->getMetaData();
        if (os->getTime() > meta_data.m_frame_cnt) {
            meta_data.m_frame_ID = meta_data.m_frame_cnt - 1;
        } else {
            meta_data.m_frame_ID = os->getTime();
        }
        cm->setMetaData(meta_data);

        this->extendContainer.boundingBox = std::make_shared<megamol::core::BoundingBoxes_2>(meta_data.m_bboxs);
        this->extendContainer.timeFramesCount = meta_data.m_frame_cnt;
        this->extendContainer.isValid = true;

    } else {

        megamol::geocalls::CallTriMeshData* cd = this->_getTrimeshDataSlot.CallAs<megamol::geocalls::CallTriMeshData>();

        if (cd == NULL)
            return false;
        if (os->getTime() > cd->FrameCount()) {
            cd->SetFrameID(cd->FrameCount() - 1, true); // isTimeForced flag set to true
        } else {
            cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
        }

        if (!(*cd)(1))
            return false;

        this->extendContainer.boundingBox = std::make_shared<core::BoundingBoxes_2>();
        this->extendContainer.boundingBox->SetBoundingBox(cd->AccessBoundingBoxes().ObjectSpaceBBox());
        this->extendContainer.timeFramesCount = cd->FrameCount();
        this->extendContainer.isValid = true;
    }
    return true;
}
