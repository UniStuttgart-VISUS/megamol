/*
 * OSPRayMeshGeometry.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "OSPRayMeshGeometry.h"
#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/Call.h"


using namespace megamol::ospray;


OSPRayMeshGeometry::OSPRayMeshGeometry(void)
        : AbstractOSPRayStructure()
        , _getMeshDataSlot("getMeshData", "Connects to the data source") {

    this->_getMeshDataSlot.SetCompatibleCall<mesh::CallMeshDescription>();
    this->MakeSlotAvailable(&this->_getMeshDataSlot);
    this->_getMeshDataSlot.SetNecessity(megamol::core::AbstractCallSlotPresentation::SLOT_REQUIRED);
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
        auto interface_dirtyness = this->InterfaceIsDirty();
        if (cm->hasUpdate() || this->time != os->getTime() || interface_dirtyness) {
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
            std::vector<size_t> tmp_prefix_count(meshes.size());
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
                    tmp_prefix_count[counter] = counter == 0 ? 0 : tmp_prefix_count[counter - 1];
                    ++counter;
                    break;
                }
                auto const c_bs = mesh::MeshDataAccessCollection::getByteSize(entry.second.indices.type);
                auto const num_el = entry.second.indices.byte_size / (c_bs * c_count);
                tmp_prefix_count[counter] = counter == 0 ? num_el : tmp_prefix_count[counter - 1] + num_el;
                ++counter;
            }

            _mesh_prefix_count.insert(_mesh_prefix_count.end(), tmp_prefix_count.begin(), tmp_prefix_count.end());

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
                        data->flags->operator[](a_idx) =
                            cur_sel == core::FlagStorageTypes::to_integral(core::FlagStorageTypes::flag_bits::ENABLED)
                                ? core::FlagStorageTypes::to_integral(core::FlagStorageTypes::flag_bits::ENABLED |
                                                                      core::FlagStorageTypes::flag_bits::SELECTED)
                                : core::FlagStorageTypes::to_integral(core::FlagStorageTypes::flag_bits::ENABLED);
                        fcw->setData(data, version + 1);
                        (*fcw)(core::FlagCallWrite_CPU::CallGetData);
                        os->setPickResult(-1, -1);
                    }
                }
            }
        }
    }

    // Write stuff into the structureContainer
    this->structureContainer.type = structureTypeEnum::GEOMETRY;
    this->structureContainer.geometryType = geometryTypeEnum::MESH;

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
    }
    return true;
}
