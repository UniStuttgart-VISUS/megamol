/*
 * OSPRaySphereGeometry.cpp
 * Copyright (C) 2021 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "OSPRaySphereGeometry.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/Call.h"
#include "mmcore/utility/log/Log.h"

using namespace megamol::ospray;


OSPRaySphereGeometry::OSPRaySphereGeometry(void)
        : AbstractOSPRayStructure()
        , getDataSlot("getdata", "Connects to the data source") {

    this->getDataSlot.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);
}


bool OSPRaySphereGeometry::readData(megamol::core::Call& call) {

    // fill material container
    this->processMaterial();

    // get transformation parameter
    this->processTransformation();

    // fill clipping plane container
    this->processClippingPlane();

    // read Data, calculate  shape parameters, fill data vectors
    CallOSPRayStructure* os = dynamic_cast<CallOSPRayStructure*>(&call);
    megamol::geocalls::MultiParticleDataCall* cd = this->getDataSlot.CallAs<megamol::geocalls::MultiParticleDataCall>();

    // check flagstorage
    auto fcw = writeFlagsSlot.CallAs<core::FlagCallWrite_CPU>();
    auto fcr = readFlagsSlot.CallAs<core::FlagCallRead_CPU>();

    bool flags_dirty = false;
    size_t flags_size = 0;
    if (fcw != nullptr && fcr != nullptr) {
        if ((*fcr)(core::FlagCallWrite_CPU::CallGetData)) {
            flags_dirty = fcr->hasUpdate();
        }
        flags_size = fcr->getData()->flags->size();
    }

    this->structureContainer.dataChanged = false;
    if (cd == NULL)
        return false;
    cd->SetTimeStamp(os->getTime());
    cd->SetFrameID(os->getTime(), true);
    if (!(*cd)(1))
        return false;
    if (!(*cd)(0))
        return false;

    auto interface_dirty = this->InterfaceIsDirty();
    if (this->datahash != cd->DataHash() || this->time != os->getTime() || interface_dirty || flags_dirty) {
        this->datahash = cd->DataHash();
        this->time = os->getTime();
        this->structureContainer.dataChanged = true;
    } else {
        return true;
    }

    if (cd->GetParticleListCount() == 0)
        return false;

    sphereStructure ss;
    ss.spheres = std::make_shared<ParticleDataAccessCollection>();

    const auto plist_count = cd->GetParticleListCount();
    int tot_part_count = 0;
    std::vector<int> offset_list(plist_count);
    for (int i = 0; i < plist_count; ++i) {
        offset_list[i] = tot_part_count;
        tot_part_count += cd->AccessParticles(i).GetCount();
    }
    _enabled_vertices.resize(plist_count);
    _enabled_colors.resize(plist_count);
    _selected_vertices.resize(plist_count);
    for (int i = 0; i < plist_count; ++i) {
        geocalls::MultiParticleDataCall::Particles& parts = cd->AccessParticles(i);

        std::vector<ParticleDataAccessCollection::VertexAttribute> attrib;

        unsigned int partCount = parts.GetCount();
        if (partCount == 0)
            continue;

        int vertex_stride = 3;
        size_t vertex_byte_stride = parts.GetVertexDataStride();
        vertex_byte_stride = vertex_byte_stride == 0
                                 ? geocalls::MultiParticleDataCall::Particles::VertexDataSize[parts.GetVertexDataType()]
                                 : vertex_byte_stride;

        size_t color_byte_stride = parts.GetColourDataStride();
        color_byte_stride = color_byte_stride == 0
                                ? geocalls::MultiParticleDataCall::Particles::ColorDataSize[parts.GetColourDataType()]
                                : color_byte_stride;

        // process flags
        if (flags_dirty) {
            fcr->getData()->validateFlagCount(tot_part_count);
            _enabled_vertices[i].clear();
            _enabled_colors[i].clear();
            _selected_vertices[i].clear();
            _enabled_colors[i].reserve(partCount);
            _enabled_vertices[i].reserve(partCount);
            _selected_vertices[i].reserve(partCount);
            vertex_byte_stride = 3 * sizeof(float);
            color_byte_stride = 4 * sizeof(float);

            for (int j = 0; j < partCount; j++) {
                if (fcr->getData()->flags->operator[](j + offset_list[i]) ==
                        core::FlagStorageTypes::to_integral(core::FlagStorageTypes::flag_bits::ENABLED)) {
                    const std::array<float, 3> current_pos = {parts.GetParticleStore().GetXAcc()->Get_f(j),
                        parts.GetParticleStore().GetYAcc()->Get_f(j),
                        parts.GetParticleStore().GetZAcc()->Get_f(j)};
                    _enabled_vertices[i].emplace_back(current_pos);
                    if (parts.GetColourDataType() == geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA) {
                        const std::array<float, 4> current_col = {parts.GetParticleStore().GetCRAcc()->Get_f(j),
                            parts.GetParticleStore().GetCGAcc()->Get_f(j),
                            parts.GetParticleStore().GetCBAcc()->Get_f(j),
                            parts.GetParticleStore().GetCAAcc()->Get_f(j)};
                        _enabled_colors[i].emplace_back(current_col);
                    }
                }
                if (fcr->getData()->flags->operator[](j + offset_list[i]) &
                    core::FlagStorageTypes::to_integral(core::FlagStorageTypes::flag_bits::SELECTED)) {
                    const std::array<float, 3> current_pos = {parts.GetParticleStore().GetXAcc()->Get_f(j),
                        parts.GetParticleStore().GetYAcc()->Get_f(j), parts.GetParticleStore().GetZAcc()->Get_f(j)};
                    _selected_vertices[i].emplace_back(current_pos);
                }
            }

            // process selected
            if (!_selected_vertices[i].empty()) {
                std::vector<ParticleDataAccessCollection::VertexAttribute> sel_attrib;
                sel_attrib.emplace_back(ParticleDataAccessCollection::VertexAttribute{
                    reinterpret_cast<const uint8_t*>(_selected_vertices[i][0].data()),
                    vertex_byte_stride * _selected_vertices[i].size(), 3, ParticleDataAccessCollection::FLOAT,
                    vertex_byte_stride, 0, ParticleDataAccessCollection::AttributeSemanticType::POSITION});

                std::string sel_identifier = std::string(FullName()) + "_spheres_selected_" + std::to_string(i);
                ss.spheres->addSphereCollection(
                    sel_identifier, sel_attrib, parts.GetGlobalRadius(), {0.8*255.0f, 0.0f, 0.0f, 255.0f});
            }

            // process enabled
            if (!_enabled_vertices[i].empty()) {
                attrib.emplace_back(ParticleDataAccessCollection::VertexAttribute{
                    reinterpret_cast<const uint8_t*>(_enabled_vertices[i][0].data()),
                    vertex_byte_stride * _enabled_vertices[i].size(), 3, ParticleDataAccessCollection::FLOAT,
                    vertex_byte_stride, 0, ParticleDataAccessCollection::AttributeSemanticType::POSITION});
            }

            if (!_enabled_colors[i].empty()) {
                attrib.emplace_back(ParticleDataAccessCollection::VertexAttribute{
                    reinterpret_cast<const uint8_t*>(_enabled_colors[i][0].data()),
                    color_byte_stride * _enabled_colors[i].size(), 4, ParticleDataAccessCollection::FLOAT,
                    color_byte_stride,
                    vertex_stride * ParticleDataAccessCollection::getByteSize(ParticleDataAccessCollection::FLOAT),
                    ParticleDataAccessCollection::AttributeSemanticType::COLOR});
            }


        } else {
            // Vertex data type check
            if (parts.GetVertexDataType() == geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ) {
                auto va = ParticleDataAccessCollection::VertexAttribute();
                va.data = static_cast<const uint8_t*>(parts.GetVertexData());
                va.byte_size = vertex_byte_stride * partCount;
                va.component_cnt = 3;
                va.component_type = ParticleDataAccessCollection::FLOAT;
                va.stride = vertex_byte_stride;
                va.offset = 0;
                va.semantic = ParticleDataAccessCollection::AttributeSemanticType::POSITION;
                attrib.emplace_back(va);
            } else if (parts.GetVertexDataType() == geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR) {
                attrib.emplace_back(
                    ParticleDataAccessCollection::VertexAttribute{static_cast<const uint8_t*>(parts.GetVertexData()),
                        vertex_byte_stride * partCount, 3, ParticleDataAccessCollection::FLOAT, vertex_byte_stride, 0,
                        ParticleDataAccessCollection::AttributeSemanticType::POSITION});

                attrib.emplace_back(
                    ParticleDataAccessCollection::VertexAttribute{static_cast<const uint8_t*>(parts.GetVertexData()),
                        vertex_byte_stride * partCount, 1, ParticleDataAccessCollection::FLOAT, vertex_byte_stride,
                        3 * ParticleDataAccessCollection::getByteSize(ParticleDataAccessCollection::FLOAT),
                        ParticleDataAccessCollection::AttributeSemanticType::RADIUS});
                vertex_stride = 4;
            }

            // Color data type check
            if (parts.GetColourDataType() == geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA) {
                if (parts.GetColourData() == parts.GetVertexData()) {
                    attrib.emplace_back(
                        ParticleDataAccessCollection::VertexAttribute{static_cast<const uint8_t*>(parts.GetColourData()),
                            color_byte_stride * partCount, 4, ParticleDataAccessCollection::FLOAT, color_byte_stride,
                            vertex_stride * ParticleDataAccessCollection::getByteSize(ParticleDataAccessCollection::FLOAT),
                            ParticleDataAccessCollection::AttributeSemanticType::COLOR});
                } else {
                    attrib.emplace_back(
                        ParticleDataAccessCollection::VertexAttribute{static_cast<const uint8_t*>(parts.GetColourData()),
                            color_byte_stride * partCount, 4, ParticleDataAccessCollection::FLOAT, color_byte_stride, 0,
                            ParticleDataAccessCollection::AttributeSemanticType::COLOR});
                }
            }
        }

        std::string identifier = std::string(FullName()) + "_spheres_" + std::to_string(i);
        auto g_col = parts.GetGlobalColour();
        std::array<float, 4> global_color = {
            g_col[0] / 255.0f, g_col[1] / 255.0f, g_col[2] / 255.0f, g_col[3] / 255.0f};
        ss.spheres->addSphereCollection(identifier, attrib, parts.GetGlobalRadius(), global_color);
    }
    // Write stuff into the structureContainer
    this->structureContainer.type = structureTypeEnum::GEOMETRY;
    this->structureContainer.geometryType = geometryTypeEnum::SPHERES;

    this->structureContainer.structure = ss;

    return true;
}


OSPRaySphereGeometry::~OSPRaySphereGeometry() {
    this->Release();
}


bool OSPRaySphereGeometry::create() {
    return true;
}


void OSPRaySphereGeometry::release() {}


/*
ospray::OSPRaySphereGeometry::InterfaceIsDirty()
*/
bool OSPRaySphereGeometry::InterfaceIsDirty() {
    CallOSPRayMaterial* cm = this->getMaterialSlot.CallAs<CallOSPRayMaterial>();
    bool material_dirty = false;
    if (cm != nullptr) {
        cm->getMaterialParameter();
        material_dirty = cm->InterfaceIsDirty();
    }
    if (material_dirty) {
        return true;
    }
    return false;
}


bool OSPRaySphereGeometry::getExtends(megamol::core::Call& call) {
    CallOSPRayStructure* os = dynamic_cast<CallOSPRayStructure*>(&call);
    megamol::geocalls::MultiParticleDataCall* cd = this->getDataSlot.CallAs<megamol::geocalls::MultiParticleDataCall>();

    if (cd == NULL)
        return false;
    cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
    // if (!(*cd)(1)) return false; // table returns flase at first attempt and breaks everything
    (*cd)(1);
    this->extendContainer.boundingBox = std::make_shared<megamol::core::BoundingBoxes_2>();
    this->extendContainer.boundingBox->SetBoundingBox(cd->AccessBoundingBoxes().ObjectSpaceBBox());
    this->extendContainer.timeFramesCount = cd->FrameCount();
    this->extendContainer.isValid = true;

    return true;
}
