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


OSPRaySphereGeometry::OSPRaySphereGeometry()
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
    if (this->datahash != cd->DataHash() || this->time != os->getTime() || interface_dirty) {
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
    this->extendContainer.boundingBox->SetClipBox(cd->AccessBoundingBoxes().ObjectSpaceClipBox());
    this->extendContainer.timeFramesCount = cd->FrameCount();
    this->extendContainer.isValid = true;

    return true;
}
