/*
 * OSPRayLineGeometry.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "OSPRayLineGeometry.h"
#include "geometry_calls/LinesDataCall.h"
#include "mesh/MeshCalls.h"
#include "mmcore/Call.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/utility/log/Log.h"
#include "mmospray/CallOSPRayStructure.h"

using namespace megamol::ospray;
using namespace megamol;


OSPRayLineGeometry::OSPRayLineGeometry(void)
        : AbstractOSPRayStructure()
        , getDataSlot("getdata", "Connects to the data source")
        , getLineDataSlot("getLineData", "")
        , globalRadiusSlot("globalRadius", "Sets the radius of the lines")
        , representationSlot("lineRepresentation", "Sets the representation type of the line structure") {

    getDataSlot.SetCompatibleCall<geocalls::LinesDataCallDescription>();
    MakeSlotAvailable(&getDataSlot);

    getLineDataSlot.SetCompatibleCall<mesh::CallMeshDescription>();
    MakeSlotAvailable(&getLineDataSlot);

    globalRadiusSlot << new core::param::FloatParam(0.01);
    MakeSlotAvailable(&globalRadiusSlot);

    auto repType = new megamol::core::param::EnumParam(curveRepresentationType::ROUND);
    repType->SetTypePair(curveRepresentationType::ROUND, "round");
    repType->SetTypePair(curveRepresentationType::FLAT, "flat");
    repType->SetTypePair(curveRepresentationType::RIBBON, "ribbon");
    repType->SetTypePair(curveRepresentationType::DISJOINT, "disjoint");
    this->representationSlot << repType;
    MakeSlotAvailable(&representationSlot);

}


bool OSPRayLineGeometry::readData(core::Call& call) {

    // fill material container
    this->processMaterial();

    // get transformation parameter
    this->processTransformation();

    // fill clipping plane container
    this->processClippingPlane();

    // read Data, calculate  shape parameters, fill data vectors
    CallOSPRayStructure* os = dynamic_cast<CallOSPRayStructure*>(&call);

    mesh::CallMesh* cm = this->getLineDataSlot.CallAs<mesh::CallMesh>();
    curveStructure cs;
    if (cm != nullptr) {
        auto meta_data = cm->getMetaData();
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
        auto interface_dirty = this->InterfaceIsDirty();
        if (cm->hasUpdate() || this->time != os->getTime() || interface_dirty) {
            this->time = os->getTime();
            this->structureContainer.dataChanged = true;
            this->extendContainer.boundingBox = std::make_shared<core::BoundingBoxes_2>(meta_data.m_bboxs);
            _converted_data = nullptr;
            _converted_data = std::make_shared<mesh::MeshDataAccessCollection>();

            _converted_attribs.clear();
            _converted_vertices.clear();


            auto num_meshes = cm->getData()->accessMeshes().size();

            auto first_mesh = cm->getData()->accessMeshes().begin();
            for (int m = 0; m < num_meshes; ++m) {
                auto current_mesh = std::next(first_mesh,m);
                if (current_mesh->second.primitive_type == mesh::MeshDataAccessCollection::LINES) {
                    _converted_vertices.emplace_back();
                    _converted_attribs.emplace_back();
                    _converted_index.emplace_back();
                    _converted_indices.emplace_back();

                    auto num_indices = first_mesh->second.indices.byte_size / mesh::MeshDataAccessCollection::getByteSize(first_mesh->second.indices.type);

                    _converted_vertices.back().resize(num_indices);
                    _converted_attribs.back().resize(1);
                    _converted_index.back().reserve(num_indices / 2);


                    auto indices = reinterpret_cast<unsigned int*>(current_mesh->second.indices.data);
                    float* vertices;

                    for (auto& attr : current_mesh->second.attributes) {
                        if (attr.semantic == mesh::MeshDataAccessCollection::POSITION) {
                            vertices = reinterpret_cast<float*>(attr.data);
                        }
                    }

                    for (int i = 0; i < num_indices/2; ++i) {

                        std::array<float, 3> const vert0 = {
                            vertices[3 * indices[2*i + 0] + 0], vertices[3 * indices[2*i + 0] + 1], vertices[3 * indices[2*i + 0] + 2]};
                        std::array<float, 3> const vert1 = {
                            vertices[3 * indices[2*i + 1] + 0], vertices[3 * indices[2*i + 1] + 1], vertices[3 * indices[2*i + 1] + 2]};
                        _converted_vertices.back()[2 * i + 0] = vert0;
                        _converted_vertices.back()[2 * i + 1] = vert1;
                        _converted_index.back().emplace_back(2 * i);

                    }


                    _converted_attribs.back().resize(1);
                    _converted_attribs.back()[0].semantic =
                        mesh::MeshDataAccessCollection::AttributeSemanticType::POSITION;
                    _converted_attribs.back()[0].component_type = mesh::MeshDataAccessCollection::ValueType::FLOAT;
                    _converted_attribs.back()[0].byte_size = _converted_vertices.back().size() * sizeof(std::array<float, 3>);
                    _converted_attribs.back()[0].component_cnt = 3;
                    _converted_attribs.back()[0].stride = 3 * sizeof(float);
                    _converted_attribs.back()[0].data =
                        reinterpret_cast<uint8_t*>(_converted_vertices.back().data());


                    _converted_indices.back().type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
                    _converted_indices.back().byte_size = _converted_index.back().size() * sizeof(uint32_t);
                    _converted_indices.back().data = reinterpret_cast<uint8_t*>(_converted_index.back().data());

                    _converted_data->addMesh(current_mesh->first, _converted_attribs.back(), _converted_indices.back(),
                        mesh::MeshDataAccessCollection::LINE_STRIP);
                } else {
                    // leave non line meshes untouched
                    _converted_data->addMesh(current_mesh->first, current_mesh->second.attributes, current_mesh->second.indices, current_mesh->second.primitive_type);
                }
            } 
            cs.mesh = _converted_data;
            structureContainer.structure = cs;
        }

    } else {

        geocalls::LinesDataCall* cd = this->getDataSlot.CallAs<geocalls::LinesDataCall>();
        if (cd == NULL)
            return false;
        cd->SetTime(os->getTime());
        cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
        auto interface_dirty = this->InterfaceIsDirty();

        if (!(*cd)(1))
            return false;
        if (!(*cd)(0))
            return false;

        if (this->datahash != cd->DataHash() || this->time != os->getTime() || interface_dirty) {
            this->datahash = cd->DataHash();
            this->time = os->getTime();
            this->structureContainer.dataChanged = true;
        } else {
            return true;
        }


        unsigned int lineCount = cd->Count();

        std::vector<float> vd;
        std::vector<float> cd_rgba;
        std::vector<unsigned int> index;

        unsigned int indexWalker = 0;

        // Generate vertex and index arrays
        for (unsigned int i = 0; i < lineCount; ++i) {
            auto line = cd->GetLines()[i];
            for (unsigned int j = 0; j < line.Count(); ++j) {
                vd.push_back(line.VertexArrayFloat()[3 * j + 0]);
                vd.push_back(line.VertexArrayFloat()[3 * j + 1]);
                vd.push_back(line.VertexArrayFloat()[3 * j + 2]);
                if (line.ColourArrayType() == geocalls::LinesDataCall::Lines::CDT_FLOAT_RGB) {
                    cd_rgba.push_back(line.ColourArrayFloat()[3 * j + 0]);
                    cd_rgba.push_back(line.ColourArrayFloat()[3 * j + 1]);
                    cd_rgba.push_back(line.ColourArrayFloat()[3 * j + 2]);
                    cd_rgba.push_back(1.0f);
                } else if (line.ColourArrayType() == geocalls::LinesDataCall::Lines::CDT_FLOAT_RGBA) {
                    cd_rgba.push_back(line.ColourArrayFloat()[4 * j + 0]);
                    cd_rgba.push_back(line.ColourArrayFloat()[4 * j + 1]);
                    cd_rgba.push_back(line.ColourArrayFloat()[4 * j + 2]);
                    cd_rgba.push_back(line.ColourArrayFloat()[4 * j + 3]);
                } else if (line.ColourArrayType() == geocalls::LinesDataCall::Lines::CDT_BYTE_RGB) {
                    cd_rgba.push_back((static_cast<unsigned int>(line.ColourArrayByte()[0]) / 255.0f));
                    cd_rgba.push_back((static_cast<unsigned int>(line.ColourArrayByte()[1]) / 255.0f));
                    cd_rgba.push_back((static_cast<unsigned int>(line.ColourArrayByte()[2]) / 255.0f));
                    cd_rgba.push_back(1.0f);
                } else if (line.ColourArrayType() == geocalls::LinesDataCall::Lines::CDT_BYTE_RGBA) {
                    cd_rgba.push_back((static_cast<unsigned int>(line.ColourArrayByte()[0]) / 255.0f));
                    cd_rgba.push_back((static_cast<unsigned int>(line.ColourArrayByte()[1]) / 255.0f));
                    cd_rgba.push_back((static_cast<unsigned int>(line.ColourArrayByte()[2]) / 255.0f));
                    cd_rgba.push_back((static_cast<unsigned int>(line.ColourArrayByte()[3]) / 255.0f));
                }
                index.push_back(indexWalker);
                indexWalker += 1;
            }
            index.pop_back();
        }

        if (cd->GetLines()->ColourArrayType() == geocalls::LinesDataCall::Lines::CDT_NONE) {
            auto col = cd->GetLines()->GlobalColour();
            float r = static_cast<unsigned int>(col.R()) / 255.0f;
            float g = static_cast<unsigned int>(col.G()) / 255.0f;
            float b = static_cast<unsigned int>(col.B()) / 255.0f;
            float a = static_cast<unsigned int>(col.A()) / 255.0f;
            for (unsigned int i = 0; i < vd.size(); ++i) {
                cd_rgba.push_back(r);
                cd_rgba.push_back(g);
                cd_rgba.push_back(b);
                cd_rgba.push_back(a);
            }
        }


        // Write stuff into the structureContainer

        cs.vertexData = std::make_shared<std::vector<float>>(std::move(vd));
        cs.colorData = std::make_shared<std::vector<float>>(std::move(cd_rgba));
        cs.vertexLength = 3;
        cs.colorLength = 4;
        cs.indexData = std::make_shared<std::vector<unsigned int>>(std::move(index));
    }



    structureContainer.type = structureTypeEnum::GEOMETRY;
    structureContainer.geometryType = geometryTypeEnum::LINES;
    cs.globalRadius = globalRadiusSlot.Param<core::param::FloatParam>()->Value();
    cs.representation = curveRepresentationType(representationSlot.Param<core::param::EnumParam>()->Value());
    structureContainer.structure = cs;

    return true;
}

OSPRayLineGeometry::~OSPRayLineGeometry() {
    this->Release();
}

bool OSPRayLineGeometry::create() {
    return true;
}

void OSPRayLineGeometry::release() {}

/*
ospray::OSPRayLineGeometry::InterfaceIsDirty()
*/
bool OSPRayLineGeometry::InterfaceIsDirty() {
    CallOSPRayMaterial* cm = getMaterialSlot.CallAs<CallOSPRayMaterial>();
    cm->getMaterialParameter();
    if (cm->InterfaceIsDirty() || globalRadiusSlot.IsDirty() || representationSlot.IsDirty()) {
        globalRadiusSlot.ResetDirty();
        representationSlot.ResetDirty();
        return true;
    } else {
        return false;
    }
}


bool OSPRayLineGeometry::getExtends(core::Call& call) {
    CallOSPRayStructure* os = dynamic_cast<CallOSPRayStructure*>(&call);

    mesh::CallMesh* cm = this->getLineDataSlot.CallAs<mesh::CallMesh>();
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
        geocalls::LinesDataCall* cd = this->getDataSlot.CallAs<geocalls::LinesDataCall>();

        if (cd == NULL)
            return false;
        cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
        // if (!(*cd)(1)) return false; // table returns flase at first attempt and breaks everything
        (*cd)(1);
        this->extendContainer.boundingBox = std::make_shared<core::BoundingBoxes_2>();
        this->extendContainer.boundingBox->SetBoundingBox(cd->AccessBoundingBoxes().ObjectSpaceBBox());
        this->extendContainer.timeFramesCount = cd->FrameCount();
        this->extendContainer.isValid = true;
    }

    return true;
}
