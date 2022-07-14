/*
 * OSPRayMeshGLGeometry.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "OSPRayMeshGLGeometry.h"
#include "geometry_calls_gl/CallTriMeshDataGL.h"
#include "mesh/MeshCalls.h"
#include "mmcore/BoundingBoxes_2.h"
#include "mmcore/Call.h"
#include "mmcore/utility/log/Log.h"


using namespace megamol::ospray;


OSPRayMeshGLGeometry::OSPRayMeshGLGeometry(void)
        : AbstractOSPRayStructure()
        , _getTrimeshDataSlot("getTrimeshData", "Connects to the data source")
        , _getMeshDataSlot("getMeshData", "Connects to the data source") {

    this->_getTrimeshDataSlot.SetCompatibleCall<geocalls_gl::CallTriMeshDataGLDescription>();
    this->MakeSlotAvailable(&this->_getTrimeshDataSlot);
}


bool OSPRayMeshGLGeometry::readData(megamol::core::Call& call) {

    // fill material container
    this->processMaterial();

    // fill transformation container
    this->processTransformation();

    //fill clipping plane container
    this->processClippingPlane();

    // read Data, calculate  shape parameters, fill data vectors
    CallOSPRayStructure* os = dynamic_cast<CallOSPRayStructure*>(&call);

    auto fcw = writeFlagsSlot.CallAs<core::FlagCallWrite_CPU>();
    auto fcr = readFlagsSlot.CallAs<core::FlagCallRead_CPU>();

    geocalls_gl::CallTriMeshDataGL* cd = this->_getTrimeshDataSlot.CallAs<geocalls_gl::CallTriMeshDataGL>();

    if (cd != nullptr) {

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
        ms.mesh = std::make_shared<std::vector<std::shared_ptr<mesh::MeshDataAccessCollection>>>();


        unsigned int triangleCount = 0;
        unsigned int vertexCount = 0;

        const unsigned int objectCount = cd->Count();


        for (unsigned int i = 0; i < objectCount; i++) {

            std::vector<mesh::MeshDataAccessCollection::VertexAttribute> attrib;
            mesh::MeshDataAccessCollection::IndexData index;


            const geocalls_gl::CallTriMeshDataGL::Mesh& obj = cd->Objects()[i];
            triangleCount = obj.GetTriCount();
            vertexCount = obj.GetVertexCount();


            // check vertex data type
            switch (obj.GetVertexDataType()) {
            case geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT:
                attrib.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute{
                    const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(obj.GetVertexPointerFloat())),
                    3 * vertexCount *
                        mesh::MeshDataAccessCollection::getByteSize(mesh::MeshDataAccessCollection::FLOAT),
                    3, mesh::MeshDataAccessCollection::FLOAT, 0, 0,
                    mesh::MeshDataAccessCollection::AttributeSemanticType::POSITION});
                break;
            // case geocalls_gl::CallTriMeshData::Mesh::DT_DOUBLE:
            default:
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[OSPRayMeshGLGeometry] Vertex: No other data types than FLOAT are supported.");
                return false;
            }

            // check normal pointer
            if (obj.HasNormalPointer()) {
                switch (obj.GetNormalDataType()) {
                case geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT:
                    attrib.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute{
                        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(obj.GetNormalPointerFloat())),
                        3 * vertexCount *
                            mesh::MeshDataAccessCollection::getByteSize(mesh::MeshDataAccessCollection::FLOAT),
                        3, mesh::MeshDataAccessCollection::FLOAT, sizeof(float) * 3, 0,
                        mesh::MeshDataAccessCollection::AttributeSemanticType::NORMAL});
                    break;
                default:
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[OSPRayMeshGLGeometry] Normals: No other data types than FLOAT are supported.");
                    return false;
                }
            }

            // check colorpointer and convert to rgba
            if (obj.HasColourPointer()) {
                _color.clear();
                switch (obj.GetColourDataType()) {
                case geocalls_gl::CallTriMeshDataGL::Mesh::DT_BYTE:
                    _color.reserve(vertexCount * 4);
                    for (unsigned int i = 0; i < 3 * obj.GetVertexCount(); i++) {
                        _color.push_back((float)obj.GetColourPointerByte()[i] / 255.0f);
                        if ((i + 1) % 3 == 0) {
                            _color.push_back(1.0f);
                        }
                    }
                    break;
                case geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT:
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
                        "[OSPRayMeshGLGeometry] Color: No other data types than BYTE or FLOAT are supported.");
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
                case geocalls_gl::CallTriMeshDataGL::Mesh::DT_FLOAT:
                    attrib.emplace_back(mesh::MeshDataAccessCollection::VertexAttribute{
                        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(obj.GetTextureCoordinatePointerFloat())),
                        2 * vertexCount *
                            mesh::MeshDataAccessCollection::getByteSize(mesh::MeshDataAccessCollection::FLOAT),
                        2, mesh::MeshDataAccessCollection::FLOAT, 2 * sizeof(float), 0,
                        mesh::MeshDataAccessCollection::AttributeSemanticType::TEXCOORD});
                    break;
                default:
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[OSPRayMeshGLGeometry] TextureCoordinate: No other data "
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
                case geocalls_gl::CallTriMeshDataGL::Mesh::DT_UINT32:
                    index.data = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(obj.GetTriIndexPointerUInt32()));
                    index.type = mesh::MeshDataAccessCollection::ValueType::UNSIGNED_INT;
                    index.byte_size = 3 * sizeof(uint32_t) * (triangleCount - 1);
                    break;

                default:
                    megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[OSPRayMeshGLGeometry] Index: No other data types than BYTE or FLOAT are supported.");
                    return false;
                }
            }
            std::string identifier = std::string(FullName()) + "_object_" + std::to_string(i);
            ms.mesh->back()->addMesh(identifier, attrib, index);

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


OSPRayMeshGLGeometry::~OSPRayMeshGLGeometry() {
    this->Release();
}

bool OSPRayMeshGLGeometry::create() {
    return true;
}

void OSPRayMeshGLGeometry::release() {}

/*
ospray::OSPRaySphereGeometry::InterfaceIsDirty()
*/
bool OSPRayMeshGLGeometry::InterfaceIsDirty() {
    return false;
}


bool OSPRayMeshGLGeometry::getExtends(megamol::core::Call& call) {
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

        megamol::geocalls_gl::CallTriMeshDataGL* cd =
            this->_getTrimeshDataSlot.CallAs<megamol::geocalls_gl::CallTriMeshDataGL>();

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
