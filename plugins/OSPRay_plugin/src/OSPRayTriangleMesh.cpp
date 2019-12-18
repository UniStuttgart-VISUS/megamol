/*
 * OSPRayTriangleMesh.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "OSPRayTriangleMesh.h"
#include <functional>
#include "geometry_calls/CallTriMeshData.h"
#include "mmcore/Call.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "vislib/sys/Log.h"
#include "mmcore/BoundingBoxes_2.h"


using namespace megamol::ospray;


OSPRayTriangleMesh::OSPRayTriangleMesh(void)
    : AbstractOSPRayStructure()
    , getTrimeshDataSlot("getTrimeshData", "Connects to the data source")
    , getMeshDataSlot("getMeshData", "Connects to the data source") {

    this->getTrimeshDataSlot.SetCompatibleCall<geocalls::CallTriMeshDataDescription>();
    this->MakeSlotAvailable(&this->getTrimeshDataSlot);

    this->getMeshDataSlot.SetCompatibleCall<mesh::CallMeshDescription>();
    this->MakeSlotAvailable(&this->getMeshDataSlot);
}


bool OSPRayTriangleMesh::readData(megamol::core::Call& call) {

    // fill material container
    this->processMaterial();

    // fill transformation container
    this->processTransformation();

    // read Data, calculate  shape parameters, fill data vectors
    CallOSPRayStructure* os = dynamic_cast<CallOSPRayStructure*>(&call);

    mesh::CallMesh* cm = this->getMeshDataSlot.CallAs<mesh::CallMesh>();

    if (cm != nullptr) {
        auto meta_data = cm->getMetaData();
        this->structureContainer.dataChanged = false;
        if (os->getTime() > meta_data.m_frame_cnt) {
            meta_data.m_frame_ID = meta_data.m_frame_cnt - 1;
        } else {
            meta_data.m_frame_ID = os->getTime();
        }
        cm->setMetaData(meta_data);
        if (!(*cm)(1)) return false;
        if (!(*cm)(0)) return false;
        meta_data = cm->getMetaData();
        if (cm->hasUpdate() || this->time != os->getTime() || this->InterfaceIsDirty()) {
            this->time = os->getTime();
            this->structureContainer.dataChanged = true;
            this->extendContainer.boundingBox = std::make_shared<megamol::core::BoundingBoxes_2>(meta_data.m_bboxs);
            this->extendContainer.timeFramesCount = meta_data.m_frame_cnt;
            this->extendContainer.isValid = true;
            this->structureContainer.mesh = cm->getData();
        }
    } else {


        geocalls::CallTriMeshData* cd = this->getTrimeshDataSlot.CallAs<geocalls::CallTriMeshData>();

        this->structureContainer.dataChanged = false;
        if (cd == NULL) return false;
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

        if (!(*cd)(1)) return false;
        if (!(*cd)(0)) return false;

        this->structureContainer.mesh = std::make_shared<mesh::MeshDataAccessCollection>();


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
                vislib::sys::Log::DefaultLog.WriteError(
                    "[OSPRayTriangleMesh] Vertex: No other data types than FLOAT are supported.");
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
                    vislib::sys::Log::DefaultLog.WriteError(
                        "[OSPRayTriangleMesh] Normals: No other data types than FLOAT are supported.");
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
                        _color.push_back((float)obj.GetColourPointerByte()[i] / 255.0f);
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
                    vislib::sys::Log::DefaultLog.WriteError(
                        "[OSPRayTriangleMesh] Color: No other data types than BYTE or FLOAT are supported.");
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
                        2, mesh::MeshDataAccessCollection::FLOAT, 2*sizeof(float), 0,
                        mesh::MeshDataAccessCollection::AttributeSemanticType::TEXCOORD});
                    break;
                default:
                    vislib::sys::Log::DefaultLog.WriteError("[OSPRayTriangleMesh] TextureCoordinate: No other data "
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
                    index.byte_size = 3 * triangleCount * sizeof(uint32_t);
                    break;

                default:
                    vislib::sys::Log::DefaultLog.WriteError(
                        "[OSPRayTriangleMesh] Index: No other data types than BYTE or FLOAT are supported.");
                    return false;
                }
            }
            this->structureContainer.mesh->addMesh(attrib, index);

        } // end for
    }


    // Write stuff into the structureContainer
    this->structureContainer.type = structureTypeEnum::GEOMETRY;
    this->structureContainer.geometryType = geometryTypeEnum::TRIANGLES;
    // this->structureContainer.vertexData = std::make_shared<std::vector<float>>(std::move(vertexD));
    // this->structureContainer.colorData = std::make_shared<std::vector<float>>(std::move(colorD));
    // this->structureContainer.normalData = std::make_shared<std::vector<float>>(std::move(normalD));
    // this->structureContainer.texData = std::make_shared<std::vector<float>>(std::move(texD));
    // this->structureContainer.indexData = std::make_shared<std::vector<uint32_t>>(std::move(indexD));
    // this->structureContainer.vertexCount = vertexCount;
    // this->structureContainer.triangleCount = triangleCount;

    return true;
}


OSPRayTriangleMesh::~OSPRayTriangleMesh() { this->Release(); }

bool OSPRayTriangleMesh::create() { return true; }

void OSPRayTriangleMesh::release() {}

/*
ospray::OSPRaySphereGeometry::InterfaceIsDirty()
*/
bool OSPRayTriangleMesh::InterfaceIsDirty() {
        return false;
}


bool OSPRayTriangleMesh::getExtends(megamol::core::Call& call) {
    CallOSPRayStructure* os = dynamic_cast<CallOSPRayStructure*>(&call);

    mesh::CallMesh* cm = this->getMeshDataSlot.CallAs<mesh::CallMesh>();

    if (cm != nullptr) {

        if (!(*cm)(1)) return false;
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

        megamol::geocalls::CallTriMeshData* cd = this->getTrimeshDataSlot.CallAs<megamol::geocalls::CallTriMeshData>();

        if (cd == NULL) return false;
        if (os->getTime() > cd->FrameCount()) {
            cd->SetFrameID(cd->FrameCount() - 1, true); // isTimeForced flag set to true
        } else {
            cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
        }

        if (!(*cd)(1)) return false;

        this->extendContainer.boundingBox = std::make_shared<core::BoundingBoxes_2>();
        this->extendContainer.boundingBox->SetBoundingBox(cd->AccessBoundingBoxes().ObjectSpaceBBox());
        this->extendContainer.timeFramesCount = cd->FrameCount();
        this->extendContainer.isValid = true;
    }
    return true;
}