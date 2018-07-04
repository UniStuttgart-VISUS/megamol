/*
 * PlyWriter.cpp
 *
 * Copyright (C) 2017 by Karsten Schatz
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "PlyWriter.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::stdplugin::datatools;
using namespace megamol::geocalls;

/*
 * io::PlyWriter::PlyWriter
 */
io::PlyWriter::PlyWriter(void) : AbstractDataWriter(),
    filenameSlot("filename", "The path to the .ply file to be written"),
    frameIDSlot("frameID", "The ID of the frame to be written"),
    meshDataSlot("meshData", "The slot requesting the data to be written") {

    this->filenameSlot.SetParameter(new param::FilePathParam(""));
    this->MakeSlotAvailable(&this->filenameSlot);

    this->frameIDSlot.SetParameter(new param::IntParam(0, 0));
    this->MakeSlotAvailable(&this->frameIDSlot);

    this->meshDataSlot.SetCompatibleCall<CallTriMeshDataDescription>();
    this->MakeSlotAvailable(&this->meshDataSlot);
}

/*
 * io::PlyWriter::~PlyWriter
 */
io::PlyWriter::~PlyWriter(void) {
    this->Release();
}

/*
 * io::PlyWriter::create
 */
bool io::PlyWriter::create(void) {
    return true;
}

/*
 * io::PlyWriter::release
 */
void io::PlyWriter::release(void) {
}

/*
 * io::PlyWriter::run
 */
bool io::PlyWriter::run(void) {
    using vislib::sys::Log;
    vislib::TString filename(this->filenameSlot.Param<param::FilePathParam>()->Value());
    if (filename.IsEmpty()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "No file name specified. Abort.");
        return false;
    }

    CallTriMeshData *ctd = this->meshDataSlot.CallAs<CallTriMeshData>();
    if (ctd == nullptr) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "No data source connected. Abort.");
        return false;
    }

    ctd->SetFrameID(this->frameIDSlot.Param<param::IntParam>()->Value());
    if (!(*ctd)(1))return false; // getExtents
    ctd->SetFrameID(this->frameIDSlot.Param<param::IntParam>()->Value());
    if (!(*ctd)(0))return false; // getData

    std::vector<std::vector<unsigned int>> faces;
    std::vector<std::vector<float>> vertices;
    std::vector<std::vector<float>> normals;
    std::vector<std::vector<float>> colors;

    // read in the complete mesh data
    for (unsigned int objID = 0; objID < ctd->Count(); objID++) {
        const CallTriMeshData::Mesh& mesh = ctd->Objects()[objID];
        
        std::vector<unsigned int> fac;
        std::vector<float> vert;
        std::vector<float> norm;
        std::vector<float> col;

        unsigned int faceCnt = mesh.GetTriCount();
        unsigned int vertCnt = mesh.GetVertexCount();

        fac.resize(faceCnt * 3);
        fac.shrink_to_fit();

        vert.resize(vertCnt * 3);
        vert.shrink_to_fit();
        
        norm.resize(vertCnt * 3);
        norm.shrink_to_fit();

        col.resize(vertCnt * 3);
        col.shrink_to_fit();

        bool verticesAvailable = false;
        bool facesAvailable = false;
        bool colorsAvailable = false;
        bool normalsAvailable = false;

        // copy the vertex data
        if (mesh.GetVertexDataType() == CallTriMeshData::Mesh::DT_FLOAT) {
            if (mesh.GetVertexPointerFloat() != nullptr) {
                std::copy(mesh.GetVertexPointerFloat(), mesh.GetVertexPointerFloat() + vertCnt * 3, vert.begin());
                verticesAvailable = true;
            }
        } else if (mesh.GetVertexDataType() == CallTriMeshData::Mesh::DT_DOUBLE) {
            if (mesh.GetVertexPointerDouble() != nullptr) {
                std::transform(mesh.GetVertexPointerDouble(), mesh.GetVertexPointerDouble() + vertCnt * 3, vert.begin(), [](double v) {
                    return static_cast<float>(v);
                });
                verticesAvailable = true;
            } 
        }

        auto dt = mesh.GetTriDataType();

        // copy the face data
        if (mesh.GetTriDataType() == CallTriMeshData::Mesh::DT_UINT32) {
            if (mesh.GetTriIndexPointerUInt32() != nullptr) {
                std::copy(mesh.GetTriIndexPointerUInt32(), mesh.GetTriIndexPointerUInt32() + faceCnt * 3, fac.begin());
                facesAvailable = true;
            }
        } else if (mesh.GetTriDataType() == CallTriMeshData::Mesh::DT_UINT16) {
            if (mesh.GetTriIndexPointerUInt16() != nullptr) {
                std::transform(mesh.GetTriIndexPointerUInt16(), mesh.GetTriIndexPointerUInt16() + faceCnt * 3, fac.begin(), [](unsigned short v) {
                    return static_cast<unsigned int>(v);
                });
                facesAvailable = true;
            }
        } else if (mesh.GetTriDataType() == CallTriMeshData::Mesh::DT_BYTE) {
            if (mesh.GetTriIndexPointerByte() != nullptr) {
                std::transform(mesh.GetTriIndexPointerByte(), mesh.GetTriIndexPointerByte() + faceCnt * 3, fac.begin(), [](unsigned char v) {
                    return static_cast<unsigned int>(v);
                });
                facesAvailable = true;
            }
        }

        // copy the normal data
        if (mesh.GetNormalDataType() == CallTriMeshData::Mesh::DT_FLOAT) {
            if (mesh.GetNormalPointerFloat() != nullptr) {
                std::copy(mesh.GetNormalPointerFloat(), mesh.GetNormalPointerFloat() + vertCnt * 3, norm.begin());
                normalsAvailable = true;
            }
        } else if (mesh.GetNormalDataType() == CallTriMeshData::Mesh::DT_DOUBLE) {
            if (mesh.GetNormalPointerDouble() != nullptr) {
                std::transform(mesh.GetNormalPointerDouble(), mesh.GetNormalPointerDouble() + vertCnt * 3, norm.begin(), [](double v) {
                    return static_cast<float>(v);
                });
                normalsAvailable = true;
            }
        }

        // copy the colour data
        if (mesh.GetColourDataType() == CallTriMeshData::Mesh::DT_FLOAT) {
            if (mesh.GetColourPointerFloat() != nullptr) {
                std::copy(mesh.GetColourPointerFloat(), mesh.GetColourPointerFloat() + vertCnt * 3, col.begin());
                colorsAvailable = true;
            }
        } else if (mesh.GetColourDataType() == CallTriMeshData::Mesh::DT_DOUBLE) {
            if (mesh.GetColourPointerDouble() != nullptr) {
                std::transform(mesh.GetColourPointerDouble(), mesh.GetColourPointerDouble() + vertCnt * 3, col.begin(), [](double v) {
                    return static_cast<float>(v);
                });
                colorsAvailable = true;
            }
        } else if (mesh.GetColourDataType() == CallTriMeshData::Mesh::DT_BYTE) {
            if (mesh.GetColourPointerByte() != nullptr) {
                std::transform(mesh.GetColourPointerByte(), mesh.GetColourPointerByte() + vertCnt * 3, col.begin(), [](unsigned char v) {
                    return static_cast<float>(v) / 255.0f;
                });
                colorsAvailable = true;
            }
        }

        if (verticesAvailable) {
            vertices.push_back(vert);
        } else {
            continue;
        }

        if (facesAvailable) {
            faces.push_back(fac);
        } else {
            // if we have no face data but vertex data, we assume that each 3 vertices form a triangle
            fac.resize(vert.size() / 3);
            for (unsigned int i = 0; i < fac.size(); i++) {
                fac[i] = i;
            }
            faces.push_back(fac);
        }

        if (normalsAvailable) {
            normals.push_back(norm);
        } else {
            norm.clear();
            normals.push_back(norm);
        }

        if (colorsAvailable) {
            colors.push_back(col);
        } else {
            col.clear();
            colors.push_back(col);
        }
    }

    // collect relevant data
    unsigned int vertexCount = 0;
    unsigned int faceCount = 0;

    bool wantNormals = true;
    vislib::math::Vector<unsigned char, 3> standardCol(255, 0, 0);

    for (unsigned int objID = 0; objID < vertices.size(); objID++) {
        vertexCount += static_cast<unsigned int>(vertices[objID].size() / 3);
        faceCount += static_cast<unsigned int>(faces[objID].size() / 3);

        if (normals[objID].size() <  vertices[objID].size()) {
            wantNormals = false;
        }
    }

    ctd->Unlock();
    // write the ply file
    vislib::sys::FastFile file;
    if (!file.Open(filename, vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE, vislib::sys::File::CREATE_OVERWRITE)) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Unable to create output file \"%s\". Abort.",
            vislib::StringA(filename).PeekBuffer());
        return false;
    }
    
    
#define ASSERT_WRITEOUT(A, S) if (file.Write((A), (S)) != (S)) { \
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Write error %d", __LINE__); \
        file.Close(); \
        return false; \
            }

    // header
    // the color type has to be uchar since most programs can only read this type (although float would be available)
    std::string header = "ply\nformat ascii 1.0\ncomment file created by MegaMol\n";
    std::string newline = "\n";
    ASSERT_WRITEOUT(header.c_str(), header.size());
    header = "element vertex " + std::to_string(vertexCount) + "\n";
    ASSERT_WRITEOUT(header.c_str(), header.size());
    header = "property float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n";
    ASSERT_WRITEOUT(header.c_str(), header.size());

    if (wantNormals) {
        header = "property float nx\nproperty float ny\nproperty float nz\n";
        ASSERT_WRITEOUT(header.c_str(), header.size());
    }

    header = "element face " + std::to_string(faceCount) + "\n";
    ASSERT_WRITEOUT(header.c_str(), header.size());
    header = "property list uchar int vertex_index\nend_header\n";
    ASSERT_WRITEOUT(header.c_str(), header.size());

    unsigned int indexOffset = 0;
    std::string text;

    // body vertices
    for (unsigned int objID = 0; objID < static_cast<unsigned int>(vertices.size()); objID++) {
        for (unsigned int i = 0; i < static_cast<unsigned int>(vertices[objID].size() / 3); i++) {
            text = std::to_string(vertices[objID][i * 3 + 0]) + " " + std::to_string(vertices[objID][i * 3 + 1]) + " " + std::to_string(vertices[objID][i * 3 + 2]);
            if (colors[objID].size() > i * 3 + 2) {
                unsigned char c1 = static_cast<unsigned char>(colors[objID][i * 3 + 0] * 255.0f);
                unsigned char c2 = static_cast<unsigned char>(colors[objID][i * 3 + 1] * 255.0f);
                unsigned char c3 = static_cast<unsigned char>(colors[objID][i * 3 + 2] * 255.0f);
                text += " " + std::to_string(c1) + " " + std::to_string(c2) + " " + std::to_string(c3);
            } else {
                text += " " + std::to_string(standardCol.GetX()) + " " + std::to_string(standardCol.GetY()) + " " + std::to_string(standardCol.GetZ());
            }
            if (wantNormals) {
                text += " " + std::to_string(normals[objID][i * 3 + 0]) + " " + std::to_string(normals[objID][i * 3 + 1]) + " " + std::to_string(normals[objID][i * 3 + 2]);
            }
            text += "\n";
            ASSERT_WRITEOUT(text.c_str(), text.size());
        }
    }

    // body faces
    for (unsigned int objID = 0; objID < static_cast<unsigned int>(faces.size()); objID++) {
        for (unsigned int i = 0; i < static_cast<unsigned int>(faces[objID].size() / 3); i++) {
            text = "3 " + std::to_string(faces[objID][i * 3 + 0] + indexOffset) + " " + 
                std::to_string(faces[objID][i * 3 + 1] + indexOffset) + " " + 
                std::to_string(faces[objID][i * 3 + 2] + indexOffset) + "\n";
            ASSERT_WRITEOUT(text.c_str(), text.size());
        }

        indexOffset += static_cast<unsigned int>(vertices[objID].size());
    }
    file.Close();

#undef ASSERT_WRITEOUT

    return true;
}

/*
 * io::PlyWriter::getCapabilities
 */
bool io::PlyWriter::getCapabilities(DataWriterCtrlCall& call) {
    call.SetAbortable(false);
    return true;
}