/*
 * PLYDataSource.cpp
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "io/PLYDataSource.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include <fstream>
#include <string>
#include <sstream>
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "geometry_calls/CallTriMeshData.h"

using namespace megamol;
using namespace megamol::core::moldyn;
using namespace megamol::geocalls;
using namespace megamol::stdplugin::datatools;
using namespace megamol::core;

/*
 * Checks whether two chars are equal, regardless of their case.
 * 
 * @param a The first char.
 * @param b The second char.
 * @return True if the two chars are equal, false otherwise.
 */
bool icompare_pred(unsigned char a, unsigned char b) { 
    return std::tolower(a) == std::tolower(b); 
}

/*
 * Checks two strings for equality, regardless of the letters case.
 * 
 * @param a The first string.
 * @param b The second string.
 * @return True if the two strings are equal, false otherwise.
 */
bool icompare(std::string const& a, std::string const& b) {
    if (a.length() == b.length()) {
        return std::equal(b.begin(), b.end(), a.begin(), icompare_pred);
    } else {
        return false;
    }
}

/**
 * Splits a string by a certain char delimiter.
 * 
 * @param s The input string.
 * @param delim The delimiter char.
 * @return Vector containing all parts of the string.
 */
std::vector<std::string> isplit(std::string const& s, char delim = ' ') {
    // TODO do this more intelligent
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string line;
    while (std::getline(ss, line, delim)) {
        result.push_back(line);
    }
    return result;
}

/**
 * Returns the size in bytes of the given tinyply data type.
 * 
 * @param tinyplyType The type of the tinyply variable.
 * @return The size of the given type in bytes.
 */
uint32_t tinyTypeSize(tinyply::Type tinyplyType) {
    switch (tinyplyType) {
    case tinyply::Type::INT8:
    case tinyply::Type::UINT8:
        return 1;
    case tinyply::Type::INT16:
    case tinyply::Type::UINT16:
        return 2;
    case tinyply::Type::INT32:
    case tinyply::Type::UINT32:
    case tinyply::Type::FLOAT32:
        return 4;
    case tinyply::Type::FLOAT64:
        return 8;
    case tinyply::Type::INVALID:
    default:
        return 0;
    }
}

/*
 * io::PLYDataSource::theUndef
 */
const char theUndef[] = "undef";

/*
 * io::PLYDataSource::PLYDataSource
 */
io::PLYDataSource::PLYDataSource(void) : core::Module(),
        filename("filename", "The path to the PLY file to load."),
        vertElemSlot("vertex element", "which element to get the vertex info from"),
        faceElemSlot("face element", "which element to get the face info from"),
        xPropSlot("x property", "which property to get the x position from"),
        yPropSlot("y property", "which property to get the y position from"),
        zPropSlot("z property", "which property to get the z position from"),
        nxPropSlot("nx property", "which property to get the normal x component from"),
        nyPropSlot("ny property", "which property to get the normal y component from"),
        nzPropSlot("nz property", "which property to get the normal z component from"),
        rPropSlot("r property", "which property to get the red component from"),
        gPropSlot("g property", "which property to get the green component from"),
        bPropSlot("b property", "which property to get the blue component from"),
        iPropSlot("i property", "which property to get the intensity from"),
        indexPropSlot("index property", "which property to get the vertex indices from"),
        radiusSlot("sphere radius", "the radius of the output spheres"),
        getSphereData("getspheredata", "Slot to request sphere data from this data source."),
        getMeshData("getmeshdata", "Slot to request mesh data from this data source."),
        data_hash(0), data_offset(0), vertex_count(0), face_count(0), hasBinaryFormat(false), isLittleEndian(true) {

    this->filename.SetParameter(new core::param::FilePathParam(""));
    this->filename.SetUpdateCallback(&PLYDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);

    this->vertElemSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->vertElemSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->vertElemSlot);
    this->faceElemSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->faceElemSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->faceElemSlot);

    this->xPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->xPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->xPropSlot);
    this->yPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->yPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->yPropSlot);
    this->zPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->zPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->zPropSlot);
    this->nxPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->nxPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->nxPropSlot);
    this->nyPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->nyPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->nyPropSlot);
    this->nzPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->nzPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->nzPropSlot);
    this->rPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->rPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->rPropSlot);
    this->gPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->gPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->gPropSlot);
    this->bPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->bPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->bPropSlot);
    this->iPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->iPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->iPropSlot);
    this->indexPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->indexPropSlot.SetUpdateCallback(&PLYDataSource::anyEnumChanged);
    this->MakeSlotAvailable(&this->indexPropSlot);

    this->radiusSlot.SetParameter(new core::param::FloatParam(1.0f));
    this->MakeSlotAvailable(&this->radiusSlot);

    this->getSphereData.SetCallback(MultiParticleDataCall::ClassName(), MultiParticleDataCall::FunctionName(0), &PLYDataSource::getSphereDataCallback);
    this->getSphereData.SetCallback(MultiParticleDataCall::ClassName(), MultiParticleDataCall::FunctionName(1), &PLYDataSource::getSphereExtentCallback);
    this->MakeSlotAvailable(&this->getSphereData);

    this->getMeshData.SetCallback(CallTriMeshData::ClassName(), CallTriMeshData::FunctionName(0), &PLYDataSource::getMeshDataCallback);
    this->getMeshData.SetCallback(CallTriMeshData::ClassName(), CallTriMeshData::FunctionName(1), &PLYDataSource::getMeshExtentCallback);
    this->MakeSlotAvailable(&this->getMeshData);
}

/*
 * io::PLYDataSource::~PLYDataSource
 */
io::PLYDataSource::~PLYDataSource(void) {
    this->Release();
}

/*
 * io::PLYDataSource::create
 */
bool io::PLYDataSource::create(void) {
    // intentionally empty
    return true;
}

/*
 * io::PLYDataSource::release
 */
void io::PLYDataSource::release(void) {
    // intentionally empty
}

/*
 * io::PLYDataSource::assertData
 */
bool io::PLYDataSource::assertData() {

    if (!instream.is_open()) return false;
    // if one of these pointers is not null, we already have read the data
    if (posPointers.pos_double != nullptr || posPointers.pos_float != nullptr) return true;

    // jump to the data in the file
    instream.seekg(this->data_offset, std::ios::beg);
    size_t vertexCount = 0;
    size_t faceCount = 0;

    // reserve the space for the data
    if (std::none_of(guessedPos.begin(), guessedPos.end(), [](std::string s) { return s.empty(); })) {
        if (std::any_of(guessedPos.begin(), guessedPos.end(), [this](std::string s) { return elementIndexMap.count(s) > 0; })) { 
            uint32_t maxSize = 0;
            uint32_t elemCount = 0;
            for (auto s : guessedPos) {
                auto idx = this->elementIndexMap[s];
                auto size = this->propertySizes[idx.first][idx.second];
                elemCount += this->elementCount[idx.first];
                vertexCount = this->elementCount[idx.first];
                maxSize = size > maxSize ? size : maxSize;
            }
            if (maxSize <= 4) {
                posPointers.pos_float = new float[elemCount];
            } else {
                posPointers.pos_double = new double[elemCount];
            }
            this->vertex_count = vertexCount;
        } else {
            vislib::sys::Log::DefaultLog.WriteWarn("One of the position labels could not be found");
        }
    }
    if (std::none_of(guessedNormal.begin(), guessedNormal.end(), [](std::string s) { return s.empty(); })) {
        if (std::any_of(guessedNormal.begin(), guessedNormal.end(), [this](std::string s) { return elementIndexMap.count(s) > 0; })) {
            uint32_t maxSize = 0;
            uint32_t elemCount = 0;
            for (auto s : guessedNormal) {
                auto idx = this->elementIndexMap[s];
                auto size = this->propertySizes[idx.first][idx.second];
                elemCount += this->elementCount[idx.first];
                maxSize = size > maxSize ? size : maxSize;
            }
            if (maxSize <= 4) {
                normalPointers.norm_float = new float[elemCount];
            } else {
                normalPointers.norm_double = new double[elemCount];
            }
        } else {
            vislib::sys::Log::DefaultLog.WriteWarn("One of the normal labels could not be found");
        }
    }
    if (std::none_of(guessedColor.begin(), guessedColor.end(), [](std::string s) { return s.empty(); })) {
        if (std::any_of(guessedColor.begin(), guessedColor.end(), [this](std::string s) { return elementIndexMap.count(s) > 0; })) {
            uint32_t maxSize = 0;
            uint32_t elemCount = 0;
            for (auto s : guessedColor) {
                auto idx = this->elementIndexMap[s];
                auto size = this->propertySizes[idx.first][idx.second];
                elemCount += this->elementCount[idx.first];
                maxSize = size > maxSize ? size : maxSize;
            }
            if (maxSize <= 1) {
                colorPointers.col_uchar = new unsigned char[elemCount];
            } else if (maxSize > 1 && maxSize < 8) {
                colorPointers.col_float = new float[elemCount];
            } else {
                colorPointers.col_double = new double[elemCount];
            }
        } else {
            vislib::sys::Log::DefaultLog.WriteWarn("One of the color labels could not be found");
        }
    }
    if (!guessedIndices.empty()) {
        if (this->elementIndexMap.count(guessedIndices) > 0) {
            auto idx = this->elementIndexMap[guessedIndices];
            auto size = this->propertySizes[idx.first][idx.second];
            auto elemCount = this->elementCount[idx.first] * 3; // TODO modify this to work with anything besides triangles
            faceCount = this->elementCount[idx.first];
            if (size <= 1) {
                facePointers.face_uchar = new unsigned char[elemCount];
            } else if (size == 2) {
                facePointers.face_u16 = new uint16_t[elemCount];
            } else {
                facePointers.face_u32 = new uint32_t[elemCount];
            }
            this->face_count = faceCount;
        } else {
            vislib::sys::Log::DefaultLog.WriteWarn("The triangle index label could not be found");
        }
    }

    if (this->hasBinaryFormat) {
        if (this->isLittleEndian) { // binary little endian

        } else { // binary big endian

        }
    } else { // ascii format
        std::string line;
        // TODO check order of the values (these here only work with normal ordered files

        this->boundingBox.Set(FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
        float * bbPointer = const_cast<float *>(boundingBox.PeekBounds()); // hackedihack

        // parse vertices
        for (size_t i = 0; i < vertexCount; i++) {
            if (std::getline(instream, line)) {
                auto split = isplit(line);
                for (size_t j = 0; j < guessedPos.size(); j++) {
                    if (elementIndexMap.count(guessedPos[j]) > 0) {
                        auto idx = elementIndexMap[guessedPos[j]];
                        if (posPointers.pos_float != nullptr) {
                            posPointers.pos_float[3 * i + j] = std::stof(split[idx.second]);
                        }
                        if (posPointers.pos_double != nullptr) {
                            posPointers.pos_double[3 * i + j] = std::stod(split[idx.second]);
                        }
                        if (std::stof(split[idx.second]) < bbPointer[j]) {
                            bbPointer[j] = std::stof(split[idx.second]);
                        }
                        if (std::stof(split[idx.second]) > bbPointer[j + 3]) {
                            bbPointer[j + 3] = std::stof(split[idx.second]);
                        }
                    }
                }
                for (size_t j = 0; j < guessedNormal.size(); j++) {
                    if (elementIndexMap.count(guessedNormal[j]) > 0) {
                        auto idx = elementIndexMap[guessedPos[j]];
                        if (normalPointers.norm_float != nullptr) {
                            normalPointers.norm_float[3 * i + j] = std::stof(split[idx.second]);
                        }
                        if (normalPointers.norm_double != nullptr) {
                            normalPointers.norm_double[3 * i + j] = std::stod(split[idx.second]);
                        }
                    }
                }
                for (size_t j = 0; j < guessedColor.size(); j++) {
                    if (elementIndexMap.count(guessedColor[j]) > 0) {
                        auto idx = elementIndexMap[guessedPos[j]];
                        if (colorPointers.col_uchar != nullptr) {
                            colorPointers.col_uchar[3 * i + j] = static_cast<unsigned char>(std::stoul(split[idx.second]));
                        }
                        if (colorPointers.col_float != nullptr) {
                            colorPointers.col_float[3 * i + j] = std::stof(split[idx.second]);
                        }
                        if (colorPointers.col_double != nullptr) {
                            colorPointers.col_double[3 * i + j] = std::stod(split[idx.second]);
                        }
                    }
                }
            } else {
                vislib::sys::Log::DefaultLog.WriteError("Unexpected file ending during vertex parsing");
                return false;
            }
        }
        // parse faces
        for (size_t i = 0; i < faceCount; i++) {
            if (std::getline(instream, line)) {
                auto split = isplit(line);
                if (elementIndexMap.count(guessedFaces)) {
                    uint32_t faceSize = static_cast<uint32_t>(std::stoul(split[0]));
                    if (faceSize != 3) {
                        vislib::sys::Log::DefaultLog.WriteError("The PlyDataSource is currently only able to handle triangular faces");
                        return false;
                    }
                    for (size_t j = 1; j < faceSize + 1; j++) {
                        if (facePointers.face_uchar != nullptr) {
                            facePointers.face_uchar[3 * i + j] = static_cast<unsigned char>(std::stoul(split[j]));
                        }
                        if (facePointers.face_u16 != nullptr) {
                            facePointers.face_u16[3 * i + j] = static_cast<uint16_t>(std::stoul(split[j]));
                        }
                        if (facePointers.face_u32 != nullptr) {
                            facePointers.face_u32[3 * i + j] = static_cast<uint32_t>(std::stoul(split[j]));
                        }
                    }
                }
            } else {
                vislib::sys::Log::DefaultLog.WriteError("Unexpected file ending during face parsing");
                return false;
            }
        }
    }

    return true;
}

/*
 * io::PLYDataSource::filenameChanged
 */
bool io::PLYDataSource::filenameChanged(core::param::ParamSlot& slot) {

    using vislib::sys::Log;

    instream.open(filename.Param<core::param::FilePathParam>()->Value(), std::ios::binary);
    if (instream.fail()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to open PLY File \"%s\".", vislib::StringA(filename.Param<core::param::FilePathParam>()->Value()).PeekBuffer());
        // TODO ?
        return true;
    }

    this->vertElemSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->faceElemSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->xPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->yPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->zPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->nxPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->nyPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->nzPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->rPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->gPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->bPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->iPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();
    this->indexPropSlot.Param<core::param::FlexEnumParam>()->ClearValues();

    guessedVertices = "";
    guessedFaces = "";
    guessedIndices = "";
    guessedPos.assign(3, "");
    guessedNormal.assign(3, "");
    guessedColor.assign(3, "");

    this->elementIndexMap.clear();
    this->elementSizes.clear();
    this->elementCount.clear();
    this->propertySizes.clear();
    this->hasBinaryFormat = false;
    this->isLittleEndian = true;
    this->data_offset = 0;

    plf.parse_header(instream);

    uint32_t element_index = 0;
    uint32_t property_index = 0;
    uint32_t element_size = 0;

    for (auto e: plf.get_elements()) {
        this->vertElemSlot.Param<core::param::FlexEnumParam>()->AddValue(e.name);
        this->faceElemSlot.Param<core::param::FlexEnumParam>()->AddValue(e.name);
        if (icompare(e.name,"vertex")) {
            guessedVertices = e.name;
        }
        if (icompare(e.name, "face")) {
            guessedFaces = e.name;
        }
        this->elementCount.push_back(static_cast<uint32_t>(e.size));
        this->propertySizes.push_back(std::vector<uint32_t>());

        property_index = 0;
        element_size = 0;
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "element: %s, %u bytes", e.name.c_str(), e.size);
        for (auto p: e.properties) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "    property: %s %s %i", p.name.c_str(), tinyply::PropertyTable[p.propertyType].str.c_str(), tinyply::PropertyTable[p.propertyType].stride);
            this->xPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);
            this->yPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);
            this->zPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);
            this->nxPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);
            this->nyPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);
            this->nzPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);
            this->rPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);
            this->gPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);
            this->bPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);
            this->iPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);
            this->indexPropSlot.Param<core::param::FlexEnumParam>()->AddValue(p.name);

            if (icompare(p.name, "x")) {
                guessedPos[0] = p.name;
            }
            if (icompare(p.name, "y")) {
                guessedPos[1] = p.name;
            }
            if (icompare(p.name, "z")) {
                guessedPos[2] = p.name;
            }
            if (icompare(p.name, "nx")) {
                guessedNormal[0] = p.name;
            }
            if (icompare(p.name, "ny")) {
                guessedNormal[1] = p.name;
            }
            if (icompare(p.name, "nz")) {
                guessedNormal[2] = p.name;
            }
            if (icompare(p.name, "r")) {
                guessedColor[0] = p.name;
            }
            if (icompare(p.name, "g")) {
                guessedColor[1] = p.name;
            }
            if (icompare(p.name, "b")) {
                guessedColor[2] = p.name;
            }
            if (icompare(p.name, "indices") || icompare(p.name, "vertex_indices")) {
                guessedIndices = p.name;
            }
            elementIndexMap[p.name] = std::make_pair(element_index, property_index);
            element_size += tinyTypeSize(p.propertyType);
            propertySizes[propertySizes.size() - 1].push_back(tinyTypeSize(p.propertyType));
            property_index++;
        }
        elementSizes.push_back(element_size);
        element_index++;
    }

    if (!guessedVertices.empty()) {
        this->vertElemSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedVertices);
        if (std::none_of(guessedPos.begin(), guessedPos.end(), [](std::string s) { return s.empty(); })) {
            this->xPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedPos[0]);
            this->yPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedPos[1]);
            this->zPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedPos[2]);
        }
        if (std::none_of(guessedNormal.begin(), guessedNormal.end(), [](std::string s) { return s.empty(); })) {
            this->nxPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedNormal[0]);
            this->nyPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedNormal[1]);
            this->nzPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedNormal[2]);
        }
        if (std::none_of(guessedColor.begin(), guessedColor.end(), [](std::string s) { return s.empty(); })) {
            this->rPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedColor[0]);
            this->gPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedColor[1]);
            this->bPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedColor[2]);
        }
    }
    if (!guessedFaces.empty()) {
        this->faceElemSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedFaces);
        if (!guessedIndices.empty()) {
            this->indexPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedIndices);
        }
    }
    
    // read the missing data ourself: file format, offset of the data to the file start
    instream.seekg(0, instream.beg);
    std::string line;
    bool done = false;
    while (std::getline(instream, line) && !done) {
        if (icompare(line.substr(0, 6), "format")) {
            if (icompare(line.substr(7, 3), "asc")) {
                this->hasBinaryFormat = false;
            } else if(icompare(line.substr(7, 3), "bin")) {
                this->hasBinaryFormat = true;
                if (icompare(line.substr(14, 3), "lit")) {
                    this->isLittleEndian = true;
                } else if (icompare(line.substr(14, 3), "big")) {
                    this->isLittleEndian = false;
                } else {
                    vislib::sys::Log::DefaultLog.WriteWarn("Endianness could not be determined, assuming little endian");
                    this->isLittleEndian = true;
                }
            } else {
                vislib::sys::Log::DefaultLog.WriteWarn("File format could not be determined, assuming ASCII");
                this->hasBinaryFormat = false;
            }
        }
        if (icompare(line.substr(0, 10), "end_header")) {
            this->data_offset = static_cast<size_t>(instream.tellg());
            done = true;
        }
    }

    return true;
}

/*
 * io::PLYDataSource::anyEnumChanged
 */
bool io::PLYDataSource::anyEnumChanged(core::param::ParamSlot& slot) {
    this->clearAllFields();
    return true;
}

/*
 * io::PLYDataSource::getSphereDataCallback
 */
bool io::PLYDataSource::getSphereDataCallback(core::Call& caller) {
    auto c2 = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&caller);
    if (c2 == nullptr) return false;

    c2->SetParticleListCount(1);
    MultiParticleDataCall::Particles& p = c2->AccessParticles(0);
    p.SetCount(this->vertex_count);
    // TODO always write data in the float pointer, since sphere data is only possible with float
    if (p.GetCount() > 0) {
        p.SetVertexData(SimpleSphericalParticles::VertexDataType::VERTDATA_FLOAT_XYZ, this->posPointers.pos_float);
        if (colorPointers.col_uchar != nullptr) {
            p.SetColourData(SimpleSphericalParticles::ColourDataType::COLDATA_UINT8_RGB, this->colorPointers.col_uchar);
        } else if (colorPointers.col_float != nullptr) {
            p.SetColourData(SimpleSphericalParticles::ColourDataType::COLDATA_FLOAT_RGB, this->colorPointers.col_float);
        } else {
            p.SetColourData(SimpleSphericalParticles::ColourDataType::COLDATA_NONE, nullptr);
        }
    }
    p.SetGlobalRadius(this->radiusSlot.Param<param::FloatParam>()->Value());

    return true;
}

/*
 * io::PLYDataSource::getSphereExtentCallback
 */
bool io::PLYDataSource::getSphereExtentCallback(core::Call& caller) {
    auto c2 = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&caller);
    if (c2 == nullptr) return false;

    if (!assertData()) return false;

    if (this->radiusSlot.IsDirty()) {
        this->radiusSlot.ResetDirty();
        this->sphereBoundingBox = this->boundingBox;
        this->sphereBoundingBox.Grow(this->radiusSlot.Param<param::FloatParam>()->Value());
        this->data_hash++;
    }

    c2->AccessBoundingBoxes().Clear();
    c2->AccessBoundingBoxes().SetObjectSpaceBBox(this->sphereBoundingBox);
    c2->AccessBoundingBoxes().SetObjectSpaceClipBox(this->sphereBoundingBox);
    c2->SetFrameCount(1);
    c2->SetDataHash(this->data_hash);

    return true;
}

/*
 * io::PLYDataSource::getMeshDataCallback
 */
bool io::PLYDataSource::getMeshDataCallback(core::Call& caller) {
    auto c2 = dynamic_cast<CallTriMeshData*>(&caller);
    if (c2 == nullptr) return false;

    // stupid if-cascade...
    // if you come up with something more beautiful, just do it
    if (posPointers.pos_float != nullptr) {
        
    } else if (posPointers.pos_double != nullptr) {

    } else {
        this->mesh.SetVertexData(0, nullptr, nullptr, nullptr, nullptr, false);
    }

    return true;
}

/*
 * io::PLYDataSource::getMeshExtentCallback
 */
bool io::PLYDataSource::getMeshExtentCallback(core::Call& caller) {
    auto c2 = dynamic_cast<CallTriMeshData*>(&caller);
    if (c2 == nullptr) return false;

    c2->SetFrameCount(1);
    c2->SetDataHash(this->data_hash);

    return true;
}

/*
 * io::PLYDataSource::clearAllFields
 */
void io::PLYDataSource::clearAllFields(void) {
    if (posPointers.pos_float != nullptr) {
        delete[] posPointers.pos_float;
        posPointers.pos_float = nullptr;
    }
    if (posPointers.pos_double != nullptr) {
        delete[] posPointers.pos_double;
        posPointers.pos_double = nullptr;
    }
    if (colorPointers.col_uchar != nullptr) {
        delete[] colorPointers.col_uchar;
        colorPointers.col_uchar = nullptr;
    }
    if (colorPointers.col_float != nullptr) {
        delete[] colorPointers.col_float;
        colorPointers.col_float = nullptr;
    }
    if (colorPointers.col_double != nullptr) {
        delete[] colorPointers.col_double;
        colorPointers.col_double = nullptr;
    }
    if (normalPointers.norm_float != nullptr) {
        delete[] normalPointers.norm_float;
        normalPointers.norm_float = nullptr;
    }
    if (normalPointers.norm_double != nullptr) {
        delete[] normalPointers.norm_double;
        normalPointers.norm_double = nullptr;
    }
    if (facePointers.face_uchar != nullptr) {
        delete[] facePointers.face_uchar;
        facePointers.face_uchar = nullptr;
    }
    if (facePointers.face_u16 != nullptr) {
        delete[] facePointers.face_u16;
        facePointers.face_u16 = nullptr;
    }
    if (facePointers.face_u32 != nullptr) {
        delete[] facePointers.face_u32;
        facePointers.face_u32 = nullptr;
    }
    this->vertex_count = 0;
    this->face_count = 0;
}