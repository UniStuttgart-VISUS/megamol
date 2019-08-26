/*
 * PLYDataSource.cpp
 *
 * Copyright (C) 2018 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "io/PLYDataSource.h"
#include <array>
#include <fstream>
#include <sstream>
#include <string>
#include "geometry_calls/CallTriMeshData.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"

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
bool icompare_pred(unsigned char a, unsigned char b) { return std::tolower(a) == std::tolower(b); }

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
uint64_t tinyTypeSize(tinyply::Type tinyplyType) {
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

/**
 * Returns whether the given tinyply type is a signed or unsigned one.
 *
 * @param tinyplyType The type of the tinyply variable.
 * @return True if the type is a signed type. False otherwise.
 */
bool tinyIsSigned(tinyply::Type tinyplyType) {
    switch (tinyplyType) {
    case tinyply::Type::INT8:
    case tinyply::Type::INT16:
    case tinyply::Type::INT32:
    case tinyply::Type::FLOAT32:
    case tinyply::Type::FLOAT64:
        return true;
    case tinyply::Type::UINT8:
    case tinyply::Type::UINT16:
    case tinyply::Type::UINT32:
    case tinyply::Type::INVALID:
    default:
        return false;
    }
}

/**
 * Changes the endianness of a given variable by reversing all bits in the field.
 *
 * @param obj Reference to the variable of which the endianness is changed.
 */
template <class T> void changeEndianness(T& obj) {
    unsigned char* mem = reinterpret_cast<unsigned char*>(&obj);
    std::reverse(mem, mem + sizeof(T));
}

/*
 * io::PLYDataSource::theUndef
 */
const char theUndef[] = "undef";

/*
 * io::PLYDataSource::PLYDataSource
 */
io::PLYDataSource::PLYDataSource(void)
    : core::Module()
    , filename("filename", "The path to the PLY file to load.")
    , vertElemSlot("vertex element", "which element to get the vertex info from")
    , faceElemSlot("face element", "which element to get the face info from")
    , xPropSlot("x property", "which property to get the x position from")
    , yPropSlot("y property", "which property to get the y position from")
    , zPropSlot("z property", "which property to get the z position from")
    , nxPropSlot("nx property", "which property to get the normal x component from")
    , nyPropSlot("ny property", "which property to get the normal y component from")
    , nzPropSlot("nz property", "which property to get the normal z component from")
    , rPropSlot("r property", "which property to get the red component from")
    , gPropSlot("g property", "which property to get the green component from")
    , bPropSlot("b property", "which property to get the blue component from")
    , iPropSlot("i property", "which property to get the intensity from")
    , indexPropSlot("index property", "which property to get the vertex indices from")
    , radiusSlot("sphere radius", "the radius of the output spheres")
    , getSphereData("getspheredata", "Slot to request sphere data from this data source.")
    , getMeshData("getmeshdata", "Slot to request mesh data from this data source.")
    , data_hash(0)
    , data_offset(0)
    , vertex_count(0)
    , face_count(0)
    , hasBinaryFormat(false)
    , isLittleEndian(true) {

    this->filename.SetParameter(new core::param::FilePathParam(""));
    this->filename.SetUpdateCallback(&PLYDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filename);

    this->vertElemSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->MakeSlotAvailable(&this->vertElemSlot);
    this->faceElemSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->MakeSlotAvailable(&this->faceElemSlot);

    this->xPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->MakeSlotAvailable(&this->xPropSlot);
    this->yPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->MakeSlotAvailable(&this->yPropSlot);
    this->zPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->MakeSlotAvailable(&this->zPropSlot);
    this->nxPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->MakeSlotAvailable(&this->nxPropSlot);
    this->nyPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->MakeSlotAvailable(&this->nyPropSlot);
    this->nzPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->MakeSlotAvailable(&this->nzPropSlot);
    this->rPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->MakeSlotAvailable(&this->rPropSlot);
    this->gPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->MakeSlotAvailable(&this->gPropSlot);
    this->bPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->MakeSlotAvailable(&this->bPropSlot);
    this->iPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->MakeSlotAvailable(&this->iPropSlot);
    this->indexPropSlot.SetParameter(new core::param::FlexEnumParam(theUndef));
    this->MakeSlotAvailable(&this->indexPropSlot);

    this->radiusSlot.SetParameter(new core::param::FloatParam(1.0f));
    this->MakeSlotAvailable(&this->radiusSlot);
    this->radiusSlot.ForceSetDirty(); // this forces the program to recompute the sphere bounding box

    this->getSphereData.SetCallback(MultiParticleDataCall::ClassName(), MultiParticleDataCall::FunctionName(0),
        &PLYDataSource::getSphereDataCallback);
    this->getSphereData.SetCallback(MultiParticleDataCall::ClassName(), MultiParticleDataCall::FunctionName(1),
        &PLYDataSource::getSphereExtentCallback);
    this->MakeSlotAvailable(&this->getSphereData);

    this->getMeshData.SetCallback(
        CallTriMeshData::ClassName(), CallTriMeshData::FunctionName(0), &PLYDataSource::getMeshDataCallback);
    this->getMeshData.SetCallback(
        CallTriMeshData::ClassName(), CallTriMeshData::FunctionName(1), &PLYDataSource::getMeshExtentCallback);
    this->MakeSlotAvailable(&this->getMeshData);

    this->resetParameterDirtyness();
}

/*
 * io::PLYDataSource::~PLYDataSource
 */
io::PLYDataSource::~PLYDataSource(void) { this->Release(); }

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
    // if one of these pointers is not null, we already have read the data
    if (posPointers.pos_double != nullptr || posPointers.pos_float != nullptr) return true;

    instream.close();
    instream.open(filename.Param<core::param::FilePathParam>()->Value(), std::ios::binary);
    if (instream.fail()) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Unable to open PLY File \"%s\".",
            vislib::StringA(filename.Param<core::param::FilePathParam>()->Value()).PeekBuffer());
        return true;
    }
    // if one of these pointers is not null, we already have read the data
    //if (posPointers.pos_double != nullptr || posPointers.pos_float != nullptr) return true;

    // jump to the data in the file
    instream.seekg(this->data_offset, instream.beg);
    size_t vertexCount = 0;
    size_t faceCount = 0;

    // reserve the space for the data
    if (std::none_of(selectedPos.begin(), selectedPos.end(), [](std::string s) { return s.empty(); })) {
        if (std::any_of(
                selectedPos.begin(), selectedPos.end(), [this](std::string s) { return elementIndexMap.count(s) > 0; })) {
            uint64_t maxSize = 0;
            uint64_t elemCount = 0;
            for (auto s : selectedPos) {
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
    if (std::none_of(selectedNormal.begin(), selectedNormal.end(), [](std::string s) { return s.empty(); })) {
        if (std::any_of(selectedNormal.begin(), selectedNormal.end(),
                [this](std::string s) { return elementIndexMap.count(s) > 0; })) {
            uint64_t maxSize = 0;
            uint64_t elemCount = 0;
            for (auto s : selectedNormal) {
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
    if (std::none_of(selectedColor.begin(), selectedColor.end(), [](std::string s) { return s.empty(); })) {
        if (std::any_of(selectedColor.begin(), selectedColor.end(),
                [this](std::string s) { return elementIndexMap.count(s) > 0; })) {
            uint64_t maxSize = 0;
            uint64_t elemCount = 0;
            for (auto s : selectedColor) {
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
    if (!selectedIndices.empty()) {
        if (this->elementIndexMap.count(selectedIndices) > 0) {
            auto idx = this->elementIndexMap[selectedIndices];
            auto size = this->propertySizes[idx.first][idx.second];
            auto elemCount =
                this->elementCount[idx.first] * 3; // TODO modify this to work with anything besides triangles
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

    auto const flt_max = std::numeric_limits<float>::max();
    auto const flt_min = std::numeric_limits<float>::lowest();
    this->boundingBox.Set(flt_max, flt_max, flt_max, flt_min, flt_min, flt_min);
    float* bbPointer = const_cast<float*>(boundingBox.PeekBounds()); // hackedihack

    if (this->hasBinaryFormat) {
        // vector storing the read data seperately
        std::vector<std::vector<char>> readData(this->elementCount.size());
        for (size_t i = 0; i < readData.size(); i++) {
            // maybe TODO: skip unnecessary elements
            uint64_t readsize = elementSizes[i];
            if (elementIndexMap.count(selectedIndices) > 0) {
                auto idx = elementIndexMap[selectedIndices];
                if (idx.first == i && listFlags[idx.first][idx.second]) {
                    // we assume that faces are always triangular
                    readsize = listSizes[idx.first][idx.second] + 3 * propertySizes[idx.first][idx.second];
                }
            }
            readData[i].resize(elementCount[i] * readsize);
            auto ms = readData[i].max_size();
            instream.read(reinterpret_cast<char*>(readData[i].data()), elementCount[i] * readsize);
            if (instream.fail()) {
                vislib::sys::Log::DefaultLog.WriteError(
                    "Reading of the field with index %i failed", static_cast<int>(i));
                this->clearAllFields();
                return false;
            }
        }

        // copy the data into the vectors (this is necessary because the data may be interleaved, which is not always
        // the case)
        // this could be done partially in parallel
        for (size_t i = 0; i < selectedPos.size(); i++) {
            if (elementIndexMap.count(selectedPos[i]) > 0) {
                auto idx = elementIndexMap[selectedPos[i]];
                auto elemSize = elementSizes[idx.first];
                auto size = propertySizes[idx.first][idx.second];
                auto stride = propertyStrides[idx.first][idx.second];
                if (posPointers.pos_float != nullptr) {
                    for (size_t v = 0; v < vertex_count; v++) {
                        std::memcpy(
                            &posPointers.pos_float[3 * v + i], &readData[idx.first][v * elemSize + stride], size);
                        if (posPointers.pos_float[3 * v + i] < bbPointer[i]) {
                            bbPointer[i] = posPointers.pos_float[3 * v + i];
                        }
                        if (posPointers.pos_float[3 * v + i] > bbPointer[i + 3]) {
                            bbPointer[i + 3] = posPointers.pos_float[3 * v + i];
                        }
                    }
                }
                if (posPointers.pos_double != nullptr) {
                    for (size_t v = 0; v < vertex_count; v++) {
                        std::memcpy(
                            &posPointers.pos_double[3 * v + i], &readData[idx.first][v * elemSize + stride], size);
                        if (posPointers.pos_double[3 * v + i] < bbPointer[i]) {
                            bbPointer[i] = static_cast<float>(posPointers.pos_double[3 * v + i]);
                        }
                        if (posPointers.pos_double[3 * v + i] > bbPointer[i + 3]) {
                            bbPointer[i + 3] = static_cast<float>(posPointers.pos_double[3 * v + i]);
                        }
                    }
                }
            }
        }

        for (size_t i = 0; i < selectedNormal.size(); i++) {
            if (elementIndexMap.count(selectedNormal[i]) > 0) {
                auto idx = elementIndexMap[selectedNormal[i]];
                auto elemSize = elementSizes[idx.first];
                auto size = propertySizes[idx.first][idx.second];
                auto stride = propertyStrides[idx.first][idx.second];
                if (normalPointers.norm_float != nullptr) {
                    for (size_t v = 0; v < vertex_count; v++) {
                        std::memcpy(
                            &normalPointers.norm_float[3 * v + i], &readData[idx.first][v * elemSize + stride], size);
                    }
                }
                if (normalPointers.norm_double != nullptr) {
                    for (size_t v = 0; v < vertex_count; v++) {
                        std::memcpy(
                            &normalPointers.norm_double[3 * v + i], &readData[idx.first][v * elemSize + stride], size);
                    }
                }
            }
        }

        for (size_t i = 0; i < selectedColor.size(); i++) {
            if (i > 2) break;
            if (elementIndexMap.count(selectedColor[i]) > 0) {
                auto idx = elementIndexMap[selectedColor[i]];
                auto elemSize = elementSizes[idx.first];
                auto size = propertySizes[idx.first][idx.second];
                auto stride = propertyStrides[idx.first][idx.second];
                if (colorPointers.col_uchar != nullptr) {
                    for (size_t v = 0; v < vertex_count; v++) {
                        std::memcpy(
                            &colorPointers.col_uchar[3 * v + i], &readData[idx.first][v * elemSize + stride], size);
                    }
                }
                if (colorPointers.col_float != nullptr) {
                    for (size_t v = 0; v < vertex_count; v++) {
                        std::memcpy(
                            &colorPointers.col_float[3 * v + i], &readData[idx.first][v * elemSize + stride], size);
                    }
                }
                if (colorPointers.col_double != nullptr) {
                    for (size_t v = 0; v < vertex_count; v++) {
                        std::memcpy(
                            &colorPointers.col_double[3 * v + i], &readData[idx.first][v * elemSize + stride], size);
                    }
                }
            }
        }

        if (elementIndexMap.count(selectedIndices) > 0) {
            auto idx = elementIndexMap[selectedIndices];
            auto elemSize = elementSizes[idx.first];
            auto size = propertySizes[idx.first][idx.second];
            auto stride = propertyStrides[idx.first][idx.second];
            auto listStartSize = listSizes[idx.first][idx.second];
            auto totSize = listStartSize + 3 * size;
            if (facePointers.face_uchar != nullptr) {
                for (size_t f = 0; f < face_count; f++) {
                    std::memcpy(&facePointers.face_uchar[f * 3],
                        &readData[idx.first][f * totSize + stride + listStartSize], 3 * size);
                }
            }
            if (facePointers.face_u16 != nullptr) {
                for (size_t f = 0; f < face_count; f++) {
                    std::memcpy(&facePointers.face_u16[f * 3],
                        &readData[idx.first][f * totSize + stride + listStartSize], 3 * size);
                }
            }
            if (facePointers.face_u32 != nullptr) {
                for (size_t f = 0; f < face_count; f++) {
                    std::memcpy(&facePointers.face_u32[f * 3],
                        &readData[idx.first][f * totSize + stride + listStartSize], 3 * size);
                }
            }
        }
        // change endianness if necessary
        if (!isLittleEndian) {
            // this could be done in parallel
            if (posPointers.pos_float != nullptr) {
                for (size_t v = 0; v < vertex_count; v++) {
                    changeEndianness(posPointers.pos_float[3 * v + 0]);
                    changeEndianness(posPointers.pos_float[3 * v + 1]);
                    changeEndianness(posPointers.pos_float[3 * v + 2]);
                }
            }
            if (posPointers.pos_double != nullptr) {
                for (size_t v = 0; v < vertex_count; v++) {
                    changeEndianness(posPointers.pos_double[3 * v + 0]);
                    changeEndianness(posPointers.pos_double[3 * v + 1]);
                    changeEndianness(posPointers.pos_double[3 * v + 2]);
                }
            }
            if (normalPointers.norm_float != nullptr) {
                for (size_t v = 0; v < vertex_count; v++) {
                    changeEndianness(normalPointers.norm_float[3 * v + 0]);
                    changeEndianness(normalPointers.norm_float[3 * v + 1]);
                    changeEndianness(normalPointers.norm_float[3 * v + 2]);
                }
            }
            if (normalPointers.norm_double != nullptr) {
                for (size_t v = 0; v < vertex_count; v++) {
                    changeEndianness(normalPointers.norm_double[3 * v + 0]);
                    changeEndianness(normalPointers.norm_double[3 * v + 1]);
                    changeEndianness(normalPointers.norm_double[3 * v + 2]);
                }
            }
            if (colorPointers.col_uchar != nullptr) {
                for (size_t v = 0; v < vertex_count; v++) {
                    changeEndianness(colorPointers.col_uchar[3 * v + 0]);
                    changeEndianness(colorPointers.col_uchar[3 * v + 1]);
                    changeEndianness(colorPointers.col_uchar[3 * v + 2]);
                }
            }
            if (colorPointers.col_float != nullptr) {
                for (size_t v = 0; v < vertex_count; v++) {
                    changeEndianness(colorPointers.col_float[3 * v + 0]);
                    changeEndianness(colorPointers.col_float[3 * v + 1]);
                    changeEndianness(colorPointers.col_float[3 * v + 2]);
                }
            }
            if (colorPointers.col_double != nullptr) {
                for (size_t v = 0; v < vertex_count; v++) {
                    changeEndianness(colorPointers.col_double[3 * v + 0]);
                    changeEndianness(colorPointers.col_double[3 * v + 1]);
                    changeEndianness(colorPointers.col_double[3 * v + 2]);
                }
            }
            if (facePointers.face_uchar != nullptr) {
                for (size_t f = 0; f < face_count; f++) {
                    changeEndianness(facePointers.face_uchar[3 * f + 0]);
                    changeEndianness(facePointers.face_uchar[3 * f + 1]);
                    changeEndianness(facePointers.face_uchar[3 * f + 2]);
                }
            }
            if (facePointers.face_u16 != nullptr) {
                for (size_t f = 0; f < face_count; f++) {
                    changeEndianness(facePointers.face_u16[3 * f + 0]);
                    changeEndianness(facePointers.face_u16[3 * f + 1]);
                    changeEndianness(facePointers.face_u16[3 * f + 2]);
                }
            }
            if (facePointers.face_u32 != nullptr) {
                for (size_t f = 0; f < face_count; f++) {
                    changeEndianness(facePointers.face_u32[3 * f + 0]);
                    changeEndianness(facePointers.face_u32[3 * f + 1]);
                    changeEndianness(facePointers.face_u32[3 * f + 2]);
                }
            }
        }
    } else { // ascii format
        std::string line;
        for (size_t elm = 0; elm < this->elementCount.size(); elm++) {
            // parse vertices
            if (icompare(elementNames[elm], selectedVertices)) {
                for (size_t i = 0; i < vertexCount; i++) {
                    if (std::getline(instream, line)) {
                        auto split = isplit(line);
                        for (size_t j = 0; j < selectedPos.size(); j++) {
                            if (elementIndexMap.count(selectedPos[j]) > 0) {
                                auto idx = elementIndexMap[selectedPos[j]];
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
                        for (size_t j = 0; j < selectedNormal.size(); j++) {
                            if (elementIndexMap.count(selectedNormal[j]) > 0) {
                                auto idx = elementIndexMap[selectedNormal[j]];
                                if (normalPointers.norm_float != nullptr) {
                                    normalPointers.norm_float[3 * i + j] = std::stof(split[idx.second]);
                                }
                                if (normalPointers.norm_double != nullptr) {
                                    normalPointers.norm_double[3 * i + j] = std::stod(split[idx.second]);
                                }
                            }
                        }
                        for (size_t j = 0; j < selectedColor.size(); j++) {
                            if (elementIndexMap.count(selectedColor[j]) > 0) {
                                auto idx = elementIndexMap[selectedColor[j]];
                                if (colorPointers.col_uchar != nullptr) {
                                    colorPointers.col_uchar[3 * i + j] =
                                        static_cast<unsigned char>(std::stoul(split[idx.second]));
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
            }
            // parse faces
            if (icompare(elementNames[elm], selectedFaces)) {
                for (size_t i = 0; i < faceCount; i++) {
                    if (std::getline(instream, line)) {
                        auto split = isplit(line);
                        if (elementIndexMap.count(selectedIndices)) {
                            uint64_t faceSize = static_cast<uint64_t>(std::stoul(split[0]));
                            if (faceSize != 3) {
                                vislib::sys::Log::DefaultLog.WriteError(
                                    "The PlyDataSource is currently only able to handle triangular faces");
                                return false;
                            }
                            for (size_t j = 1; j < faceSize + 1; j++) {
                                unsigned long bla = std::stoul(split[j]);
                                if (facePointers.face_uchar != nullptr) {
                                    facePointers.face_uchar[3 * i + j - 1] = static_cast<unsigned char>(bla);
                                }
                                if (facePointers.face_u16 != nullptr) {
                                    facePointers.face_u16[3 * i + j - 1] = static_cast<uint16_t>(bla);
                                }
                                if (facePointers.face_u32 != nullptr) {
                                    facePointers.face_u32[3 * i + j - 1] = static_cast<uint32_t>(bla);
                                }
                            }
                        }
                    } else {
                        vislib::sys::Log::DefaultLog.WriteError("Unexpected file ending during face parsing");
                        return false;
                    }
                }
            }
        }
    }

    instream.close();
    return true;
}

/*
 * io::PLYDataSource::filenameChanged
 */
bool io::PLYDataSource::filenameChanged(core::param::ParamSlot& slot) {

    using vislib::sys::Log;

    this->clearAllFields();
    this->data_hash++;

    instream.close();
    instream.open(filename.Param<core::param::FilePathParam>()->Value(), std::ios::binary);
    if (instream.fail()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to open PLY File \"%s\".",
            vislib::StringA(filename.Param<core::param::FilePathParam>()->Value()).PeekBuffer());
        this->clearAllFields();
        return true;
    }

    this->clearAllFields();

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

    this->guessedVertices = "";
    this->guessedFaces = "";
    this->guessedIndices = "";
    this->guessedPos.assign(3, "");
    this->guessedNormal.assign(3, "");
    this->guessedColor.assign(3, "");

    this->selectedVertices = "";
    this->selectedFaces = "";
    this->selectedIndices = "";
    this->selectedPos.assign(3, "");
    this->selectedNormal.assign(3, "");
    this->selectedColor.assign(3, "");

    this->elementIndexMap.clear();
    this->elementSizes.clear();
    this->elementCount.clear();
    this->elementNames.clear();
    this->propertySizes.clear();
    this->propertyStrides.clear();
    this->propertySigns.clear();
    this->listFlags.clear();
    this->listSigns.clear();
    this->listSizes.clear();
    this->hasBinaryFormat = false;
    this->isLittleEndian = true;
    this->data_offset = 0;
    this->face_count = 0;
    this->vertex_count = 0;

    tinyply::PlyFile plf;
    plf.parse_header(this->instream);

    uint64_t element_index = 0;
    uint64_t property_index = 0;
    uint64_t element_size = 0;

    for (auto e : plf.get_elements()) {
        this->vertElemSlot.Param<core::param::FlexEnumParam>()->AddValue(e.name);
        this->faceElemSlot.Param<core::param::FlexEnumParam>()->AddValue(e.name);
        if (icompare(e.name, "vertex")) {
            guessedVertices = e.name;
        }
        if (icompare(e.name, "face")) {
            guessedFaces = e.name;
        }
        this->elementCount.push_back(static_cast<uint64_t>(e.size));
        this->elementNames.push_back(e.name);
        this->propertySizes.push_back(std::vector<uint64_t>());
        this->propertyStrides.push_back(std::vector<uint64_t>());
        this->propertySigns.push_back(std::vector<bool>());
        this->listFlags.push_back(std::vector<bool>());
        this->listSigns.push_back(std::vector<bool>());
        this->listSizes.push_back(std::vector<uint64_t>());

        property_index = 0;
        element_size = 0;
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "element: %s, %u bytes", e.name.c_str(), e.size);
        for (auto p : e.properties) {
            Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "    property: %s %s %i", p.name.c_str(),
                tinyply::PropertyTable[p.propertyType].str.c_str(), tinyply::PropertyTable[p.propertyType].stride);
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
            if (icompare(p.name, "r") || icompare(p.name, "red") || icompare(p.name, "diffuse_red")) {
                guessedColor[0] = p.name;
            }
            if (icompare(p.name, "g") || icompare(p.name, "green") || icompare(p.name, "diffuse_green")) {
                guessedColor[1] = p.name;
            }
            if (icompare(p.name, "b") || icompare(p.name, "blue") || icompare(p.name, "diffuse_blue")) {
                guessedColor[2] = p.name;
            }
            if (icompare(p.name, "indices") || icompare(p.name, "vertex_indices") || icompare(p.name, "vertex_index")) {
                guessedIndices = p.name;
            }
            elementIndexMap[p.name] = std::make_pair(element_index, property_index);
            propertyStrides[propertyStrides.size() - 1].push_back(element_size);
            element_size += tinyTypeSize(p.propertyType);
            propertySizes[propertySizes.size() - 1].push_back(tinyTypeSize(p.propertyType));
            propertySigns[propertySigns.size() - 1].push_back(tinyIsSigned(p.propertyType));
            property_index++;

            listFlags[listFlags.size() - 1].push_back(p.isList);
            listSizes[listSizes.size() - 1].push_back(tinyTypeSize(p.listType));
            listSigns[listSigns.size() - 1].push_back(tinyIsSigned(p.listType));
        }
        elementSizes.push_back(element_size);
        element_index++;
    }

    this->selectedPos = guessedPos;
    this->selectedNormal = guessedNormal;
    this->selectedColor = guessedColor;
    this->selectedIndices = guessedIndices;
    this->selectedVertices = guessedVertices;
    this->selectedFaces = guessedFaces;

    if (!guessedVertices.empty()) {
        this->vertElemSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedVertices);
        if (std::none_of(guessedPos.begin(), guessedPos.end(), [](std::string s) { return s.empty(); })) {
            this->xPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedPos[0], false);
            this->yPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedPos[1], false);
            this->zPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedPos[2], false);
        }
        if (std::none_of(guessedNormal.begin(), guessedNormal.end(), [](std::string s) { return s.empty(); })) {
            this->nxPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedNormal[0], false);
            this->nyPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedNormal[1], false);
            this->nzPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedNormal[2], false);
        }
        if (std::none_of(guessedColor.begin(), guessedColor.end(), [](std::string s) { return s.empty(); })) {
            this->rPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedColor[0], false);
            this->gPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedColor[1], false);
            this->bPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedColor[2], false);
        }
    }
    if (!guessedFaces.empty()) {
        this->faceElemSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedFaces, false);
        if (!guessedIndices.empty()) {
            this->indexPropSlot.Param<core::param::FlexEnumParam>()->SetValue(guessedIndices, false);
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
            } else if (icompare(line.substr(7, 3), "bin")) {
                this->hasBinaryFormat = true;
                if (icompare(line.substr(14, 3), "lit")) {
                    this->isLittleEndian = true;
                } else if (icompare(line.substr(14, 3), "big")) {
                    this->isLittleEndian = false;
                } else {
                    vislib::sys::Log::DefaultLog.WriteWarn(
                        "Endianness could not be determined, assuming little endian");
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
    instream.close();

    // we have to have a clean parameter state
    this->resetParameterDirtyness();

    return true;
}

/*
 * io::PLYDataSource::fileUpdate
 */
bool io::PLYDataSource::fileUpdate(core::param::ParamSlot& slot) {
    using vislib::sys::Log;

    this->clearAllFields();

    instream.close();
    instream.open(filename.Param<core::param::FilePathParam>()->Value(), std::ios::binary);
    if (instream.fail()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Unable to open PLY File \"%s\".",
            vislib::StringA(filename.Param<core::param::FilePathParam>()->Value()).PeekBuffer());
        this->clearAllFields();
        return true;
    }

    for (size_t i = 0; i < this->guessedPos.size(); i++) {
        if (this->guessedPos[i].length() > 0) {
            if (i == 0) {
                this->selectedPos[i] = this->xPropSlot.Param<param::FlexEnumParam>()->Value();
            } else if (i == 1) {
                this->selectedPos[i] = this->yPropSlot.Param<param::FlexEnumParam>()->Value();
            } else if (i == 2) {
                this->selectedPos[i] = this->zPropSlot.Param<param::FlexEnumParam>()->Value();
            }
        }
    }

    for (size_t i = 0; i < this->guessedNormal.size(); i++) {
        if (this->guessedNormal[i].length() > 0) {
            if (i == 0) {
                this->selectedNormal[i] = this->nxPropSlot.Param<param::FlexEnumParam>()->Value();
            }
            else if (i == 1) {
                this->selectedNormal[i] = this->nyPropSlot.Param<param::FlexEnumParam>()->Value();
            }
            else if (i == 2) {
                this->selectedNormal[i] = this->nzPropSlot.Param<param::FlexEnumParam>()->Value();
            }
        }
    }

    for (size_t i = 0; i < this->guessedColor.size(); i++) {
        if (this->guessedColor[i].length() > 0) {
            if (i == 0) {
                this->selectedColor[i] = this->rPropSlot.Param<param::FlexEnumParam>()->Value();
            }
            else if (i == 1) {
                this->selectedColor[i] = this->gPropSlot.Param<param::FlexEnumParam>()->Value();
            }
            else if (i == 2) {
                this->selectedColor[i] = this->bPropSlot.Param<param::FlexEnumParam>()->Value();
            }
        }
    }

    if (this->guessedIndices.length() > 0) {
        this->selectedIndices = this->indexPropSlot.Param<param::FlexEnumParam>()->Value();
    }

    if (this->guessedVertices.length() > 0) {
        this->selectedVertices = this->vertElemSlot.Param<param::FlexEnumParam>()->Value();
    }

    if (this->guessedFaces.length() > 0) {
        this->selectedFaces = this->faceElemSlot.Param<param::FlexEnumParam>()->Value();
    }

    // we have to have a clean parameter state
    this->resetParameterDirtyness();
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
    // Always write data in the float pointer, since sphere data is only possible with float
    bool copyfloat = false;
    if (copyfloat && this->posPointers.pos_double != nullptr && this->posPointers.pos_float == nullptr) {
        this->posPointers.pos_float = new float[this->vertex_count * 3];
        for (size_t i = 0; i < this->vertex_count * 3; i++) {
            this->posPointers.pos_float[i] = static_cast<float>(this->posPointers.pos_double[i]);
        }
    }

    if (p.GetCount() > 0) {
        if (this->posPointers.pos_float != nullptr) {
            p.SetVertexData(SimpleSphericalParticles::VertexDataType::VERTDATA_FLOAT_XYZ, this->posPointers.pos_float);
        } else if (this->posPointers.pos_double != nullptr) {
            p.SetVertexData(
                SimpleSphericalParticles::VertexDataType::VERTDATA_DOUBLE_XYZ, this->posPointers.pos_double);
        } else {
            p.SetCount(0);
        }
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

    if (this->checkParameterDirtyness()) {
        this->fileUpdate(this->filename);
        this->data_hash++;
    }

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
        if (normalPointers.norm_float != nullptr) {
            if (colorPointers.col_uchar) {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_float,
                    normalPointers.norm_float, colorPointers.col_uchar, nullptr, false);
            } else if (colorPointers.col_float) {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_float,
                    normalPointers.norm_float, colorPointers.col_float, nullptr, false);
            } else if (colorPointers.col_double) {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_float,
                    normalPointers.norm_float, colorPointers.col_double, nullptr, false);
            } else {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_float,
                    normalPointers.norm_float, nullptr, nullptr, false);
            }
        } else if (normalPointers.norm_double != nullptr) {
            if (colorPointers.col_uchar) {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_float,
                    normalPointers.norm_double, colorPointers.col_uchar, nullptr, false);
            } else if (colorPointers.col_float) {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_float,
                    normalPointers.norm_double, colorPointers.col_float, nullptr, false);
            } else if (colorPointers.col_double) {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_float,
                    normalPointers.norm_double, colorPointers.col_double, nullptr, false);
            } else {
                this->mesh.SetVertexData(
                    static_cast<uint32_t>(this->vertex_count), posPointers.pos_float, nullptr, nullptr, nullptr, false);
            }
        } else {
            if (colorPointers.col_uchar) {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_float, nullptr,
                    colorPointers.col_uchar, nullptr, false);
            } else if (colorPointers.col_float) {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_float, nullptr,
                    colorPointers.col_float, nullptr, false);
            } else if (colorPointers.col_double) {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_float, nullptr,
                    colorPointers.col_double, nullptr, false);
            } else {
                this->mesh.SetVertexData(
                    static_cast<uint32_t>(this->vertex_count), posPointers.pos_float, nullptr, nullptr, nullptr, false);
            }
        }
    } else if (posPointers.pos_double != nullptr) {
        if (normalPointers.norm_float != nullptr) {
            if (colorPointers.col_uchar) {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_double,
                    normalPointers.norm_float, colorPointers.col_uchar, nullptr, false);
            } else if (colorPointers.col_float) {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_double,
                    normalPointers.norm_float, colorPointers.col_float, nullptr, false);
            } else if (colorPointers.col_double) {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_double,
                    normalPointers.norm_float, colorPointers.col_double, nullptr, false);
            } else {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_double,
                    normalPointers.norm_float, nullptr, nullptr, false);
            }
        } else if (normalPointers.norm_double != nullptr) {
            if (colorPointers.col_uchar) {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_double,
                    normalPointers.norm_double, colorPointers.col_uchar, nullptr, false);
            } else if (colorPointers.col_float) {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_double,
                    normalPointers.norm_double, colorPointers.col_float, nullptr, false);
            } else if (colorPointers.col_double) {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_double,
                    normalPointers.norm_double, colorPointers.col_double, nullptr, false);
            } else {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_double, nullptr,
                    nullptr, nullptr, false);
            }
        } else {
            if (colorPointers.col_uchar) {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_double, nullptr,
                    colorPointers.col_uchar, nullptr, false);
            } else if (colorPointers.col_float) {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_double, nullptr,
                    colorPointers.col_float, nullptr, false);
            } else if (colorPointers.col_double) {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_double, nullptr,
                    colorPointers.col_double, nullptr, false);
            } else {
                this->mesh.SetVertexData(static_cast<uint32_t>(this->vertex_count), posPointers.pos_double, nullptr,
                    nullptr, nullptr, false);
            }
        }
    } else {
        this->mesh.SetVertexData(0, nullptr, nullptr, nullptr, nullptr, false);
    }

    if (facePointers.face_uchar != nullptr) {
        this->mesh.SetTriangleData(static_cast<uint32_t>(this->face_count), facePointers.face_uchar, false);
    } else if (facePointers.face_u16 != nullptr) {
        this->mesh.SetTriangleData(static_cast<uint32_t>(this->face_count), facePointers.face_u16, false);
    } else if (facePointers.face_u32 != nullptr) {
        this->mesh.SetTriangleData(static_cast<uint32_t>(this->face_count), facePointers.face_u32, false);
        // this->mesh.SetTriangleData(0, facePointers.face_uchar, false);
    } else {
        this->mesh.SetTriangleData(0, facePointers.face_uchar, false);
    }

    c2->SetObjects(1, &this->mesh);

    return true;
}

/*
 * io::PLYDataSource::getMeshExtentCallback
 */
bool io::PLYDataSource::getMeshExtentCallback(core::Call& caller) {
    auto c2 = dynamic_cast<CallTriMeshData*>(&caller);
    if (c2 == nullptr) return false;

    if (this->checkParameterDirtyness()) {
        this->fileUpdate(this->filename);
        this->data_hash++;
    }

    if (!assertData()) return false;

    c2->AccessBoundingBoxes().Clear();
    c2->AccessBoundingBoxes().SetObjectSpaceBBox(this->boundingBox);
    c2->AccessBoundingBoxes().SetObjectSpaceClipBox(this->boundingBox);
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

/*
 * io::PLYDataSource::checkParameterDirtyness
 */
bool io::PLYDataSource::checkParameterDirtyness(void) {
    bool isDirty = false;

    isDirty = isDirty || this->vertElemSlot.IsDirty();
    isDirty = isDirty || this->faceElemSlot.IsDirty();
    isDirty = isDirty || this->xPropSlot.IsDirty();
    isDirty = isDirty || this->yPropSlot.IsDirty();
    isDirty = isDirty || this->zPropSlot.IsDirty();
    isDirty = isDirty || this->nxPropSlot.IsDirty();
    isDirty = isDirty || this->nyPropSlot.IsDirty();
    isDirty = isDirty || this->nzPropSlot.IsDirty();
    isDirty = isDirty || this->rPropSlot.IsDirty();
    isDirty = isDirty || this->gPropSlot.IsDirty();
    isDirty = isDirty || this->bPropSlot.IsDirty();
    isDirty = isDirty || this->iPropSlot.IsDirty();
    isDirty = isDirty || this->indexPropSlot.IsDirty();

    this->resetParameterDirtyness();

    return isDirty;
}

/*
 * io::PLYDataSource::resetParameterDirtyness
 */
void io::PLYDataSource::resetParameterDirtyness(void) {
    this->vertElemSlot.ResetDirty();
    this->faceElemSlot.ResetDirty();
    this->xPropSlot.ResetDirty();
    this->yPropSlot.ResetDirty();
    this->zPropSlot.ResetDirty();
    this->nxPropSlot.ResetDirty();
    this->nyPropSlot.ResetDirty();
    this->nzPropSlot.ResetDirty();
    this->rPropSlot.ResetDirty();
    this->gPropSlot.ResetDirty();
    this->bPropSlot.ResetDirty();
    this->iPropSlot.ResetDirty();
    this->indexPropSlot.ResetDirty();
}
