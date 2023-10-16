//
// VTILoader.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 12, 2013
//     Author: scharnkn
//

//#define VERBOSE // Toggle debugging messages

#include "VTILoader.h"
#include "Base64.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"
#include "protein_calls/VTIDataCall.h"
#include "vislib/Exception.h"
#include "vislib/String.h"
#include "vislib/sys/File.h"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <ctype.h>
#include <sstream>
#include <string>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::core::utility::log;

#pragma push_macro("min")
#undef min
#pragma push_macro("max")
#undef max

/*
 * VTILoader::VTILoader
 */
VTILoader::VTILoader()
        : AnimDataModule()
        , dataOutSlot("dataOut", "The slot providing the loaded data")
        , filenameSlot("vtiPath", "The path to the VTI data file to be loaded")
        , maxFramesSlot("maxFrames", "The maximum number of frames to be loaded")
        , frameStartSlot("frameStart", "The first frame to be loaded")
        , maxCacheSizeSlot("maxCacheSize", "The maximum size of the cache")
        , hash(0)
        , byteOrder(protein_calls::VTKImageData::VTI_LITTLE_ENDIAN)
        , version(0, 0)
        , wholeExtent(0, 0, 0, 0, 0, 0)
        , origin(0.0f, 0.0f, 0.0f)
        , spacing(0.0f, 0.0f, 0.0f)
        , nPieces(0)
        , filenamesDigits(0)
        , nFrames(0) {

    this->dataOutSlot.SetCallback(protein_calls::VTIDataCall::ClassName(),
        protein_calls::VTIDataCall::FunctionName(protein_calls::VTIDataCall::CallForGetData), &VTILoader::getData);
    this->dataOutSlot.SetCallback(protein_calls::VTIDataCall::ClassName(),
        protein_calls::VTIDataCall::FunctionName(protein_calls::VTIDataCall::CallForGetExtent), &VTILoader::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);

    this->filenameSlot.SetParameter(new core::param::FilePathParam(""));
    this->MakeSlotAvailable(&this->filenameSlot);

    this->maxFramesSlot.SetParameter(new core::param::IntParam(11, 1));
    this->MakeSlotAvailable(&this->maxFramesSlot);

    this->frameStartSlot.SetParameter(new core::param::IntParam(0, 0));
    this->MakeSlotAvailable(&this->frameStartSlot);

    this->maxCacheSizeSlot.SetParameter(new core::param::IntParam(10, 1));
    this->MakeSlotAvailable(&this->maxCacheSizeSlot);
}


/*
 * VTILoader::~VTILoader
 */
VTILoader::~VTILoader() {
    this->Release();
}


/*
 * VTILoader::create
 */
bool VTILoader::create() {
    return true;
}


/*
 * VTILoader::release
 */
void VTILoader::release() {
    // TODO What to do here?
}


/*
 * VTILoader::getData
 */
bool VTILoader::getData(core::Call& call) {


    // Get data call
    protein_calls::VTIDataCall* dc = dynamic_cast<protein_calls::VTIDataCall*>(&call);
    if (dc == NULL) {
        return false;
    }

    if (!(*dc)(protein_calls::VTIDataCall::CallForGetExtent)) {
        return false;
    }

    //    printf("Frame requested: %u\n", dc->FrameID()); // DEBUG

    if (dc->FrameID() >= this->FrameCount()) {
        Log::DefaultLog.WriteError(
            "%s: Frame %u requested (nFrames %u)", this->ClassName(), dc->FrameID(), this->FrameCount());
        return false;
    }

    // Request the frame
    Frame* fr = NULL;
    fr = dynamic_cast<VTILoader::Frame*>(this->requestLockedFrame(dc->FrameID()));

    if (fr == NULL) {
        return false;
    }

    //    printf("Vorher: %.16f\n", (const float*)(fr->GetPointDataByIdx(0, 0))[10000]);

    //    // DEBUG print texture values
    //    for (int i = 0; i < fr->GetPiecePointArraySize(0, 0); ++i) {
    //        printf("%.16f\n", (const float*)(fr->GetPointDataByIdx(0, 0))[i]);
    //    }
    // printf("Size %u\n", fr->GetPiecePointArraySize(0, 0));
    // END DEBUG

    // If the 'force' flag is set, check whether the frame number is correct,
    // if not re-request the frame
    if (dc->IsFrameForced()) {
        while (dc->FrameID() != fr->FrameNumber()) {
            dc->Unlock();
            int frameBefore = ((static_cast<int>(dc->FrameID() - 1) + static_cast<int>(this->FrameCount()))) %
                              static_cast<int>(this->FrameCount());

            // scharnkn:
            // Request the frame before the actual requested frame (modulo
            // framenumber) to trigger loading of the actually requested frame
            fr = dynamic_cast<VTILoader::Frame*>(this->requestLockedFrame(frameBefore));
            dc->SetUnlocker(new Unlocker(*fr));
            dc->Unlock();
            fr = dynamic_cast<VTILoader::Frame*>(this->requestLockedFrame(dc->FrameID()));
            dc->SetUnlocker(new Unlocker(*fr));
            if (fr == NULL) {
                return false;
            }
        }
    }

    // Set unlocker object for the frame
    dc->SetUnlocker(new Unlocker(*fr));

    // Set data of the call
    dc->SetData(fr->GetData());

    // Set the bounding ox of the frame's data
    Cubef bbox(fr->GetOrigin().GetX(), fr->GetOrigin().GetY(), fr->GetOrigin().GetZ(),
        fr->GetOrigin().GetX() + (fr->GetWholeExtent().Right()) * fr->GetSpacing().GetX(),
        fr->GetOrigin().GetY() + (fr->GetWholeExtent().Top()) * fr->GetSpacing().GetY(),
        fr->GetOrigin().GetZ() + (fr->GetWholeExtent().Front()) * fr->GetSpacing().GetZ());
    dc->AccessBoundingBoxes().Clear();
    dc->AccessBoundingBoxes().SetObjectSpaceBBox(bbox);
    dc->AccessBoundingBoxes().SetObjectSpaceClipBox(bbox);

    // Set data hash value
    dc->SetDataHash(this->hash);

    //printf("Frame loaded: %u\n", fr->FrameNumber()); // DEBUG

    //    // DEBUG print texture values
    //    for (int i = 0; i < dc->GetPiecePointArraySize(0, 0); ++i) {
    //        printf("%f\n", (const float*)(dc->GetPointDataByIdx(0, 0))[i]);
    //    }
    //    // END DEBUG
    //    printf("Get data: Size %u\n", dc->GetPiecePointArraySize(0, 0));

    return true;
}


/*
 * VTILoader::getExtent
 */
bool VTILoader::getExtent(core::Call& call) {

    //    printf("Getextent started\n");


    // Check parameters
    if (this->filenameSlot.IsDirty()) { // Files have to be loaded first
        this->filenameSlot.ResetDirty();
        if (!this->loadFile(
                this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str())) {
            printf("Loading file failed");
            return false;
        }
        this->hash++; // Change data hash
        this->hash = this->hash % 10;
    }
    if (this->maxFramesSlot.IsDirty()) {
        this->maxFramesSlot.ResetDirty();
        this->resetFrameCache();
        if (this->nFrames > 0) {
            // Set number of frames
            this->setFrameCount(std::min(
                static_cast<uint>(this->maxFramesSlot.Param<core::param::IntParam>()->Value()), this->nFrames));
            // Start the loading thread
            this->initFrameCache(this->maxCacheSizeSlot.Param<core::param::IntParam>()->Value());
        }
    }
    if (this->frameStartSlot.IsDirty()) {
        this->frameStartSlot.ResetDirty();
        // TODO Not used atm
    }
    if (this->maxCacheSizeSlot.IsDirty()) {
        this->maxCacheSizeSlot.ResetDirty();
        this->resetFrameCache();
        // Set number of frames
        this->setFrameCount(
            std::min(static_cast<uint>(this->maxFramesSlot.Param<core::param::IntParam>()->Value()), this->nFrames));

        // Start the loading thread
        this->initFrameCache(this->maxCacheSizeSlot.Param<core::param::IntParam>()->Value());
    }

    // Get data call
    protein_calls::VTIDataCall* dc = dynamic_cast<protein_calls::VTIDataCall*>(&call);
    if (dc == NULL) {
        return false;
    }

    // Request the frame
    Frame* fr = NULL;
    fr = dynamic_cast<VTILoader::Frame*>(this->requestLockedFrame(dc->FrameID()));

    if (fr == NULL) {
        return false;
    }

    // Set frame count
    dc->SetFrameCount(
        std::min(static_cast<uint>(this->maxFramesSlot.Param<core::param::IntParam>()->Value()), this->nFrames));

    //    dc->SetWholeExtent(fr->GetWholeExtent());
    ////    printf("wholeExtent %u %u %u %u %u %u\n",
    ////            this->wholeExtent.Left(),
    ////            this->wholeExtent.Bottom(),
    ////            this->wholeExtent.Back(),
    ////            this->wholeExtent.Right(),
    ////            this->wholeExtent.Top(),
    ////            this->wholeExtent.Front());
    //    dc->SetOrigin(fr->GetOrigin());
    ////    printf("origin %f %f %f\n",
    ////            this->origin.GetX(),
    ////            this->origin.GetY(),
    ////            this->origin.GetZ());
    //    dc->SetSpacing(fr->GetSpacing());
    ////    printf("spacing %f %f %f\n",
    ////            this->spacing.GetX(),
    ////            this->spacing.GetY(),
    ////            this->spacing.GetZ());
    //    dc->SetNumberOfPieces(this->nPieces);
    //    //printf("pieces %u\n", this->nPieces);

    //    vislib::math::Cuboid<float> bbox(
    //            this->origin.GetX(),
    //            this->origin.GetY(),
    //            this->origin.GetZ(),
    //            this->origin.GetX() + (this->wholeExtent.Right()-1)*this->spacing.GetX(),
    //            this->origin.GetY() + (this->wholeExtent.Top()-1)*this->spacing.GetY(),
    //            this->origin.GetZ() + (this->wholeExtent.Front()-1)*this->spacing.GetZ());
    //
    //    dc->AccessBoundingBoxes().Clear();
    //    dc->AccessBoundingBoxes().SetObjectSpaceBBox(bbox);
    //    dc->AccessBoundingBoxes().SetObjectSpaceClipBox(bbox);

    return true;
}


/*
 * VTILoader::loadFile
 */
bool VTILoader::loadFile(const vislib::StringA& filename) {


    // Test whether the filename is invalid or empty
    if (filename.IsEmpty()) {
        Log::DefaultLog.WriteInfo("%s: No file to load (filename empty)", this->ClassName());
        return true;
    }
    vislib::sys::File file;
    if (!file.Open(
            filename, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
        Log::DefaultLog.WriteError("%s: Unable to open file '%s'", this->ClassName(), filename.PeekBuffer());
        return false;
    }

    // Generate filename pattern
    this->filenamesDigits = 0;
    this->filenamesPrefix.Clear();
    this->filenamesPrefix = filename.Substring(0, filename.Find('.'));
    this->filenamesSuffix.Clear();
    this->filenamesSuffix = filename.Substring(filename.FindLast('.') + 1, filename.Length());

    if (filename.Find('.') != filename.FindLast('.')) { // File name contains at least two '.', file series is assumed
        this->filenamesDigits = filename.Length() - 2 - this->filenamesPrefix.Length() - this->filenamesSuffix.Length();
        // Search for frame files (determines maximum frame idx)
        bool search_done = false;
        vislib::StringA frameFile;
        this->nFrames = 0;
        while (!search_done) {
            std::stringstream ss;
            ss.width(this->filenamesDigits);
            ss.fill('0');
            std::string digits;
            ss << this->nFrames;
            frameFile = this->filenamesPrefix;
            frameFile.Append('.');
            frameFile.Append((ss.str()).c_str());
            frameFile.Append('.');
            frameFile.Append(this->filenamesSuffix);

            //            // Try to open frame files
            //            Log::DefaultLog.WriteMsg(
            //                    Log::LEVEL_INFO, "%s: Checking for %s ...",
            //                    this->ClassName(),
            //                    frameFile.PeekBuffer());

            if (file.Exists(frameFile)) {
                this->nFrames++;
            } else {
                search_done = true;
            }
        }

        vislib::StringA pattern;
        pattern = this->filenamesPrefix;
        // TODO Slash only works for linux
        pattern = pattern.Substring(pattern.FindLast("/") + 1);
        pattern.Append('.');
        for (uint i = 0; i < this->filenamesDigits; ++i) {
            pattern.Append("%");
        }
        pattern.Append('.');
        pattern.Append(this->filenamesSuffix);
#ifdef VERBOSE
        Log::DefaultLog.WriteInfo(
            "%s: %u frame file(s) found using pattern %s", this->ClassName(), this->nFrames, pattern.PeekBuffer());
#endif
    } else { // Single file
        this->nFrames = 1;
    }
    this->setFrameCount(this->nFrames);

    vislib::sys::File::FileSize fileSize =
        vislib::sys::File::GetSize(this->filenameSlot.Param<core::param::FilePathParam>()->Value().native().c_str());

#ifdef VERBOSE
    time_t t = clock(); // DEBUG
#endif                  // VERBOSE

#ifdef VERBOSE
    Log::DefaultLog.WriteInfo("%s: Parsing file '%s' (%u Bytes) ...", this->ClassName(),
        this->filenameSlot.Param<core::param::FilePathParam>()->Value().PeekBuffer(),
        fileSize); // DEBUG
#endif

    // Read data file to char buffer
    char* buffer = new char[(unsigned int) fileSize];
    if (!file.Open(this->filenameSlot.Param<core::param::FilePathParam>()->Value().native().c_str(),
            vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_EXCLUSIVE, vislib::sys::File::OPEN_ONLY))
        return false;
    file.Read(buffer, fileSize);
    file.Close();

    this->nPieces = 0;

    // Parse buffer
    char *pt = buffer, *pt_end;
    vislib::StringA entity;
    while (pt < buffer + fileSize - 2) {

        // Get next xml entity string
        while (*pt != '<') {
            pt++;
        }
        pt_end = pt;
        while (*pt_end != '>') {
            pt_end++;
        }
        entity = vislib::StringA(pt + 1, (int) (pt_end - pt));

        // Parse and store relevant attributes
        if (entity.StartsWith("VTKFile")) {

            vislib::StringA dataType, version, byteOrder;
            //printf("%s\n", entity.PeekBuffer()); // DEBUG
            protein_calls::VTKImageData::ByteOrder b;

            // type
            dataType = entity.Substring(entity.Find("type", 0) + 6);
            dataType = dataType.Substring(0, dataType.Find("\"", 0));
            if (dataType != vislib::StringA("ImageData")) {
                Log::DefaultLog.WriteError("%s: Unable to load file '%s' (wrong 'type' attribute)", this->ClassName(),
                    this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str(),
                    fileSize); // DEBUG
                return false;
            }
            //printf("    type        : %s\n", dataType.PeekBuffer()); // DEBUG

            // version
            version = entity.Substring(entity.Find("version", 0) + 9);
            version = version.Substring(0, version.Find("\"", 0));
            if (version.Length() > 3) {
                Log::DefaultLog.WriteError("%s: Unable to load file '%s' (wrong 'version' attribute)",
                    this->ClassName(),
                    this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str(),
                    fileSize); // DEBUG
                return false;
            }
            this->version.SetX(static_cast<int>(version.PeekBuffer()[0]) - 48);
            this->version.SetY(static_cast<int>(version.PeekBuffer()[2]) - 48);
            //printf("    version     : %s\n", version.PeekBuffer()); // DEBUG

            // byte order
            byteOrder = entity.Substring(entity.Find("byte_order", 0) + 12);
            byteOrder = byteOrder.Substring(0, byteOrder.Find("\"", 0));
            if (byteOrder == vislib::StringA("LittleEndian")) {
                b = protein_calls::VTKImageData::VTI_LITTLE_ENDIAN;
            } else if (byteOrder == vislib::StringA("BigEndian")) {
                b = protein_calls::VTKImageData::VTI_BIG_ENDIAN;
            } else {
                Log::DefaultLog.WriteError("%s: Unable to load file '%s' (wrong 'byte_order' attribute)",
                    this->ClassName(),
                    this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str(),
                    fileSize); // DEBUG
                return false;
            }
            //printf("    byte_order  : %s\n", byteOrder.PeekBuffer()); // DEBUG

        } else if (entity.StartsWith("ImageData")) {

            vislib::StringA extendStr, originStr, spacingStr;
            unsigned int extent[6];

            //printf("    -----------------\n"); // DEBUG
            //printf("%s\n", entity.PeekBuffer()); // DEBUG

            // whole extend
            extendStr = entity.Substring(entity.Find("WholeExtent", 0) + 13);
            extendStr = extendStr.Substring(0, extendStr.Find("\"", 0));
            extent[0] = string2int(extendStr.Substring(0, extendStr.Find(' ')));
            extendStr = extendStr.Substring(extendStr.Find(' ') + 1);
            extent[1] = string2int(extendStr.Substring(0, extendStr.Find(' ')));
            extendStr = extendStr.Substring(extendStr.Find(' ') + 1);
            extent[2] = string2int(extendStr.Substring(0, extendStr.Find(' ')));
            extendStr = extendStr.Substring(extendStr.Find(' ') + 1);
            extent[3] = string2int(extendStr.Substring(0, extendStr.Find(' ')));
            extendStr = extendStr.Substring(extendStr.Find(' ') + 1);
            extent[4] = string2int(extendStr.Substring(0, extendStr.Find(' ')));
            extendStr = extendStr.Substring(extendStr.Find(' ') + 1);
            extent[5] = string2int(extendStr);
            // printf("    Whole extend : %u %u %u %u %u %u\n",
            //        extent[0], extent[1],
            //        extent[2], extent[3],
            //        extent[4], extent[5]); // DEBUG

            this->wholeExtent.Set(extent[0], extent[2], extent[4], extent[1], extent[3], extent[5]);

            // origin
            originStr = entity.Substring(entity.Find("Origin", 0) + 8);
            originStr = originStr.Substring(0, originStr.Find("\"", 0));
            this->origin[0] = string2float(originStr.Substring(0, originStr.Find(' ')));
            originStr = originStr.Substring(originStr.Find(' ') + 1);
            this->origin[1] = string2float(originStr.Substring(0, originStr.Find(' ')));
            originStr = originStr.Substring(originStr.Find(' ') + 1);
            this->origin[2] = string2float(originStr);
            //printf("    Origin       : %f %f %f\n", this->origin[0],
            //       this->origin[1], this->origin[2]); // DEBUG

            // spacing
            spacingStr = entity.Substring(entity.Find("Spacing", 0) + 9);
            spacingStr = spacingStr.Substring(0, spacingStr.Find("\"", 0));
            this->spacing[0] = string2float(spacingStr.Substring(0, spacingStr.Find(' ')));
            spacingStr = spacingStr.Substring(spacingStr.Find(' ') + 1);
            this->spacing[1] = string2float(spacingStr.Substring(0, spacingStr.Find(' ')));
            spacingStr = spacingStr.Substring(spacingStr.Find(' ') + 1);
            this->spacing[2] = string2float(spacingStr);
            //printf("    Spacing      : %f %f %f\n", this->spacing[0],
            //        this->spacing[1], this->spacing[2]); // DEBUG

        } else if (entity.StartsWith("Piece")) {
            //printf("    -----------------\n"); // DEBUG
            this->nPieces++;
        }

        pt = pt_end + 1;
        entity.Clear();
    }
    delete[] buffer;

#ifdef VERBOSE
    Log::DefaultLog.WriteInfo("%s: ... done (%f s), found %u pieces", this->ClassName(),
        (double(clock() - t) / double(CLOCKS_PER_SEC)), this->nPieces); // DEBUG
#endif                                                                  // VERBOSE

    // Set number of frames
    this->setFrameCount(
        std::min(static_cast<uint>(this->maxFramesSlot.Param<core::param::IntParam>()->Value()), this->nFrames));

    // Start the loading thread
    this->initFrameCache(this->maxCacheSizeSlot.Param<core::param::IntParam>()->Value());

    return true;
}


/*
 * VTILoader::readDataAscii2Float
 */
void VTILoader::readDataAscii2Float(char* buffIn, float* buffOut, SIZE_T sizeOut) {

    char num[64];
    char *pt_nums, *pt_end = buffIn;
    unsigned int numCount = 0;
    while (numCount < sizeOut) {
        while (isspace(*pt_end)) { // Omit whitespace chars
            pt_end++;
        }
        pt_nums = pt_end;
        while (!isspace(*pt_end)) { // Get chars containing number
            pt_end++;
        }
        memset(num, 0, 64);
        memcpy(num, pt_nums, pt_end - pt_nums + 1);
        buffOut[numCount] = static_cast<float>(atof(num));
        numCount++;
        while (isspace(*pt_end)) { // Omit whitespace chars
            pt_end++;
        }
    }
}


/*
 * VTILoader::readDataBinary2Float
 */
void VTILoader::readDataBinary2Float(char* buffIn, float* buffOut, // TODO Other data types
    SIZE_T sizeOut) {

    char* pt_end = buffIn;

    while (isspace(*pt_end)) { // Omit whitespace chars
        pt_end++;
    }
    // TODO Decode first 8 bytes (=size of the data following in bytes)
    //pt_end +=8;
    // Decode actual data
    Base64::Decode(pt_end, (char*) buffOut, sizeOut * sizeof(float));
}


/*
 * VTILoader::constructFrame
 */
view::AnimDataModule::Frame* VTILoader::constructFrame() const {
    Frame* f = new Frame(*const_cast<VTILoader*>(this));
    //    f->SetNumberOfPieces(this->nPieces);
    return f;
}


/*
 * VTILoader::loadFrame
 */
void VTILoader::loadFrame(view::AnimDataModule::Frame* frame, unsigned int idx) {

    using namespace vislib;

    VTILoader::Frame* fr = dynamic_cast<VTILoader::Frame*>(frame);
    if (fr == NULL)
        return;

    // Set the frame index of the frame
    fr->SetFrameIdx(idx);

    // Generate filename based on frame idx and pattern
    vislib::StringA frameFile;
    if (this->filenamesDigits == 0) {
        frameFile = this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str();
    } else {
        std::stringstream ss;
        ss.width(this->filenamesDigits);
        ss.fill('0');
        std::string digits;
        ss << idx;
        frameFile = this->filenamesPrefix;
        frameFile.Append('.');
        frameFile.Append(ss.str().c_str());
        frameFile.Append('.');
        frameFile.Append(this->filenamesSuffix);
    }

    // Try to open the current frames file
    vislib::sys::File file;
    vislib::sys::File::FileSize fileSize = vislib::sys::File::GetSize(frameFile);

#if defined(VERBOSE)
    time_t t = clock();
#endif // defined(VERBOSE)

    // Read data file to char buffer
    char* buffer = new char[(unsigned int) fileSize];
    file.Open(
        frameFile, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_EXCLUSIVE, vislib::sys::File::OPEN_ONLY);
    file.Read(buffer, fileSize);
    file.Close();

    uint pieceCounter = 0;

    // Parse buffer
    char *pt = buffer, *pt_end;
    vislib::StringA entity;
    while (pt < buffer + fileSize - 2) {

        // Get next xml entity string
        while (*pt != '<') {
            pt++;
        }
        pt_end = pt;
        while (*pt_end != '>') {
            pt_end++;
        }
        entity = vislib::StringA(std::string(pt + 1, pt_end - pt).c_str());

        // Parse and store pieces
        if (entity.StartsWith("ImageData")) {

            vislib::StringA extendStr, originStr, spacingStr;
            unsigned int extent[6];

            //printf("    -----------------\n"); // DEBUG
            //printf("%s\n", entity.PeekBuffer()); // DEBUG

            // whole extend
            vislib::math::Cuboid<uint> wholeExtent;
            extendStr = entity.Substring(entity.Find("WholeExtent", 0) + 13);
            extendStr = extendStr.Substring(0, extendStr.Find("\"", 0));
            extent[0] = string2int(extendStr.Substring(0, extendStr.Find(' ')));
            extendStr = extendStr.Substring(extendStr.Find(' ') + 1);
            extent[1] = string2int(extendStr.Substring(0, extendStr.Find(' ')));
            extendStr = extendStr.Substring(extendStr.Find(' ') + 1);
            extent[2] = string2int(extendStr.Substring(0, extendStr.Find(' ')));
            extendStr = extendStr.Substring(extendStr.Find(' ') + 1);
            extent[3] = string2int(extendStr.Substring(0, extendStr.Find(' ')));
            extendStr = extendStr.Substring(extendStr.Find(' ') + 1);
            extent[4] = string2int(extendStr.Substring(0, extendStr.Find(' ')));
            extendStr = extendStr.Substring(extendStr.Find(' ') + 1);
            extent[5] = string2int(extendStr);
            //             printf("    Whole extend : %u %u %u %u %u %u\n",
            //                    extent[0], extent[1],
            //                    extent[2], extent[3],
            //                    extent[4], extent[5]); // DEBUG

            wholeExtent.Set(extent[0], extent[2], extent[4], extent[1], extent[3], extent[5]);
            fr->SetWholeExtent(wholeExtent);

            // origin
            Vec3f origin;
            originStr = entity.Substring(entity.Find("Origin", 0) + 8);
            originStr = originStr.Substring(0, originStr.Find("\"", 0));
            origin[0] = string2float(originStr.Substring(0, originStr.Find(' ')));
            originStr = originStr.Substring(originStr.Find(' ') + 1);
            origin[1] = string2float(originStr.Substring(0, originStr.Find(' ')));
            originStr = originStr.Substring(originStr.Find(' ') + 1);
            origin[2] = string2float(originStr);
            //            printf("    Origin       : %f %f %f\n", origin[0],
            //                    origin[1], origin[2]); // DEBUG
            fr->SetOrigin(origin);

            // spacing
            Vec3f spacing;
            spacingStr = entity.Substring(entity.Find("Spacing", 0) + 9);
            spacingStr = spacingStr.Substring(0, spacingStr.Find("\"", 0));
            spacing[0] = string2float(spacingStr.Substring(0, spacingStr.Find(' ')));
            spacingStr = spacingStr.Substring(spacingStr.Find(' ') + 1);
            spacing[1] = string2float(spacingStr.Substring(0, spacingStr.Find(' ')));
            spacingStr = spacingStr.Substring(spacingStr.Find(' ') + 1);
            spacing[2] = string2float(spacingStr);
            //            printf("    Spacing      : %f %f %f\n", this->spacing[0],
            //                    this->spacing[1], this->spacing[2]); // DEBUG
            fr->SetSpacing(spacing);

        } else if (entity.StartsWith("Piece")) {

            pieceCounter++;
            fr->SetNumberOfPieces(pieceCounter);

            vislib::math::Vector<int, 6> extent;
            vislib::StringA extentStr;

            extentStr = entity.Substring(entity.Find("Extent", 0) + 8);
            extentStr = extentStr.Substring(0, extentStr.Find("\"", 0));
            extent[0] = static_cast<int>(string2float(extentStr.Substring(0, extentStr.Find(' '))));
            extentStr = extentStr.Substring(extentStr.Find(' ') + 1);
            extent[1] = static_cast<int>(string2float(extentStr.Substring(0, extentStr.Find(' '))));
            extentStr = extentStr.Substring(extentStr.Find(' ') + 1);
            extent[2] = static_cast<int>(string2float(extentStr.Substring(0, extentStr.Find(' '))));
            extentStr = extentStr.Substring(extentStr.Find(' ') + 1);
            extent[3] = static_cast<int>(string2float(extentStr.Substring(0, extentStr.Find(' '))));
            extentStr = extentStr.Substring(extentStr.Find(' ') + 1);
            extent[4] = static_cast<int>(string2float(extentStr.Substring(0, extentStr.Find(' '))));
            extentStr = extentStr.Substring(extentStr.Find(' ') + 1);
            extent[5] = static_cast<int>(string2float(extentStr));
            //            printf("    Piece extend : %i %i %i %i %i %i\n", extent[0],
            //                    extent[1], extent[2], extent[3], extent[4], extent[5]); // DEBUG

            // TODO Use string2int instead

            // Set the pieces extent
            fr->SetPieceExtent(
                pieceCounter - 1, Cubeu(extent[0], extent[1], extent[2], extent[3], extent[4], extent[5]));

        } else if (entity.StartsWith("DataArray")) {

            //printf("    -----------------\n"); // DEBUG

            // type
            vislib::StringA dataType;
            protein_calls::VTKImageData::DataArray::DataType t; // TODO Actually use data type
            protein_calls::VTKImageData::DataFormat f;
            dataType = entity.Substring(entity.Find("type", 0) + 6);
            dataType = dataType.Substring(0, dataType.Find("\"", 0));
            if (dataType == vislib::StringA("Float32")) {
                t = protein_calls::VTKImageData::DataArray::VTI_FLOAT;
            } else {
                Log::DefaultLog.WriteError("%s: Unable to load file '%s' (wrong data type in data array)",
                    this->ClassName(),
                    this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str(),
                    fileSize); // DEBUG
                return;
            }
            //printf("    type  : %s\n", dataType.PeekBuffer()); // DEBUG

            // Name
            vislib::StringA name;
            name = entity.Substring(entity.Find("Name", 0) + 6);
            name = name.Substring(0, name.Find("\"", 0));
            //            printf("    Name  : %s\n", name.PeekBuffer()); // DEBUG

            // Format
            vislib::StringA format;
            format = entity.Substring(entity.Find("format", 0) + 8);
            format = format.Substring(0, format.Find("\"", 0));
            if (format == vislib::StringA("ascii")) {
                f = protein_calls::VTKImageData::VTISOURCE_ASCII;
            } else if (format == vislib::StringA("binary")) {
                f = protein_calls::VTKImageData::VTISOURCE_BINARY;
            } else {
                Log::DefaultLog.WriteError("%s: Unable to load file '%s' (unsupported data format %s)",
                    this->ClassName(),
                    this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str(),
                    format.PeekBuffer()); // DEBUG
                return;
            }
            //printf("    Format: %s\n", format.PeekBuffer()); // DEBUG

            // Minimum value
            vislib::StringA rangeMin;
            float min;
            rangeMin = entity.Substring(entity.Find("RangeMin", 0) + 10);
            rangeMin = rangeMin.Substring(0, rangeMin.Find("\"", 0));
            min = this->string2float(rangeMin);
            //            printf("    Min  : %f\n", min); // DEBUG

            // Maximum value
            vislib::StringA rangeMax;
            float max;
            rangeMax = entity.Substring(entity.Find("RangeMax", 0) + 10);
            rangeMax = rangeMax.Substring(0, rangeMax.Find("\"", 0));
            max = this->string2float(rangeMax);
            //            printf("    Max  : %f\n", max); // DEBUG

            // Try to get number of components
            // TODO look for a better, more robust solution! NumberOfCom is sometimes missing.
            vislib::StringA numComp;
            unsigned int numComponents = 0;
            StringA::Size numCompIdx = entity.Find("NumberOfComponents", 0);
            if (numCompIdx != StringA::INVALID_POS) {
                numComp = entity.Substring(numCompIdx + 20);
                numComp = numComp.Substring(0, numComp.Find("\"", 0));
                numComponents = atoi(numComp.PeekBuffer());
            }
            if (numComponents == 0)
                numComponents = 1;

            pt_end++; // Omit next '>'

            // Get overall grid size of the current piece
            uint gridSize = (fr->GetPieceExtent(pieceCounter - 1).Width() + 1) *
                            (fr->GetPieceExtent(pieceCounter - 1).Depth() + 1) *
                            (fr->GetPieceExtent(pieceCounter - 1).Height() + 1) * numComponents;

            float* data = 0;
            if (f == protein_calls::VTKImageData::VTISOURCE_ASCII) {
                data = new float[gridSize];
                this->readDataAscii2Float(pt_end, data, gridSize);
                fr->SetPointData((const char*) data, min, max, protein_calls::VTKImageData::DataArray::VTI_FLOAT, name,
                    1, pieceCounter - 1); // TODO Use real ID AND NUMBER OF COMPONENTS!!
            } else if (f == protein_calls::VTKImageData::VTISOURCE_BINARY) {
                data = new float[gridSize + 1];
                this->readDataBinary2Float(pt_end, data, gridSize);
                const float* dataplus = data + 1;
                fr->SetPointData((const char*) (dataplus), min, max, protein_calls::VTKImageData::DataArray::VTI_FLOAT,
                    name, numComponents, pieceCounter - 1); // TODO Use real ID AND NUMBER OF COMPONENTS!!

                //                // DEBUG print texture values
                //                for (int i = 0; i < fr->GetPiecePointArraySize(0, 0); ++i) {
                //                    printf("%i: %.16f\n", i, data[i]);
                //                }
                //                // END DEBUG
            }
            if (data)
                delete[] data;
        }

        pt = pt_end + 1;
        entity.Clear();
    }
    delete[] buffer;

#if defined(VERBOSE)
    vislib::StringA frameFileShortPath;
#if defined(_WIN32)
    frameFileShortPath = frameFile.Substring(frameFile.FindLast('\\') + 1, frameFile.Length() - 1);
#else
    frameFileShortPath = frameFile.Substring(frameFile.FindLast('/') + 1, frameFile.Length() - 1);
#endif
    Log::DefaultLog.WriteInfo("%s: '%s' done (%u Bytes, %f s)", this->ClassName(), frameFileShortPath.PeekBuffer(),
        fileSize,
        (double(clock() - t) / double(CLOCKS_PER_SEC))); // DEBUG
#endif                                                   // defined(VERBOSE)


    //    // DEBUG print texture values
    //    for (int i = 0; i < fr->GetPiecePointArraySize(0, 0); ++i) {
    //        printf("%i: %.16f\n", i, ((const float*)(fr->GetPointDataByIdx(0, 0)))[i]);
    //    }
    //    printf("Size %u\n", fr->GetPiecePointArraySize(0, 0));
    //    // END DEBUG
}


/*
 * VTILoader::string2int
 */
int VTILoader::string2int(vislib::StringA str) {
    using namespace std;

    std::stringstream ss(std::string(str.PeekBuffer()));
    int i;

    if ((ss >> i).fail()) {
        //error
    }
    return i;
}


/*
 * VTILoader::string2float
 */
float VTILoader::string2float(vislib::StringA str) {
    using namespace std;

    std::stringstream ss(std::string(str.PeekBuffer()));
    float f;

    if ((ss >> f).fail()) {
        //error
    }

    return f;
}


/*
 * VTILoader::Frame::Frame
 */
VTILoader::Frame::Frame(megamol::core::view::AnimDataModule& owner) : view::AnimDataModule::Frame(owner) {}


/*
 * VTILoader::Frame::~Frame
 */
VTILoader::Frame::~Frame() {
    // Release all data arrays of all pieces
    this->data.Release();
}
