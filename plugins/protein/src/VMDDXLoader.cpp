//
// VMDDXLoader.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: May 7, 2013
//     Author: scharnkn
//


#include "VMDDXLoader.h"
#include "Base64.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"
#include "protein_calls/VTIDataCall.h"
#include "vislib/Exception.h"
#include "vislib/String.h"
#include "vislib/sys/ASCIIFileBuffer.h"
#include <cmath>
#include <ctime>
#include <ctype.h>
#include <sstream>
#include <string>
//#include "vislib_vector_typedefs.h"
#include "vislib/math/Cuboid.h"
typedef vislib::math::Cuboid<float> Cubef;

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::core::utility::log;


/*
 * VMDDXLoader::VMDDXLoader
 */
VMDDXLoader::VMDDXLoader()
        : Module()
        , dataOutSlot("dataout", "The slot providing the loaded data")
        , filenameSlot("filename", "The path to the *.dx data file to be loaded")
        , hash(0)
        ,
        //        extent(0, 0, 0, 0, 0, 0),
        //        origin(0.0f, 0.0f, 0.0f),
        //        spacing(0.0f, 0.0f, 0.0f),
        filenamesDigits(0)
        , nFrames(-1)
//        min(0.0f),
//        max(0.0f)
{

    this->dataOutSlot.SetCallback(protein_calls::VTIDataCall::ClassName(),
        protein_calls::VTIDataCall::FunctionName(protein_calls::VTIDataCall::CallForGetData), &VMDDXLoader::getData);
    this->dataOutSlot.SetCallback(protein_calls::VTIDataCall::ClassName(),
        protein_calls::VTIDataCall::FunctionName(protein_calls::VTIDataCall::CallForGetExtent),
        &VMDDXLoader::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);

    this->filenameSlot.SetParameter(new core::param::FilePathParam(""));
    this->MakeSlotAvailable(&this->filenameSlot);
}


/*
 * VMDDXLoader::~VMDDXLoader
 */
VMDDXLoader::~VMDDXLoader() {
    this->Release();
}


/*
 * VMDDXLoader::create
 */
bool VMDDXLoader::create() {
    return true;
}


/*
 * VMDDXLoader::release
 */
void VMDDXLoader::release() {
    this->data.Release();
}


/*
 * VMDDXLoader::getData
 */
bool VMDDXLoader::getData(core::Call& call) {
    using namespace vislib::sys;

    // Get data call
    protein_calls::VTIDataCall* dc = dynamic_cast<protein_calls::VTIDataCall*>(&call);
    if (dc == NULL)
        return false;

    //    Log::DefaultLog.WriteInfo( "%s: Frame requested: %u",
    //            this->ClassName(),
    //            dc->FrameID()); // DEBUG

    // Generate filename based on frame idx and pattern
    vislib::StringA frameFile;
    if (this->filenamesDigits == 0) {
        frameFile = this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str();
    } else {
        std::stringstream ss;
        ss.width(this->filenamesDigits);
        ss.fill('0');
        std::string digits;
        ss << dc->FrameID();
        frameFile = this->filenamesPrefix;
        frameFile.Append('.');
        frameFile.Append((ss.str()).c_str());
        frameFile.Append('.');
        frameFile.Append(this->filenamesSuffix);
    }

    if (!this->loadFile(frameFile)) {
        return false;
    }

    // Set image data
    dc->SetData(&this->imgdata);

    Cubef bbox(this->imgdata.GetOrigin().GetX(), this->imgdata.GetOrigin().GetY(), this->imgdata.GetOrigin().GetZ(),
        this->imgdata.GetOrigin().GetX() +
            (this->imgdata.GetWholeExtent().Right() - 1) * this->imgdata.GetSpacing().GetX(),
        this->imgdata.GetOrigin().GetY() +
            (this->imgdata.GetWholeExtent().Top() - 1) * this->imgdata.GetSpacing().GetY(),
        this->imgdata.GetOrigin().GetZ() +
            (this->imgdata.GetWholeExtent().Front() - 1) * this->imgdata.GetSpacing().GetZ());

    dc->AccessBoundingBoxes().Clear();
    dc->AccessBoundingBoxes().SetObjectSpaceBBox(bbox);
    dc->AccessBoundingBoxes().SetObjectSpaceClipBox(bbox);

    // Set data hash value
    dc->SetDataHash(this->hash);

    return true;
}


/*
 * VMDDXLoader::getExtent
 */
bool VMDDXLoader::getExtent(core::Call& call) {

    using namespace vislib::sys;

    // Check parameters
    if (this->filenameSlot.IsDirty()) { // Files have to be loaded first
        this->filenameSlot.ResetDirty();
        this->scanFolder(); // (Re-)scan the folder
        this->hash++;       // Change data hash
        this->hash = this->hash % 10;
    }

    // Get data call
    protein_calls::VTIDataCall* dc = dynamic_cast<protein_calls::VTIDataCall*>(&call);
    if (dc == NULL)
        return false;

    // Set frame count
    dc->SetFrameCount(this->nFrames);

    return true;
}


/*
 * VMDDXLoader::loadFile
 */
bool VMDDXLoader::loadFile(const vislib::StringA& filename) {
    using namespace vislib;

    vislib::sys::File testFile;
    vislib::sys::ASCIIFileBuffer file(vislib::sys::ASCIIFileBuffer::PARSING_WORDS);
    StringA word;
    uint lineCnt;

    // Test whether the filename is invalid or empty
    if (filename.IsEmpty()) {
        Log::DefaultLog.WriteInfo("%s: No file to load (filename empty)", this->ClassName());
        return true;
    }

    if (!file.LoadFile(filename)) {
        Log::DefaultLog.WriteError("%s: Unable to open file '%s'", this->ClassName(), filename.PeekBuffer());
        return false;
    }

    vislib::sys::File::FileSize fileSize = vislib::sys::File::GetSize(filename);

    time_t t = clock(); // DEBUG

    Log::DefaultLog.WriteInfo("%s: Parsing file '%s' (%u Bytes) ...", this->ClassName(), filename.PeekBuffer(),
        fileSize); // DEBUG

    // File successfully loaded, get extent spacing and origin
    lineCnt = 0;
    double min = 0.0f;
    double max = 0.0f;
    bool readFloatData = false;
    uint floatDataCounter = 0;
    while (lineCnt < file.Count()) {

        if (file[lineCnt].Count() >= 4) {
            if (StringA(file[lineCnt].Word(3)).Equals("gridpositions")) {
                // Get extent of the data
                this->imgdata.SetWholeExtent(Cubeu(0, 0, 0, this->string2int(TString(file[lineCnt].Word(5))) - 1,
                    this->string2int(TString(file[lineCnt].Word(6))) - 1,
                    this->string2int(TString(file[lineCnt].Word(7))) - 1));
            }
        }

        if (file[lineCnt].Count() >= 1) {
            if (StringA(file[lineCnt].Word(0)).Equals("origin")) {
                // Get origin of the data
                this->imgdata.SetOrigin(Vec3f(this->string2float(TString(file[lineCnt].Word(1))),
                    this->string2float(TString(file[lineCnt].Word(2))),
                    this->string2float(TString(file[lineCnt].Word(3)))));
            }
        }

        if (file[lineCnt].Count() >= 1) {
            if (StringA(file[lineCnt].Word(0)).Equals("delta")) {
                this->imgdata.SetSpacing(Vec3f(this->string2float(TString(file[lineCnt + 0].Word(1))),
                    this->string2float(TString(file[lineCnt + 1].Word(2))),
                    this->string2float(TString(file[lineCnt + 2].Word(3)))));
                lineCnt += 2;
            }
        }

        // Stop reading float data
        if (file[lineCnt].Count() >= 1) {
            if ((readFloatData) && (StringA(file[lineCnt].Word(0)).Equals("attribute"))) {
                readFloatData = false;
            }
        }

        if (readFloatData) {
            //printf("Number of words in the current line %s\n", file[lineCnt].Count());
            for (uint w = 0; w < file[lineCnt].Count(); ++w) { // Loop through all words
                //printf("Current word %s\n", file[lineCnt].Word(w));
                this->data.Peek()[floatDataCounter] = this->string2float(file[lineCnt].Word(w));
                if (min > this->data.Peek()[floatDataCounter]) {
                    min = this->data.Peek()[floatDataCounter];
                }
                if (max < this->data.Peek()[floatDataCounter]) {
                    max = this->data.Peek()[floatDataCounter];
                }
                floatDataCounter++;
            }
        }

        // Start reading float data
        if ((!readFloatData) && (file[lineCnt].Count() >= 4)) {
            if (StringA(file[lineCnt].Word(3)).Equals("array")) {
                // Read data from now on
                readFloatData = true;
                this->data.Validate((this->imgdata.GetWholeExtent().Width() + 1) *
                                    (this->imgdata.GetWholeExtent().Depth() + 1) *
                                    (this->imgdata.GetWholeExtent().Height() + 1));
            }
        }

        lineCnt++;
    }

    // Change ordering from row major to column major
    this->dataTmp.Validate(this->data.GetCount());
    //#pragma omp parallel for
    for (int cnt = 0; cnt < static_cast<int>(this->data.GetCount()); ++cnt) {
        int x = cnt % int((this->imgdata.GetWholeExtent().Width() + 1));
        int y = (cnt / int((this->imgdata.GetWholeExtent().Width() + 1))) %
                int((this->imgdata.GetWholeExtent().Height() + 1));
        int z = (cnt / int((this->imgdata.GetWholeExtent().Width() + 1))) /
                int((this->imgdata.GetWholeExtent().Height() + 1));
        this->dataTmp.Peek()[int((this->imgdata.GetWholeExtent().Depth() + 1)) *
                                 (int((this->imgdata.GetWholeExtent().Height() + 1)) * x + y) +
                             z] = this->data.Peek()[cnt];
    }
    memcpy(this->data.Peek(), this->dataTmp.Peek(), this->data.GetSize() * sizeof(float));

    // Setup data array
    this->imgdata.SetNumberOfPieces(1);
    this->imgdata.SetPointData(
        (const char*) this->data.Peek(), min, max, protein_calls::VTKImageData::DataArray::VTI_FLOAT, "vmddata", 1, 0);

    Log::DefaultLog.WriteInfo("%s: ... done (%f s)", this->ClassName(),
        (double(clock() - t) / double(CLOCKS_PER_SEC))); // DEBUG

    return true;
}


/*
 * VMDDXLoader::readDataAscii2Float
 */
void VMDDXLoader::readDataAscii2Float(char* buffIn, float* buffOut, SIZE_T sizeOut) {

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
 * VMDDXLoader::scanFolder
 */
void VMDDXLoader::scanFolder() {
    using namespace vislib;

    TString filename = this->filenameSlot.Param<core::param::FilePathParam>()->Value().generic_u8string().c_str();

    vislib::sys::File testFile;

    // Generate filename pattern
    this->filenamesDigits = 0;
    this->filenamesPrefix.Clear();
    this->filenamesPrefix = filename.Substring(0, filename.Find('.'));
    this->filenamesSuffix.Clear();
    this->filenamesSuffix = filename.Substring(filename.FindLast('.') + 1, filename.Length());

    // Determine number of frames if necessary
    if (filename.Find('.') != filename.FindLast('.')) { // File name contains at least two '.', file series is assumed
        this->filenamesDigits = filename.Length() - 2 - this->filenamesPrefix.Length() - this->filenamesSuffix.Length();
        //        Log::DefaultLog.WriteInfo( "%s: Generated filename pattern %s %u DIGITS %s", // TODO
        //                this->ClassName(), this->filenamesPrefix.PeekBuffer(), this->filenamesDigits,
        //                this->filenamesSuffix.PeekBuffer());

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

            // Try to open frame files
            //            Log::DefaultLog.WriteMsg(
            //                    Log::LEVEL_INFO, "%s: Checking for %s ...",
            //                    this->ClassName(),
            //                    frameFile.PeekBuffer());

            if (testFile.Exists(frameFile)) {
                this->nFrames++;
            } else {
                search_done = true;
            }
        }

        Log::DefaultLog.WriteInfo("%s: %u frame files found", this->ClassName(), this->nFrames);
    } else { // Single file
        this->nFrames = 1;
    }
}


/*
 * VMDDXLoader::string2int
 */
int VMDDXLoader::string2int(vislib::StringA str) {
    using namespace std;

    std::stringstream ss(std::string(str.PeekBuffer()));
    int i;

    if ((ss >> i).fail()) {
        //error
    }
    return i;
}


/*
 * VMDDXLoader::string2float
 */
float VMDDXLoader::string2float(vislib::StringA str) {
    using namespace std;

    // TODO Make own header for this

    std::stringstream ss(std::string(str.PeekBuffer()));
    float f;

    if ((ss >> f).fail()) {
        //error
    }

    return f;
}
