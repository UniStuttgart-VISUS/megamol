//
// VTKLegacyDataLoaderUnstructuredGrid.cpp
//
// Copyright (C) 2013-2018 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on : Sep 23, 2013
// Author     : scharnkn
//

#include "VTKLegacyDataLoaderUnstructuredGrid.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/log/Log.h"
#include "protein/VTKLegacyDataCallUnstructuredGrid.h"
#include "vislib/sys/File.h"
#include <algorithm>
#include <cmath>
#include <ctype.h>
#include <sstream>


using namespace megamol;
using namespace megamol::protein;
using namespace megamol::core::utility::log;

#define VERBOSE
#define SWAP_BYTES
//#define NORMALIZE_RADIUS

#pragma push_macro("min")
#undef min
#pragma push_macro("max")
#undef max

/*
 * VTKLegacyDataLoaderUnstructuredGrid::VTKLegacyDataLoaderUnstructuredGrid
 */
VTKLegacyDataLoaderUnstructuredGrid::VTKLegacyDataLoaderUnstructuredGrid()
        : core::view::AnimDataModule()
        , dataOutSlot("dataOut", "The slot providing the loaded data")
        , filenameSlot("vtkPath", "The path to the first VTK data file to be loaded")
        , maxFramesSlot("maxFrames", "The maximum number of frames to be loaded")
        , frameStartSlot("frameStart", "The first frame to be loaded")
        , maxCacheSizeSlot("maxCacheSize", "The maximum size of the cache")
        , mpdcAttributeSlot(
              "mpdcAttribute", "The name of the point data attribute to be sent with MultiParticleDataCall")
        , globalRadiusParam("globalRadius", "The global radius to be sent with MultiParticleDataCall")
        , hash(0)
        , filenamesDigits(0)
        , nFrames(0)
        , readPointData(false)
        , readCellData(false)
        , bbox(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) {

    // Unstructured grid data
    this->dataOutSlot.SetCallback(VTKLegacyDataCallUnstructuredGrid::ClassName(),
        VTKLegacyDataCallUnstructuredGrid::FunctionName(VTKLegacyDataCallUnstructuredGrid::CallForGetData),
        &VTKLegacyDataLoaderUnstructuredGrid::getData);
    this->dataOutSlot.SetCallback(VTKLegacyDataCallUnstructuredGrid::ClassName(),
        VTKLegacyDataCallUnstructuredGrid::FunctionName(VTKLegacyDataCallUnstructuredGrid::CallForGetExtent),
        &VTKLegacyDataLoaderUnstructuredGrid::getExtent);

    // Multi stream particle data
    this->dataOutSlot.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(0), &VTKLegacyDataLoaderUnstructuredGrid::getData);
    this->dataOutSlot.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(1), &VTKLegacyDataLoaderUnstructuredGrid::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);

    this->filenameSlot.SetParameter(new core::param::FilePathParam(""));
    this->MakeSlotAvailable(&this->filenameSlot);

    this->maxFramesSlot.SetParameter(new core::param::IntParam(11, 1));
    this->MakeSlotAvailable(&this->maxFramesSlot);

    this->frameStartSlot.SetParameter(new core::param::IntParam(0, 0));
    this->MakeSlotAvailable(&this->frameStartSlot);

    this->maxCacheSizeSlot.SetParameter(new core::param::IntParam(10, 1));
    this->MakeSlotAvailable(&this->maxCacheSizeSlot);

    this->mpdcAttributeSlot.SetParameter(new core::param::StringParam(""));
    this->MakeSlotAvailable(&this->mpdcAttributeSlot);

    this->globalRadiusParam.SetParameter(new core::param::FloatParam(1.0f, 0.0f));
    this->MakeSlotAvailable(&this->globalRadiusParam);
}


/*
 * VTKLegacyDataLoaderUnstructuredGrid::~VTKLegacyDataLoaderUnstructuredGrid
 */
VTKLegacyDataLoaderUnstructuredGrid::~VTKLegacyDataLoaderUnstructuredGrid() {
    this->Release();
}


/*
 * VTKLegacyDataLoaderUnstructuredGrid::create
 */
bool VTKLegacyDataLoaderUnstructuredGrid::create(void) {
    return true;
}


/*
 * VTKLegacyDataLoaderUnstructuredGrid::release
 */
void VTKLegacyDataLoaderUnstructuredGrid::release(void) {}


/*
 * VTKLegacyDataLoaderUnstructuredGrid::getData
 */
bool VTKLegacyDataLoaderUnstructuredGrid::getData(core::Call& call) {

    // Get unstructured grid data call
    VTKLegacyDataCallUnstructuredGrid* dc = dynamic_cast<VTKLegacyDataCallUnstructuredGrid*>(&call);
    if (dc != NULL) {
        if (dc->FrameID() >= this->FrameCount()) {
#ifdef VERBOSE
            Log::DefaultLog.WriteError( "%s: Frame %u requested (nFrames %u)", this->ClassName(),
                dc->FrameID(), this->FrameCount());
#endif
            return false;
        }

        // Request the frame
        Frame* fr = NULL;
        fr = dynamic_cast<VTKLegacyDataLoaderUnstructuredGrid::Frame*>(this->requestLockedFrame(dc->FrameID()));

        if (fr == NULL) {
            return false;
        }

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
                fr = dynamic_cast<VTKLegacyDataLoaderUnstructuredGrid::Frame*>(this->requestLockedFrame(frameBefore));
                dc->SetUnlocker(new VTKUnlocker(*fr));
                dc->Unlock();
                fr = dynamic_cast<VTKLegacyDataLoaderUnstructuredGrid::Frame*>(this->requestLockedFrame(dc->FrameID()));
                dc->SetUnlocker(new VTKUnlocker(*fr));
                if (fr == NULL) {
                    return false;
                }
            }
        }

        // Set unlocker object for the frame
        dc->SetUnlocker(new VTKUnlocker(*fr));

        // Set data of the call
        dc->SetData(fr->GetData());

        // Set data hash value
        dc->SetDataHash(this->hash);

        // printf("Frame loaded: %u\n", fr->FrameNumber()); // DEBUG
    } else {
        // Try to get pointer to unstructured grid call
        geocalls::MultiParticleDataCall* mpdc = dynamic_cast<geocalls::MultiParticleDataCall*>(&call);
        if (mpdc != NULL) {

            // Request the frame
            Frame* fr = NULL;
            fr = dynamic_cast<VTKLegacyDataLoaderUnstructuredGrid::Frame*>(this->requestLockedFrame(mpdc->FrameID()));

            if (fr == NULL) {
                return false;
            }

            // If the 'force' flag is set, check whether the frame number is correct,
            // if not re-request the frame
            if (mpdc->IsFrameForced()) {
                while (mpdc->FrameID() != fr->FrameNumber()) {
                    mpdc->Unlock();
                    int frameBefore = ((static_cast<int>(mpdc->FrameID() - 1) + static_cast<int>(this->FrameCount()))) %
                                      static_cast<int>(this->FrameCount());

                    // scharnkn:
                    // Request the frame before the actual requested frame (modulo
                    // framenumber) to trigger loading of the actually requested frame
                    fr = dynamic_cast<VTKLegacyDataLoaderUnstructuredGrid::Frame*>(
                        this->requestLockedFrame(frameBefore));
                    mpdc->SetUnlocker(new VTKUnlocker(*fr));
                    mpdc->Unlock();
                    fr = dynamic_cast<VTKLegacyDataLoaderUnstructuredGrid::Frame*>(
                        this->requestLockedFrame(mpdc->FrameID()));
                    mpdc->SetUnlocker(new VTKUnlocker(*fr));
                    if (fr == NULL) {
                        return false;
                    }
                }
            }

            mpdc->SetDataHash(this->hash);
            mpdc->SetParticleListCount(1); // Only one particle list

            // Loop through all particle lists

            // Set global radius to 1 TODO ?

#ifndef NORMALIZE_RADIUS
            mpdc->AccessParticles(0).SetGlobalRadius(this->globalRadiusParam.Param<core::param::FloatParam>()->Value());
#else
            mpdc->AccessParticles(0).SetGlobalRadius(1.0f);
#endif
            // Set number of frames
            mpdc->AccessParticles(0).SetCount(fr->GetNumberOfPoints());
            // Set particle type
            mpdc->AccessParticles(0).SetGlobalType(0); // TODO What is this?

//#define CONTEST2016
#ifndef CONTEST2016
            // Set attribute array as float 'color' value
            if (!this->mpdcAttributeSlot.Param<core::param::StringParam>()->Value().empty()) {
                mpdc->AccessParticles(0).SetColourData(geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_I,
                    fr->PeekPointDataByName(this->mpdcAttributeSlot.Param<core::param::StringParam>()->Value().c_str()),
                    0);
            }
#else
            std::vector<float> dataVec;
            dataVec.resize(fr->GetNumberOfPoints() * 4);

            const float* velPtr = reinterpret_cast<const float*>(fr->PeekPointDataByName("velocity")->PeekData());
            const float* conPtr = reinterpret_cast<const float*>(fr->PeekPointDataByName("concentration")->PeekData());

            /*for (int i = 0; i < fr->GetNumberOfPoints(); i++) {
                std::cout << (int)fr->PeekPointDataByName("concentration")->PeekData()[i] << std::endl;
            }*/

            for (int64_t i = 0; i < (int64_t)fr->GetNumberOfPoints(); i++) {
                // std::cout << conPtr[i] << std::endl;
                dataVec[(unsigned int)i * 4 + 0] = velPtr[(unsigned int)i * 3 + 0];
                dataVec[(unsigned int)i * 4 + 1] = velPtr[(unsigned int)i * 3 + 1];
                dataVec[(unsigned int)i * 4 + 2] = velPtr[(unsigned int)i * 3 + 2];
                dataVec[(unsigned int)i * 4 + 3] = conPtr[(unsigned int)i];

                /*if (i > 1) {
                    std::cout << dataVec[i * 4] << " " << dataVec[i * 4 + 1] << " " << dataVec[i * 4 + 2] << " " <<
                dataVec[i * 4 + 3] << std::endl;
                }*/
            }
            mpdc->AccessParticles(0).SetColourData(
                geocalls::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA, dataVec.data(), 0);
#endif
            // Set vertex positions
            mpdc->AccessParticles(0).SetVertexData(geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ,
                (const void*)(fr->GetData()->PeekPoints()));

            mpdc->SetUnlocker(new VTKUnlocker(*fr));


        } else {
            return false;
        }
    }

    return true;
}


/*
 * VTKLegacyDataLoaderUnstructuredGrid::getExtent
 */
bool VTKLegacyDataLoaderUnstructuredGrid::getExtent(core::Call& call) {

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
            this->setFrameCount((unsigned int)std::min(
                static_cast<size_t>(this->maxFramesSlot.Param<core::param::IntParam>()->Value()), this->nFrames));
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
        this->setFrameCount((unsigned int)std::min(
            static_cast<size_t>(this->maxFramesSlot.Param<core::param::IntParam>()->Value()), this->nFrames));

        // Start the loading thread
        this->initFrameCache(this->maxCacheSizeSlot.Param<core::param::IntParam>()->Value());
    }

    // Try to get pointer to unstructured grid call
    VTKLegacyDataCallUnstructuredGrid* dc = dynamic_cast<VTKLegacyDataCallUnstructuredGrid*>(&call);
    if (dc != NULL) {

        // Set frame count
        dc->SetFrameCount((unsigned int)std::min(
            static_cast<size_t>(this->maxFramesSlot.Param<core::param::IntParam>()->Value()), this->nFrames));

        dc->AccessBoundingBoxes().Clear();
        dc->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
        dc->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox);
    } else {
        // Try to get pointer to unstructured grid call
        geocalls::MultiParticleDataCall* mpdc = dynamic_cast<geocalls::MultiParticleDataCall*>(&call);
        if (mpdc != NULL) {
            // Set frame count
            mpdc->SetFrameCount((unsigned int)std::min(
                static_cast<size_t>(this->maxFramesSlot.Param<core::param::IntParam>()->Value()), this->nFrames));

            mpdc->AccessBoundingBoxes().Clear();
            mpdc->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
            mpdc->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox);
        } else {
            return false;
        }
    }

    return true;
}


/*
 * VTKLegacyDataLoaderUnstructuredGrid::loadFile
 */
bool VTKLegacyDataLoaderUnstructuredGrid::loadFile(const vislib::StringA& filename) {
    /* Test whether the filename is invalid or empty */

    if (filename.IsEmpty()) {
        Log::DefaultLog.WriteInfo( "%s: No file to load (filename empty)", this->ClassName());
        return true;
    }

    vislib::sys::File file;

    /* Generate filename pattern based on the given filename */

    // If single file given
    if (!filename.Contains('%')) {
        this->nFrames = 1;
    } else {
        // Generate filename pattern
        size_t a = filename.Find('%');
        size_t b = filename.FindLast('%');
        this->filenamesDigits = b - a + 1;

        this->filenamesPrefix.Clear();
        this->filenamesPrefix = filename.Substring(0, (int)a);
        this->filenamesSuffix.Clear();
        this->filenamesSuffix = filename.Substring((int)b + 2);

        //        printf("DIGITS %i\n", this->filenamesDigits);
        //        printf("PREFIX %s\n", this->filenamesPrefix.PeekBuffer());
        //        printf("SUFFIX %s\n", this->filenamesSuffix.PeekBuffer());

        // Search for frame files in the folder
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
            frameFile.Append((ss.str()).c_str());
            frameFile.Append('.');
            frameFile.Append(this->filenamesSuffix);

            // Try to open frame files
#if defined(VERBOSE)
            vislib::StringA frameFileShortPath;
#if defined(_WIN32)
            frameFileShortPath = frameFile.Substring(frameFile.FindLast('\\') + 1, frameFile.Length() - 1);
#else
            frameFileShortPath = frameFile.Substring(frameFile.FindLast('/') + 1, frameFile.Length() - 1);
#endif
            Log::DefaultLog.WriteInfo( "%s: checking for  '%s'", this->ClassName(),
                frameFileShortPath.PeekBuffer()); // DEBUG
#endif                                            // defined(VERBOSE)


            if (vislib::sys::File::Exists(frameFile)) {
                this->nFrames++;
            } else {
                search_done = true;
            }
        }
    }

    Log::DefaultLog.WriteInfo( "%s: found %i frame files", this->ClassName(),
        this->nFrames); // DEBUG

    // Set number of frames
    this->setFrameCount((unsigned int)std::min(
        static_cast<size_t>(this->maxFramesSlot.Param<core::param::IntParam>()->Value()), this->nFrames));


    /* Start the loading thread */

    this->initFrameCache(this->maxCacheSizeSlot.Param<core::param::IntParam>()->Value());


    /* Compute an approximated bounding box */ // TODO Own method with optional exact bbox

    // Generate filename of the first frame
    vislib::StringA frameFile;

    if (filename.Contains('%')) {
        std::stringstream ss;
        ss.width(this->filenamesDigits);
        ss.fill('0');
        std::string digits;
        ss << 0; // TODO USe first frame parameter
        frameFile = this->filenamesPrefix;
        frameFile.Append((ss.str()).c_str());
        frameFile.Append('.');
        frameFile.Append(this->filenamesSuffix);
        //    printf("FILE %s\n", frameFile.PeekBuffer());
    } else {
        frameFile = filename;
    }

    // Try to open the first frames file
    vislib::sys::File::FileSize fileSize = vislib::sys::File::GetSize(frameFile);

#if defined(VERBOSE)
    time_t t = clock();
#endif // defined(VERBOSE)

    // Read data file to char buffer
    char* buffer = new char[(unsigned int)fileSize];
    file.Open(
        frameFile, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_EXCLUSIVE, vislib::sys::File::OPEN_ONLY);
    file.Read(buffer, fileSize);
    file.Close();


    // Parse buffer
    char* buffPt = buffer;
    vislib::StringA line;
    AbstractVTKLegacyData::DataEncoding encoding = AbstractVTKLegacyData::ASCII;

    // Loop through all chars
    while (buffPt < &buffer[fileSize - 1]) {

        // Read the current line
        line = this->readCurrentLine(buffPt);
        //        printf("LINE: %s\n", line.PeekBuffer());

        // Check file encoding
        if (line.Contains("# vtk DataFile Version")) {
            this->seekNextLine(buffPt); // Skip title
            this->seekNextLine(buffPt); // Skip data set info TODO Check whether this is UNSTRUCTURED_GRID
            if (this->readNextToken(buffPt) == "ASCII") {
                printf("ASCII\n");
                encoding = AbstractVTKLegacyData::ASCII;
            } else {
                encoding = AbstractVTKLegacyData::BINARY;
            }
        } else if (line.Contains("POINTS")) { // Get the vertex positions
            this->seekNextToken(buffPt);
            size_t vertexCnt = atoi(this->readNextToken(buffPt));
            //            printf("vertex cnt %i\n", vertexCnt);
            this->seekNextLine(buffPt);
            if (encoding == AbstractVTKLegacyData::ASCII) {
                float* tmpBuff = new float[vertexCnt * 3];
                this->readASCIIFloats(buffPt, tmpBuff, vertexCnt * 3);

                // Loop through all vertices to approximate bounding box
                for (size_t v = 0; v < vertexCnt; ++v) {
                    if (v == 0) { // Init bbox
//                        printf("VERTEX %f %f %f\n",
//                                tmpBuff[3*v+0],
//                                tmpBuff[3*v+1],
//                                tmpBuff[3*v+2]);
#ifdef NORMALIZE_RADIUS
                        this->bbox.Set(
                            tmpBuff[3 * v + 0] / this->globalRadiusParam.Param<core::param::FloatParam>()->Value(),
                            tmpBuff[3 * v + 1] / this->globalRadiusParam.Param<core::param::FloatParam>()->Value(),
                            tmpBuff[3 * v + 2] / this->globalRadiusParam.Param<core::param::FloatParam>()->Value(),
                            tmpBuff[3 * v + 0] / this->globalRadiusParam.Param<core::param::FloatParam>()->Value(),
                            tmpBuff[3 * v + 1] / this->globalRadiusParam.Param<core::param::FloatParam>()->Value(),
                            tmpBuff[3 * v + 2] / this->globalRadiusParam.Param<core::param::FloatParam>()->Value());
#else
                        this->bbox.Set(tmpBuff[3 * v + 0], tmpBuff[3 * v + 1], tmpBuff[3 * v + 2], tmpBuff[3 * v + 0],
                            tmpBuff[3 * v + 1], tmpBuff[3 * v + 2]);
#endif
                        //                        printf("Bounding box %f %f %f %f %f %f\n",
                        //                                this->bbox.Left(),
                        //                                this->bbox.Bottom(),
                        //                                this->bbox.Back(),
                        //                                this->bbox.Right(),
                        //                                this->bbox.Top(),
                        //                                this->bbox.Front());

                    } else {
//                        printf("VERTEX %f %f %f\n",
//                                tmpBuff[3*v+0],
//                                tmpBuff[3*v+1],
//                                tmpBuff[3*v+2]);
#ifdef NORMALIZE_RADIUS
                        this->bbox.GrowToPoint(
                            tmpBuff[3 * v + 0] / this->globalRadiusParam.Param<core::param::FloatParam>()->Value(),
                            tmpBuff[3 * v + 1] / this->globalRadiusParam.Param<core::param::FloatParam>()->Value(),
                            tmpBuff[3 * v + 2] / this->globalRadiusParam.Param<core::param::FloatParam>()->Value());
#else
                        this->bbox.GrowToPoint(tmpBuff[3 * v + 0], tmpBuff[3 * v + 1], tmpBuff[3 * v + 2]);
#endif
                    }
                    buffPt += 12;
                }

                delete[] tmpBuff;
            } else {
#ifdef SWAP_BYTES
                this->swapBytes(buffPt, 4, vertexCnt * 3); // TODO Assumes 4 byte values
#endif
                // Loop through all vertices to approximate bounding box
                for (size_t v = 0; v < vertexCnt; ++v) {
                    if (v == 0) { // Init bbox
//                        printf("VERTEX %f %f %f\n",
//                                ((float*)(buffPt))[v*3+0],
//                                ((float*)(buffPt))[v*3+1],
//                                ((float*)(buffPt))[v*3+2]);
#ifdef NORMALIZE_RADIUS
                        this->bbox.Set(((float*)(buffPt))[v * 3 + 0] /
                                           this->globalRadiusParam.Param<core::param::FloatParam>()->Value(),
                            ((float*)(buffPt))[v * 3 + 1] /
                                this->globalRadiusParam.Param<core::param::FloatParam>()->Value(),
                            ((float*)(buffPt))[v * 3 + 2] /
                                this->globalRadiusParam.Param<core::param::FloatParam>()->Value(),
                            ((float*)(buffPt))[v * 3 + 0] /
                                this->globalRadiusParam.Param<core::param::FloatParam>()->Value(),
                            ((float*)(buffPt))[v * 3 + 1] /
                                this->globalRadiusParam.Param<core::param::FloatParam>()->Value(),
                            ((float*)(buffPt))[v * 3 + 2] /
                                this->globalRadiusParam.Param<core::param::FloatParam>()->Value());
#else
                        this->bbox.Set(((float*)(buffPt))[v * 3 + 0], ((float*)(buffPt))[v * 3 + 1],
                            ((float*)(buffPt))[v * 3 + 2], ((float*)(buffPt))[v * 3 + 0], ((float*)(buffPt))[v * 3 + 1],
                            ((float*)(buffPt))[v * 3 + 2]);
#endif

                        //                        printf("Bounding box %f %f %f %f %f %f\n",
                        //                                this->bbox.Left(),
                        //                                this->bbox.Bottom(),
                        //                                this->bbox.Back(),
                        //                                this->bbox.Right(),
                        //                                this->bbox.Top(),
                        //                                this->bbox.Front());

                    } else {
#ifdef NORMALIZE_RADIUS
                        this->bbox.GrowToPoint(((float*)(buffPt))[v * 3 + 0] /
                                                   this->globalRadiusParam.Param<core::param::FloatParam>()->Value(),
                            ((float*)(buffPt))[v * 3 + 1] /
                                this->globalRadiusParam.Param<core::param::FloatParam>()->Value(),
                            ((float*)(buffPt))[v * 3 + 2] /
                                this->globalRadiusParam.Param<core::param::FloatParam>()->Value());
#else
                        this->bbox.GrowToPoint(((float*)(buffPt))[v * 3 + 0], ((float*)(buffPt))[v * 3 + 1],
                            ((float*)(buffPt))[v * 3 + 2]);
#endif
                        //                        printf("VERTEX %f %f %f\n",
                        //                                ((float*)(buffPt))[v*3+0],
                        //                                ((float*)(buffPt))[v*3+1],
                        //                                ((float*)(buffPt))[v*3+2]);
                    }
                }
            }
            break;
        } else {
            this->seekNextLine(buffPt);
        }
    }
    //    printf("Bounding box %f %f %f %f %f %f\n",
    //            this->bbox.Left(),
    //            this->bbox.Bottom(),
    //            this->bbox.Back(),
    //            this->bbox.Right(),
    //            this->bbox.Top(),
    //            this->bbox.Front());

    // this->bbox.Grow(4.0); // Add fixed offset TODO

    delete[] buffer;

    return true;
}


/*
 * VTKLegacyDataLoaderUnstructuredGrid::constructFrame
 */
core::view::AnimDataModule::Frame* VTKLegacyDataLoaderUnstructuredGrid::constructFrame(void) const {
    Frame* f = new Frame(*const_cast<VTKLegacyDataLoaderUnstructuredGrid*>(this));
    return f;
}


/*
 * VTKLegacyDataLoaderUnstructuredGrid::loadFrame
 */
void VTKLegacyDataLoaderUnstructuredGrid::loadFrame(core::view::AnimDataModule::Frame* frame, unsigned int idx) {
    using namespace vislib;

    size_t dataCnt = 0, fieldArrayCnt;

    VTKLegacyDataLoaderUnstructuredGrid::Frame* fr = dynamic_cast<VTKLegacyDataLoaderUnstructuredGrid::Frame*>(frame);
    if (fr == NULL) {
        return;
    }

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
    char* buffer = new char[(unsigned int)fileSize];
    file.Open(
        frameFile, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_EXCLUSIVE, vislib::sys::File::OPEN_ONLY);
    file.Read(buffer, fileSize);
    file.Close();

    // Parse buffer
    char* buffPt = buffer;
    vislib::StringA line;

    // Loop through all chars
    while (buffPt < &buffer[fileSize - 1]) {

        // Read the current line
        line = this->readCurrentLine(buffPt);
        //        printf("LINE: %s\n", line.PeekBuffer());

        // Get header data
        if (line.Contains("# vtk DataFile Version")) {
            this->readHeaderData(buffPt, frame);
        } else if (line.Contains("POINTS")) { // Get vertex positions
            this->readPoints(buffPt, frame);
        } else if (line.Contains("CELLS")) { // Get cell indices
            this->readCells(buffPt, frame);
        } else if (line.Contains("CELL_TYPES")) { // Get cell types
            this->readCellTypes(buffPt, frame);
        } else if (line.Contains("POINT_DATA")) { // Get point data
            this->readPointData = true;
            this->readCellData = false;
            this->seekNextToken(buffPt);
            dataCnt = atoi(this->readNextToken(buffPt));
            this->seekNextToken(buffPt);         // Skip to next line
        } else if (line.Contains("CELL_DATA")) { // Get point data
            this->readCellData = true;
            this->readPointData = false;
            this->seekNextToken(buffPt);
            dataCnt = atoi(this->readNextToken(buffPt));
            this->seekNextToken(buffPt);     // Skip to next line
        } else if (line.Contains("FIELD")) { // Get field data
            this->seekNextToken(buffPt);
            this->seekNextToken(buffPt);
            fieldArrayCnt = atoi(this->readNextToken(buffPt));
            this->readFieldData(buffPt, frame, fieldArrayCnt);
        } else if (line.Contains("SCALAR")) { // Get scalar data
            if (this->readPointData) {
                this->readDataArray(buffPt, frame, dataCnt, 1, AbstractVTKLegacyData::POINT_DATA);
            }
            if (this->readCellData) {
                this->readDataArray(buffPt, frame, dataCnt, 1, AbstractVTKLegacyData::CELL_DATA);
            }
        } else if (line.Contains("VECTORS")) { // Get vector data
            if (this->readPointData) {
                this->readDataArray(buffPt, frame, dataCnt, 3, AbstractVTKLegacyData::POINT_DATA);
            }
            if (this->readCellData) {
                this->readDataArray(buffPt, frame, dataCnt, 3, AbstractVTKLegacyData::CELL_DATA);
            }
        } else if (line.Contains("NORMALS")) { // Get normals
            if (this->readPointData) {
                this->readDataArray(buffPt, frame, dataCnt, 3, AbstractVTKLegacyData::POINT_DATA);
            }
            if (this->readCellData) {
                this->readDataArray(buffPt, frame, dataCnt, 3, AbstractVTKLegacyData::CELL_DATA);
            }
        } else if (line.Contains("TEXTURE_COORDINATES")) { // Get texture coordinates
            if (this->readPointData) {
                this->readDataArray(buffPt, frame, dataCnt, 3, AbstractVTKLegacyData::POINT_DATA);
            }
            if (this->readCellData) {
                this->readDataArray(buffPt, frame, dataCnt, 3, AbstractVTKLegacyData::CELL_DATA);
            }
        } else if (line.Contains("TENSOR_DATA")) { // Get tensor data
            if (this->readPointData) {
                this->readDataArray(buffPt, frame, dataCnt, 9, AbstractVTKLegacyData::POINT_DATA);
            }
            if (this->readCellData) {
                this->readDataArray(buffPt, frame, dataCnt, 9, AbstractVTKLegacyData::CELL_DATA);
            }
        } else {
            this->seekNextLine(buffPt);
        }
    }

    delete[] buffer;

#if defined(VERBOSE)
    vislib::StringA frameFileShortPath;
#if defined(_WIN32)
    frameFileShortPath = frameFile.Substring(frameFile.FindLast('\\') + 1, frameFile.Length() - 1);
#else
    frameFileShortPath = frameFile.Substring(frameFile.FindLast('/') + 1, frameFile.Length() - 1);
#endif
    Log::DefaultLog.WriteInfo( "%s: done loading '%s' (%u Bytes, %f s)", this->ClassName(),
        frameFileShortPath.PeekBuffer(), fileSize,
        (double(clock() - t) / double(CLOCKS_PER_SEC))); // DEBUG
#endif                                                   // defined(VERBOSE)
}


/*
 * VTKLegacyDataLoaderUnstructuredGrid::readASCIIFloats
 */
void VTKLegacyDataLoaderUnstructuredGrid::readASCIIFloats(char*& buffPt, float* out, size_t cnt) {
    vislib::StringA token;
    for (size_t i = 0; i < cnt; ++i) {
        token = this->readNextToken(buffPt);
        //        if (i%1000==0) printf("Token %i: %f\n", i, atof(token.PeekBuffer()));
        //        printf("Token %i: %f\n", i, atof(token.PeekBuffer()));
        out[i] = (float)atof(token.PeekBuffer());
        this->seekNextToken(buffPt);
    }
}


/*
 * VTKLegacyDataLoaderUnstructuredGrid::readASCIIInts
 */
void VTKLegacyDataLoaderUnstructuredGrid::readASCIIInts(char*& buffPt, int* out, size_t cnt) {
    vislib::StringA token;
    for (size_t i = 0; i < cnt; ++i) {
        token = this->readNextToken(buffPt);
        out[i] = atoi(token.PeekBuffer());
        this->seekNextToken(buffPt);
    }
}


/*
 * VTKLegacyDataLoaderUnstructuredGrid::readCells
 */
void VTKLegacyDataLoaderUnstructuredGrid::readCells(char*& buffPt, core::view::AnimDataModule::Frame* frame) {

    size_t cellDataSize;
    vislib::StringA cellCntStr, cellDataSizeStr;

    vislib::StringA line;

    VTKLegacyDataLoaderUnstructuredGrid::Frame* fr = dynamic_cast<VTKLegacyDataLoaderUnstructuredGrid::Frame*>(frame);
    if (fr == NULL) {
        return;
    }


    /* Get the number of cells */

    line = this->readCurrentLine(buffPt);
    this->seekNextToken(buffPt); // Skip token

    // Get number of cells
    cellCntStr = this->readNextToken(buffPt);
    this->seekNextToken(buffPt);

    // Get size of cell index data
    cellDataSizeStr = this->readCurrentLine(buffPt);
    cellDataSize = atoi(cellDataSizeStr.PeekBuffer());

    this->seekNextLine(buffPt); // Set pointer to the beginning of the next line


    /* Read actual numbers */

    // NOTE: Buffer pointer should now point to first char of the cell indices

    // ASCII
    if (fr->GetEncoding() == AbstractVTKLegacyData::ASCII) {
        int* tempBuff = new int[cellDataSize];
        this->readASCIIInts(buffPt, tempBuff, cellDataSize);
        fr->SetCellIndexData(tempBuff, cellDataSize);
        delete[] tempBuff;
    } else {
#ifdef SWAP_BYTES
        this->swapBytes(buffPt, 4, cellDataSize);
#endif
        fr->SetCellIndexData((const int*)(buffPt), cellDataSize);
    }

    // Increment buffer pointer
    buffPt += cellDataSize * sizeof(int);
}


/*
 * VTKLegacyDataLoaderUnstructuredGrid::readCelltypes
 */
void VTKLegacyDataLoaderUnstructuredGrid::readCellTypes(char*& buffPt, core::view::AnimDataModule::Frame* frame) {

    size_t cellCnt;
    vislib::StringA cellCntStr, cellDataSizeStr;
    vislib::StringA line;

    VTKLegacyDataLoaderUnstructuredGrid::Frame* fr = dynamic_cast<VTKLegacyDataLoaderUnstructuredGrid::Frame*>(frame);
    if (fr == NULL) {
        return;
    }


    /* Get the number of cells */

    line = this->readCurrentLine(buffPt);
    this->seekNextToken(buffPt); // Skip token

    // Get number of cells
    cellCntStr = this->readNextToken(buffPt);
    cellCnt = atoi(cellCntStr.PeekBuffer());

    this->seekNextLine(buffPt); // Set pointer to the beginning of the next line

    /* Read actual numbers */

    // NOTE: Buffer pointer should now point to first char of the cell indices

    // ASCII
    if (fr->GetEncoding() == AbstractVTKLegacyData::ASCII) {
        int* tempBuff = new int[cellCnt];
        this->readASCIIInts(buffPt, tempBuff, cellCnt);
        fr->SetCellTypes(tempBuff, cellCnt);
        delete[] tempBuff;
    } else {
#ifdef SWAP_BYTES
        this->swapBytes(buffPt, 4, cellCnt);
#endif
        fr->SetCellTypes((const int*)(buffPt), cellCnt);
    }

    buffPt += cellCnt * sizeof(int);
}


/*
 * VTKLegacyDataLoaderUnstructuredGrid::readHeaderData
 */
void VTKLegacyDataLoaderUnstructuredGrid::readHeaderData(char*& buffPt, core::view::AnimDataModule::Frame* frame) {
    VTKLegacyDataLoaderUnstructuredGrid::Frame* fr = dynamic_cast<VTKLegacyDataLoaderUnstructuredGrid::Frame*>(frame);
    if (fr == NULL) {
        return;
    }


    /* Get file VTK version */

    vislib::StringA line = this->readCurrentLine(buffPt);
    vislib::StringA versionStr = line.Substring(line.Find("Version") + 8, 3);
    int version[2];
    version[0] = atoi(&versionStr[0]);
    version[1] = atoi(&versionStr[2]);
    this->seekNextLine(buffPt);
    this->seekNextLine(buffPt); // Skip title string


    /* Get file encoding (ASCII or BINARY) */

    line = this->readCurrentLine(buffPt);
    if (line == "ASCII") {
        fr->SetEncoding(AbstractVTKLegacyData::ASCII);
    } else if (line == "BINARY") {
        fr->SetEncoding(AbstractVTKLegacyData::BINARY);
    } else {
        Log::DefaultLog.WriteError( "%s: invalid file syntax",
            this->ClassName()); // DEBUG
        return;
    }
    this->seekNextLine(buffPt);


    /* Check topology/geometry information (needs to be UNSTRUCTURED_GRID */

    line = this->readCurrentLine(buffPt);
    vislib::StringA topologyStr = line.Substring(8);
    //    printf("%s\n", topologyStr.PeekBuffer());
    if (!(topologyStr == "UNSTRUCTURED_GRID")) {
        Log::DefaultLog.WriteError( "%s: data is not of type 'UNSTRUCTURED_GRID'",
            this->ClassName()); // DEBUG
        return;
    } else {
        fr->SetTopology(AbstractVTKLegacyData::UNSTRUCTURED_GRID);
    }
    this->seekNextLine(buffPt);
}


/*
 * VTKLegacyDataLoaderUnstructuredGrid::ReadPoints
 */
void VTKLegacyDataLoaderUnstructuredGrid::readPoints(char*& buffPt, core::view::AnimDataModule::Frame* frame) {

    vislib::StringA line;

    VTKLegacyDataLoaderUnstructuredGrid::Frame* fr = dynamic_cast<VTKLegacyDataLoaderUnstructuredGrid::Frame*>(frame);
    if (fr == NULL) {
        return;
    }


    /* Get the number of vertices */

    this->seekNextToken(buffPt);
    size_t vertexCnt = atoi(this->readNextToken(buffPt));
    //    printf("vertex cnt %i\n", vertexCnt);

    // TODO Throw error if type is not float
    //    // Get the type of the coordinates
    //    vislib::StringA vertexTypeStr = line.Substring(line.Find(" ")+1);
    //    vertexTypeStr = vertexTypeStr.Substring(vertexTypeStr.Find(" ")+1);
    //    vertexTypeStr.Remove(" ");
    //    printf("vertex type %s\n", vertexTypeStr.PeekBuffer());


    this->seekNextLine(buffPt);

    /* Read actual numbers */

    // NOTE: Buffer pointer should now point to first char of the vertex data

    // ASCII
    if (fr->GetEncoding() == AbstractVTKLegacyData::ASCII) {
        float* tempBuff = new float[vertexCnt * 3];
        this->readASCIIFloats(buffPt, tempBuff, vertexCnt * 3);
        fr->SetPoints(tempBuff, vertexCnt);
        delete[] tempBuff;
        // BINARY TODO atm assumes float!!!!
    } else {
#ifdef SWAP_BYTES
        this->swapBytes(buffPt, 4, vertexCnt * 3);
#endif
#ifdef NORMALIZE_RADIUS
        for (unsigned int i = 0; i < vertexCnt * 3; i++) {
            ((float*)(buffPt))[i] /= this->globalRadiusParam.Param<core::param::FloatParam>()->Value();
        }
#endif
        fr->SetPoints((const float*)(buffPt), vertexCnt);
        // Increment buffer pointer
        buffPt += vertexCnt * 3 * sizeof(float); // TODO assumes float
    }
}


/*
 * VTKLegacyDataLoaderUnstructuredGrid::readPointData
 */
void VTKLegacyDataLoaderUnstructuredGrid::readFieldData(
    char*& buffPt, core::view::AnimDataModule::Frame* frame, size_t numArrays) {

    vislib::StringA line, fieldId, nComponentsStr, nTupelStr, dataTypeStr;
    size_t nComponents, nTupel;
    AbstractVTKLegacyData::DataType t;

    VTKLegacyDataLoaderUnstructuredGrid::Frame* fr = dynamic_cast<VTKLegacyDataLoaderUnstructuredGrid::Frame*>(frame);
    if (fr == NULL) {
        return;
    }

    /* Get the number of vertices */

    line = this->readCurrentLine(buffPt);
    this->seekNextLine(buffPt); // Skip to next line
    // std::cout << "Line: " << line << std::endl;

    // Loop through all field arrays
    for (size_t a = 0; a < numArrays; ++a) {

        fieldId = this->readNextToken(buffPt); // Read field name
        // std::cout << "Field: " <<  fieldId << std::endl;
        this->seekNextToken(buffPt);

        nComponentsStr = this->readNextToken(buffPt); // Read number of components
        nComponents = static_cast<size_t>(atoi(nComponentsStr.PeekBuffer()));
        // printf("Number of components %i\n", nComponents);
        this->seekNextToken(buffPt);

        nTupelStr = this->readNextToken(buffPt); // Read number of tupels
        nTupel = static_cast<size_t>(atoi(nTupelStr.PeekBuffer()));
        // printf("Number of tupel %i\n", nTupel);
        this->seekNextToken(buffPt);

        dataTypeStr = this->readNextToken(buffPt); // Read number of tupels
                                                   // printf("Data type %s\n", dataTypeStr.PeekBuffer());
        t = AbstractVTKLegacyData::GetDataTypeByString(dataTypeStr);
        this->seekNextToken(buffPt);

        // ASCII
        if (fr->GetEncoding() == AbstractVTKLegacyData::ASCII) {
            if (t == AbstractVTKLegacyData::FLOAT) {
                float* tempBuff = new float[nComponents * nTupel];
                this->readASCIIFloats(buffPt, tempBuff, nComponents * nTupel);
                fr->AddPointData((const char*)(tempBuff), nTupel, nComponents, t, fieldId);
                delete[] tempBuff;
            } else {
                printf("TODO\n"); // TODO?
            }
        } else {
#ifdef SWAP_BYTES
            this->swapBytes(buffPt, 4, nComponents * nTupel); // TODO Assumes 4 byte data type!
#endif
            fr->AddPointData(buffPt, nTupel, nComponents, t, fieldId); // TODO Handle cell data aswell
        }

        // Increment buffPt
        buffPt += nComponents * nTupel * 4; // Assumes float

        // TODO: (gralkapk) This looks like hazard
        // TODO: For binary files, there will be no \n at the end of an field
        // TODO: Added fr->GetEncoding() == AbstractVTKLegacyData::ASCII as possible fix
        if (fr->GetEncoding() == AbstractVTKLegacyData::ASCII && a != numArrays - 1)
            this->seekNextLine(buffPt);
    }
}


/*
 * VTKLegacyDataLoaderUnstructuredGrid::readDataArray
 *
 * Note: for SCALAR, VECTOR, TENSOR etc. data
 */
void VTKLegacyDataLoaderUnstructuredGrid::readDataArray(char*& buffPt, core::view::AnimDataModule::Frame* frame,
    size_t nTupels, size_t nComponents, AbstractVTKLegacyData::DataAssociation) {

    vislib::StringA fieldId, dataTypeStr, line;
    AbstractVTKLegacyData::DataType t;

    VTKLegacyDataLoaderUnstructuredGrid::Frame* fr = dynamic_cast<VTKLegacyDataLoaderUnstructuredGrid::Frame*>(frame);
    if (fr == NULL) {
        return;
    }

    /* Get the id of the scalar data */

    this->seekNextToken(buffPt);           // Skip the first token of the line
    fieldId = this->readNextToken(buffPt); // Read field name
    this->seekNextToken(buffPt);


    /* Get type of the scalar data */

    dataTypeStr = this->readNextToken(buffPt); // Read number of tupels
    t = AbstractVTKLegacyData::GetDataTypeByString(dataTypeStr);
    this->seekNextToken(buffPt);


    // Get scalar data
    if (this->readNextToken(buffPt) == "LOOKUP_TABLE") {
        // Skip next line specifying the lookup table type TODO custom lookup_tables?
        this->seekNextLine(buffPt);
    }

    // ASCII
    if (fr->GetEncoding() == AbstractVTKLegacyData::ASCII) {
        if (t == AbstractVTKLegacyData::FLOAT) {
            float* tempBuff = new float[nTupels * nComponents];
            this->readASCIIFloats(buffPt, tempBuff, nTupels);
            fr->AddPointData((const char*)(tempBuff), nTupels, nComponents, t, fieldId);
            delete[] tempBuff;
        } else {
            printf("TODO\n"); // TODO?
        }
    } else {
#ifdef SWAP_BYTES
        this->swapBytes(buffPt, 4, nTupels * nComponents); // TODO Assumes 4 byte data type!
#endif
        fr->AddPointData(buffPt, nTupels, nComponents, t, fieldId);
    }

    // Increment buffer
    buffPt += nTupels * sizeof(float) * nComponents; // TODO assumes float4
}


/*
 * VTKLegacyDataLoaderUnstructuredGrid::swapBytes
 */
void VTKLegacyDataLoaderUnstructuredGrid::swapBytes(char* buffPt, size_t stride, size_t cnt) {
    char tmpChar;
    for (size_t c = 0; c < cnt; ++c) {
        tmpChar = buffPt[c * stride + 0];
        buffPt[c * stride + 0] = buffPt[c * stride + 3];
        buffPt[c * stride + 3] = tmpChar;
        tmpChar = buffPt[c * stride + 1];
        buffPt[c * stride + 1] = buffPt[c * stride + 2];
        buffPt[c * stride + 2] = tmpChar;
    }
}


/*
 * VTKLegacyDataLoaderUnstructuredGrid::Frame::Frame
 */
VTKLegacyDataLoaderUnstructuredGrid::Frame::Frame(megamol::core::view::AnimDataModule& owner)
        : core::view::AnimDataModule::Frame(owner) {}


/*
 * VTKLegacyDataLoaderUnstructuredGrid::Frame::~Frame
 */
VTKLegacyDataLoaderUnstructuredGrid::Frame::~Frame(void) {}
