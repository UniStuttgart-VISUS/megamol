/*
 * WavefrontObjWriter.cpp
 *
 * Copyright (C) 2016 by Karsten Schatz
 * Copyright (C) 2016 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "WavefrontObjWriter.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include <thread>

using namespace megamol;
using namespace megamol::core;

/*
 * trisoup::WavefrontObjWriter::WavefrontObjWriter
 */
trisoup::WavefrontObjWriter::WavefrontObjWriter()
        : AbstractDataWriter()
        , filenameSlot("filename", "The path to the .obj file to be written")
        , frameIDSlot("frameID", "The ID of the frame to be written")
        , dataSlot("data", "The slot requesting the data to be written") {

    this->filenameSlot.SetParameter(
        new param::FilePathParam("", megamol::core::param::FilePathParam::Flag_File_ToBeCreated));
    this->MakeSlotAvailable(&this->filenameSlot);

    this->frameIDSlot.SetParameter(new param::IntParam(0, 0));
    this->MakeSlotAvailable(&this->frameIDSlot);

    this->dataSlot.SetCompatibleCall<megamol::geocalls::LinesDataCallDescription>();
    this->MakeSlotAvailable(&this->dataSlot);

    // TODO also accept CallTriMeshData
}


/*
 * trisoup::WavefrontObjWriter::~WavefrontObjWriter
 */
trisoup::WavefrontObjWriter::~WavefrontObjWriter() {
    this->Release();
}


/*
 * trisoup::WavefrontObjWriter::create
 */
bool trisoup::WavefrontObjWriter::create() {
    return true;
}


/*
 * trisoup::WavefrontObjWriter::release
 */
void trisoup::WavefrontObjWriter::release() {}


/*
 * trisoup::WavefrontObjWriter::run
 */
bool trisoup::WavefrontObjWriter::run() {
    using megamol::core::utility::log::Log;
    auto filename = this->filenameSlot.Param<param::FilePathParam>()->Value();
    if (filename.empty()) {
        Log::DefaultLog.WriteError("No file name specified. Abort.");
        return false;
    }

    megamol::geocalls::LinesDataCall* ldc = this->dataSlot.CallAs<megamol::geocalls::LinesDataCall>();
    if (ldc == NULL) {
        Log::DefaultLog.WriteError("No data source connected. Abort.");
        return false;
    }

    /*if (vislib::sys::File::Exists(filename.native().c_str())) {
        Log::DefaultLog.WriteWarn(
            "File %s already exists and will be overwritten.",
            filename.generic_string().c_str());
    }*/

    return writeLines(ldc);
}


/*
 * trisoup::WavefrontObjWriter::getCapabilities
 */
bool trisoup::WavefrontObjWriter::getCapabilities(DataWriterCtrlCall& call) {
    call.SetAbortable(false);
    return true;
}


bool trisoup::WavefrontObjWriter::writeLines(megamol::geocalls::LinesDataCall* ldc) {
    using megamol::core::utility::log::Log;
    vislib::TString filename(this->filenameSlot.Param<param::FilePathParam>()->Value().generic_string().c_str());
    vislib::math::Cuboid<float> bbox;
    vislib::math::Cuboid<float> cbox;
    UINT32 frameCnt = 1;

    /*unsigned int myFrame = static_cast<unsigned int>(this->frameIDSlot.Param<param::IntParam>()->Value());
    ldc->SetFrameID(myFrame, true);*/

    if (!(*ldc)(1)) {
        Log::DefaultLog.WriteWarn("Unable to query data extents.");
        bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        cbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    } else {
        if (ldc->AccessBoundingBoxes().IsObjectSpaceBBoxValid() ||
            ldc->AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
            if (ldc->AccessBoundingBoxes().IsObjectSpaceBBoxValid()) {
                bbox = ldc->AccessBoundingBoxes().ObjectSpaceBBox();
            } else {
                bbox = ldc->AccessBoundingBoxes().ObjectSpaceClipBox();
            }
            if (ldc->AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
                cbox = ldc->AccessBoundingBoxes().ObjectSpaceClipBox();
            } else {
                cbox = ldc->AccessBoundingBoxes().ObjectSpaceBBox();
            }
        } else {
            Log::DefaultLog.WriteWarn("Object space bounding boxes not valid. Using defaults");
            bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
            cbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        }
        frameCnt = ldc->FrameCount();
        if (frameCnt == 0) {
            Log::DefaultLog.WriteError("WavefronObjWriter: Data source counts zero frames. Abort.");
            ldc->Unlock();
            return false;
        }
    }

    /*if (myFrame >= frameCnt) {
        Log::DefaultLog.WriteError(
            "The requested frame does not exist. Abort.");
        ldc->Unlock();
        return false;
    }*/


    ldc->Unlock();
    for (UINT32 myFrame = 0; myFrame < frameCnt; myFrame++) {
        vislib::sys::FastFile file;
        vislib::TString outFilename = filename + vislib::TString("_") +
                                      vislib::TString(std::to_string(myFrame).c_str()) + vislib::TString(".obj");
        if (!file.Open(outFilename, vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_EXCLUSIVE,
                vislib::sys::File::CREATE_OVERWRITE)) {
            Log::DefaultLog.WriteError(
                "Unable to create output file \"%s\". Abort.", vislib::StringA(outFilename).PeekBuffer());
            ldc->Unlock();
            return false;
        }

        Log::DefaultLog.WriteInfo("Started writing data for frame %u\n", myFrame);
        int missCnt = -9;
        do {
            ldc->Unlock();
            ldc->SetFrameID(myFrame, true);
            if (!(*ldc)(0)) {
                Log::DefaultLog.WriteError("Cannot get data frame %u. Abort.\n", myFrame);
                file.Close();
                return false;
            }
            if (ldc->FrameID() != myFrame) {
                if ((missCnt % 10) == 0) {
                    Log::DefaultLog.WriteWarn("Frame %u returned on request for frame %u\n", ldc->FrameID(), myFrame);
                }
                ++missCnt;
                std::this_thread::sleep_for(std::chrono::milliseconds(1 + std::max<int>(missCnt, 0) * 100));
            }
        } while (ldc->FrameID() != myFrame);

#define ASSERT_WRITEOUT(A, S)                                   \
    if (file.Write((A), (S)) != (S)) {                          \
        Log::DefaultLog.WriteError("Write error %d", __LINE__); \
        file.Close();                                           \
        return false;                                           \
    }

        std::string header = "# OBJ file created by the MegaMol WavefrontObjWriter\n#\n";
        ASSERT_WRITEOUT(header.c_str(), header.size());

        unsigned int firstVertex = 1;       // index of the first vertex for the next line (.obj indices start at 1)
        std::vector<float> vertexPositions; // vector containing the positions of all vertices
        std::vector<std::vector<unsigned int>> indices; // vector containing the vertex indices for each line
        std::vector<size_t> lineIDs;
        indices.resize(ldc->Count());

        auto theLines = ldc->GetLines();

        // reorganize the data into the vectors
        for (unsigned int i = 0; i < ldc->Count(); i++) {
            auto line = theLines[i];
            if (line.Count() < 2) {
                Log::DefaultLog.WriteWarn(
                    "Skipping base line with index %u because of having too few vertices (%u)\n", i, line.Count());
                continue;
            }

            if (line.VertexArrayDataType() != megamol::geocalls::LinesDataCall::Lines::DT_FLOAT &&
                line.VertexArrayDataType() != megamol::geocalls::LinesDataCall::Lines::DT_DOUBLE) {
                Log::DefaultLog.WriteWarn("Skipping base line with index %u due to missing vertex data\n", i);
                continue;
            }

            lineIDs.push_back(line.ID());

            bool useIndexArray = true;
            if (line.IndexArrayDataType() != megamol::geocalls::LinesDataCall::Lines::DT_BYTE &&
                line.IndexArrayDataType() != megamol::geocalls::LinesDataCall::Lines::DT_UINT16 &&
                line.IndexArrayDataType() != megamol::geocalls::LinesDataCall::Lines::DT_UINT32) {
                useIndexArray = false;
            }

            if (line.IndexArrayDataType() == megamol::geocalls::LinesDataCall::Lines::DT_BYTE &&
                line.IndexArrayByte() == NULL) {
                useIndexArray = false;
            }
            if (line.IndexArrayDataType() == megamol::geocalls::LinesDataCall::Lines::DT_UINT16 &&
                line.IndexArrayUInt16() == NULL) {
                useIndexArray = false;
            }
            if (line.IndexArrayDataType() == megamol::geocalls::LinesDataCall::Lines::DT_UINT32 &&
                line.IndexArrayUInt32() == NULL) {
                useIndexArray = false;
            }

            for (unsigned int j = 0; j < line.Count(); j++) {
                if (useIndexArray) {
                    if (line.IndexArrayDataType() == megamol::geocalls::LinesDataCall::Lines::DT_BYTE) {
                        indices[i].push_back(firstVertex + static_cast<unsigned int>(line.IndexArrayByte()[j]));
                    } else if (line.IndexArrayDataType() == megamol::geocalls::LinesDataCall::Lines::DT_UINT16) {
                        indices[i].push_back(firstVertex + static_cast<unsigned int>(line.IndexArrayUInt16()[j]));
                    } else if (line.IndexArrayDataType() == megamol::geocalls::LinesDataCall::Lines::DT_UINT32) {
                        indices[i].push_back(firstVertex + static_cast<unsigned int>(line.IndexArrayUInt32()[j]));
                    } // else should never happen (catched the possibility before)
                } else {
                    indices[i].push_back(firstVertex + j);
                }
            }

            for (unsigned int j = 0; j < line.Count(); j++) {
                if (line.VertexArrayDataType() == megamol::geocalls::LinesDataCall::Lines::DT_FLOAT) {
                    const float* ptr = line.VertexArrayFloat();
                    vertexPositions.push_back(ptr[j * 3 + 0]);
                    vertexPositions.push_back(ptr[j * 3 + 1]);
                    vertexPositions.push_back(ptr[j * 3 + 2]);
                } else if (line.VertexArrayDataType() == megamol::geocalls::LinesDataCall::Lines::DT_DOUBLE) {
                    const double* ptr = line.VertexArrayDouble();
                    vertexPositions.push_back(static_cast<float>(ptr[j * 3 + 0]));
                    vertexPositions.push_back(static_cast<float>(ptr[j * 3 + 1]));
                    vertexPositions.push_back(static_cast<float>(ptr[j * 3 + 2]));
                } // else should never happen (catched the possibility before)
                firstVertex++;
            }
        }

        ldc->Unlock();

        // write all vertices
        for (unsigned int i = 0; i < vertexPositions.size(); i = i + 3) {
            std::string vertexString = "v ";
            vertexString += std::to_string(vertexPositions[i + 0]) + " ";
            vertexString += std::to_string(vertexPositions[i + 1]) + " ";
            vertexString += std::to_string(vertexPositions[i + 2]) + "\n";
            ASSERT_WRITEOUT(vertexString.c_str(), vertexString.size());
        }
        std::string newline = "\n";
        ASSERT_WRITEOUT(newline.c_str(), newline.size());

        // write all lines
        for (unsigned int i = 0; i < indices.size(); i++) { // loop over all lines
            bool write = false;
            std::string lineStart = "l ";
            std::string lineString = lineStart;
            for (unsigned int j = 0; j < indices[i].size(); j++) { // loop over all indices along the line
                lineString += std::to_string(indices[i][j]) + " ";

                if (j % 2 == 1) { // write a line after two vertices
                    lineString += std::to_string(lineIDs[i]) + " ";
                    lineString += newline;
                    ASSERT_WRITEOUT(lineString.c_str(), lineString.size());
                    lineString = lineStart;
                }
            }
        }
        file.Close();
    }

    Log::DefaultLog.WriteInfo("WavefrontObjWriter: Completed writing data\n");

    return true;
}
