//
// VTIWriter.cpp
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 19, 2013
//     Author: scharnkn
//

#include "VTIWriter.h"
#include "Base64.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/log/Log.h"
#include "protein_calls/VTIDataCall.h"
#include "stdafx.h"
#include "sys/stat.h"
#include "vislib/StringConverter.h"
#include "vislib/sys/File.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#if defined(_WIN32)
#include <direct.h>
#endif

using namespace megamol;
using namespace megamol::protein;
using namespace megamol::core::utility::log;


/*
 * VTIWriter::VTIWriter
 */
VTIWriter::VTIWriter()
        : AbstractJob()
        , Module()
        , dataCallerSlot("getdata", "Connects the writer module with the data source.")
        , minFrameSlot("minFrame", "Parameter to determine the first frame to be written")
        , nFramesSlot("nFrames", "Parameter to determine the number of frames to be written")
        , strideSlot("stride", "Parameter to determine the stride used when writing frames")
        , filenamePrefixSlot("filenamePrefix", "Parameter for the filename prefix")
        , outDirSlot("outputFolder", "Parameter for the output folder")
        , dataFormatSlot("dataFormat", "Parameter for the output format of the data")
        , jobDone(false)
        , filenameDigits(0) {

    // Make data caller slot available
    this->dataCallerSlot.SetCompatibleCall<protein_calls::VTIDataCallDescription>();
    this->MakeSlotAvailable(&this->dataCallerSlot);

    // Parameter to determine the first frame to be written
    this->minFrameSlot << new core::param::IntParam(0);
    this->MakeSlotAvailable(&this->minFrameSlot);

    // Parameter to determine the number of frames to be written
    this->nFramesSlot << new core::param::IntParam(1);
    this->MakeSlotAvailable(&this->nFramesSlot);

    // Parameter to determine the stride used when writing frames
    this->strideSlot << new core::param::IntParam(1);
    this->MakeSlotAvailable(&this->strideSlot);

    // Parameter for the filename prefix
    this->filenamePrefixSlot << new core::param::StringParam("out");
    this->MakeSlotAvailable(&this->filenamePrefixSlot);

    // Parameter for the output folder
    this->outDirSlot << new core::param::StringParam(".");
    this->MakeSlotAvailable(&this->outDirSlot);

    // Parameter for the output format of the data
    megamol::core::param::EnumParam* fp =
        new megamol::core::param::EnumParam((int)protein_calls::VTKImageData::VTISOURCE_ASCII);
    fp->SetTypePair(protein_calls::VTKImageData::VTISOURCE_BINARY, "Binary");
    fp->SetTypePair(protein_calls::VTKImageData::VTISOURCE_ASCII, "Ascii");
    //fp->SetTypePair(VTKImageData::VTISOURCE_APPENDED, "Appended"); // TODO Not implemented yet
    this->dataFormatSlot << fp;
    this->MakeSlotAvailable(&this->dataFormatSlot);
}


/*
 * VTIWriter::~VTIWriter
 */
VTIWriter::~VTIWriter() {
    this->Release();
}


/*
 * VTIWriter::IsRunning
 */
bool VTIWriter::IsRunning(void) const {
    return (!(this->jobDone));
}


/*
 * VTIWriter::Start
 */
bool VTIWriter::Start(void) {

    protein_calls::VTIDataCall* dc = this->dataCallerSlot.CallAs<protein_calls::VTIDataCall>();
    if (dc == NULL) {
        this->jobDone = true;
        return false;
    }

    // Get extent of the data set
    if (!(*dc)(protein_calls::VTIDataCall::CallForGetExtent))
        return false;

    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: Number of frames %u", this->ClassName(), dc->FrameCount());

    // Determine maximum frame to be written
    unsigned int maxFrame = this->minFrameSlot.Param<core::param::IntParam>()->Value() +
                            this->strideSlot.Param<core::param::IntParam>()->Value() *
                                (this->nFramesSlot.Param<core::param::IntParam>()->Value() - 1);

    // Check whether the selected frames are valid
    if (maxFrame >= dc->FrameCount()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "%s: Invalid frame selection (max frame is %u, but number of frames is %u", this->ClassName(), maxFrame,
            dc->FrameCount());
        this->jobDone = true;
        return false;
    }

    // Determine number of digits to be used in generated filenames
    unsigned int counter = 0;
    this->filenameDigits = 0;
#if defined(_WIN32)
    this->filenameDigits = 6; // TODO Do not hardcode number of digits
#else
    do {
        this->filenameDigits += 1;
        counter += ::exp10f(this->filenameDigits);
    } while (counter <= dc->FrameCount());
#endif

    // Create output directories if necessary
    if (!this->createDirectories(this->outDirSlot.Param<core::param::StringParam>()->Value().c_str())) {
        this->jobDone = true;
        return false;
    }

    // Loop through all the selected frames
    for (int fr = this->minFrameSlot.Param<core::param::IntParam>()->Value(); fr <= static_cast<int>(maxFrame);
         fr += this->strideSlot.Param<core::param::IntParam>()->Value()) {

        // Request frame
        dc->SetFrameID(fr, true); // Set 'force' flag
        if (!(*dc)(protein_calls::VTIDataCall::CallForGetExtent)) {
            this->jobDone = true;
            return false;
        }
        if (!(*dc)(protein_calls::VTIDataCall::CallForGetData)) {
            this->jobDone = true;
            return false;
        }

        // Write frame to file
        if (!this->writeFile(dc)) {
            return false;
        }

        // Unlock frame
        dc->Unlock();
    }

    this->jobDone = true;
    return true; // TODO
}


/*
 * VTIWriter::Terminate
 */
bool VTIWriter::Terminate(void) {
    return true;
}


/*
 * VTIWriter::create
 */
bool VTIWriter::create(void) {
    return true;
}


/*
 * VTIWriter::release
 */
void VTIWriter::release(void) {
    this->buffEn.Release();
    this->buffDec.Release();
}


/*
 * VTIWriter::createDirectories
 */
bool VTIWriter::createDirectories(vislib::StringA folder) {
    using namespace vislib;
    using namespace vislib::sys;

    if (File::IsDirectory(folder)) {
        return true;
    } else {
        if (folder.Contains("/")) {
            if (this->createDirectories(folder.Substring(0, folder.FindLast("/")))) {
#if defined(_WIN32)
                _mkdir(folder.PeekBuffer());
#else
                mkdir(folder, 777);
#endif
            }
        } else {
#if defined(_WIN32)
            _mkdir(folder.PeekBuffer());
#else
            mkdir(folder, 777);
#endif
        }
    }
    return true;
}


/*
 * VTIWriter::GetFormatStr
 */
vislib::TString VTIWriter::getFormatStr(protein_calls::VTKImageData::DataFormat f) {
    switch (f) {
    case protein_calls::VTKImageData::VTISOURCE_BINARY:
        return vislib::TString("binary");
    case protein_calls::VTKImageData::VTISOURCE_ASCII:
        return vislib::TString("ascii");
    case protein_calls::VTKImageData::VTISOURCE_APPENDED:
        return vislib::TString("appended");
    default:
        return vislib::TString("");
    }
}


/*
 * VTIWriter::GetTypeStr
 */
vislib::TString VTIWriter::getTypeStr(protein_calls::VTKImageData::DataArray::DataType t) {
    switch (t) {
    case protein_calls::VTKImageData::DataArray::VTI_FLOAT:
        return vislib::TString("Float32");
    case protein_calls::VTKImageData::DataArray::VTI_INT:
        return vislib::TString("Int32");
    case protein_calls::VTKImageData::DataArray::VTI_UINT:
        return vislib::TString("UInt32");
    case protein_calls::VTKImageData::DataArray::VTI_DOUBLE:
        return vislib::TString("Float64");
    case protein_calls::VTKImageData::DataArray::VTI_UNKNOWN:
        return vislib::TString("Unknown");
    default:
        return vislib::TString("");
    }
}


/*
 * VTIWriter::writeDataAscii
 */
bool VTIWriter::writeDataAscii(
    const void* data, size_t size, std::ofstream& outfile, protein_calls::VTKImageData::DataArray::DataType t) {
    switch (t) {
    case protein_calls::VTKImageData::DataArray::VTI_FLOAT:
        this->writeDataAsciiFloat((const float*)data, size, outfile);
        return true;
    case protein_calls::VTKImageData::DataArray::VTI_INT:
        return true;
    case protein_calls::VTKImageData::DataArray::VTI_UINT:
        return true;
    case protein_calls::VTKImageData::DataArray::VTI_DOUBLE:
        return true;
    case protein_calls::VTKImageData::DataArray::VTI_UNKNOWN:
        return true;
    default:
        return "";
    }
}


/*
 * VTIWriter::writeDataBinary
 */
bool VTIWriter::writeDataBinary(
    const void* data, size_t size, std::ofstream& outfile, protein_calls::VTKImageData::DataArray::DataType t) {
    switch (t) {
    case protein_calls::VTKImageData::DataArray::VTI_FLOAT:
        this->writeDataBinaryFloat((const float*)data, size, outfile);
        return true;
    case protein_calls::VTKImageData::DataArray::VTI_INT:
        return true;
    case protein_calls::VTKImageData::DataArray::VTI_UINT:
        return true;
    case protein_calls::VTKImageData::DataArray::VTI_DOUBLE:
        return true;
    case protein_calls::VTKImageData::DataArray::VTI_UNKNOWN:
        return true;
    default:
        return "";
    }
}


/*
 * VTIWriter::writeDataAsciiFloat
 */
bool VTIWriter::writeDataAsciiFloat(const float* data, size_t size, std::ofstream& outfile) {
    for (size_t i = 0; i < size - 1; ++i) {
        outfile << std::scientific << data[i] << " ";
    }
    outfile << std::scientific << data[size - 1];
    return true;
}

/*
 * VTIWriter::writeDataBinaryFloat
 */
bool VTIWriter::writeDataBinaryFloat(const float* data, size_t size, std::ofstream& outfile) {

    int sizeBytes = static_cast<int>(size * sizeof(float));
    size_t sizeFillerBytes = (sizeBytes + 4) + (3 - (sizeBytes + 4) % 3) % 3;
    //printf("Data needs %u filler bytes\n", (3-(sizeBytes+4)%3)%3);
    //char sizeBuffEncoded[8];

    // First four bytes is the number of bytes in the following encoded data,
    // it is encoded separately from the rest of the data, two filler bytes
    // are appended --> 6 bytes --> 8 bytes when encoded
    //Base64::Encode((const char *)(&sizeBytes), &sizeBuffEncoded[0], 4);
    //outfile.write(&sizeBuffEncoded[0], 8);

    // Now encode the actual data
    this->buffEn.Validate((sizeFillerBytes / 3) * 4); // Buffer for encoded data
    this->buffDec.Validate(sizeBytes + 4);            // Buffer for decoded data + size

    memcpy(this->buffDec.Peek(), (const char*)(&sizeBytes), 4);
    memcpy(this->buffDec.Peek() + 4, data, sizeBytes);

    Base64::Encode((const char*)this->buffDec.Peek(), this->buffEn.Peek(), sizeBytes + 4);
    outfile.write(this->buffEn.Peek(), this->buffEn.GetCount());

    return true;
}


/*
 * VTIWriter::writeDataArray
 */
bool VTIWriter::writeDataArray(const protein_calls::VTIDataCall* dc, bool isPointData, unsigned int dataIdx,
    unsigned int pieceIdx, std::ofstream& outfile) {

    // Write point data
    if (isPointData) {
        outfile << "        <DataArray type=\"";
        outfile << this->getTypeStr(dc->GetPiecePointArrayType(dataIdx, pieceIdx));
        outfile << "\" Name=\"" << dc->GetPointDataArrayId(dataIdx, pieceIdx);
        outfile << "\" format=\"";
        outfile << this->getFormatStr(static_cast<protein_calls::VTKImageData::DataFormat>(
            this->dataFormatSlot.Param<core::param::EnumParam>()->Value()));
        outfile << "\" RangeMin=\"" << std::scientific << dc->GetPointDataArrayMin(dataIdx, pieceIdx);
        outfile << "\" RangeMax=\"" << std::scientific << dc->GetPointDataArrayMax(dataIdx, pieceIdx);
        outfile << "\">" << std::endl;
        // Write data according to parameter
        switch (this->dataFormatSlot.Param<core::param::EnumParam>()->Value()) {
        case protein_calls::VTKImageData::VTISOURCE_ASCII:
            this->writeDataAscii(dc->GetPointDataByIdx(dataIdx, pieceIdx),
                dc->GetPiecePointArraySize(dataIdx, pieceIdx), outfile, dc->GetPiecePointArrayType(dataIdx, pieceIdx));
            break;
        case protein_calls::VTKImageData::VTISOURCE_BINARY:
            this->writeDataBinary(dc->GetPointDataByIdx(dataIdx, pieceIdx),
                dc->GetPiecePointArraySize(dataIdx, pieceIdx), outfile, dc->GetPiecePointArrayType(dataIdx, pieceIdx));
            break;
        case protein_calls::VTKImageData::VTISOURCE_APPENDED:
            break; // TODO
        }
        outfile << std::endl;
        outfile << "        </DataArray>" << std::endl;
    } else { // Write cell data
        outfile << "        <DataArray type=\"";
        outfile << this->getTypeStr(dc->GetPieceCellArrayType(dataIdx, pieceIdx));
        outfile << "\" Name=\"" << dc->GetCellDataArrayId(dataIdx, pieceIdx);
        outfile << "\" format=\"";
        outfile << this->getFormatStr(static_cast<protein_calls::VTKImageData::DataFormat>(
            this->dataFormatSlot.Param<core::param::EnumParam>()->Value()));
        outfile << "\" RangeMin=\"" << std::scientific << dc->GetCellDataArrayMin(dataIdx, pieceIdx);
        outfile << "\" RangeMax=\"" << std::scientific << dc->GetCellDataArrayMax(dataIdx, pieceIdx);
        outfile << "\">" << std::endl;
        // Write data according to parameter
        switch (this->dataFormatSlot.Param<core::param::EnumParam>()->Value()) {
        case protein_calls::VTKImageData::VTISOURCE_ASCII:
            this->writeDataAscii(dc->GetCellDataByIdx(dataIdx, pieceIdx), dc->GetPieceCellArraySize(dataIdx, pieceIdx),
                outfile, dc->GetPiecePointArrayType(dataIdx, pieceIdx));
            break;
        case protein_calls::VTKImageData::VTISOURCE_BINARY:
            this->writeDataBinary(dc->GetCellDataByIdx(dataIdx, pieceIdx), dc->GetPieceCellArraySize(dataIdx, pieceIdx),
                outfile, dc->GetPieceCellArrayType(dataIdx, pieceIdx));
            break;
        case protein_calls::VTKImageData::VTISOURCE_APPENDED:
            break; // TODO
        }
        outfile << std::endl;
        outfile << "        </DataArray>" << std::endl;
    }

    return true;
}


/*
 * VTIWriter::writeFile
 */
bool VTIWriter::writeFile(protein_calls::VTIDataCall* dc) {

    // Generate filename based on frame number
    std::string filename;
    std::stringstream ss;
    ss.width(this->filenameDigits);
    ss.fill('0');
    std::string digits;
    ss << dc->FrameID();
    filename.append(
        (const char*)(this->outDirSlot.Param<core::param::StringParam>()->Value().c_str())); // Set output folder
    filename.append("/");
    filename.append(
        (const char*)(this->filenamePrefixSlot.Param<core::param::StringParam>()->Value().c_str())); // Set prefix
    filename.append(".");
    filename.append((ss.str()).c_str());
    filename.append(".vti");

    Log::DefaultLog.WriteMsg(
        Log::LEVEL_INFO, "%s: Writing frame %u to file '%s'", this->ClassName(), dc->FrameID(), filename.data());

    // Try to open the output file
    std::ofstream outfile;
    outfile.open(filename.data(), std::ios::out | std::ios::binary);
    if (!outfile.good()) {
        Log::DefaultLog.WriteMsg(
            Log::LEVEL_ERROR, "%s: Unable to open file '%s'\n", this->ClassName(), filename.data());
        return false;
    }


    /* Write image data to file */

    // Header data
    outfile << "<?xml version=\"1.0\"?>" << std::endl;
    outfile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">"
            << std::endl; // TODO Format for byte order
    // Whole extent
    outfile << "  <ImageData WholeExtent=\"";
    outfile << dc->GetWholeExtent().Left() << " ";
    outfile << dc->GetWholeExtent().Right() << " ";
    outfile << dc->GetWholeExtent().Bottom() << " ";
    outfile << dc->GetWholeExtent().Top() << " ";
    outfile << dc->GetWholeExtent().Back() << " ";
    outfile << dc->GetWholeExtent().Front() << "\"";
    // Origin
    outfile << " Origin=\"" << std::scientific << dc->GetOrigin().X() << " " << dc->GetOrigin().Y() << " "
            << dc->GetOrigin().Z() << "\"";
    outfile << " Spacing=\"" << std::scientific << dc->GetSpacing().X() << " " << dc->GetSpacing().Y() << " "
            << dc->GetSpacing().Z() << "\"";
    outfile << ">" << std::endl;


    /* Write pieces */

    for (uint i = 0; i < dc->GetNumberOfPieces(); ++i) {
        if (!this->writePiece(dc, i, outfile)) {
            return false;
        }
    }


    /* Close tags */

    outfile << "  </ImageData>" << std::endl;
    outfile << "</VTKFile>" << std::endl;

    // Close the output file
    outfile.close();

    return true;
}


/*
 * VTIWriter::writePiece
 */
bool VTIWriter::writePiece(const protein_calls::VTIDataCall* dc, uint idx, std::ofstream& outfile) {


    /* Write piece data header */

    outfile << "    <Piece ";
    // Extent
    outfile << "Extent=\"";
    outfile << dc->GetPieceExtent(idx).Left() << " ";
    outfile << dc->GetPieceExtent(idx).Right() << " ";
    outfile << dc->GetPieceExtent(idx).Bottom() << " ";
    outfile << dc->GetPieceExtent(idx).Top() << " ";
    outfile << dc->GetPieceExtent(idx).Back() << " ";
    outfile << dc->GetPieceExtent(idx).Front() << "\">" << std::endl;

    /* Write point data */

    // Header
    outfile << "      <PointData";
    // Loop through all data arrays in this pieces point data to write the
    for (size_t p = 0; p < dc->GetArrayCntOfPiecePointData(idx); ++p) {
        // Write either scalar, vector, or tensor data
        size_t nComponents = dc->GetPointDataArrayNumberOfComponents((unsigned int)p, idx);
        if (nComponents == 1) {
            outfile << " Scalars=\"";
        } else if (nComponents == 3) {
            outfile << " Vectors=\"";
        } else {
            outfile << " Tensors=\"";
        }
        printf("ID: %s\n", dc->GetPointDataArrayId((unsigned int)p, idx).PeekBuffer());
        outfile << dc->GetPointDataArrayId((unsigned int)p, idx);
        outfile << "\"";
    }
    outfile << ">" << std::endl;
    // Actual data
    for (size_t p = 0; p < dc->GetArrayCntOfPiecePointData(idx); ++p) {
        // Write data
        this->writeDataArray(dc, true, (unsigned int)p, idx, outfile);
    }
    // End point data
    outfile << "      </PointData>" << std::endl;

    /* Write cell data */

    // Header
    outfile << "      <CellData";
    // Loop through all data arrays in this pieces point data to write the
    for (size_t p = 0; p < dc->GetArrayCntOfPieceCellData(idx); ++p) {
        // Write either scalar, vector, or tensor data
        size_t nComponents = dc->GetCellDataArrayNumberOfComponents((unsigned int)p, idx);
        if (nComponents == 1) {
            outfile << " Scalars=\"";
        } else if (nComponents == 3) {
            outfile << " Vectors=\"";
        } else {
            outfile << " Tensors=\"";
        }
        outfile << dc->GetCellDataArrayId((unsigned int)p, idx);
        outfile << "\"";
    }
    outfile << ">" << std::endl;
    // Actual data
    for (size_t p = 0; p < dc->GetArrayCntOfPieceCellData(idx); ++p) {
        // Write data
        this->writeDataArray(dc, true, (unsigned int)p, idx, outfile);
    }
    // End cell data
    outfile << "      </CellData>" << std::endl;

    // End piece
    outfile << "    </Piece>" << std::endl;

    return true;
}
