/*
 * DatRawWriter.cpp
 *
 * Copyright (C) 2019 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */
#include "DatRawWriter.h"

#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/utility/log/Log.h"
#include <iomanip>
#include <sstream>
#include <string>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::volume;

/*
 * DatRawWriter::DatRawWriter
 */
DatRawWriter::DatRawWriter()
        : AbstractDataWriter()
        , filenameSlot("filepathPrefix", "The path prefix of the folder and file the files will be written to. To this "
                                         "path the ending .dat and .raw will be added")
        , frameIDSlot("frameID", "The id of the data frame that will be written")
        , dataSlot("data", "The slot requesting the data to be written") {

    this->filenameSlot.SetParameter(
        new param::FilePathParam("", megamol::core::param::FilePathParam::Flag_File_ToBeCreated));
    this->MakeSlotAvailable(&this->filenameSlot);

    this->frameIDSlot.SetParameter(new param::IntParam(0, 0));
    this->MakeSlotAvailable(&this->frameIDSlot);

    this->dataSlot.SetCompatibleCall<geocalls::VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->dataSlot);
}

/*
 * DatRawWriter::~DatRawWriter
 */
DatRawWriter::~DatRawWriter() {
    this->Release();
}

/*
 * DatRawWriter::create
 */
bool DatRawWriter::create() {
    return true;
}

/*
 * DatRawWriter::release
 */
void DatRawWriter::release() {}

/*
 * DatRawWriter::run
 */
bool DatRawWriter::run() {
    using megamol::core::utility::log::Log;
    std::string filepath(this->filenameSlot.Param<param::FilePathParam>()->Value().generic_string());
    if (filepath.empty()) {
        Log::DefaultLog.WriteError("No file path specified. Abort.");
        return false;
    }
    if (filepath[filepath.length() - 1] == '/') {
        Log::DefaultLog.WriteError("Only a path, no filename prefix given. Abort.");
        return false;
    }

    geocalls::VolumetricDataCall* vdc = this->dataSlot.CallAs<geocalls::VolumetricDataCall>();
    if (vdc == nullptr) {
        Log::DefaultLog.WriteError("No data source connected. Abort.");
        return false;
    }

    unsigned int frame = static_cast<unsigned int>(this->frameIDSlot.Param<param::IntParam>()->Value());
    vdc->SetFrameID(frame, true);

    if (!(*vdc)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS)) {
        Log::DefaultLog.WriteError("Bounding box retrieval failed. Abort");
        return false;
    }
    if (frame >= vdc->FrameCount()) {
        Log::DefaultLog.WriteError("Selected frame %u not available (total %u frames)", frame, vdc->FrameCount());
        return false;
    }
    if (!(*vdc)(geocalls::VolumetricDataCall::IDX_GET_METADATA)) {
        Log::DefaultLog.WriteError("Metadata retrieval failed. Abort.");
        return false;
    }
    if (!(*vdc)(geocalls::VolumetricDataCall::IDX_GET_DATA)) {
        Log::DefaultLog.WriteError("Data retrieval failed. Abort.");
        return false;
    }

    std::stringstream datpath;
    datpath << filepath << std::setw(4) << std::setfill('0') << std::to_string(frame) << ".dat";
    std::stringstream rawpath;
    rawpath << filepath << std::setw(4) << std::setfill('0') << std::to_string(frame) << ".raw";
    return writeFrame(datpath.str(), rawpath.str(), *vdc);
}

/*
 * DatRawWriter::writeFrame
 */
bool DatRawWriter::writeFrame(std::string datpath, std::string rawpath, geocalls::VolumetricDataCall& data) {
    using megamol::core::utility::log::Log;
    auto lastPos = rawpath.find_last_of("/\\");
    std::string writestring = rawpath.substr(lastPos + 1);

    auto meta = data.GetMetadata();

    std::ofstream datfile(datpath, std::ios_base::binary);
    if (datfile.is_open()) {
        datfile << "ObjectFileName: " << writestring << std::endl;
        datfile << "Format:         ";
        switch (meta->ScalarType) {
        case geocalls::VolumetricDataCall::ScalarType::BITS:
            datfile << "UCHAR" << std::endl;
            break;
        case geocalls::VolumetricDataCall::ScalarType::FLOATING_POINT:
            datfile << "FLOAT" << std::endl;
            break;
        case geocalls::VolumetricDataCall::ScalarType::SIGNED_INTEGER:
            datfile << "INT" << std::endl;
            break;
        case geocalls::VolumetricDataCall::ScalarType::UNSIGNED_INTEGER:
            datfile << "UINT" << std::endl;
            break;
        default:
            Log::DefaultLog.WriteError("No Proper file format selected. Dat output is corrupted.");
            return false;
        }
        datfile << "GridType:       ";
        switch (meta->GridType) {
        case geocalls::VolumetricDataCall::GridType::CARTESIAN:
            datfile << "EQUIDISTANT" << std::endl;
            break;
        case geocalls::VolumetricDataCall::GridType::RECTILINEAR:
            datfile << "RECTILINEAR" << std::endl;
            break;
        case geocalls::VolumetricDataCall::GridType::TETRAHEDRAL:
            datfile << "TETRAHEDRA" << std::endl;
            break;
        default:
            Log::DefaultLog.WriteError("No Proper grid type selected. Dat output is corrupted.");
            return false;
        }
        datfile << "Components:     " << meta->Components << std::endl;
        datfile << "Dimensions:     "
                << "3" << std::endl; // The input only supports 3D volumes
        datfile << "TimeSteps:      "
                << "1" << std::endl; // Not all readers support this, so it is always 1
        datfile << "ByteOrder:      "
                << "LITTLE_ENDIAN" << std::endl; // On normal systems alway little endian
        datfile << "Resolution:     " << meta->Resolution[0] << " " << meta->Resolution[1] << " " << meta->Resolution[2]
                << std::endl;
        datfile << "SliceThickness: " << meta->SliceDists[0][0] << " " << meta->SliceDists[1][0] << " "
                << meta->SliceDists[1][0] << std::endl;
        datfile << "Origin:         " << meta->Origin[0] << " " << meta->Origin[1] << " " << meta->Origin[2]
                << std::endl;
        datfile << "Time:           " << data.FrameID() << std::endl;
        datfile.close();
        Log::DefaultLog.WriteInfo("Dat file successfully written to \"%s\"", datpath.c_str());
    } else {
        Log::DefaultLog.WriteError("Dat file \"%s\" could not be opened", datpath.c_str());
        return false;
    }

    std::ofstream rawfile(rawpath, std::ios_base::binary);
    if (rawfile.is_open()) {
        rawfile.write(reinterpret_cast<const char*>(data.GetData()), data.GetVoxelSize() * data.GetVoxelsPerFrame());
        rawfile.close();
        Log::DefaultLog.WriteInfo("Raw file successfully written to \"%s\"", rawpath.c_str());
    } else {
        Log::DefaultLog.WriteError("Raw file \"%s\" could not be opened", rawpath.c_str());
        return false;
    }

    return true;
}
