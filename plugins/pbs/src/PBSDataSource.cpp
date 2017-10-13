/*
 * PBSDataSource.cpp
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "PBSDataSource.h"

#include <fstream>
#include <vector>

#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include "vislib/sys/Log.h"


using namespace megamol;
using namespace megamol::pbs;


PBSDataSource::PBSDataSource(void) : core::Module(),
filenameSlot("filename", "The path to the PBS file to load."),
datatypeSlot("datatype", "The datatype of the field in the PBS file."),
toleranceSlot("tolerance", "The ZFP tolerance used for compressing the PBS file."),
numElementsSlot("numElements", "The number of elements stored in the PBS file."),
getData("getdata", "Slot to request data from this data source.") {

}


PBSDataSource::~PBSDataSource(void) {
    this->Release();
}


bool PBSDataSource::create(void) {
    return true;
}


void PBSDataSource::release(void) {

}


bool PBSDataSource::filenameChanged(core::param::ParamSlot& slot) {
    auto vl_filename = this->filenameSlot.Param<core::param::FilePathParam>()->Value();

    std::ifstream file(vl_filename, std::ios::binary | std::ios::ate);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(file_size);
    if (!file.read(buffer.data(), file_size)) {
        // error occurred during read
        vislib::sys::Log::DefaultLog.WriteError("PBSDataSource::filenameChanged: Could not read file %s\n", vl_filename);
        file.close();
        return false;
    }

    file.close();

    zfp_type type = static_cast<zfp_type>(this->datatypeSlot.Param<core::param::EnumParam>()->Value());

    unsigned int num_elements = this->numElementsSlot.Param<core::param::IntParam>()->Value();

    this->data.clear();
    this->data.resize(num_elements*this->datatype_size[type]);

    zfp_field* field = zfp_field_1d(this->data.data(), type, num_elements);

    zfp_stream *zfp = zfp_stream_open(nullptr);

    zfp_stream_set_accuracy(zfp, this->toleranceSlot.Param<core::param::FloatParam>()->Value());

    bitstream* stream = stream_open(buffer.data(), buffer.size());

    zfp_stream_set_bit_stream(zfp, stream);

    if (!zfp_decompress(zfp, field)) {
        // error occurred during decompression
        vislib::sys::Log::DefaultLog.WriteError("PBSDataSource::filenameChanged: Could not decompress ZFP stream\n");

        zfp_field_free(field);
        zfp_stream_close(zfp);
        stream_close(stream);

        return false;
    }

    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);

    return true;
}


bool PBSDataSource::getDataCallback(core::Call& caller) {
    return false;
}


bool PBSDataSource::getExtentCallback(core::Call& caller) {
    return false;
}
