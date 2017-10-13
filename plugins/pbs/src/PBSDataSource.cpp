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
start_idx_slot("start_idx", "The start idx of chunks to read."),
end_idx_slot("end_idx", "The end idx of chunks to read."),
start_region_idx_slot("start_region_idx", "The start idx of chunks to read."),
end_region_idx_slot("end_idx", "The end idx of chunks to read."),
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


void PBSDataSource::clearBuffers(void) {
    this->x_data.clear();
    this->y_data.clear();
    this->z_data.clear();
    this->nx_data.clear();
    this->ny_data.clear();
    this->cr_data.clear();
    this->cg_data.clear();
    this->cb_data.clear();
}


bool PBSDataSource::readPBSFile(const std::string& filename, std::vector<char>& data, const zfp_type type, const unsigned int num_elements, const double tol) {
    //unsigned int num_elements = 0;
    //double tol = 0.0;
    //int _type = 0;

    //std::ifstream header(filename + ".txt");
    //if (!(header >> num_elements >> tol >> _type)) {
    //    // error occurred during header processing
    //    vislib::sys::Log::DefaultLog.WriteError("PBSDataSource::filenameChanged: Could not read file %s\n", filename + ".txt");
    //    header.close();
    //    return false;
    //}
    //zfp_type type = static_cast<zfp_type>(_type);
    //header.close();

    std::ifstream file(filename + ".zfp", std::ios::binary | std::ios::ate);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(file_size);
    if (!file.read(buffer.data(), file_size)) {
        // error occurred during read
        vislib::sys::Log::DefaultLog.WriteError("PBSDataSource::readPBSFile: Could not read file %s\n", filename + ".zfp");
        file.close();
        return false;
    }

    file.close();

    data.clear();
    data.resize(num_elements*this->datatype_size[type]);

    zfp_field* field = zfp_field_1d(this->data.data(), type, num_elements);

    zfp_stream* zfp = zfp_stream_open(nullptr);

    zfp_stream_set_accuracy(zfp, tol);

    bitstream* stream = stream_open(buffer.data(), buffer.size());

    zfp_stream_set_bit_stream(zfp, stream);

    if (!zfp_decompress(zfp, field)) {
        // error occurred during decompression
        vislib::sys::Log::DefaultLog.WriteError("PBSDataSource::readPBSFile: Could not decompress ZFP stream\n");

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


bool PBSDataSource::filenameChanged(core::param::ParamSlot& slot) {
    const std::string path_to_pbs = this->filenameSlot.Param<core::param::FilePathParam>()->Value();

    const auto start_idx = this->start_idx_slot.Param<core::param::IntParam>()->Value();
    const auto end_idx = this->end_idx_slot.Param<core::param::IntParam>()->Value();

    const auto start_region_idx = this->start_idx_slot.Param<core::param::IntParam>()->Value();
    const auto end_region_idx = this->end_idx_slot.Param<core::param::IntParam>()->Value();

    unsigned int num_elements = 0;
    double tol = 0.0;
    int _type = 0;

    for (int idx = start_idx; idx <= end_idx; idx++) {
        for (unsigned int attr = 0; attr < max_num_attributes; attr++) {
            // construct final filepath
            const std::string filename = path_to_pbs + "/" + this->filename_prefixes[attr] + "_" + std::to_string(idx);

            std::ifstream header(filename + ".txt");
            if (header.good()) {
                if (!(header >> num_elements >> tol >> _type)) {
                    // error occurred during header processing
                    vislib::sys::Log::DefaultLog.WriteError("PBSDataSource::filenameChanged: Could not read file %s\n", filename + ".txt");
                    header.close();
                    return false;
                }
                zfp_type type = static_cast<zfp_type>(_type);
                header.close();

                // read file
                std::vector<char> buffer;
                if (!this->readPBSFile(filename, buffer, type, num_elements, tol)) {
                    return false;
                }

                if (buffer.size() > 0) {
                    auto attr_type = static_cast<attribute_type>(attr);
                    try {
                        switch (attr_type) {
                        case attribute_type::x:
                            this->insertElements(this->x_data, buffer, num_elements);
                            break;
                        case attribute_type::y:
                            this->insertElements(this->y_data, buffer, num_elements);
                            break;
                        case attribute_type::z:
                            this->insertElements(this->z_data, buffer, num_elements);
                            break;
                        case attribute_type::nx:
                            this->insertElements(this->nx_data, buffer, num_elements);
                            break;
                        case attribute_type::ny:
                            this->insertElements(this->ny_data, buffer, num_elements);
                            break;
                        case attribute_type::cr:
                            this->insertElements(this->cr_data, buffer, num_elements);
                            break;
                        case attribute_type::cg:
                            this->insertElements(this->cg_data, buffer, num_elements);
                            break;
                        case attribute_type::cb:
                            this->insertElements(this->cb_data, buffer, num_elements);
                            break;
                        }
                    } catch (std::out_of_range &e) {
                        vislib::sys::Log::DefaultLog.WriteError("%s", e.what());
                        return false;
                    }
                }
            } else {
                // print info that attr is missing
                vislib::sys::Log::DefaultLog.WriteInfo("PBSDataSource::filenameChanged: Attribute %s for chunk %d is missing in path %s\n",
                    this->filename_prefixes[attr], idx, path_to_pbs);
            }
        }

        // check whether we are in render region and set flags accordingly
        if (start_region_idx <= idx && idx <= end_region_idx) {
            // renderable -> set flags to one
            this->render_flag.insert(this->render_flag.end(), num_elements, true);
        } else {
            this->render_flag.insert(this->render_flag.end(), num_elements, false);
        }
    }

    return true;
}


bool PBSDataSource::getDataCallback(core::Call& caller) {
    return false;
}


bool PBSDataSource::getExtentCallback(core::Call& caller) {
    return false;
}
