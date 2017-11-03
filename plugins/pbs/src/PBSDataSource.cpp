/*
 * PBSDataSource.cpp
 *
 * Copyright (C) 2017 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "PBSDataSource.h"

#include <chrono>
#include <fstream>
#include <vector>
#include <sstream>

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"

#include "vislib/sys/Log.h"

#include "pbs/PBSDataCall.h"


using namespace megamol;
using namespace megamol::pbs;


PBSDataSource::PBSDataSource(void)
    : core::Module(),
      filenameSlot("filename", "The path to the PBS file to load."),
      start_idx_slot("start_idx", "The start idx of chunks to read."),
      end_idx_slot("end_idx", "The end idx of chunks to read."),
      start_region_idx_slot("start_region_idx", "The start idx of chunks to read."),
      end_region_idx_slot("end_region_idx", "The end idx of chunks to read."),
      getData("getdata", "Slot to request data from this data source."),
      with_normals(false),
      with_colors(false) {
    this->getData.SetCallback(PBSDataCall::ClassName(), PBSDataCall::FunctionName(0), &PBSDataSource::getDataCallback);
    this->getData.SetCallback(
        PBSDataCall::ClassName(), PBSDataCall::FunctionName(1), &PBSDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getData);

    this->start_idx_slot << new core::param::IntParam(-1);
    this->MakeSlotAvailable(&this->start_idx_slot);

    this->end_idx_slot << new core::param::IntParam(-1);
    this->MakeSlotAvailable(&this->end_idx_slot);

    this->start_region_idx_slot << new core::param::IntParam(-1);
    this->MakeSlotAvailable(&this->start_region_idx_slot);

    this->end_region_idx_slot << new core::param::IntParam(-1);
    this->MakeSlotAvailable(&this->end_region_idx_slot);

    this->filenameSlot << new core::param::FilePathParam("");
    this->filenameSlot.SetUpdateCallback(&PBSDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->filenameSlot);
}


PBSDataSource::~PBSDataSource(void) {
    this->Release();
}


bool PBSDataSource::create(void) {
    this->x_data = std::make_shared<std::vector<PBSStorage::pbs_coord_t>>();
    this->y_data = std::make_shared<std::vector<PBSStorage::pbs_coord_t>>();
    this->z_data = std::make_shared<std::vector<PBSStorage::pbs_coord_t>>();
    this->nx_data = std::make_shared<std::vector<PBSStorage::pbs_normal_t>>();
    this->ny_data = std::make_shared<std::vector<PBSStorage::pbs_normal_t>>();
    this->cr_data = std::make_shared<std::vector<PBSStorage::pbs_color_t>>();
    this->cg_data = std::make_shared<std::vector<PBSStorage::pbs_color_t>>();
    this->cb_data = std::make_shared<std::vector<PBSStorage::pbs_color_t>>();

    this->g_bbox = std::shared_ptr<double>(new double[6], std::default_delete<double[]>());
    this->l_bbox = std::shared_ptr<double>(new double[6], std::default_delete<double[]>());

    return true;
}


void PBSDataSource::release(void) {
}


void PBSDataSource::clearBuffers(void) {
    /*this->x_data->clear();
    this->y_data->clear();
    this->z_data->clear();
    this->nx_data->clear();
    this->ny_data->clear();
    this->cr_data->clear();
    this->cg_data->clear();
    this->cb_data->clear();*/

    this->with_normals = false;
    this->with_colors = false;
}


bool PBSDataSource::readPBSFile(std::ifstream &file, const size_t file_buffer_size, std::vector<char> &data,
    std::vector<char> &tmp, const zfp_type type, const unsigned int num_elements, const double tol) {
    // unsigned int num_elements = 0;
    // double tol = 0.0;
    // int _type = 0;

    // std::ifstream header(filename + ".txt");
    // if (!(header >> num_elements >> tol >> _type)) {
    //    // error occurred during header processing
    //    vislib::sys::Log::DefaultLog.WriteError("PBSDataSource::filenameChanged: Could not read file %s\n", filename +
    //    ".txt"); header.close(); return false;
    //}
    // zfp_type type = static_cast<zfp_type>(_type);
    // header.close();

    /*std::ifstream file(filename + ".zfp", std::ios::binary | std::ios::ate);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);*/


    // data.clear();

    if (type == zfp_type::zfp_type_none) {
        data.resize(file_buffer_size);
        if (!file.read(data.data(), file_buffer_size)) {
            // error occurred during read
            vislib::sys::Log::DefaultLog.WriteError("PBSDataSource::readPBSFile: Could not read file\n");
            // file.close();
            return false;
        }
    } else {

        auto read_start = std::chrono::high_resolution_clock::now();
        // std::vector<char> buffer(file_buffer_size);
        tmp.resize(file_buffer_size);
        if (!file.read(tmp.data(), file_buffer_size)) {
            // error occurred during read
            vislib::sys::Log::DefaultLog.WriteError("PBSDataSource::readPBSFile: Could not read file\n");
            // file.close();
            return false;
        }
        auto read_duration = std::chrono::high_resolution_clock::now() - read_start;
        vislib::sys::Log::DefaultLog.WriteInfo(
            "READ duration: %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(read_duration));

        auto zfp_start = std::chrono::high_resolution_clock::now();
        // file.close();

        data.resize(num_elements * this->datatype_size[type]);

        zfp_field *field = zfp_field_1d(data.data(), type, num_elements);

        zfp_stream *zfp = zfp_stream_open(nullptr);

        zfp_stream_set_accuracy(zfp, tol);

        bitstream *stream = stream_open(tmp.data(), tmp.size());

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

        auto zfp_duration = std::chrono::high_resolution_clock::now() - zfp_start;
        vislib::sys::Log::DefaultLog.WriteInfo(
            "ZFP duration: %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(zfp_duration));
    }
    return true;
}


bool PBSDataSource::read(void) {
    if (!this->isDirty()) {
        // required, to guard for a call to this function before params are initialized
        return true;
    }

    this->clearBuffers();

    const std::string path_to_pbs = this->filenameSlot.Param<core::param::FilePathParam>()->Value();


    const auto start_idx = this->start_idx_slot.Param<core::param::IntParam>()->Value();
    const auto end_idx = this->end_idx_slot.Param<core::param::IntParam>()->Value();

    const auto start_region_idx = this->start_region_idx_slot.Param<core::param::IntParam>()->Value();
    const auto end_region_idx = this->end_region_idx_slot.Param<core::param::IntParam>()->Value();

    unsigned int num_elements = 0;
    // double gBBox[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    /*std::vector<char> buffer(5000000000);
    std::vector<char> tmp_buffer(5000000000);*/

    std::vector<char> buffer(1);
    std::vector<char> tmp_buffer(1);

    for (int idx = start_idx; idx <= end_idx; idx++) {
        for (unsigned int attr = 0; attr < max_num_attributes; attr++) {
            // construct final filepath
            const std::string filename = path_to_pbs + "/" + this->filename_prefixes[attr] + "_" + std::to_string(idx);

            double tol = 0.0;
            int _type = 0;
            unsigned int num_buffers = 0;

            // double lBBox[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

            std::ifstream file(filename + ".zfp", std::ios::binary | std::ios::ate);
            std::ifstream header(filename + ".txt");
            if (file.good() && header.good()) {
                std::streamsize file_size = file.tellg();
                file.seekg(0, std::ios::beg);

                /*file.read(reinterpret_cast<char*>(this->g_bbox.get()), 6 * sizeof(double));
                file.read(reinterpret_cast<char*>(this->l_bbox.get()), 6 * sizeof(double));
                file.read(reinterpret_cast<char*>(&num_elements), sizeof(unsigned int));
                file.read(reinterpret_cast<char*>(&tol), sizeof(double));
                file.read(reinterpret_cast<char*>(&_type), sizeof(int));
                file.read(reinterpret_cast<char*>(&num_buffers), sizeof(unsigned int));*/

                std::string line;
                std::getline(header, line);
                std::istringstream stream(line);
                for (int i = 0; i < 6; i++) {
                    stream >> this->g_bbox.get()[i];
                }
                std::getline(header, line);
                stream = std::istringstream(line);
                for (int i = 0; i < 6; i++) {
                    stream >> this->l_bbox.get()[i];
                }
                header >> num_elements;
                header.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                header >> tol;
                header.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                header >> _type;
                header.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                header >> num_buffers;
                header.close();

                auto type = static_cast<zfp_type>(_type);

                auto file_buffer_size = file_size - file.tellg();

                // read file
                // std::vector<char> buffer;
                if (!this->readPBSFile(file, file_buffer_size, buffer, tmp_buffer, type, num_elements, tol)) {
                    file.close();
                    return false;
                }
                file.close();

                auto copy_start = std::chrono::high_resolution_clock::now();

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
                            this->with_normals = true;
                            break;
                        case attribute_type::ny:
                            this->insertElements(this->ny_data, buffer, num_elements);
                            this->with_normals = true;
                            break;
                        case attribute_type::cr:
                            this->insertElements(this->cr_data, buffer, num_elements);
                            this->with_colors = true;
                            break;
                        case attribute_type::cg:
                            this->insertElements(this->cg_data, buffer, num_elements);
                            this->with_colors = true;
                            break;
                        case attribute_type::cb:
                            this->insertElements(this->cb_data, buffer, num_elements);
                            this->with_colors = true;
                            break;
                        }
                    } catch (std::out_of_range &e) {
                        vislib::sys::Log::DefaultLog.WriteError("%s", e.what());
                        return false;
                    }
                }

                auto copy_duration = std::chrono::high_resolution_clock::now() - copy_start;
                vislib::sys::Log::DefaultLog.WriteInfo(
                    "Copy duration: %d\n", std::chrono::duration_cast<std::chrono::milliseconds>(copy_duration));
            } else {
                vislib::sys::Log::DefaultLog.WriteInfo(
                    "PBSDataSource::read: Attribute %s is missing\n", this->filename_prefixes[attr]);
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


bool PBSDataSource::isDirty(void) {
    return this->start_idx_slot.IsDirty() || this->end_idx_slot.IsDirty() || this->start_region_idx_slot.IsDirty() ||
           this->end_region_idx_slot.IsDirty();
}


void PBSDataSource::resetDirty(void) {
    this->start_idx_slot.ResetDirty();
    this->end_idx_slot.ResetDirty();
    this->start_region_idx_slot.ResetDirty();
    this->end_region_idx_slot.ResetDirty();
}


bool PBSDataSource::filenameChanged(core::param::ParamSlot &slot) {
    return this->read();
}


bool PBSDataSource::getDataCallback(core::Call &c) {
    try {
        if (this->isDirty()) {
            this->read();
            this->resetDirty();
        }

        PBSDataCall *pdc = dynamic_cast<PBSDataCall *>(&c);

        std::shared_ptr<PBSStorage> ret = std::make_shared<PBSStorage>();

        if (this->with_normals && this->with_colors) {
            ret = std::make_shared<PBSDataCall::PNCStorage>();
        } else if (this->with_colors) {
            ret = std::make_shared<PBSDataCall::CStorage>();
        } else if (this->with_normals) {
            ret = std::make_shared<PBSDataCall::NStorage>();
        }

        ret->SetRenderableFlags(std::make_shared<std::vector<bool>>(this->render_flag));
        ret->SetX(this->x_data);
        ret->SetY(this->y_data);
        ret->SetZ(this->z_data);
        ret->SetNX(this->nx_data);
        ret->SetNY(this->ny_data);
        ret->SetCR(this->cr_data);
        ret->SetCG(this->cg_data);
        ret->SetCB(this->cb_data);

        pdc->SetGlobalBBox(this->g_bbox);
        pdc->SetLocalBBox(this->l_bbox);

        pdc->SetData(std::move(ret));
    } catch (...) {
        return false;
    }

    return true;
}


bool PBSDataSource::getExtentCallback(core::Call &c) {
    try {
        if (this->isDirty()) {
            this->read();
            this->resetDirty();
        }
    } catch (...) {
        return false;
    }

    return true;
}
