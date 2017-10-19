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

#include "pbs/PBSDataCall.h"


using namespace megamol;
using namespace megamol::pbs;


PBSDataSource::PBSDataSource(void) : core::Module(),
filenameSlot("filename", "The path to the PBS file to load."),
start_idx_slot("start_idx", "The start idx of chunks to read."),
end_idx_slot("end_idx", "The end idx of chunks to read."),
start_region_idx_slot("start_region_idx", "The start idx of chunks to read."),
end_region_idx_slot("end_region_idx", "The end idx of chunks to read."),
getData("getdata", "Slot to request data from this data source."),
with_normals(false), with_colors(false) {
    this->getData.SetCallback(PBSDataCall::ClassName(), PBSDataCall::FunctionName(0), &PBSDataSource::getDataCallback);
    this->getData.SetCallback(PBSDataCall::ClassName(), PBSDataCall::FunctionName(1), &PBSDataSource::getExtentCallback);
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
    this->x_data = std::make_shared<std::vector<double>>();
    this->y_data = std::make_shared<std::vector<double>>();
    this->z_data = std::make_shared<std::vector<double>>();
    this->nx_data = std::make_shared<std::vector<float>>();
    this->ny_data = std::make_shared<std::vector<float>>();
    this->cr_data = std::make_shared<std::vector<unsigned int>>();
    this->cg_data = std::make_shared<std::vector<unsigned int>>();
    this->cb_data = std::make_shared<std::vector<unsigned int>>();

    return true;
}


void PBSDataSource::release(void) {

}


void PBSDataSource::clearBuffers(void) {
    this->x_data->clear();
    this->y_data->clear();
    this->z_data->clear();
    this->nx_data->clear();
    this->ny_data->clear();
    this->cr_data->clear();
    this->cg_data->clear();
    this->cb_data->clear();

    this->with_normals = false;
    this->with_colors = false;
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

    zfp_field* field = zfp_field_1d(data.data(), type, num_elements);

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


bool megamol::pbs::PBSDataSource::read(void) {
    this->clearBuffers();

    const std::string path_to_pbs = this->filenameSlot.Param<core::param::FilePathParam>()->Value();

    if (!this->isDirty()) {
        return true;
    }

    const auto start_idx = this->start_idx_slot.Param<core::param::IntParam>()->Value();
    const auto end_idx = this->end_idx_slot.Param<core::param::IntParam>()->Value();

    const auto start_region_idx = this->start_region_idx_slot.Param<core::param::IntParam>()->Value();
    const auto end_region_idx = this->end_region_idx_slot.Param<core::param::IntParam>()->Value();

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


bool PBSDataSource::isDirty(void) {
    return this->start_idx_slot.IsDirty() ||
        this->end_idx_slot.IsDirty() ||
        this->start_region_idx_slot.IsDirty() ||
        this->end_region_idx_slot.IsDirty();
}


void PBSDataSource::resetDirty(void) {
    this->start_idx_slot.ResetDirty();
    this->end_idx_slot.ResetDirty();
    this->start_region_idx_slot.ResetDirty();
    this->end_region_idx_slot.ResetDirty();
}


bool PBSDataSource::filenameChanged(core::param::ParamSlot& slot) {
    return this->read();
}


bool PBSDataSource::getDataCallback(core::Call& c) {
    try {
        if (this->isDirty()) {
            this->read();
        }

        PBSDataCall* pdc = dynamic_cast<PBSDataCall*>(&c);

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

        pdc->SetData(std::move(ret));

        /*switch (this->generatePBSType) {
        case pbs_type::P:
        {
            PBSStorage ret;
            ret.SetX(this->x_data);
            ret.SetY(this->y_data);
            ret.SetZ(this->z_data);
            pdc->SetData(std::make_shared<PBSStorage>(ret));
        }
            break;
        case pbs_type::PN:
        {
            PBSDataCall::PNStorage ret;
            ret.SetX(this->x_data);
            ret.SetY(this->y_data);
            ret.SetZ(this->z_data);
            ret.SetNX(this->nx_data);
            ret.SetNY(this->ny_data);
            pdc->SetData(std::make_shared<PBSDataCall::PNStorage>(ret));
        }
            break;
        case pbs_type::PC:
        {
            PBSDataCall::PCStorage ret;
            ret.SetX(this->x_data);
            ret.SetY(this->y_data);
            ret.SetZ(this->z_data);
            ret.SetCR(this->cr_data);
            ret.SetCG(this->cg_data);
            ret.SetCB(this->cb_data);
            pdc->SetData(std::make_shared<PBSDataCall::PCStorage>(ret));
        }
        case pbs_type::PNC:
        {
            PBSDataCall::PNCStorage ret;
            ret.SetX(this->x_data);
            ret.SetY(this->y_data);
            ret.SetZ(this->z_data);
            ret.SetNX(this->nx_data);
            ret.SetNY(this->ny_data);
            ret.SetCR(this->cr_data);
            ret.SetCG(this->cg_data);
            ret.SetCB(this->cb_data);
            pdc->SetData(std::make_shared<PBSDataCall::PNCStorage>(ret));
        }
        default:
            break;
        }*/
    } catch (...) {
        return false;
    }

    return true;
}


bool PBSDataSource::getExtentCallback(core::Call& c) {
    return true;
}
