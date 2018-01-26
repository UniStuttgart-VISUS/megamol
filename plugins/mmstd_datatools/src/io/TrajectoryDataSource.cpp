#include "stdafx.h"
#include "TrajectoryDataSource.h"

#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"

#include "geometry_calls/LinesDataCall.h"


megamol::stdplugin::datatools::io::TrajectoryDataSource::TrajectoryDataSource()
    : megamol::core::Module()
    , trajOutSlot("trajOut", "Trajectory output")
    , trajFilepath("filename", "Trajectory file to read")
    , minFrameSlot("frame::min", "minimal frame id to load")
    , maxFrameSlot("frame::max", "maximal frame id to load")
    , minIDSlot("id::min", "minimal particle id to load")
    , maxIDSlot("id::max", "maximal particle id to load")
    , datahash(0)
    , data_param_changed_(true) {
    this->trajOutSlot.SetCallback(megamol::geocalls::LinesDataCall::ClassName(),
        megamol::geocalls::LinesDataCall::FunctionName(0),
        &TrajectoryDataSource::getDataCallback);
    this->trajOutSlot.SetCallback(megamol::geocalls::LinesDataCall::ClassName(),
        megamol::geocalls::LinesDataCall::FunctionName(1),
        &TrajectoryDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->trajOutSlot);

    this->trajFilepath << new megamol::core::param::FilePathParam("traj.raw");
    this->trajFilepath.SetUpdateCallback(&TrajectoryDataSource::filenameChanged);
    this->MakeSlotAvailable(&this->trajFilepath);

    this->minFrameSlot << new megamol::core::param::IntParam(0, 0, std::numeric_limits<int>::max());
    this->minFrameSlot.SetUpdateCallback(&TrajectoryDataSource::dataParamChanged);
    this->MakeSlotAvailable(&this->minFrameSlot);

    this->maxFrameSlot << new megamol::core::param::IntParam(std::numeric_limits<int>::max(), 0, std::numeric_limits<int>::max());
    this->maxFrameSlot.SetUpdateCallback(&TrajectoryDataSource::dataParamChanged);
    this->MakeSlotAvailable(&this->maxFrameSlot);

    this->minIDSlot << new megamol::core::param::IntParam(0, 0, std::numeric_limits<int>::max());
    this->minIDSlot.SetUpdateCallback(&TrajectoryDataSource::dataParamChanged);
    this->MakeSlotAvailable(&this->minIDSlot);

    this->maxIDSlot << new megamol::core::param::IntParam(std::numeric_limits<int>::max(), 0, std::numeric_limits<int>::max());
    this->maxIDSlot.SetUpdateCallback(&TrajectoryDataSource::dataParamChanged);
    this->MakeSlotAvailable(&this->maxIDSlot);
}


megamol::stdplugin::datatools::io::TrajectoryDataSource::~TrajectoryDataSource() {
    this->Release();
}


bool megamol::stdplugin::datatools::io::TrajectoryDataSource::create() {
    this->file_header_.bbox = new float[6];

    return true;
}


void megamol::stdplugin::datatools::io::TrajectoryDataSource::release() {
    delete[] this->file_header_.bbox;
}


bool megamol::stdplugin::datatools::io::TrajectoryDataSource::getDataCallback(megamol::core::Call& c) {
    megamol::geocalls::LinesDataCall* outCall = dynamic_cast<megamol::geocalls::LinesDataCall*>(&c);
    if (outCall == nullptr) return false;

    if (!this->assertData()) return false;

    outCall->SetFrameCount(1);
    outCall->SetFrameID(0);
    outCall->SetDataHash(this->datahash);
    outCall->SetData(this->lines_data.size(), this->lines_data.data());

    return true;
}


bool megamol::stdplugin::datatools::io::TrajectoryDataSource::getExtentCallback(megamol::core::Call& c) {
    megamol::geocalls::LinesDataCall* outCall = dynamic_cast<megamol::geocalls::LinesDataCall*>(&c);
    if (outCall == nullptr) return false;

    if (!this->assertData()) return false;

    outCall->SetFrameCount(1);
    outCall->SetFrameID(0);
    outCall->SetDataHash(this->datahash);
    outCall->AccessBoundingBoxes().SetObjectSpaceBBox(this->file_header_.bbox[0], this->file_header_.bbox[1], this->file_header_.bbox[2],
        this->file_header_.bbox[3], this->file_header_.bbox[4], this->file_header_.bbox[5]);
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(this->file_header_.bbox[0], this->file_header_.bbox[1], this->file_header_.bbox[2],
        this->file_header_.bbox[3], this->file_header_.bbox[4], this->file_header_.bbox[5]);
    outCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);

    return true;
}


bool megamol::stdplugin::datatools::io::TrajectoryDataSource::filenameChanged(megamol::core::param::ParamSlot& p) {
    this->filepath_ = this->trajFilepath.Param<megamol::core::param::FilePathParam>()->Value();

    FILE* file = fopen(this->filepath_.c_str(), "rb");

    fread(&this->file_header_.particle_count, sizeof(uint64_t), 1, file);
    fread(&this->file_header_.frame_count, sizeof(unsigned int), 1, file);
    fread(this->file_header_.bbox, sizeof(float), 6, file);

    this->particle_file_offsets_.clear();
    this->particle_file_offsets_.reserve(this->file_header_.particle_count);
    this->particle_frame_begin_end_.clear();
    this->particle_frame_begin_end_.reserve(this->file_header_.particle_count);
    this->sorted_id_list.clear();
    this->sorted_id_list.reserve(this->file_header_.particle_count);

    size_t file_offset = sizeof(uint64_t) + sizeof(unsigned int) + 6 * sizeof(float);
    size_t offset_incr = sizeof(uint64_t) + 2 * sizeof(unsigned int) + 3 * sizeof(float)*this->file_header_.frame_count
        + sizeof(char)*this->file_header_.frame_count;

    this->index_dummy.resize(this->file_header_.particle_count * 2);

    for (uint64_t pi = 0; pi < this->file_header_.particle_count; ++pi) {
        fseek(file, file_offset, SEEK_SET);
        uint64_t id{0};
        fread(&id, sizeof(uint64_t), 1, file);
        unsigned int frame_begin{0};
        fread(&frame_begin, sizeof(unsigned int), 1, file);
        unsigned int frame_end{0};
        fread(&frame_end, sizeof(unsigned int), 1, file);
        this->particle_file_offsets_[id] = file_offset;
        this->particle_frame_begin_end_[id] = std::make_pair(frame_begin, frame_end);
        file_offset += offset_incr;
        this->index_dummy[pi * 2] = pi;
        this->index_dummy[pi * 2 + 1] = pi + 1;
        this->sorted_id_list.push_back(id);
    }

    std::sort(this->sorted_id_list.begin(), this->sorted_id_list.end());

    fclose(file);

    ++this->datahash;

    return true;
}


bool megamol::stdplugin::datatools::io::TrajectoryDataSource::dataParamChanged(megamol::core::param::ParamSlot& p) {
    this->data_param_changed_ = true;

    this->frame_begin_end_.first = this->minFrameSlot.Param<megamol::core::param::IntParam>()->Value();
    this->frame_begin_end_.second = this->maxFrameSlot.Param<megamol::core::param::IntParam>()->Value() < this->file_header_.frame_count - 1 ?
        this->maxFrameSlot.Param<megamol::core::param::IntParam>()->Value() : this->file_header_.frame_count - 1;

    this->id_begin_end_.first = this->minIDSlot.Param<megamol::core::param::IntParam>()->Value();
    this->id_begin_end_.second = this->maxIDSlot.Param<megamol::core::param::IntParam>()->Value();// < this->file_header_.particle_count - 1 ?
        //this->maxIDSlot.Param<megamol::core::param::IntParam>()->Value() : this->file_header_.particle_count - 1;

    return true;
}


bool megamol::stdplugin::datatools::io::TrajectoryDataSource::assertData() {
    static size_t const frame_element_offset = 3 * sizeof(float);

    float const half_x_dim = (this->file_header_.bbox[3] - this->file_header_.bbox[0]) / 2.0f;
    float const half_y_dim = (this->file_header_.bbox[4] - this->file_header_.bbox[1]) / 2.0f;
    float const half_z_dim = (this->file_header_.bbox[5] - this->file_header_.bbox[2]) / 2.0f;

    if (this->data_param_changed_) {
        this->data_param_changed_ = false;

        this->data.clear();
        this->data.reserve(this->id_begin_end_.second - this->id_begin_end_.first + 1);

        this->col_data.clear();
        this->col_data.reserve(this->id_begin_end_.second - this->id_begin_end_.first + 1);

        this->is_fluid_data.clear();
        this->is_fluid_data.reserve(this->id_begin_end_.second - this->id_begin_end_.first + 1);

        this->lines_data.clear();
        this->lines_data.reserve(this->id_begin_end_.second - this->id_begin_end_.first + 1);

        FILE* file = fopen(this->filepath_.c_str(), "rb");

        auto start_it = std::find_if(this->sorted_id_list.cbegin(), this->sorted_id_list.cend(),
            [&](auto const& a) {return a >= this->id_begin_end_.first; });
        auto end_it = std::find_if(this->sorted_id_list.cbegin(), this->sorted_id_list.cend(),
            [&](auto const& a) {return a >= this->id_begin_end_.second; });

        //for (uint64_t pid = this->id_begin_end_.first; pid <= this->id_begin_end_.second; ++pid) {
        for (auto it = start_it; it != end_it; ++it) {
            auto pid_it = this->particle_file_offsets_.find(*it);
            if (pid_it != this->particle_file_offsets_.end()) {
                auto id_offset = pid_it->second;
                fseek(file, id_offset, SEEK_SET);
                uint64_t id{0};
                fread(&id, sizeof(uint64_t), 1, file);
                unsigned int p_fr_begin{0}, p_fr_end{0};
                fread(&p_fr_begin, sizeof(unsigned int), 1, file);
                fread(&p_fr_end, sizeof(unsigned int), 1, file);
                // jump to first requested (and valid frame)
                size_t frame_begin_offset = p_fr_begin > this->frame_begin_end_.first ?
                    static_cast<size_t>(p_fr_begin) * frame_element_offset : static_cast<size_t>(this->frame_begin_end_.first)*frame_element_offset;
                size_t frame_end_offset = p_fr_end < this->frame_begin_end_.second ?
                    static_cast<size_t>(p_fr_end + 1)*frame_element_offset : static_cast<size_t>(this->frame_begin_end_.second + 1)*frame_element_offset;
                fseek(file, frame_begin_offset, SEEK_CUR);
                std::vector<float> cur_data((frame_end_offset - frame_begin_offset) / sizeof(float));
                std::vector<uint8_t> cur_col_data((frame_end_offset - frame_begin_offset) / (3*sizeof(float))*4, 255);
                std::vector<char> cur_is_fluid_data((frame_end_offset - frame_begin_offset) / (3 * sizeof(float)));
                fread(cur_data.data(), sizeof(char), frame_end_offset - frame_begin_offset, file);
                fseek(file, id_offset + sizeof(uint64_t) + 2 * sizeof(unsigned int)
                    + 3 * sizeof(float)*this->file_header_.frame_count + frame_begin_offset / (frame_element_offset), SEEK_SET);
                fread(cur_is_fluid_data.data(), sizeof(char), cur_is_fluid_data.size(), file);
                for (size_t i = 0; i < cur_data.size() / 3 - 1; ++i) {
                    if (cur_is_fluid_data[i] == cur_is_fluid_data[i + 1] == 0) {
                        // no transition, is gas
                        cur_col_data[i * 4] = 255;
                        cur_col_data[i * 4 + 1] = 0;
                        cur_col_data[i * 4 + 2] = 0;
                    } else if (cur_is_fluid_data[i] == cur_is_fluid_data[i + 1] == 1) {
                        // no transition, is fluid
                        cur_col_data[i * 4] = 0;
                        cur_col_data[i * 4 + 1] = 0;
                        cur_col_data[i * 4 + 2] = 255;
                    } else if (cur_is_fluid_data[i] != cur_is_fluid_data[i + 1]) {
                        // transition
                        cur_col_data[i * 4] = 0;
                        cur_col_data[i * 4 + 1] = 255;
                        cur_col_data[i * 4 + 2] = 0;
                    }

                    if (std::abs(cur_data[(i + 1) * 3] - cur_data[i * 3]) > half_x_dim
                        || std::abs(cur_data[(i + 1) * 3 + 1] - cur_data[i * 3 + 1]) > half_y_dim
                        || std::abs(cur_data[(i + 1) * 3 + 2] - cur_data[i * 3 + 2]) > half_z_dim) {
                        cur_col_data[i * 4 + 3] = 42;
                        cur_col_data[(i + 1) * 4 + 3] = 42;
                    }
                }
                this->data.push_back(cur_data);
                this->col_data.push_back(cur_col_data);
                // set the line
                megamol::geocalls::LinesDataCall::Lines l;
                l.Set(((this->data.back().size() / 3) - 1) * 2, this->index_dummy.data(), this->data.back().data(),
                    this->col_data.back().data(), true);
                    //vislib::graphics::ColourRGBAu8{255, 255, 255, 255});
                this->lines_data.push_back(l);
            }
        }

        fclose(file);
    }

    ++this->datahash;

    return true;
}
