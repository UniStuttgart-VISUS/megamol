#include "stdafx.h"
#include "TrajectoryFactory.h"

#include "vislib/sys/ConsoleProgressBar.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/moldyn/DirectionalParticleDataCall.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/IntParam.h"


megamol::stdplugin::datatools::TrajectoryFactory::TrajectoryFactory()
    : megamol::core::AbstractDataWriter()
    , inDataSlot("inData", "Data input")
    , filepathSlot("filepath", "Path where the trajectory files should be written")
    , maxFramesInMemSlot("maxFramesInMem", "Set the maximal number of frames kept in memory")
    , datahash{std::numeric_limits<size_t>::max()} {
    this->inDataSlot.SetCompatibleCall<megamol::core::moldyn::MultiParticleDataCallDescription>();
    this->inDataSlot.SetCompatibleCall<megamol::core::moldyn::DirectionalParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);

    this->filepathSlot << new megamol::core::param::FilePathParam("traj");
    this->MakeSlotAvailable(&this->filepathSlot);

    this->maxFramesInMemSlot << new megamol::core::param::IntParam(10, 1);
    this->MakeSlotAvailable(&this->maxFramesInMemSlot);
}


megamol::stdplugin::datatools::TrajectoryFactory::~TrajectoryFactory() {
    this->Release();
}


bool megamol::stdplugin::datatools::TrajectoryFactory::create() {
    return true;
}


void megamol::stdplugin::datatools::TrajectoryFactory::release() {

}


bool megamol::stdplugin::datatools::TrajectoryFactory::run() {
    megamol::core::AbstractGetData3DCall* inCall = this->inDataSlot.CallAs<megamol::core::AbstractGetData3DCall>();
    if (inCall == nullptr) return false;

    (*inCall)(1);

    if (inCall->DataHash() != this->datahash) {
        this->datahash = inCall->DataHash();

        return this->assertData(*inCall);
    }

    return true;
}


bool megamol::stdplugin::datatools::TrajectoryFactory::getCapabilities(megamol::core::DataWriterCtrlCall& call) {
    call.SetAbortable(false);
    return true;
}


bool megamol::stdplugin::datatools::TrajectoryFactory::assertData(megamol::core::Call& c) {
    megamol::core::moldyn::DirectionalParticleDataCall* dirInCall
        = dynamic_cast<megamol::core::moldyn::DirectionalParticleDataCall*>(&c);

    megamol::core::moldyn::MultiParticleDataCall* simInCall
        = dynamic_cast<megamol::core::moldyn::MultiParticleDataCall*>(&c);

    if (dirInCall == nullptr && simInCall == nullptr) return false;

    megamol::core::AbstractGetData3DCall* inCall = dirInCall ? dynamic_cast<megamol::core::AbstractGetData3DCall*>(dirInCall)
        : dynamic_cast<megamol::core::AbstractGetData3DCall*>(simInCall);


    (*inCall)(0);

    auto const frameCount = inCall->FrameCount();

    if (frameCount == 0) return false;

    auto const parListCount = dirInCall ? dirInCall->GetParticleListCount() : simInCall->GetParticleListCount();

    if (parListCount == 0) return false;

    static size_t start_offset = sizeof(uint64_t) + sizeof(unsigned int) + 6 * sizeof(float);

    std::vector<std::unordered_map<uint64_t /*id*/, uint64_t /*offset*/>> file_id_offsets(parListCount);

    std::vector<std::unordered_map<uint64_t, std::pair<unsigned int, unsigned int>>> frame_id_assoc(parListCount);

    std::vector<std::unordered_map<uint64_t, trajectory_t<float>>> data(parListCount);

    std::vector<size_t> max_offset(parListCount, start_offset);

    static size_t const max_line_size = sizeof(uint64_t) + 2 * sizeof(unsigned int) + 3 * frameCount * sizeof(float);

    std::vector<char> zero_out_buf(max_line_size, 0);

    std::string filepath = this->filepathSlot.Param<megamol::core::param::FilePathParam>()->Value();
    unsigned int max_frames_in_mem = this->maxFramesInMemSlot.Param<megamol::core::param::IntParam>()->Value();

    // init files
    for (unsigned int pli = 0; pli < parListCount; ++pli) {
        FILE* file = fopen((filepath + "traj_" + std::to_string(pli) + ".raw").c_str(), "wb");

        uint64_t dummy = 0;
        fwrite(&dummy, sizeof(uint64_t), 1, file);
        fwrite(&frameCount, sizeof(unsigned int), 1, file);
        auto bbox = inCall->AccessBoundingBoxes().ObjectSpaceBBox();
        fwrite(bbox.PeekBounds(), sizeof(float), 6, file);

        fclose(file);
    }

    vislib::sys::ConsoleProgressBar cpb;
    cpb.Start("TrajectoryFactory", frameCount);

    for (unsigned int fi = 0; fi < frameCount; ++fi) {
        do {
            inCall->SetFrameID(fi, true);
            (*inCall)(0);
        } while (inCall->FrameID() != fi);

        for (unsigned int pli = 0; pli < parListCount; ++pli) {
            auto const& parts = dirInCall ? dirInCall->AccessParticles(pli) : simInCall->AccessParticles(pli);
            //if (!parts.HasID()) continue;

            auto const part_count = parts.GetCount();

            auto& cur_data = data[pli];
            auto& cur_frame_id_assoc = frame_id_assoc[pli];
            auto& cur_file_id_offsets = file_id_offsets[pli];

            cur_data.reserve(part_count);
            cur_frame_id_assoc.reserve(part_count);
            cur_file_id_offsets.reserve(part_count);

            for (uint64_t par_i = 0; par_i < part_count; ++par_i) {
                megamol::core::moldyn::SimpleSphericalParticles::particle_t par = parts[par_i];

                cur_data[par.id.GetIDu64()].push_back({
                    par.vert.GetXf(), par.vert.GetYf(), par.vert.GetZf()});

                // set frame_ids
                auto it = cur_frame_id_assoc.find(par.id.GetIDu64());
                if (it != cur_frame_id_assoc.end()) {
                    it->second.second = fi;
                } else {
                    cur_frame_id_assoc.insert(std::make_pair(par.id.GetIDu64(), std::make_pair(fi, fi)));
                }
            }

            // check whether cache is full
            // if it is, write cache to disk and clear the cache

            if (!(fi%max_frames_in_mem)) {
                // write to disk
                write(filepath, pli, cur_data, cur_file_id_offsets, cur_frame_id_assoc, max_offset, max_line_size, zero_out_buf);

                // clear data
                cur_data.clear();
            }

            if (fi == frameCount - 1) {
                // write the rest
                write(filepath, pli, cur_data, cur_file_id_offsets, cur_frame_id_assoc, max_offset, max_line_size, zero_out_buf);

                // write particle count
                FILE* file = fopen((filepath + "traj_" + std::to_string(pli) + ".raw").c_str(), "r+b");
                size_t count = cur_file_id_offsets.size();
                fwrite(&count, sizeof(size_t), 1, file);
                fclose(file);
            }
        }

        cpb.Set(fi);
    }

    cpb.Stop();

    return true;
}


void megamol::stdplugin::datatools::TrajectoryFactory::write(std::string const& filepath, unsigned int const pli,
    std::unordered_map<uint64_t, trajectory_t<float>>& cur_data,
    std::unordered_map<uint64_t, uint64_t>& cur_file_id_offsets,
    std::unordered_map<uint64_t, std::pair<unsigned int, unsigned int>>& cur_frame_id_assoc,
    std::vector<size_t>& max_offset, size_t const max_line_size,
    std::vector<char> const& zero_out_buf) const {
    FILE* file = fopen((filepath + "traj_" + std::to_string(pli) + ".raw").c_str(), "r+b");

    for (auto const& el : cur_data) {
        auto it_offset = cur_file_id_offsets.find(el.first);
        if (it_offset != cur_file_id_offsets.end()) {
            // id already exists
            writeParticle(file, it_offset->second, cur_frame_id_assoc[el.first],
                el.first, cur_data[el.first]);
        } else {
            // new id and therefore new particle
            cur_file_id_offsets.insert(std::make_pair(el.first, max_offset[pli]));
            fseek(file, max_offset[pli], SEEK_SET);
            // init new line for particle
            fwrite(zero_out_buf.data(), sizeof(char), zero_out_buf.size(), file);

            writeParticle(file, max_offset[pli], cur_frame_id_assoc[el.first],
                el.first, cur_data[el.first], true);
            max_offset[pli] += max_line_size;
        }
    }

    fclose(file);
}


void megamol::stdplugin::datatools::TrajectoryFactory::writeParticle(FILE* file, size_t const base_offset,
    std::pair<unsigned int, unsigned int> const& frame_start_end, uint64_t const id, trajectory_t<float> const& toWrite, bool const new_par) const {
    //static size_t start_offset = sizeof(uint64_t) + 2 * sizeof(unsigned int);

    fseek(file, base_offset, SEEK_SET);
    if (new_par) {
        fwrite(&id, sizeof(uint64_t), 1, file);

        fwrite(&frame_start_end.first, sizeof(unsigned int), 1, file);
    } else {
        fseek(file, sizeof(uint64_t) + sizeof(unsigned int), SEEK_CUR);
    }
    fwrite(&frame_start_end.second, sizeof(unsigned int), 1, file);

    fseek(file, 3 * sizeof(float)*frame_start_end.first, SEEK_CUR);

    fwrite(toWrite.data(), sizeof(float) * 3, toWrite.size(), file);
}
