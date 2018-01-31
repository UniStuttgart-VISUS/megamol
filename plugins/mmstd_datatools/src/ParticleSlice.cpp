#include "stdafx.h"
#include "ParticleSlice.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/moldyn/DirectionalParticleDataCall.h"

#include "mmcore/param/FloatParam.h"
#include "mmcore/param/EnumParam.h"

megamol::stdplugin::datatools::ParticleSlice::ParticleSlice()
    : megamol::core::Module()
    , dataOutSlot("dataOut", "data output")
    , dataInSlot("dataIn", "data input")
    , axisSlot("axis", "select axis to slice")
    , thicknessSlot("thickness", "select thickness of slice as factor of particle radius")
    , positionSlot("position", "select position of slice")
    , datahash{std::numeric_limits<size_t>::max()}
    , frameID{std::numeric_limits<unsigned int>::max()} {
    this->dataOutSlot.SetCallback(megamol::core::moldyn::MultiParticleDataCall::ClassName(),
        megamol::core::moldyn::MultiParticleDataCall::FunctionName(0),
        &ParticleSlice::getDataCallback);
    this->dataOutSlot.SetCallback(megamol::core::moldyn::MultiParticleDataCall::ClassName(),
        megamol::core::moldyn::MultiParticleDataCall::FunctionName(1),
        &ParticleSlice::getExtentCallback);
    this->dataOutSlot.SetCallback(megamol::core::moldyn::DirectionalParticleDataCall::ClassName(),
        megamol::core::moldyn::DirectionalParticleDataCall::FunctionName(0),
        &ParticleSlice::getDataCallback);
    this->dataOutSlot.SetCallback(megamol::core::moldyn::DirectionalParticleDataCall::ClassName(),
        megamol::core::moldyn::DirectionalParticleDataCall::FunctionName(1),
        &ParticleSlice::getExtentCallback);
    this->MakeSlotAvailable(&this->dataOutSlot);

    this->dataInSlot.SetCompatibleCall<megamol::core::moldyn::MultiParticleDataCallDescription>();
    this->dataInSlot.SetCompatibleCall<megamol::core::moldyn::DirectionalParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->dataInSlot);

    auto ep = new megamol::core::param::EnumParam(0);
    ep->SetTypePair(0, "x-axis");
    ep->SetTypePair(1, "y-axis");
    ep->SetTypePair(2, "z-axis");
    this->axisSlot << ep;
    this->MakeSlotAvailable(&this->axisSlot);

    this->thicknessSlot << new megamol::core::param::FloatParam(1.0f, std::numeric_limits<float>::min(),
        std::numeric_limits<float>::max());
    this->MakeSlotAvailable(&this->thicknessSlot);

    this->positionSlot << new megamol::core::param::FloatParam(1.0f, std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::max());
    this->MakeSlotAvailable(&this->positionSlot);
}


megamol::stdplugin::datatools::ParticleSlice::~ParticleSlice() {
    this->Release();
}


bool megamol::stdplugin::datatools::ParticleSlice::create() {
    return true;
}


void megamol::stdplugin::datatools::ParticleSlice::release() {

}


bool megamol::stdplugin::datatools::ParticleSlice::getDataCallback(megamol::core::Call &c) {
    megamol::core::moldyn::DirectionalParticleDataCall* dirOutCall
        = dynamic_cast<megamol::core::moldyn::DirectionalParticleDataCall*>(&c);

    megamol::core::moldyn::MultiParticleDataCall* simOutCall
        = dynamic_cast<megamol::core::moldyn::MultiParticleDataCall*>(&c);

    if (dirOutCall == nullptr && simOutCall == nullptr) return false;

    megamol::core::AbstractGetData3DCall* outCall = dirOutCall ? dynamic_cast<megamol::core::AbstractGetData3DCall*>(dirOutCall)
        : dynamic_cast<megamol::core::AbstractGetData3DCall*>(simOutCall);

    megamol::core::moldyn::DirectionalParticleDataCall* dirInCall
        = this->dataInSlot.CallAs<megamol::core::moldyn::DirectionalParticleDataCall>();

    megamol::core::moldyn::MultiParticleDataCall* simInCall
        = this->dataInSlot.CallAs<megamol::core::moldyn::MultiParticleDataCall>();

    if (dirInCall == nullptr && simInCall == nullptr) return false;

    megamol::core::AbstractGetData3DCall* inCall = dirInCall ? dynamic_cast<megamol::core::AbstractGetData3DCall*>(dirInCall)
        : dynamic_cast<megamol::core::AbstractGetData3DCall*>(simInCall);

    inCall->SetFrameID(outCall->FrameID(), outCall->IsFrameForced());
    if (!(*inCall)(1)) return false;
    if (!(*inCall)(0)) return false;

    if (!this->assertData()) return false;

    //(*outCall) = (*inCall);

    outCall->SetDataHash(this->datahash);
    outCall->SetFrameCount(inCall->FrameCount());
    outCall->SetFrameID(inCall->FrameID());

    if (dirOutCall) {
        if (dirInCall) {
            auto const& partListCount = dirInCall->GetParticleListCount();
            dirOutCall->SetParticleListCount(partListCount);
            for (unsigned int pli = 0; pli < partListCount; ++pli) {
                auto const& parts_in = dirInCall->AccessParticles(pli);
                auto& parts_out = dirOutCall->AccessParticles(pli);
                parts_out.SetCount(this->positions[pli].size() / 3);
                if (this->positions[pli].size() / 3 > 0) {
                    parts_out.SetVertexData(parts_in.GetVertexDataType(), this->positions[pli].data());
                    parts_out.SetColourData(parts_in.GetColourDataType(), this->colors[pli].data());
                    parts_out.SetDirData(parts_in.GetDirDataType(), this->directions[pli].data());
                }
            }
        } else {
            auto const& partListCount = simInCall->GetParticleListCount();
            dirOutCall->SetParticleListCount(partListCount);
            for (unsigned int pli = 0; pli < partListCount; ++pli) {
                auto const& parts_in = simInCall->AccessParticles(pli);
                auto& parts_out = dirOutCall->AccessParticles(pli);
                parts_out.SetCount(this->positions[pli].size() / 3);
                if (this->positions[pli].size() / 3 > 0) {
                    parts_out.SetVertexData(parts_in.GetVertexDataType(), this->positions[pli].data());
                    parts_out.SetColourData(parts_in.GetColourDataType(), this->colors[pli].data());
                    parts_out.SetDirData(megamol::core::moldyn::DirectionalParticles::DIRDATA_NONE, nullptr);
                }
            }
        }
    } else {
        if (dirInCall) {
            auto const& partListCount = dirInCall->GetParticleListCount();
            simOutCall->SetParticleListCount(partListCount);
            for (unsigned int pli = 0; pli < partListCount; ++pli) {
                auto const& parts_in = dirInCall->AccessParticles(pli);
                auto& parts_out = simOutCall->AccessParticles(pli);
                parts_out.SetCount(this->positions[pli].size() / 3);
                if (this->positions[pli].size() / 3 > 0) {
                    parts_out.SetVertexData(parts_in.GetVertexDataType(), this->positions[pli].data());
                    parts_out.SetColourData(parts_in.GetColourDataType(), this->colors[pli].data());
                }
            }
        } else {
            auto const& partListCount = simInCall->GetParticleListCount();
            simOutCall->SetParticleListCount(partListCount);
            for (unsigned int pli = 0; pli < partListCount; ++pli) {
                auto const& parts_in = simInCall->AccessParticles(pli);
                auto& parts_out = simOutCall->AccessParticles(pli);
                parts_out.SetCount(this->positions[pli].size() / 3);
                if (this->positions[pli].size() / 3 > 0) {
                    parts_out.SetVertexData(parts_in.GetVertexDataType(), this->positions[pli].data());
                    parts_out.SetColourData(parts_in.GetColourDataType(), this->colors[pli].data());
                }
            }
        }
    }

    return true;
}


bool megamol::stdplugin::datatools::ParticleSlice::getExtentCallback(megamol::core::Call& c) {
    megamol::core::moldyn::DirectionalParticleDataCall* dirOutCall
        = dynamic_cast<megamol::core::moldyn::DirectionalParticleDataCall*>(&c);

    megamol::core::moldyn::MultiParticleDataCall* simOutCall
        = dynamic_cast<megamol::core::moldyn::MultiParticleDataCall*>(&c);

    if (dirOutCall == nullptr && simOutCall == nullptr) return false;

    megamol::core::AbstractGetData3DCall* outCall = dirOutCall ? dynamic_cast<megamol::core::AbstractGetData3DCall*>(dirOutCall)
        : dynamic_cast<megamol::core::AbstractGetData3DCall*>(simOutCall);

    megamol::core::moldyn::DirectionalParticleDataCall* dirInCall
        = this->dataInSlot.CallAs<megamol::core::moldyn::DirectionalParticleDataCall>();

    megamol::core::moldyn::MultiParticleDataCall* simInCall
        = this->dataInSlot.CallAs<megamol::core::moldyn::MultiParticleDataCall>();

    if (dirInCall == nullptr && simInCall == nullptr) return false;

    megamol::core::AbstractGetData3DCall* inCall = dirInCall ? dynamic_cast<megamol::core::AbstractGetData3DCall*>(dirInCall)
        : dynamic_cast<megamol::core::AbstractGetData3DCall*>(simInCall);

    inCall->SetFrameID(outCall->FrameID(), outCall->IsFrameForced());
    if (!(*inCall)(1)) return false;
    /*if (!(*inCall)(0)) return false;

    this->assertData();*/

    outCall->SetDataHash(this->datahash);
    outCall->SetFrameCount(inCall->FrameCount());
    outCall->SetFrameID(this->frameID);

    outCall->AccessBoundingBoxes().SetObjectSpaceBBox(inCall->AccessBoundingBoxes().ObjectSpaceBBox());
    outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(inCall->AccessBoundingBoxes().ObjectSpaceClipBox());

    //(*outCall) = (*inCall);

    return true;
}


bool megamol::stdplugin::datatools::ParticleSlice::assertData() {
    megamol::core::moldyn::DirectionalParticleDataCall* dirInCall
        = this->dataInSlot.CallAs<megamol::core::moldyn::DirectionalParticleDataCall>();

    megamol::core::moldyn::MultiParticleDataCall* simInCall
        = this->dataInSlot.CallAs<megamol::core::moldyn::MultiParticleDataCall>();

    if (dirInCall == nullptr && simInCall == nullptr) return false;

    megamol::core::AbstractGetData3DCall* inCall = dirInCall ? dynamic_cast<megamol::core::AbstractGetData3DCall*>(dirInCall)
        : dynamic_cast<megamol::core::AbstractGetData3DCall*>(simInCall);

    /*(*inCall)(1);
    (*inCall)(0);*/
    bool param_changed = this->thicknessSlot.IsDirty()
        || this->axisSlot.IsDirty()
        || this->positionSlot.IsDirty();


    if (this->datahash != inCall->DataHash() || this->frameID != inCall->FrameID() || param_changed) {
        this->datahash = inCall->DataHash();
        this->frameID  = inCall->FrameID();

        this->thicknessSlot.ResetDirty();
        this->axisSlot.ResetDirty();
        this->positionSlot.ResetDirty();

        unsigned int const slice_axis = this->axisSlot.Param<megamol::core::param::EnumParam>()->Value();
        float const thickness_factor  = this->thicknessSlot.Param<megamol::core::param::FloatParam>()->Value();
        float const slice_pos         = this->positionSlot.Param<megamol::core::param::FloatParam>()->Value();

        auto const bbox       = inCall->AccessBoundingBoxes().ObjectSpaceBBox();
        auto const min_extent = bbox.PeekBounds()[slice_axis];
        auto const max_extent = bbox.PeekBounds()[slice_axis + 3];

        auto const parListCount = dirInCall ? dirInCall->GetParticleListCount() : simInCall->GetParticleListCount();

        float min_rad{std::numeric_limits<float>::max()};
        for (unsigned int pli = 0; pli < parListCount; ++pli) {
            auto const& parts = dirInCall ? dirInCall->AccessParticles(pli) : simInCall->AccessParticles(pli);

            if (parts.GetGlobalRadius() < min_rad) {
                min_rad = parts.GetGlobalRadius();
            }
        }

        // clamp slice bounds
        float tmp_slice_min = slice_pos - min_rad * thickness_factor;
        float tmp_slice_max = slice_pos + min_rad * thickness_factor;

        if (tmp_slice_max <= min_extent) {
            tmp_slice_min = min_extent;
            tmp_slice_max = tmp_slice_min + min_rad * thickness_factor;
        }
        if (tmp_slice_min >= max_extent) {
            tmp_slice_max = max_extent;
            tmp_slice_min = tmp_slice_max - min_rad * thickness_factor;
        }
        if (tmp_slice_max >= max_extent) {
            tmp_slice_max = max_extent;
        }
        if (tmp_slice_min <= min_extent) {
            tmp_slice_min = min_extent;
        }
        float const slice_min{tmp_slice_min};
        float const slice_max{tmp_slice_max};

        this->positions.resize(parListCount);
        this->colors.resize(parListCount);
        this->directions.resize(parListCount);

        for (unsigned int pli = 0; pli < parListCount; ++pli) {
            auto const& sim_parts = dirInCall ? dirInCall->AccessParticles(pli) : simInCall->AccessParticles(pli);
            auto const part_count = sim_parts.GetCount();

            auto& cur_pos_vec = this->positions[pli];
            auto& cur_col_vec = this->colors[pli];
            auto& cur_dir_vec = this->directions[pli];

            cur_pos_vec.clear();
            cur_col_vec.clear();
            cur_dir_vec.clear();

            cur_pos_vec.reserve(part_count * 3);
            if (sim_parts.GetColourDataType() != sim_parts.COLDATA_NONE) {
                if (sim_parts.GetColourDataType() == sim_parts.COLDATA_FLOAT_I) {
                    cur_col_vec.reserve(part_count);
                } else {
                    cur_col_vec.reserve(part_count * 3);
                }
            }
            if (dirInCall) {
                auto const& dir_parts = dirInCall->AccessParticles(pli);
                if (dir_parts.GetDirDataType() != dir_parts.DIRDATA_NONE) {
                    cur_dir_vec.reserve(part_count * 3);
                }
            }
            if (sim_parts.GetColourDataType() != sim_parts.COLDATA_NONE && sim_parts.GetColourDataType() != sim_parts.COLDATA_FLOAT_I) {
                if (dirInCall) {
                    auto const& dir_parts = dirInCall->AccessParticles(pli);
                    for (uint64_t par_i = 0; par_i < part_count; ++par_i) {
                        megamol::core::moldyn::DirectionalParticles::dir_particle_t par = dir_parts[par_i];
                        float par_pos[3] = {par.vert.GetXf(), par.vert.GetYf(), par.vert.GetZf()};
                        if (par_pos[slice_axis] >= slice_min && par_pos[slice_axis] <= slice_max) {
                            float par_col[3] = {par.col.GetRf(), par.col.GetGf(), par.col.GetBf()};
                            float par_dir[3] = {par.dir.GetDirXf(), par.dir.GetDirYf(), par.dir.GetDirZf()};

                            cur_pos_vec.push_back(par_pos[0]);
                            cur_pos_vec.push_back(par_pos[1]);
                            cur_pos_vec.push_back(par_pos[2]);
                            cur_col_vec.push_back(par_col[0]);
                            cur_col_vec.push_back(par_col[1]);
                            cur_col_vec.push_back(par_col[2]);
                            cur_dir_vec.push_back(par_dir[0]);
                            cur_dir_vec.push_back(par_dir[1]);
                            cur_dir_vec.push_back(par_dir[2]);
                        }
                    }
                } else {
                    for (uint64_t par_i = 0; par_i < part_count; ++par_i) {
                        megamol::core::moldyn::SimpleSphericalParticles::particle_t par = sim_parts[par_i];
                        float par_pos[3] = {par.vert.GetXf(), par.vert.GetYf(), par.vert.GetZf()};
                        if (par_pos[slice_axis] >= slice_min && par_pos[slice_axis] <= slice_max) {
                            float par_col[3] = {par.col.GetRf(), par.col.GetGf(), par.col.GetBf()};

                            cur_pos_vec.push_back(par_pos[0]);
                            cur_pos_vec.push_back(par_pos[1]);
                            cur_pos_vec.push_back(par_pos[2]);
                            cur_col_vec.push_back(par_col[0]);
                            cur_col_vec.push_back(par_col[1]);
                            cur_col_vec.push_back(par_col[2]);
                        }
                    }
                }
            } else {
                if (dirInCall) {
                    auto const& dir_parts = dirInCall->AccessParticles(pli);
                    for (uint64_t par_i = 0; par_i < part_count; ++par_i) {
                        megamol::core::moldyn::DirectionalParticles::dir_particle_t par = dir_parts[par_i];
                        float par_pos[3] = {par.vert.GetXf(), par.vert.GetYf(), par.vert.GetZf()};
                        if (par_pos[slice_axis] >= slice_min && par_pos[slice_axis] <= slice_max) {
                            float par_col = {par.col.GetIf()};
                            float par_dir[3] = {par.dir.GetDirXf(), par.dir.GetDirYf(), par.dir.GetDirZf()};

                            cur_pos_vec.push_back(par_pos[0]);
                            cur_pos_vec.push_back(par_pos[1]);
                            cur_pos_vec.push_back(par_pos[2]);
                            cur_col_vec.push_back(par_col);
                            cur_dir_vec.push_back(par_dir[0]);
                            cur_dir_vec.push_back(par_dir[1]);
                            cur_dir_vec.push_back(par_dir[2]);
                        }
                    }
                } else {
                    for (uint64_t par_i = 0; par_i < part_count; ++par_i) {
                        megamol::core::moldyn::SimpleSphericalParticles::particle_t par = sim_parts[par_i];
                        float par_pos[3] = {par.vert.GetXf(), par.vert.GetYf(), par.vert.GetZf()};
                        if (par_pos[slice_axis] >= slice_min && par_pos[slice_axis] <= slice_max) {
                            float par_col = {par.col.GetIf()};

                            cur_pos_vec.push_back(par_pos[0]);
                            cur_pos_vec.push_back(par_pos[1]);
                            cur_pos_vec.push_back(par_pos[2]);
                            cur_col_vec.push_back(par_col);
                        }
                    }
                }
            }
        }
    }

    return true;
}
