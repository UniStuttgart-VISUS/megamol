#include "MPDCGrid.h"
#include "stdafx.h"

#include "mmcore/param/IntParam.h"


megamol::datatools::MPDCGrid::MPDCGrid()
        : data_out_slot_("dataOut", "")
        , data_in_slot_("dataIn", "")
        , max_size_slot_("maxSize", "Maximum size of each cell")
        , data_out_hash_(std::numeric_limits<size_t>::max())
        , data_in_hash_(std::numeric_limits<size_t>::max())
        , out_frame_id_(-1) {
    data_out_slot_.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(0), &MPDCGrid::getDataCallback);
    data_out_slot_.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(1), &MPDCGrid::getExtentCallback);
    MakeSlotAvailable(&data_out_slot_);

    data_in_slot_.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    MakeSlotAvailable(&data_in_slot_);

    max_size_slot_ << new core::param::IntParam(1000, 1);
    MakeSlotAvailable(&max_size_slot_);
}


megamol::datatools::MPDCGrid::~MPDCGrid() {
    this->Release();
}


bool megamol::datatools::MPDCGrid::create() {
    return true;
}


void megamol::datatools::MPDCGrid::release() {}


bool megamol::datatools::MPDCGrid::getDataCallback(core::Call& c) {

    auto outData = dynamic_cast<geocalls::MultiParticleDataCall*>(&c);
    if (outData == nullptr)
        return false;

    auto inData = data_in_slot_.CallAs<geocalls::MultiParticleDataCall>();
    if (inData == nullptr)
        return false;

    if (data_in_hash_ != inData->DataHash() || out_frame_id_ != inData->FrameID()) {
        data_in_hash_ = inData->DataHash();
        out_frame_id_ = inData->FrameID();

        if (!(*inData)(0))
            return false;

        auto const plc = inData->GetParticleListCount();
        auto const cbbox = inData->AccessBoundingBoxes().ObjectSpaceClipBox();

        data_.resize(plc);
        output_.clear();

        for (size_t plidx = 0; plidx < plc; ++plidx) {
            auto const& particles = inData->AccessParticles(plidx);

            auto const parStore = particles.GetParticleStore();
            auto const xAcc = parStore.GetXAcc();
            auto const yAcc = parStore.GetYAcc();
            auto const zAcc = parStore.GetZAcc();
            auto const crAcc = parStore.GetCRAcc();
            auto const cgAcc = parStore.GetCGAcc();
            auto const cbAcc = parStore.GetCBAcc();
            auto const caAcc = parStore.GetCAAcc();

            auto const pcount = particles.GetCount();

            data_[plidx].resize(pcount);

            for (size_t pidx = 0; pidx < pcount; ++pidx) {
                data_[plidx][pidx] = {{xAcc->Get_f(pidx), yAcc->Get_f(pidx), zAcc->Get_f(pidx)}, crAcc->Get_u8(pidx),
                    cgAcc->Get_u8(pidx), cbAcc->Get_u8(pidx), caAcc->Get_u8(pidx)};
            }

            auto const maxSize = max_size_slot_.Param<core::param::IntParam>()->Value();
            auto grid = gridify(data_[plidx],
                {{cbbox.Left(), cbbox.Bottom(), cbbox.Back()}, {cbbox.Right(), cbbox.Top(), cbbox.Front()}}, maxSize, 0,
                pcount);

            auto ssps = separate(data_[plidx], grid, particles.GetGlobalRadius());

            ssps.erase(
                std::remove_if(ssps.begin(), ssps.end(), [](auto const& a) { return a.GetCount() == 0; }), ssps.end());

            output_.insert(output_.end(), ssps.cbegin(), ssps.cend());
        }

        outData->SetParticleListCount(output_.size());
        for (size_t plidx = 0; plidx < output_.size(); ++plidx) {
            outData->AccessParticles(plidx) = output_[plidx];
        }

        ++data_out_hash_;
    }

    outData->SetDataHash(data_out_hash_);
    outData->SetFrameID(inData->FrameID());

    return true;
}


bool megamol::datatools::MPDCGrid::getExtentCallback(core::Call& c) {
    auto outData = dynamic_cast<geocalls::MultiParticleDataCall*>(&c);
    if (outData == nullptr)
        return false;

    auto inData = data_in_slot_.CallAs<geocalls::MultiParticleDataCall>();
    if (inData == nullptr)
        return false;

    inData->SetFrameID(outData->FrameID(), outData->IsFrameForced());
    if (!(*inData)(1))
        return false;

    outData->SetFrameCount(inData->FrameCount());
    outData->SetFrameID(inData->FrameID());

    outData->AccessBoundingBoxes().SetObjectSpaceBBox(inData->AccessBoundingBoxes().ObjectSpaceBBox());
    outData->AccessBoundingBoxes().SetObjectSpaceClipBox(inData->AccessBoundingBoxes().ObjectSpaceClipBox());

    outData->SetDataHash(data_out_hash_);

    return true;
}


std::vector<megamol::datatools::MPDCGrid::BrickLet> megamol::datatools::MPDCGrid::gridify(
    std::vector<megamol::datatools::MPDCGrid::Particle>& particles, Box const& bbox, size_t maxSize, size_t begin,
    size_t end) {
    auto const pcount = end - begin;

    std::vector<BrickLet> ret;
    ret.reserve((pcount / maxSize) + 1);

    if (pcount <= maxSize) {
        ret.push_back({begin, end, bbox});
        return ret;
    } else {
        auto const bspan = bbox.span();
        auto const dim = arg_max(bspan);

        auto const splitPos = bbox.calc_center()[dim];

        /*for (size_t pidx = 0; pidx < pcount; ++pidx) {
        }*/
        auto it = std::partition(particles.begin() + begin, particles.begin() + end,
            [dim, splitPos](auto const& a) { return a.pos[dim] < splitPos; });
        auto mid = std::distance(particles.begin(), it);

        auto lbounds = bbox;
        lbounds.upper[dim] = splitPos;
        auto rbounds = bbox;
        rbounds.lower[dim] = splitPos;

        auto left = gridify(particles, lbounds, maxSize, begin, mid);
        auto right = gridify(particles, rbounds, maxSize, mid, end);

        ret.insert(ret.end(), left.cbegin(), left.cend());
        ret.insert(ret.end(), right.cbegin(), right.cend());
    }

    return ret;
}


std::vector<megamol::geocalls::SimpleSphericalParticles> megamol::datatools::MPDCGrid::separate(
    std::vector<megamol::datatools::MPDCGrid::Particle> const& particles,
    std::vector<megamol::datatools::MPDCGrid::BrickLet> const& bricks, float radius) {
    std::vector<geocalls::SimpleSphericalParticles> ret(bricks.size());
    auto vert_base = reinterpret_cast<char const*>(particles.data());
    auto col_base = vert_base + sizeof(vislib::math::Point<float, 3>);
    auto particle_size = sizeof(Particle);
    size_t i = 0;
    for (auto const& brick : bricks) {
        auto& ssp = ret[i];
        ssp.SetCount(brick.end - brick.begin);
        ssp.SetVertexData(geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ,
            vert_base + brick.begin * particle_size, particle_size);
        ssp.SetColourData(geocalls::SimpleSphericalParticles::COLDATA_UINT8_RGBA,
            col_base + brick.begin * particle_size, particle_size);
        ssp.SetGlobalRadius(radius);
        ++i;
    }
    return ret;
}
