#include "stdafx.h"
#include "ParticlesToPaths.h"

#include "mmcore/moldyn/DirectionalParticleDataCall.h"
#include "mmcore/param/BoolParam.h"

#include "vislib/sys/Log.h"

#include <cfenv>
#include "thermodyn/PathLineDataCall.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/IntParam.h"


megamol::thermodyn::ParticlesToPaths::ParticlesToPaths()
    : dataInSlot_("dataIn", "Input of particle data")
    , dataOutSlot_("dataOut", "Output of particle pathlines")
    , cyclXSlot("cyclX", "Considers cyclic boundary conditions in X direction")
    , cyclYSlot("cyclY", "Considers cyclic boundary conditions in Y direction")
    , cyclZSlot("cyclZ", "Considers cyclic boundary conditions in Z direction")
    , bboxSlot("line_bbox", "True, if bbox of lines should be propagated")
    , projectionSlot_("projection", "Select dimension for projection")
    , frameSkipSlot_("frameSkip", "Frames to skip for sub-sampling") {
    dataInSlot_.SetCompatibleCall<core::moldyn::DirectionalParticleDataCallDescription>();
    MakeSlotAvailable(&dataInSlot_);

    dataOutSlot_.SetCallback(
        PathLineDataCall::ClassName(), PathLineDataCall::FunctionName(0), &ParticlesToPaths::getDataCallback);
    dataOutSlot_.SetCallback(
        PathLineDataCall::ClassName(), PathLineDataCall::FunctionName(1), &ParticlesToPaths::getExtentCallback);
    MakeSlotAvailable(&dataOutSlot_);

    this->cyclXSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclXSlot);

    this->cyclYSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclYSlot);

    this->cyclZSlot.SetParameter(new core::param::BoolParam(true));
    this->MakeSlotAvailable(&this->cyclZSlot);

    this->bboxSlot << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->bboxSlot);

    auto ep = new core::param::EnumParam(-1);
    ep->SetTypePair(-1, "None");
    ep->SetTypePair(0, "x");
    ep->SetTypePair(1, "y");
    ep->SetTypePair(2, "z");
    projectionSlot_ << ep;
    MakeSlotAvailable(&projectionSlot_);

    this->frameSkipSlot_ << new core::param::IntParam(1, 1, std::numeric_limits<int>::max());
    MakeSlotAvailable(&frameSkipSlot_);
}


megamol::thermodyn::ParticlesToPaths::~ParticlesToPaths() { this->Release(); }


bool megamol::thermodyn::ParticlesToPaths::create() { return true; }


void megamol::thermodyn::ParticlesToPaths::release() {}


bool megamol::thermodyn::ParticlesToPaths::getDataCallback(core::Call& c) {
    auto outCall = dynamic_cast<PathLineDataCall*>(&c);
    if (outCall == nullptr) return false;

    auto inCall = dataInSlot_.CallAs<core::moldyn::DirectionalParticleDataCall>();
    if (inCall == nullptr) return false;

    if (!(*inCall)(0)) return false;

    auto const frameSkip = frameSkipSlot_.Param<core::param::IntParam>()->Value();

    auto const plc = inCall->GetParticleListCount();
    auto const frameCount = inCall->FrameCount()/frameSkip;

    if (inCall->DataHash() != inDataHash_) {
        inDataHash_ = inCall->DataHash();

        pathStore_.clear();
        pathStore_.resize(plc);

        entrySizes_.resize(plc);
        colsPresent_.resize(plc);
        dirsPresent_.resize(plc);

        auto const bbox = inCall->AccessBoundingBoxes().ObjectSpaceBBox();
        auto const bbwidth = bbox.Width();
        auto const bbheight = bbox.Height();
        auto const bbdepth = bbox.Depth();
        auto const hbbwidth = 0.5f * bbwidth;
        auto const hbbheight = 0.5f * bbheight;
        auto const hbbdepth = 0.5f * bbdepth;
        auto const xo = bbox.GetLeft();
        auto const yo = bbox.GetBottom();
        auto const zo = bbox.GetBack();

        bool const cycl_x = this->cyclXSlot.Param<megamol::core::param::BoolParam>()->Value();
        bool const cycl_y = this->cyclYSlot.Param<megamol::core::param::BoolParam>()->Value();
        bool const cycl_z = this->cyclZSlot.Param<megamol::core::param::BoolParam>()->Value();

        bool const lBBoxFl = this->bboxSlot.Param<core::param::BoolParam>()->Value();

        auto const projection = this->projectionSlot_.Param<core::param::EnumParam>()->Value();

        float xMax = std::numeric_limits<float>::lowest();
        float yMax = std::numeric_limits<float>::lowest();
        float zMax = std::numeric_limits<float>::lowest();
        float xMin = std::numeric_limits<float>::max();
        float yMin = std::numeric_limits<float>::max();
        float zMin = std::numeric_limits<float>::max();

        for (unsigned int plidx = 0; plidx < plc; ++plidx) {
            // step over all time frames to create complete pathline
            auto const& part = inCall->AccessParticles(plidx);
            auto const& pCount = part.GetCount();
            auto& storeEntry = pathStore_[plidx];
            storeEntry.reserve(pCount);

            auto const idDT = part.GetIDDataType();
            if (idDT == core::moldyn::SimpleSphericalParticles::IDDATA_NONE) {
                vislib::sys::Log::DefaultLog.WriteWarn(
                    "ParticlesToPaths: Particlelist entry %d does not contain particle IDs ... Skipping entry\n");
                continue;
            }

            auto const vertDT = part.GetVertexDataType();
            if (vertDT == core::moldyn::SimpleSphericalParticles::VERTDATA_NONE ||
                vertDT == core::moldyn::SimpleSphericalParticles::VERTDATA_SHORT_XYZ) {
                vislib::sys::Log::DefaultLog.WriteWarn("ParticlesToPaths: Particlelist entry %d does not contain "
                                                       "compatible particle positions ... Skipping entry\n");
                continue;
            }

            bool dirPresent = false;
            auto const dirDT = part.GetDirDataType();
            if (dirDT != core::moldyn::DirectionalParticles::DIRDATA_NONE) dirPresent = true;
            bool colPresent = false;
            auto const colDT = part.GetColourDataType();
            if (colDT != core::moldyn::SimpleSphericalParticles::COLDATA_NONE) colPresent = true;

            auto& xAcc = part.GetParticleStore().GetXAcc();
            auto& yAcc = part.GetParticleStore().GetYAcc();
            auto& zAcc = part.GetParticleStore().GetZAcc();
            auto& dxAcc = part.GetParticleStore().GetDXAcc();
            auto& dyAcc = part.GetParticleStore().GetDYAcc();
            auto& dzAcc = part.GetParticleStore().GetDZAcc();
            auto& rAcc = part.GetParticleStore().GetCRAcc();
            auto& gAcc = part.GetParticleStore().GetCGAcc();
            auto& bAcc = part.GetParticleStore().GetCBAcc();
            auto& aAcc = part.GetParticleStore().GetCAAcc();
            auto& idAcc = part.GetParticleStore().GetIDAcc();

            int entrySize = 3;
            if (dirPresent) entrySize += 3;
            if (colPresent) entrySize += 4;

            entrySizes_[plidx] = entrySize;
            colsPresent_[plidx] = colPresent;
            dirsPresent_[plidx] = dirPresent;

            std::vector<float> old_pos(pCount * 3);
            std::vector<float> dec_pos(pCount * 3);

            for (size_t pidx = 0; pidx < pCount; ++pidx) {
                storeEntry[idAcc->Get_u64(pidx)].reserve(frameCount * entrySize);
            }

            auto const r_mode = fegetround();
            fesetround(FE_TONEAREST);

            for (unsigned int fidx = 0; fidx < frameCount; ++fidx) {
                do {
                    inCall->SetFrameID(fidx*frameSkip, true);
                    (*inCall)(1);
                } while (fidx*frameSkip != inCall->FrameID());

                if (!(*inCall)(0)) return false;

                for (size_t pidx = 0; pidx < pCount; ++pidx) {
                    auto idx = idAcc->Get_u64(pidx);

                    auto x = xAcc->Get_f(pidx);
                    auto y = yAcc->Get_f(pidx);
                    auto z = zAcc->Get_f(pidx);

                    if (fidx == 0) {
                        old_pos[pidx * 3 + 0] = xAcc->Get_f(pidx);
                        old_pos[pidx * 3 + 1] = yAcc->Get_f(pidx);
                        old_pos[pidx * 3 + 2] = zAcc->Get_f(pidx);
                        dec_pos[pidx * 3 + 0] = xAcc->Get_f(pidx);
                        dec_pos[pidx * 3 + 1] = yAcc->Get_f(pidx);
                        dec_pos[pidx * 3 + 2] = zAcc->Get_f(pidx);
                    }

                    auto const px = old_pos[pidx * 3 + 0];
                    auto const py = old_pos[pidx * 3 + 1];
                    auto const pz = old_pos[pidx * 3 + 2];

                    // auto dis = std::sqrtf(std::powf(px-x, 2.0f) + std::powf(py-y, 2.0f) + std::powf(pz-z, 2.0f));
                    auto xdis = std::fabs(px - x);
                    auto ydis = std::fabs(py - y);
                    auto zdis = std::fabs(pz - z);

                    /*auto xdis = (px - x);
                    auto ydis = (py - y);
                    auto zdis = (pz - z);*/

                    if (cycl_x && xdis >= hbbwidth) {
                        // x += bbwidth * std::copysign(std::ceil(xdis / bbwidth), px - x);
                        xdis = (px - x) - bbwidth * std::nearbyint((px - x) / bbwidth);
                    }
                    if (cycl_y && ydis >= hbbheight) {
                        // y += bbheight * std::copysign(std::ceil(ydis / bbheight), py - y);
                        ydis = (py - y) - bbheight * std::nearbyint((py - y) / bbheight);
                    }
                    if (cycl_z && zdis >= hbbdepth) {
                        // z += bbwidth * std::copysign(std::ceil(zdis / bbdepth), pz - z);
                        zdis = (pz - z) - bbdepth * std::nearbyint((pz - z) / bbdepth);
                    }

                    if (cycl_x) x = dec_pos[pidx * 3 + 0] + xdis * std::copysign(1.0f, px - x);
                    if (cycl_y) y = dec_pos[pidx * 3 + 1] + ydis * std::copysign(1.0f, py - y);
                    if (cycl_z) z = dec_pos[pidx * 3 + 2] + zdis * std::copysign(1.0f, pz - z);

                    auto& entry = storeEntry[idx];
                    /*entry.push_back(xAcc->Get_f(pidx));
                    entry.push_back(yAcc->Get_f(pidx));
                    entry.push_back(zAcc->Get_f(pidx));*/
                    entry.push_back(x);
                    entry.push_back(y);
                    entry.push_back(z);
                    if (lBBoxFl) {
                        xMax = std::max(x, xMax);
                        yMax = std::max(y, yMax);
                        zMax = std::max(z, zMax);
                        xMin = std::min(x, xMin);
                        yMin = std::min(y, yMin);
                        zMin = std::min(z, zMin);
                    }
                    if (colPresent) {
                        entry.push_back(rAcc->Get_f(pidx));
                        entry.push_back(gAcc->Get_f(pidx));
                        entry.push_back(bAcc->Get_f(pidx));
                        entry.push_back(aAcc->Get_f(pidx));
                    }
                    if (dirPresent) {
                        entry.push_back(dxAcc->Get_f(pidx));
                        entry.push_back(dyAcc->Get_f(pidx));
                        entry.push_back(dzAcc->Get_f(pidx));
                    }
                    old_pos[pidx * 3 + 0] = xAcc->Get_f(pidx);
                    old_pos[pidx * 3 + 1] = yAcc->Get_f(pidx);
                    old_pos[pidx * 3 + 2] = zAcc->Get_f(pidx);
                    dec_pos[pidx * 3 + 0] = x;
                    dec_pos[pidx * 3 + 1] = y;
                    dec_pos[pidx * 3 + 2] = z;
                }
            }
            fesetround(r_mode);
        }
        if (projection > -1) {
            for (unsigned int plidx = 0; plidx < plc; ++plidx) {
                auto& storeEntry = pathStore_[plidx];
                auto entrysize = entrySizes_[plidx];
                for (auto& entry : storeEntry) {
                    for (size_t idx = 0; idx < entry.second.size(); idx += entrysize) {
                        entry.second[idx+projection] = 0.0f;
                    }
                }
            }
            if (projection == 0) {
                xMin = -1.0f;
                xMax = 1.0f;
            }
            if (projection == 1) {
                yMin = -1.0f;
                yMax = 1.0f;
            }
            if (projection == 2) {
                zMin = -1.0f;
                zMax = 1.0f;
            }
        }
        if (lBBoxFl) {
            this->bbox.Set(xMin, yMin, zMin, xMax, yMax, zMax);
        }
    }

    // all data is serialized
    outCall->SetEntrySizes(entrySizes_);
    outCall->SetColorFlags(colsPresent_);
    outCall->SetDirFlags(dirsPresent_);
    outCall->SetPathStore(&pathStore_);
    outCall->SetTimeSteps(frameCount);
    outCall->SetDataHash(inDataHash_);

    if (this->bboxSlot.Param<core::param::BoolParam>()->Value()) {
        outCall->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
        outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox); //< TODO Not the right bbox
        outCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);
    }

    return true;
}


bool megamol::thermodyn::ParticlesToPaths::getExtentCallback(core::Call& c) {
    auto outCall = dynamic_cast<PathLineDataCall*>(&c);
    if (outCall == nullptr) return false;

    auto inCall = dataInSlot_.CallAs<core::moldyn::DirectionalParticleDataCall>();
    if (inCall == nullptr) return false;

    if (!(*inCall)(1)) return false;

    if (this->bboxSlot.Param<core::param::BoolParam>()->Value()) {
        outCall->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
        outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox); //< TODO Not the right bbox
        outCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);
    } else {
        outCall->AccessBoundingBoxes().SetObjectSpaceBBox(inCall->AccessBoundingBoxes().ObjectSpaceBBox());
        outCall->AccessBoundingBoxes().SetObjectSpaceClipBox(inCall->AccessBoundingBoxes().ObjectSpaceClipBox());
        outCall->AccessBoundingBoxes().MakeScaledWorld(1.0f);
    }

    outCall->SetFrameCount(1);

    outCall->SetDataHash(inDataHash_);

    return true;
}
