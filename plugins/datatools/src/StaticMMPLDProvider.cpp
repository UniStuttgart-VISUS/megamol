#include "StaticMMPLDProvider.h"
#include "stdafx.h"

#include "mmcore/param/StringParam.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"


megamol::datatools::StaticMMPLDProvider::StaticMMPLDProvider()
        : outDataSlot("outData", "Output")
        , filenamesSlot("filenames", "Set of filenames separated with ';'") {
    outDataSlot.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(0), &StaticMMPLDProvider::getDataCallback);
    outDataSlot.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(1), &StaticMMPLDProvider::getExtentCallback);
    MakeSlotAvailable(&outDataSlot);

    filenamesSlot << new core::param::StringParam("");
    MakeSlotAvailable(&filenamesSlot);
}


megamol::datatools::StaticMMPLDProvider::~StaticMMPLDProvider() {
    this->Release();
}


bool megamol::datatools::StaticMMPLDProvider::create() {
    return true;
}


void megamol::datatools::StaticMMPLDProvider::release() {}


bool megamol::datatools::StaticMMPLDProvider::assertData(geocalls::MultiParticleDataCall& outCall) {
    if (filenamesSlot.IsDirty()) {

        auto const filenames = vislib::TString(filenamesSlot.Param<core::param::StringParam>()->Value().c_str());

        vislib::Array<vislib::TString> filenamesArray = vislib::TStringTokeniser::Split(filenames, _T(";"), true);

        unsigned int list_count = 0;

        output_frames.clear();
        output_frames.reserve(filenamesArray.Count());

        for (unsigned int fidx = 0; fidx < filenamesArray.Count(); ++fidx) {
            auto mmpld_file = mmpld::mmpld(std::string(T2A(filenamesArray[fidx])));

            output_frames.push_back(mmpld_file.ReadFrame(0));

            list_count += output_frames.back().data.size();

            auto box = mmpld_file.GetBBox();
            auto tmp_box = vislib::math::Cuboid<float>(box[0], box[1], box[2], box[3], box[4], box[5]);

            if (fidx == 0) {
                gbbox = tmp_box;
            } else {
                gbbox.Union(tmp_box);
            }
        }

        outCall.SetParticleListCount(list_count);
        auto counter = 0u;
        for (auto const& frame : output_frames) {
            for (auto idx = 0u; idx < frame.data.size(); ++idx) {
                auto& particles = outCall.AccessParticles(counter);
                auto const& entry = frame.data[idx];

                particles.SetCount(entry.list_header.particle_count);
                vislib::math::Cuboid<float> bbox(entry.list_header.lbox[0], entry.list_header.lbox[1],
                    entry.list_header.lbox[2], entry.list_header.lbox[3], entry.list_header.lbox[4],
                    entry.list_header.lbox[5]);
                particles.SetBBox(bbox);
                particles.SetVertexData(
                    static_cast<geocalls::SimpleSphericalParticles::VertexDataType>(entry.list_header.vert_type),
                    entry.data.data() + entry.vertex_offset, entry.vertex_stride + entry.color_stride);
                particles.SetColourData(ColorTypeTranslator(entry.list_header.col_type),
                    entry.data.data() + entry.color_offset, entry.vertex_stride + entry.color_stride);
                particles.SetGlobalRadius(entry.list_header.global_radius);
                particles.SetGlobalColour(entry.list_header.global_color[0], entry.list_header.global_color[1],
                    entry.list_header.global_color[2], entry.list_header.global_color[3]);
                particles.SetColourMapIndexValues(
                    entry.list_header.intensity_range[0], entry.list_header.intensity_range[1]);

                ++counter;
            }
        }

        outCall.AccessBoundingBoxes().SetObjectSpaceBBox(gbbox);
        outCall.AccessBoundingBoxes().SetObjectSpaceClipBox(gbbox);
        outCall.AccessBoundingBoxes().MakeScaledWorld(1.0f);

        outCall.SetFrameCount(1);
        outCall.SetDataHash(++hash);

        filenamesSlot.ResetDirty();
    }


    return !output_frames.empty();
}


bool megamol::datatools::StaticMMPLDProvider::getDataCallback(core::Call& c) {
    auto outCall = dynamic_cast<geocalls::MultiParticleDataCall*>(&c);
    if (outCall == nullptr)
        return false;

    return assertData(*outCall);
}


bool megamol::datatools::StaticMMPLDProvider::getExtentCallback(core::Call& c) {
    auto outCall = dynamic_cast<geocalls::MultiParticleDataCall*>(&c);
    if (outCall == nullptr)
        return false;

    return assertData(*outCall);
}
