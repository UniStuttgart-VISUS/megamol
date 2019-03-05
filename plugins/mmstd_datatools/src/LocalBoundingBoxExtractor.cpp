#include "LocalBoundingBoxExtractor.h"
#include "geometry_calls/CallTriMeshData.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "vislib/math/Cuboid.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"

namespace megamol {
namespace stdplugin {
namespace datatools {


LocalBoundingBoxExtractor::LocalBoundingBoxExtractor()
    : core::Module()
    , inDataSlot("inDataSlot","MPDC connection")
    , outLinesSlot("outLinesSlot","Line connection")
    , outMeshSlot("outMeshSlot","Mesh connection")
    , colorSlot ("colorSlot", "Sets color of bounding box rendering") {

    this->outLinesSlot.SetCallback(geocalls::LinesDataCall::ClassName(), "GetData", &LocalBoundingBoxExtractor::getDataCallback);
    this->outLinesSlot.SetCallback(
        geocalls::LinesDataCall::ClassName(), "GetExtent", &LocalBoundingBoxExtractor::getExtentCallback);
    this->MakeSlotAvailable(&this->outLinesSlot);

    this->outMeshSlot.SetCallback(
        geocalls::CallTriMeshData::ClassName(), "GetData", &LocalBoundingBoxExtractor::getDataCallback);
    this->outMeshSlot.SetCallback(
        geocalls::CallTriMeshData::ClassName(), "GetExtent", &LocalBoundingBoxExtractor::getExtentCallback);
    this->MakeSlotAvailable(&this->outMeshSlot);

    this->inDataSlot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);

    this->colorSlot << new megamol::core::param::StringParam(_T("gray"));
    this->MakeSlotAvailable(&this->colorSlot);



}

LocalBoundingBoxExtractor::~LocalBoundingBoxExtractor() { this->Release(); }

bool LocalBoundingBoxExtractor::create() { return true; }

void LocalBoundingBoxExtractor::release() {
    // empty
}

bool LocalBoundingBoxExtractor::getDataCallback(megamol::core::Call& c) {
    
    geocalls::LinesDataCall* ldc = dynamic_cast<geocalls::LinesDataCall*>(&c);
    geocalls::CallTriMeshData* ctmd = dynamic_cast<geocalls::CallTriMeshData*>(&c);
    core::moldyn::MultiParticleDataCall* mpdc = this->inDataSlot.CallAs<core::moldyn::MultiParticleDataCall>();
    if (mpdc == nullptr) return false;

    if (!(*mpdc)(0)) return false;


    core::moldyn::MultiParticleDataCall::Particles& parts =  mpdc->AccessParticles(0);

    vislib::math::Cuboid<float> a = parts.GetBBox();
    typedef vislib::math::AbstractCuboid<float, float[6]> Super;

    std::vector<float> lbf = {a.Front(), a.Left(), a.Bottom()};
    std::vector<float> lbb = {a.Back(), a.Left(), a.Bottom()};
    std::vector<float> ltf = {a.Front(), a.Left(), a.Top()};
    std::vector<float> ltb = {a.Back(), a.Left(), a.Top()};
    std::vector<float> rbf = {a.Front(), a.Right(), a.Bottom()};
    std::vector<float> rbb = {a.Back(), a.Right(), a.Bottom()};
    std::vector<float> rtf = {a.Front(), a.Right(), a.Top()};
    std::vector<float> rtb = {a.Back(), a.Right(), a.Top()};
                                                    
    // set line data
    if (ldc != nullptr) {
        this->lines.clear();
        this->lines.resize(12); // edges of a cube

        // 12 combinations 
        lineMap.clear();
        lineMap["l1"] = lbf;
        lineMap["l1"].insert(lineMap["l1"].end(), rbf.begin(), rbf.end());
        lineMap["l2"] = lbf;
        lineMap["l2"].insert(lineMap["l2"].end(), ltf.begin(), ltf.end());
        lineMap["l3"] = lbf;
        lineMap["l3"].insert(lineMap["l3"].end(), lbb.begin(), lbb.end());
        lineMap["l4"] = rtf;
        lineMap["l4"].insert(lineMap["l4"].end(), rbf.begin(), rbf.end());
        lineMap["l5"] = rtf;
        lineMap["l5"].insert(lineMap["l5"].end(), ltf.begin(), ltf.end());
        lineMap["l6"] = rtf;
        lineMap["l6"].insert(lineMap["l6"].end(), rtb.begin(), rtb.end());
        lineMap["l7"] = ltb;
        lineMap["l7"].insert(lineMap["l7"].end(), ltf.begin(), ltf.end());
        lineMap["l8"] = ltb;
        lineMap["l8"].insert(lineMap["l8"].end(), rtb.begin(), rtb.end());
        lineMap["l9"] = ltb;
        lineMap["l9"].insert(lineMap["l9"].end(), lbb.begin(), lbb.end());
        lineMap["l10"] = rbb;
        lineMap["l10"].insert(lineMap["l10"].end(), lbb.begin(), lbb.end());
        lineMap["l11"] = rbb;
        lineMap["l11"].insert(lineMap["l11"].end(), rtb.begin(), rtb.end());
        lineMap["l12"] = rbb;
        lineMap["l12"].insert(lineMap["l12"].end(), rbf.begin(), rbf.end());

        auto it = lineMap.begin();
        for (auto loop = 0; loop < lineMap.size(); loop++) {
            unsigned char rgba[4];
            core::utility::ColourParser::FromString(
                this->colorSlot.Param<core::param::StringParam>()->Value(), 4, rgba);
            lines[loop].Set(static_cast<unsigned int>(it->second.size() / 3), it->second.data(),
                vislib::graphics::ColourRGBAu8(rgba[0], rgba[1], rgba[2], rgba[3]));
            std::advance(it, 1);
        }


        ldc->SetFrameCount(1);
        ldc->SetFrameID(0);
        ldc->SetDataHash(mpdc->DataHash());
        ldc->SetData(lines.size(), lines.data());
    }

    // set trimesh data
    if (ctmd != nullptr) {
    
    }


    return true;
}

bool LocalBoundingBoxExtractor::getExtentCallback(megamol::core::Call& c) {
    
    geocalls::LinesDataCall* ldc = dynamic_cast<geocalls::LinesDataCall*>(&c);
    geocalls::CallTriMeshData* ctmd = dynamic_cast<geocalls::CallTriMeshData*>(&c);
    core::moldyn::MultiParticleDataCall* mpdc = this->inDataSlot.CallAs<core::moldyn::MultiParticleDataCall>();
    if (mpdc == nullptr) return false;

    if (!(*mpdc)(1)) return false;

    auto globalBB = mpdc->GetBoundingBoxes().ObjectSpaceBBox();

    if (ldc != nullptr) {
        ldc->SetExtent(1, globalBB.Left(),globalBB.Bottom(), globalBB.Front(), globalBB.Right(), globalBB.Top(), globalBB.Back());
    }



    return true;
}

} // namespace datatools
} // namespace stdplugin
} // namespace megamol