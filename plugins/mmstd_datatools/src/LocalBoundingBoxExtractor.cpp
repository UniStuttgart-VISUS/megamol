#include "LocalBoundingBoxExtractor.h"
#include "vislib/math/Cuboid.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/param/ColorParam.h"

namespace megamol {
namespace stdplugin {
namespace datatools {


LocalBoundingBoxExtractor::LocalBoundingBoxExtractor()
    : core::Module()
    , inDataSlot("inDataSlot","MPDC connection")
    , outLinesSlot("outLinesSlot","Line connection")
    , outMeshSlot("outMeshSlot","Mesh connection")
    , colorSlot ("color", "Sets color of bounding box rendering") {

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

    this->colorSlot << new megamol::core::param::ColorParam(0.8 ,0.8, 0.8, 1.0);
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
        
    if (a.IsEmpty()) {
        this->calcLocalBox(parts, a);
        if (!(parts.GetCount() > 0)) {
            a = mpdc->GetBoundingBoxes().ObjectSpaceBBox();
        }
    }

    typedef vislib::math::AbstractCuboid<float, float[6]> Super;

    std::array<float, 3> lbf = {a.Left(), a.Bottom(), a.Front()};
    std::array<float, 3> lbb = {a.Left(), a.Bottom(), a.Back()};
    std::array<float, 3> ltf = {a.Left(), a.Top(), a.Front()};
    std::array<float, 3> ltb = {a.Left(), a.Top(), a.Back()};
    std::array<float, 3> rbf = {a.Right(), a.Bottom(), a.Front()};
    std::array<float, 3> rbb = {a.Right(), a.Bottom(), a.Back()};
    std::array<float, 3> rtf = {a.Right(), a.Top(), a.Front()};
    std::array<float, 3> rtb = {a.Right(), a.Top(), a.Back()};
                                                    
    // set line data
    if (ldc != nullptr) {
        this->lines.clear();
        this->lines.resize(12); // edges of a cube

        // 12 combinations 
        lineMap.clear();
        lineMap["l1"] = std::array<float, 6>();
        lineMap["l2"] = std::array<float, 6>();
        lineMap["l3"] = std::array<float, 6>();
        lineMap["l4"] = std::array<float, 6>();
        lineMap["l5"] = std::array<float, 6>();
        lineMap["l6"] = std::array<float, 6>();
        lineMap["l7"] = std::array<float, 6>();
        lineMap["l8"] = std::array<float, 6>();
        lineMap["l9"] = std::array<float, 6>();
        lineMap["l10"] = std::array<float, 6>();
        lineMap["l11"] = std::array<float, 6>();
        lineMap["l12"] = std::array<float, 6>();
        std::copy(lbf.begin(), lbf.end(), lineMap["l1"].begin());
        std::copy(rbf.begin(), rbf.end(), lineMap["l1"].begin() + 3);
        std::copy(lbf.begin(), lbf.end(), lineMap["l2"].begin());
        std::copy(ltf.begin(), ltf.end(), lineMap["l2"].begin() + 3);
        std::copy(lbf.begin(), lbf.end(), lineMap["l3"].begin());
        std::copy(lbb.begin(), lbb.end(), lineMap["l3"].begin() + 3);
        std::copy(rtf.begin(), rtf.end(), lineMap["l4"].begin());
        std::copy(rbf.begin(), rbf.end(), lineMap["l4"].begin() + 3);
        std::copy(rtf.begin(), rtf.end(), lineMap["l5"].begin());
        std::copy(ltf.begin(), ltf.end(), lineMap["l5"].begin() + 3);
        std::copy(rtf.begin(), rtf.end(), lineMap["l6"].begin());
        std::copy(rtb.begin(), rtb.end(), lineMap["l6"].begin() + 3);
        std::copy(ltb.begin(), ltb.end(), lineMap["l7"].begin());
        std::copy(ltf.begin(), ltf.end(), lineMap["l7"].begin() + 3);
        std::copy(ltb.begin(), ltb.end(), lineMap["l8"].begin());
        std::copy(rtb.begin(), rtb.end(), lineMap["l8"].begin() + 3);
        std::copy(ltb.begin(), ltb.end(), lineMap["l9"].begin());
        std::copy(lbb.begin(), lbb.end(), lineMap["l9"].begin() + 3);
        std::copy(rbb.begin(), rbb.end(), lineMap["l10"].begin());
        std::copy(lbb.begin(), lbb.end(), lineMap["l10"].begin() + 3);
        std::copy(rbb.begin(), rbb.end(), lineMap["l11"].begin());
        std::copy(rtb.begin(), rtb.end(), lineMap["l11"].begin() + 3);
        std::copy(rbb.begin(), rbb.end(), lineMap["l12"].begin());
        std::copy(rbf.begin(), rbf.end(), lineMap["l12"].begin() + 3);

        std::array<unsigned char,4> rgba = {
            static_cast<unsigned char>(this->colorSlot.Param<core::param::ColorParam>()->Value()[0] * 255),
            static_cast<unsigned char>(this->colorSlot.Param<core::param::ColorParam>()->Value()[1] * 255),
            static_cast<unsigned char>(this->colorSlot.Param<core::param::ColorParam>()->Value()[2] * 255),
            static_cast<unsigned char>(this->colorSlot.Param<core::param::ColorParam>()->Value()[3] * 255)};

        auto it = lineMap.begin();
        for (auto loop = 0; loop < lineMap.size(); loop++) {
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
        int triCount = 12;
        int vertCount = 8;

        allVerts.clear();
        allVerts.insert(allVerts.end(), lbf.begin(), lbf.end());
        allVerts.insert(allVerts.end(), lbb.begin(), lbb.end());
        allVerts.insert(allVerts.end(), ltf.begin(), ltf.end());
        allVerts.insert(allVerts.end(), ltb.begin(), ltb.end());
        allVerts.insert(allVerts.end(), rbf.begin(), rbf.end());
        allVerts.insert(allVerts.end(), rbb.begin(), rbb.end());
        allVerts.insert(allVerts.end(), rtf.begin(), rtf.end());
        allVerts.insert(allVerts.end(), rtb.begin(), rtb.end());

        enum cornerMap {
            LBF = 0,
            LBB = 1,
            LTF = 2,
            LTB = 3,
            RBF = 4,
            RBB = 5,
            RTF = 6,
            RTB = 7
        };


        allCols.clear();
        int colCount = 3;
        allCols.resize(colCount * vertCount);
        for (auto i = 0; i < vertCount; i++) {
            allCols[colCount * i + 0] = this->colorSlot.Param<core::param::ColorParam>()->Value()[0];
            allCols[colCount * i + 1] = this->colorSlot.Param<core::param::ColorParam>()->Value()[1];
            allCols[colCount * i + 2] = this->colorSlot.Param<core::param::ColorParam>()->Value()[2];
           // allCols[colCount * i + 3] = this->colorSlot.Param<core::param::ColorParam>()->Value()[3];
        }

        allIdx.clear();
        allIdx = {
            LBF, RBF, LTF,
            LBF, RBF, LBB,
            LBF, LTF, LBB,
            RTF, RBF, LTF,
            RTF, RBF, RTB,
            RTF, LTF, RTB,
            RBB, RTB, RBF,
            RBB, RTB, LBB,
            RBB, RBF, LBB,
            LTB, LBB, RTB,
            LTB, LBB, LTF,
            LTB, RTB, LTF
        };

        this->mesh.SetVertexData(vertCount, allVerts.data(), nullptr, allCols.data(), nullptr, false);
        this->mesh.SetTriangleData(triCount, allIdx.data(), false);

        ctmd->SetFrameCount(1);
        ctmd->SetFrameID(0);
        ctmd->SetObjects(1, &this->mesh);
        ctmd->SetDataHash(mpdc->DataHash());
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

    if (ctmd != nullptr) {
        ctmd->SetExtent(1, globalBB.Left(), globalBB.Bottom(), globalBB.Front(), globalBB.Right(), globalBB.Top(), globalBB.Back());
     }

    return true;
}

void LocalBoundingBoxExtractor::calcLocalBox(
    core::moldyn::MultiParticleDataCall::Particles& parts, vislib::math::Cuboid<float>& box) {
    
    if (!(parts.GetCount() > 0)) return;

    for (int i = 0; i < parts.GetCount(); i++) {
        box.SetLeft(std::min(box.GetLeft(),parts.GetParticleStore().GetXAcc()->Get_f(i)));
        box.SetRight(std::max(box.GetRight(), parts.GetParticleStore().GetXAcc()->Get_f(i)));
        box.SetBottom(std::min(box.GetBottom(), parts.GetParticleStore().GetYAcc()->Get_f(i)));
        box.SetTop(std::max(box.GetTop(), parts.GetParticleStore().GetYAcc()->Get_f(i)));
        box.SetFront(std::min(box.GetFront(), parts.GetParticleStore().GetZAcc()->Get_f(i)));
        box.SetBack(std::max(box.GetBack(), parts.GetParticleStore().GetZAcc()->Get_f(i)));
    }
}

} // namespace datatools
} // namespace stdplugin
} // namespace megamol