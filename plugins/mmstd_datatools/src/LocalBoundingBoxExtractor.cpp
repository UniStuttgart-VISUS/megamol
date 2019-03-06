#include "LocalBoundingBoxExtractor.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "vislib/math/Cuboid.h"
#include "mmcore/utility/ColourParser.h"
#include "mmcore/param/ColorParam.h"
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
    typedef vislib::math::AbstractCuboid<float, float[6]> Super;

    std::vector<float> lbf = { a.Left(), a.Bottom() ,a.Front()};
    std::vector<float> lbb = {a.Left(), a.Bottom()  ,a.Back() };
    std::vector<float> ltf = { a.Left(), a.Top()    ,a.Front()};
    std::vector<float> ltb = {a.Left(), a.Top()     ,a.Back() };
    std::vector<float> rbf = { a.Right(), a.Bottom(),a.Front()};
    std::vector<float> rbb = {a.Right(), a.Bottom() ,a.Back() };
    std::vector<float> rtf = { a.Right(), a.Top()   ,a.Front()};
    std::vector<float> rtb = {a.Right(), a.Top()    ,a.Back() };
                                                    
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

        std::vector<unsigned char> rgba = { 
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

} // namespace datatools
} // namespace stdplugin
} // namespace megamol