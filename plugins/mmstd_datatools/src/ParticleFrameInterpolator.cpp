#include "stdafx.h"
#include "ParticleFrameInterpolator.h"

#include <algorithm>
#include <map>

#include "mmcore/moldyn/DirectionalParticleDataCall.h"

#include "mmcore/param/BoolParam.h"

#include "vislib/math/Quaternion.h"
#include "vislib/math/ShallowPoint.h"

using namespace megamol;
using namespace megamol::stdplugin::datatools;


/*
 * ParticleFrameInterpolator::ParticleFrameInterpolator
 */
ParticleFrameInterpolator::ParticleFrameInterpolator()
    : core::Module()
    , dataInSlot("dataIn", "MPDC input")
    , supplementalSlot("suppIn", "Supplemental input")
    , dataOutSlot("dataOut", "MPDC output")
    , doSortParam("doSort", "If True, particles will be sorted")
    , directionalParam("directional", "If True, particles will be displaced along given directions")
    , initialized(false)
    , startFrameID(0)
    , currentTimeStamp(-1)
    , requestedTime(0) {
    this->dataInSlot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    this->dataInSlot.SetCompatibleCall<core::moldyn::DirectionalParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->dataInSlot);

    this->supplementalSlot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    this->supplementalSlot.SetCompatibleCall<core::moldyn::DirectionalParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->supplementalSlot);

    this->dataOutSlot.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &ParticleFrameInterpolator::getDataCB);
    this->dataOutSlot.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &ParticleFrameInterpolator::getExtentCB);
    this->dataOutSlot.SetCallback(core::moldyn::DirectionalParticleDataCall::ClassName(),
        core::moldyn::DirectionalParticleDataCall::FunctionName(0), &ParticleFrameInterpolator::getDataCB);
    this->dataOutSlot.SetCallback(core::moldyn::DirectionalParticleDataCall::ClassName(),
        core::moldyn::DirectionalParticleDataCall::FunctionName(1), &ParticleFrameInterpolator::getExtentCB);
    this->MakeSlotAvailable(&this->dataOutSlot);

    this->doSortParam << new core::param::BoolParam(true);
    this->MakeSlotAvailable(&this->doSortParam);

    this->directionalParam << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->directionalParam);
}


/*
 * ParticleFrameInterpolator::~ParticleFrameInterpolator
 */
ParticleFrameInterpolator::~ParticleFrameInterpolator() { this->Release(); }

/*
 * ParticleFrameInterpolator::create
 */
bool ParticleFrameInterpolator::create(void) { return true; }

/*
 * ParticleFrameInterpolator::release
 */
void ParticleFrameInterpolator::release(void) {}

/*
 * ParticleFrameInterpolator::getDataCB
 */
bool ParticleFrameInterpolator::getDataCB(core::Call& c) {
    core::moldyn::MultiParticleDataCall* outMPDC = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    core::moldyn::DirectionalParticleDataCall* outDPDC = dynamic_cast<core::moldyn::DirectionalParticleDataCall*>(&c);
    if ((outMPDC == nullptr) && (outDPDC == nullptr)) return false;

    core::moldyn::DirectionalParticleDataCall* inDPDC =
        this->dataInSlot.CallAs<core::moldyn::DirectionalParticleDataCall>();
    core::moldyn::MultiParticleDataCall* inMPDC = this->dataInSlot.CallAs<core::moldyn::MultiParticleDataCall>();
    if ((inMPDC == nullptr) && (inDPDC == nullptr)) return false;

    core::moldyn::MultiParticleDataCall* suppMPDC =
        this->supplementalSlot.CallAs<core::moldyn::MultiParticleDataCall>();

    if (inDPDC != nullptr) {
        this->checkForTransition(*inDPDC, c);
    } else {
        this->checkForTransition(*inMPDC, c);
    }

    if (outDPDC != nullptr) {
        if (inDPDC != nullptr) {
            outDPDC->SetParticleListCount(this->outData.size());

            for (unsigned int pl = 0; pl < this->outData.size(); pl++) {
                auto& outPart = outDPDC->AccessParticles(pl);
                auto& inPart = inDPDC->AccessParticles(pl);

                outPart.SetCount(this->outData[pl].size() / CROWBAR_DIMS);
                outPart.SetGlobalRadius(inPart.GetGlobalRadius());
                // outPart.SetGlobalColour(inPart.GetGlobalColour());
                outPart.SetVertexData(
                    inPart.GetVertexDataType(), this->outData[pl].data(), CROWBAR_DIMS * sizeof(float));
                outPart.SetColourData(
                    inPart.GetColourDataType(), &((this->outData[pl].data())[4]), CROWBAR_DIMS * sizeof(float));
                outPart.SetDirData(
                    inPart.GetDirDataType(), &((this->outData[pl].data())[8]), CROWBAR_DIMS * sizeof(float));

                outPart.SetColourMapIndexValues(inPart.GetMinColourIndexValue(), inPart.GetMaxColourIndexValue());
            }

            outDPDC->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
            outDPDC->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox);

            outDPDC->SetTimeStamp(this->requestedTime);
        } else if (inMPDC != nullptr) {
            outDPDC->SetParticleListCount(this->outData.size());

            for (unsigned int pl = 0; pl < this->outData.size(); pl++) {
                auto& outPart = outDPDC->AccessParticles(pl);
                auto& inPart = inMPDC->AccessParticles(pl);

                outPart.SetCount(this->outData[pl].size() / CROWBAR_DIMS);
                outPart.SetGlobalRadius(inPart.GetGlobalRadius());
                // outPart.SetGlobalColour(inPart.GetGlobalColour());
                outPart.SetVertexData(
                    inPart.GetVertexDataType(), this->outData[pl].data(), CROWBAR_DIMS * sizeof(float));
                outPart.SetColourData(
                    inPart.GetColourDataType(), &((this->outData[pl].data())[4]), CROWBAR_DIMS * sizeof(float));
                outPart.SetDirData(core::moldyn::DirectionalParticles::DIRDATA_NONE, NULL);

                outPart.SetColourMapIndexValues(inPart.GetMinColourIndexValue(), inPart.GetMaxColourIndexValue());
            }

            outDPDC->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
            outDPDC->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox);

            outDPDC->SetTimeStamp(this->requestedTime);
        }
    } else if (outMPDC != nullptr) {
        if (inDPDC != nullptr) {
            outMPDC->SetParticleListCount(this->outData.size());

            for (unsigned int pl = 0; pl < this->outData.size(); pl++) {
                auto& outPart = outMPDC->AccessParticles(pl);
                auto& inPart = inDPDC->AccessParticles(pl);

                outPart.SetCount(this->outData[pl].size() / CROWBAR_DIMS);
                outPart.SetGlobalRadius(inPart.GetGlobalRadius());
                // outPart.SetGlobalColour(inPart.GetGlobalColour());
                outPart.SetVertexData(
                    inPart.GetVertexDataType(), this->outData[pl].data(), CROWBAR_DIMS * sizeof(float));
                outPart.SetColourData(
                    inPart.GetColourDataType(), &((this->outData[pl].data())[4]), CROWBAR_DIMS * sizeof(float));

                outPart.SetColourMapIndexValues(inPart.GetMinColourIndexValue(), inPart.GetMaxColourIndexValue());
            }

            outMPDC->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
            outMPDC->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox);
        } else if (inMPDC != nullptr) {
            outMPDC->SetParticleListCount(this->outData.size());

            for (unsigned int pl = 0; pl < this->outData.size(); pl++) {
                auto& outPart = outMPDC->AccessParticles(pl);
                auto& inPart = inMPDC->AccessParticles(pl);

                outPart.SetCount(this->outData[pl].size() / CROWBAR_DIMS);
                outPart.SetGlobalRadius(inPart.GetGlobalRadius());
                // outPart.SetGlobalColour(inPart.GetGlobalColour());
                outPart.SetVertexData(
                    inPart.GetVertexDataType(), this->outData[pl].data(), CROWBAR_DIMS * sizeof(float));
                outPart.SetColourData(
                    inPart.GetColourDataType(), &((this->outData[pl].data())[4]), CROWBAR_DIMS * sizeof(float));

                if (suppMPDC == nullptr) {
                    outPart.SetColourMapIndexValues(inPart.GetMinColourIndexValue(), inPart.GetMaxColourIndexValue());
                } else {
                    outPart.SetColourMapIndexValues(suppMPDC->AccessParticles(pl).GetMinColourIndexValue(),
                        suppMPDC->AccessParticles(pl).GetMaxColourIndexValue());
                }
            }

            outMPDC->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
            outMPDC->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox);
        }
    }

    return true;
}

/*
 * ParticleFrameInterpolator::getExtentCB
 */
bool ParticleFrameInterpolator::getExtentCB(core::Call& c) {
    core::moldyn::MultiParticleDataCall* outMPDC = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&c);
    core::moldyn::DirectionalParticleDataCall* outDPDC = dynamic_cast<core::moldyn::DirectionalParticleDataCall*>(&c);
    if ((outMPDC == nullptr) && (outDPDC == nullptr)) return false;

    core::moldyn::DirectionalParticleDataCall* inDPDC =
        this->dataInSlot.CallAs<core::moldyn::DirectionalParticleDataCall>();
    core::moldyn::MultiParticleDataCall* inMPDC = this->dataInSlot.CallAs<core::moldyn::MultiParticleDataCall>();
    if ((inMPDC == nullptr) && (inDPDC == nullptr)) return false;

    if (outDPDC != nullptr) {
        if (inDPDC != nullptr) {
            if (!(*inDPDC)(1)) return false;

            this->bbox = inDPDC->AccessBoundingBoxes().ObjectSpaceBBox();

            outDPDC->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
            outDPDC->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox);

            outDPDC->SetFrameCount(inDPDC->FrameCount());
            outDPDC->SetDataHash(inDPDC->DataHash());

            outDPDC->SetTimeStamp(this->requestedTime);
        } else {
            if (!(*inMPDC)(1)) return false;

            this->bbox = inMPDC->AccessBoundingBoxes().ObjectSpaceBBox();

            outDPDC->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
            outDPDC->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox);

            outDPDC->SetFrameCount(inMPDC->FrameCount());
            outDPDC->SetDataHash(inMPDC->DataHash());

            outDPDC->SetTimeStamp(this->requestedTime);
        }
    } else if (outMPDC) {
        if (inDPDC != nullptr) {
            if (!(*inDPDC)(1)) return false;

            this->bbox = inDPDC->AccessBoundingBoxes().ObjectSpaceBBox();

            outMPDC->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
            outMPDC->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox);

            outMPDC->SetFrameCount(inDPDC->FrameCount());
            outMPDC->SetDataHash(inDPDC->DataHash());
        } else {
            if (!(*inMPDC)(1)) return false;

            this->bbox = inMPDC->AccessBoundingBoxes().ObjectSpaceBBox();

            outMPDC->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
            outMPDC->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox);

            outMPDC->SetFrameCount(inMPDC->FrameCount());
            outMPDC->SetDataHash(inMPDC->DataHash());
        }
    }

    if (inDPDC != nullptr) {
        this->checkForTransition(*inDPDC, c);
    } else {
        this->checkForTransition(*inMPDC, c);
    }

    return true;
}

/*
 * ParticleFrameInterpolator::serialiseParticleList
 */
bool ParticleFrameInterpolator::serialiseParticleList(
    const core::moldyn::SimpleSphericalParticles& part, std::vector<particle_t>& out) {
    auto vertexType = part.GetVertexDataType();
    unsigned int numVertComp = 0;
    unsigned int vertBytes = 0;
    unsigned int vertCompByte = 0;
    switch (vertexType) {
    case core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ:
        numVertComp = 3;
        vertBytes = 3 * sizeof(float);
        vertCompByte = sizeof(float);
        break;
    case core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZR:
        numVertComp = 4;
        vertBytes = 4 * sizeof(float);
        vertCompByte = sizeof(float);
        break;
    case core::moldyn::SimpleSphericalParticles::VERTDATA_SHORT_XYZ:
        numVertComp = 3;
        vertBytes = 3 * sizeof(short);
        vertCompByte = sizeof(short);
        break;
    case core::moldyn::SimpleSphericalParticles::VERTDATA_NONE:
    default:
        // hate
        printf("PFI: HATE!!!\n");
        return false;
    }

    auto colorType = part.GetColourDataType();
    unsigned int numColorComp = 0;
    unsigned int colorBytes = 0;
    unsigned int colorCompByte = 0;
    bool idExists = false;
    switch (colorType) {
    case core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGB:
        numColorComp = 3;
        colorBytes = 3 * sizeof(float);
        colorCompByte = sizeof(float);
        break;
    case core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGBA:
        numColorComp = 4;
        colorBytes = 4 * sizeof(float);
        colorCompByte = sizeof(float);
        break;
    case core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I:
        numColorComp = 1;
        colorBytes = sizeof(float);
        colorCompByte = sizeof(float);
        idExists = true;
        break;
    case core::moldyn::SimpleSphericalParticles::COLDATA_UINT8_RGB:
        numColorComp = 3;
        colorBytes = 3 * sizeof(uint8_t);
        colorCompByte = sizeof(uint8_t);
        break;
    case core::moldyn::SimpleSphericalParticles::COLDATA_UINT8_RGBA:
        numColorComp = 4;
        colorBytes = 4 * sizeof(uint8_t);
        colorCompByte = sizeof(uint8_t);
        break;
    case core::moldyn::SimpleSphericalParticles::COLDATA_NONE:
        break;
    default:
        break;
        // dont care
    }


    auto particleCount = part.GetCount();
    auto vertexStride = part.GetVertexDataStride();
    auto colorStride = part.GetColourDataStride();

    if (vertexStride == 0) {
        vertexStride = vertBytes;
    }
    if (colorStride == 0) {
        colorStride = colorBytes;
    }

    out.clear();
    out.resize(particleCount);
    out.shrink_to_fit();

    const char* vHook = reinterpret_cast<const char*>(part.GetVertexData());
    const char* cHook = reinterpret_cast<const char*>(part.GetColourData());

    for (size_t i = 0; i < particleCount; i++, vHook += vertexStride, cHook += colorStride) {
        for (size_t v = 0; v < numVertComp; v++) {
            if (vertCompByte == sizeof(float)) {
                out[i][v] = *reinterpret_cast<const float*>(vHook + v * sizeof(float));
            } else {
                out[i][v] = *reinterpret_cast<const short*>(vHook + v * sizeof(short));
            }
        }

        for (size_t c = 0; c < numColorComp; c++) {
            if (colorCompByte == sizeof(float)) {
                out[i][c + 4] = *reinterpret_cast<const float*>(cHook + c * sizeof(float));
            } else {
                out[i][c + 4] = *reinterpret_cast<const uint8_t*>(cHook + c * sizeof(uint8_t));
            }
        }
    }

    if (idExists && this->doSortParam.Param<core::param::BoolParam>()->Value()) {
        std::sort(out.begin(), out.end(), [](particle_t& a, particle_t& b) { return a[4] < b[4]; });
        /*std::map<size_t, particle_t> sortMap;
        for (size_t i = 0; i < out.size(); i++) {
            sortMap.insert(std::make_pair(out[i][4], out[i]));
        }
        for (size_t i = 0; i < out.size(); i++) {
            out[i] = sortMap[i];
        }*/
    }

    return true;
}

/*
 * ParticleFrameInterpolator::serialiseParticleList
 */
bool ParticleFrameInterpolator::serialiseParticleList(
    const core::moldyn::DirectionalParticles& part, std::vector<particle_t>& out) {
    auto vertexType = part.GetVertexDataType();
    unsigned int numVertComp = 0;
    unsigned int vertBytes = 0;
    unsigned int vertCompByte = 0;
    switch (vertexType) {
    case core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ:
        numVertComp = 3;
        vertBytes = 3 * sizeof(float);
        vertCompByte = sizeof(float);
        break;
    case core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZR:
        numVertComp = 4;
        vertBytes = 4 * sizeof(float);
        vertCompByte = sizeof(float);
        break;
    case core::moldyn::SimpleSphericalParticles::VERTDATA_SHORT_XYZ:
        numVertComp = 3;
        vertBytes = 3 * sizeof(short);
        vertCompByte = sizeof(short);
        break;
    case core::moldyn::SimpleSphericalParticles::VERTDATA_NONE:
    default:
        // hate
        printf("PFI: HATE!!!\n");
        return false;
    }

    auto colorType = part.GetColourDataType();
    unsigned int numColorComp = 0;
    unsigned int colorBytes = 0;
    unsigned int colorCompByte = 0;
    bool idExists = false;
    switch (colorType) {
    case core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGB:
        numColorComp = 3;
        colorBytes = 3 * sizeof(float);
        colorCompByte = sizeof(float);
        break;
    case core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGBA:
        numColorComp = 4;
        colorBytes = 4 * sizeof(float);
        colorCompByte = sizeof(float);
        break;
    case core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I:
        numColorComp = 1;
        colorBytes = sizeof(float);
        colorCompByte = sizeof(float);
        idExists = true;
        break;
    case core::moldyn::SimpleSphericalParticles::COLDATA_UINT8_RGB:
        numColorComp = 3;
        colorBytes = 3 * sizeof(uint8_t);
        colorCompByte = sizeof(uint8_t);
        break;
    case core::moldyn::SimpleSphericalParticles::COLDATA_UINT8_RGBA:
        numColorComp = 4;
        colorBytes = 4 * sizeof(uint8_t);
        colorCompByte = sizeof(uint8_t);
        break;
    case core::moldyn::SimpleSphericalParticles::COLDATA_NONE:
        break;
    default:
        break;
        // dont care
    }

    auto dirType = part.GetDirDataType();
    unsigned int numDirComp = 0;
    unsigned int dirBytes = 0;
    unsigned int dirCompBytes = 0;
    switch (dirType) {
    case core::moldyn::DirectionalParticles::DIRDATA_FLOAT_XYZ:
        numDirComp = 3;
        dirBytes = 3 * sizeof(float);
        dirCompBytes = sizeof(float);
        break;
    case core::moldyn::DirectionalParticles::DIRDATA_NONE:
        break;
    default:
        break;
    }


    auto particleCount = part.GetCount();
    auto vertexStride = part.GetVertexDataStride();
    auto colorStride = part.GetColourDataStride();
    auto dirStride = part.GetDirDataStride();

    if (vertexStride == 0) {
        vertexStride = vertBytes;
    }
    if (colorStride == 0) {
        colorStride = colorBytes;
    }
    if (dirStride == 0) {
        dirStride = dirBytes;
    }

    out.clear();
    out.resize(particleCount);
    out.shrink_to_fit();

    const char* vHook = reinterpret_cast<const char*>(part.GetVertexData());
    const char* cHook = reinterpret_cast<const char*>(part.GetColourData());
    const char* dHook = reinterpret_cast<const char*>(part.GetDirData());

    for (size_t i = 0; i < particleCount; i++, vHook += vertexStride, cHook += colorStride, dHook += dirStride) {
        for (size_t v = 0; v < numVertComp; v++) {
            if (vertCompByte == sizeof(float)) {
                out[i][v] = *reinterpret_cast<const float*>(vHook + v * sizeof(float));
            } else {
                out[i][v] = *reinterpret_cast<const short*>(vHook + v * sizeof(short));
            }
        }

        for (size_t c = 0; c < numColorComp; c++) {
            if (colorCompByte == sizeof(float)) {
                out[i][c + 4] = *reinterpret_cast<const float*>(cHook + c * sizeof(float));
            } else {
                out[i][c + 4] = *reinterpret_cast<const uint8_t*>(cHook + c * sizeof(uint8_t));
            }
        }

        for (size_t d = 0; d < numDirComp; d++) {
            out[i][d + 8] = *reinterpret_cast<const float*>(dHook + d * sizeof(float));
        }
    }

    if (idExists && this->doSortParam.Param<core::param::BoolParam>()->Value()) {
        std::sort(out.begin(), out.end(), [](particle_t& a, particle_t& b) { return a[4] < b[4]; });
        /*std::map<size_t, particle_t> sortMap;
        for (size_t i = 0; i < out.size(); i++) {
        sortMap.insert(std::make_pair(out[i][4], out[i]));
        }
        for (size_t i = 0; i < out.size(); i++) {
        out[i] = sortMap[i];
        }*/
    }

    return true;
}

/*
 * ParticleFrameInterpolator::serialiseForOutput
 */
bool ParticleFrameInterpolator::serialiseForOutput(void) {
    this->outData.resize(this->outPoints.size());
    this->outData.shrink_to_fit();

    size_t counter = 0;

    for (size_t pl = 0; pl < this->outPoints.size(); pl++) {
        this->outData[pl].resize(this->outPoints[pl].size() * 11);
        this->outData[pl].shrink_to_fit();
        for (size_t p = 0; p < this->outPoints[pl].size(); p++) {
            if (this->bbox.Contains(vislib::math::Point<float, 3>(
                    this->outPoints[pl][p][0], this->outPoints[pl][p][1], this->outPoints[pl][p][2]))) {
                for (unsigned int d = 0; d < CROWBAR_DIMS; d++) {
                    this->outData[pl][p * CROWBAR_DIMS + d] = this->outPoints[pl][p][d];
                }
                // counter++;
            } else {
                for (unsigned int d = 0; d < CROWBAR_DIMS; d++) {
                    this->outData[pl][p * CROWBAR_DIMS + d] = this->startPoints[pl][p][d];
                }
            }
            // this->outData[pl][3] = 1.0f;
        }
        /*this->outData[pl].resize(counter * 11);
        this->outData[pl].shrink_to_fit();*/
    }

    return true;
}

/*
 * ParticleFrameInterpolator::interpolateOnTrajectories
 */
bool ParticleFrameInterpolator::interpolateOnTrajectories(
    float inter, std::vector<std::vector<particle_t>>& nextFrame) {
    this->outPoints.resize(this->startPoints.size());
    this->outPoints.shrink_to_fit();

    /*if (inter > 1.0f || inter < 0.0f) {
        printf("PFI: Inter wrong %d\n", inter);
    }*/

    for (size_t pl = 0; pl < this->startPoints.size(); pl++) {
        this->outPoints[pl].resize(this->startPoints[pl].size());
        this->outPoints[pl].shrink_to_fit();
        if (this->directionalParam.Param<core::param::BoolParam>()->Value()) {
            for (size_t p = 0; p < this->startPoints[pl].size(); p++) {
                this->outPoints[pl][p] = this->startPoints[pl][p] + (inter * this->currentTrajectories[pl][p]);

                vislib::math::Vector<float, 3> a(
                    this->startPoints[pl][p][8], this->startPoints[pl][p][9], this->startPoints[pl][p][10]);
                vislib::math::Vector<float, 3> b(this->startPoints[pl][p][8] + this->currentTrajectories[pl][p][8],
                    this->startPoints[pl][p][9] + this->currentTrajectories[pl][p][9],
                    this->startPoints[pl][p][10] + this->currentTrajectories[pl][p][10]);

                auto dir = this->slerp(inter, a, b);

                this->outPoints[pl][p][8] = dir[0];
                this->outPoints[pl][p][9] = dir[1];
                this->outPoints[pl][p][10] = dir[2];

                // this->outPoints[pl][p][0] = nextFrame[pl][p][0];// +(inter*this->currentTrajectories[pl][p][0]);
                // this->outPoints[pl][p][1] = nextFrame[pl][p][1];// +(inter*this->currentTrajectories[pl][p][1]);
                // this->outPoints[pl][p][2] = nextFrame[pl][p][2];// +(inter*this->currentTrajectories[pl][p][2]);
            }
        } else {
            for (size_t p = 0; p < this->startPoints[pl].size(); p++) {
                this->outPoints[pl][p] = this->startPoints[pl][p] + (inter * this->currentTrajectories[pl][p]);

                /*this->outPoints[pl][p] = this->startPoints[pl][p];

                this->outPoints[pl][p][0] = this->startPoints[pl][p][0] + (inter*this->startPoints[pl][p][8]);
                this->outPoints[pl][p][1] = this->startPoints[pl][p][1] + (inter*this->startPoints[pl][p][9]);
                this->outPoints[pl][p][2] = this->startPoints[pl][p][2] + (inter*this->startPoints[pl][p][10]);*/
            }
        }
    }

    return true;
}

/*
 * ParticleFrameInterpolator::findTrajectories
 */
bool ParticleFrameInterpolator::findTrajectories(std::vector<std::vector<particle_t>>& nextFrame) {
    std::map<float, trajectory_t> candidates;

    this->currentTrajectories.resize(nextFrame.size());
    this->currentTrajectories.shrink_to_fit();

    if (this->lastTrajectories.size() > 0) {
        for (size_t pl = 0; pl < nextFrame.size(); pl++) {
            this->currentTrajectories[pl].resize(nextFrame[pl].size());
            this->currentTrajectories[pl].shrink_to_fit();
            for (size_t p = 0; p < nextFrame[pl].size(); p++) {
                // if (this->startPoints[pl][p][4] == nextFrame[pl][p][4]) {
                //    this->enumerateTrajectories(this->startPoints[pl][p], nextFrame[pl][p],
                //    this->lastTrajectories[pl][p], candidates); this->currentTrajectories[pl][p] =
                //    (*(candidates.begin())).second;
                //} else {
                //    this->currentTrajectories[pl][p] = trajectory_t();
                //    //printf("PFI: No ID match\n");
                //}
                this->enumerateTrajectories(
                    this->startPoints[pl][p], nextFrame[pl][p], this->lastTrajectories[pl][p], candidates);
                this->currentTrajectories[pl][p] = (*(candidates.begin())).second;
            }
        }
    } else {
        for (size_t pl = 0; pl < nextFrame.size(); pl++) {
            this->currentTrajectories[pl].resize(nextFrame[pl].size());
            this->currentTrajectories[pl].shrink_to_fit();
            for (size_t p = 0; p < nextFrame[pl].size(); p++) {
                // if (this->startPoints[pl][p][4] == nextFrame[pl][p][4]) {
                //    this->enumerateTrajectories(this->startPoints[pl][p], nextFrame[pl][p], trajectory_t(),
                //    candidates); this->currentTrajectories[pl][p] = (*(candidates.begin())).second;
                //} else {
                //    this->currentTrajectories[pl][p] = trajectory_t();
                //    //printf("PFI: No ID match\n");
                //}
                this->enumerateTrajectories(this->startPoints[pl][p], nextFrame[pl][p], trajectory_t(), candidates);
                this->currentTrajectories[pl][p] = (*(candidates.begin())).second;
            }
        }
    }

    return true;
}

/*
 * ParticleFrameInterpolator::findTrajectories
 */
bool ParticleFrameInterpolator::findTrajectories(
    std::vector<std::vector<particle_t>>& nextFrame, particleIDMap_t& startMap, particleIDMap_t& nextMap) {
    std::map<float, trajectory_t> candidates;

    this->currentTrajectories.resize(nextFrame.size());
    this->currentTrajectories.shrink_to_fit();

    if (this->lastTrajectories.size() > 0) {
        for (size_t pl = 0; pl < nextFrame.size(); pl++) {
            this->currentTrajectories[pl].resize(this->startPoints[pl].size());
            this->currentTrajectories[pl].shrink_to_fit();
            for (auto& i : startMap[pl]) {
                auto startIDX = i.second;
                auto ni = nextMap[pl].find(i.first);
                if (ni != nextMap[pl].end()) {
                    auto nextIDX = ni->second;

                    auto li = this->lastMap[pl].find(i.first);
                    if (li != this->lastMap[pl].end()) {
                        auto lastIDX = li->second;

                        this->enumerateTrajectories(this->startPoints[pl][startIDX], nextFrame[pl][nextIDX],
                            this->lastTrajectories[pl][lastIDX], candidates);
                    } else {
                        this->enumerateTrajectories(
                            this->startPoints[pl][startIDX], nextFrame[pl][nextIDX], trajectory_t(), candidates);
                    }
                    this->currentTrajectories[pl][startIDX] = (*(candidates.begin())).second;
                } else {
                    this->currentTrajectories[pl][startIDX] = trajectory_t();
                }
            }
        }
    } else {
        for (size_t pl = 0; pl < nextFrame.size(); pl++) {
            this->currentTrajectories[pl].resize(this->startPoints[pl].size());
            this->currentTrajectories[pl].shrink_to_fit();
            for (auto& i : startMap[pl]) {
                auto startIDX = i.second;
                auto ni = nextMap[pl].find(i.first);
                if (ni != nextMap[pl].end()) {
                    auto nextIDX = ni->second;

                    this->enumerateTrajectories(
                        this->startPoints[pl][startIDX], nextFrame[pl][nextIDX], trajectory_t(), candidates);
                    this->currentTrajectories[pl][startIDX] = (*(candidates.begin())).second;
                } else {
                    this->currentTrajectories[pl][startIDX] = trajectory_t();
                }
            }
        }
    }

    return true;
}

/*
 * ParticleFrameInterpolator::enumerateTrajectories
 */
bool ParticleFrameInterpolator::enumerateTrajectories(const particle_t& start, const particle_t& end,
    const trajectory_t& startTraject, std::map<float, trajectory_t>& out) {
    out.clear();

    float startLenght = startTraject.Length();

    trajectory_t offset;
    // offset.SetNull();

    auto tmp = this->calcTrajectory(start, end);
    float disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset.SetNull();
    offset[0] = -bbox.Width();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset.SetNull();
    offset[0] = bbox.Width();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset.SetNull();
    offset[1] = -bbox.Height();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset.SetNull();
    offset[1] = bbox.Height();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset.SetNull();
    offset[2] = -bbox.Depth();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset.SetNull();
    offset[2] = bbox.Depth();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset.SetNull();
    offset[0] = -bbox.Width();
    offset[1] = -bbox.Height();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset.SetNull();
    offset[0] = -bbox.Width();
    offset[1] = bbox.Height();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset.SetNull();
    offset[0] = bbox.Width();
    offset[1] = -bbox.Height();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset.SetNull();
    offset[0] = bbox.Width();
    offset[1] = bbox.Height();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset.SetNull();
    offset[0] = -bbox.Width();
    offset[2] = -bbox.Depth();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset.SetNull();
    offset[0] = -bbox.Width();
    offset[2] = bbox.Depth();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset.SetNull();
    offset[0] = bbox.Width();
    offset[2] = -bbox.Depth();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset.SetNull();
    offset[0] = bbox.Width();
    offset[2] = bbox.Depth();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset.SetNull();
    offset[1] = -bbox.Height();
    offset[2] = -bbox.Depth();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset.SetNull();
    offset[1] = -bbox.Height();
    offset[2] = bbox.Depth();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset.SetNull();
    offset[1] = bbox.Height();
    offset[2] = -bbox.Depth();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset.SetNull();
    offset[1] = bbox.Height();
    offset[2] = bbox.Depth();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset[0] = -bbox.Width();
    offset[1] = -bbox.Height();
    offset[2] = -bbox.Depth();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset[0] = bbox.Width();
    offset[1] = -bbox.Height();
    offset[2] = -bbox.Depth();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset[0] = -bbox.Width();
    offset[1] = bbox.Height();
    offset[2] = -bbox.Depth();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset[0] = -bbox.Width();
    offset[1] = -bbox.Height();
    offset[2] = bbox.Depth();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset[0] = bbox.Width();
    offset[1] = bbox.Height();
    offset[2] = -bbox.Depth();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset[0] = -bbox.Width();
    offset[1] = bbox.Height();
    offset[2] = bbox.Depth();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset[0] = bbox.Width();
    offset[1] = -bbox.Height();
    offset[2] = bbox.Depth();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    offset[0] = bbox.Width();
    offset[1] = bbox.Height();
    offset[2] = bbox.Depth();
    tmp = this->calcTrajectory(start, end - offset);
    disp = std::fabsf(startLenght - tmp.Length());
    out.insert(std::make_pair(disp, tmp));

    return true;
}

/*
 * ParticleFrameInterpolator::calcTrajectory
 */
ParticleFrameInterpolator::trajectory_t ParticleFrameInterpolator::calcTrajectory(
    const particle_t& start, const particle_t& end) {
    return end - start;
}

/*
 * ParticleFrameInterpolator::checkForTransition
 */
bool ParticleFrameInterpolator::checkForTransition(core::Call& inCall, core::Call& outCall) {
    core::moldyn::MultiParticleDataCall* mpdc = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&inCall);
    core::moldyn::DirectionalParticleDataCall* dpdc = dynamic_cast<core::moldyn::DirectionalParticleDataCall*>(&inCall);


    core::moldyn::MultiParticleDataCall* suppMPDC =
        this->supplementalSlot.CallAs<core::moldyn::MultiParticleDataCall>();


    // printf("Requested Time: %f\n", requestedTime);
    // float requestedTime = -2;

    std::vector<std::vector<particle_t>> nextFrame;
    std::vector<std::vector<particle_t>> nextSupp;

    if (dpdc == nullptr) {

        if (mpdc == nullptr) return false;

        core::moldyn::MultiParticleDataCall& inMPDC = *mpdc;

        mpdc = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&outCall);
        dpdc = dynamic_cast<core::moldyn::DirectionalParticleDataCall*>(&outCall);
        if (dpdc != nullptr) {
            // requestedTime = dpdc->GetTimeStamp();
        } else if (mpdc != nullptr) {
            requestedTime = mpdc->GetTimeStamp();
        } else
            return false;

        printf("PFI: Requested Time: %f\n", requestedTime);

        if (requestedTime == this->currentTimeStamp) return true;

        if (requestedTime - static_cast<float>(this->startFrameID) < 1.0f && this->initialized) {
            // only interpolate
            this->interpolateOnTrajectories(requestedTime - static_cast<float>(this->startFrameID), nextFrame);
            this->serialiseForOutput();
            this->currentTimeStamp = requestedTime;
            printf("PFI: Interpolating\n");
            return true;
        }

        // do transition
        this->lastMap = this->startMap;
        this->lastTrajectories = this->currentTrajectories;
        // fetch next frames
        auto tmp = this->startFrameID;
        this->startFrameID = std::floorf(requestedTime);
        if (this->startFrameID - tmp > 1) this->lastTrajectories.clear();
        inMPDC.SetFrameID(this->startFrameID, true);
        do {
            if (!inMPDC(1)) return false;
            if (!inMPDC(0)) return false;
        } while (inMPDC.FrameID() != this->startFrameID);
        this->startPoints.resize(inMPDC.GetParticleListCount());
        for (unsigned int pl = 0; pl < inMPDC.GetParticleListCount(); pl++) {
            this->serialiseParticleList(inMPDC.AccessParticles(pl), this->startPoints[pl]);
        }
        if (suppMPDC != nullptr) {
            suppMPDC->SetFrameID(this->startFrameID, true);
            do {
                if (!(*suppMPDC)(1)) return false;
                if (!(*suppMPDC)(0)) return false;
            } while (suppMPDC->FrameID() != this->startFrameID);
            this->startSupp.resize(suppMPDC->GetParticleListCount());
            for (unsigned int pl = 0; pl < suppMPDC->GetParticleListCount(); pl++) {
                this->serialiseParticleList(suppMPDC->AccessParticles(pl), this->startSupp[pl]);
            }

            this->mergeInputs(this->startPoints, this->startSupp, this->startMap);
        }

        inMPDC.SetFrameID(this->startFrameID + 1, true);
        do {
            if (!inMPDC(1)) return false;
            if (!inMPDC(0)) return false;
        } while (inMPDC.FrameID() != (this->startFrameID + 1));
        nextFrame.resize(inMPDC.GetParticleListCount());
        nextFrame.shrink_to_fit();
        for (unsigned int pl = 0; pl < inMPDC.GetParticleListCount(); pl++) {
            this->serialiseParticleList(inMPDC.AccessParticles(pl), nextFrame[pl]);
        }
        if (suppMPDC != nullptr) {
            suppMPDC->SetFrameID(this->startFrameID + 1, true);
            do {
                if (!(*suppMPDC)(1)) return false;
                if (!(*suppMPDC)(0)) return false;
            } while (suppMPDC->FrameID() != this->startFrameID + 1);
            nextSupp.resize(suppMPDC->GetParticleListCount());
            for (unsigned int pl = 0; pl < suppMPDC->GetParticleListCount(); pl++) {
                this->serialiseParticleList(suppMPDC->AccessParticles(pl), nextSupp[pl]);
            }

            this->mergeInputs(nextFrame, nextSupp, this->nextMap);
        }

        /*printf("PFI: PL0 %d %d %d\n", (int)(this->startPoints[0].size()) - (int)(nextFrame[0].size()),
        this->startPoints[0].size(), nextFrame[0].size()); printf("PFI: PL1 %d %d %d\n",
        (int)(this->startPoints[1].size()) - (int)(nextFrame[1].size()), this->startPoints[1].size(),
        nextFrame[1].size()); printf("PFI: PL2 %d %d %d\n", (int)(this->startPoints[2].size()) -
        (int)(nextFrame[2].size()), this->startPoints[2].size(), nextFrame[2].size()); printf("PFI: PL3 %d %d %d\n",
        (int)(this->startPoints[3].size()) - (int)(nextFrame[3].size()), this->startPoints[3].size(),
        nextFrame[3].size());*/
    } else {
        // serialise directional data
        core::moldyn::DirectionalParticleDataCall& inDPDC = *dpdc;

        mpdc = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&outCall);
        dpdc = dynamic_cast<core::moldyn::DirectionalParticleDataCall*>(&outCall);
        if (dpdc != nullptr) {
            // requestedTime = dpdc->GetTimeStamp();
        } else if (mpdc != nullptr) {
            requestedTime = mpdc->GetTimeStamp();
        } else
            return false;

        printf("PFI: Requested Time: %f\n", requestedTime);

        if (requestedTime == this->currentTimeStamp) return true;

        if (requestedTime - static_cast<float>(this->startFrameID) < 1.0f && this->initialized) {
            // only interpolate
            this->interpolateOnTrajectories(requestedTime - static_cast<float>(this->startFrameID), nextFrame);
            this->serialiseForOutput();
            this->currentTimeStamp = requestedTime;
            printf("PFI: Interpolating\n");
            return true;
        }

        // do transition
        this->lastMap = this->startMap;
        this->lastTrajectories = this->currentTrajectories;
        // fetch next frames
        auto tmp = this->startFrameID;
        this->startFrameID = std::floorf(requestedTime);
        if (this->startFrameID - tmp > 1) this->lastTrajectories.clear();
        inDPDC.SetFrameID(this->startFrameID, true);
        do {
            if (!inDPDC(1)) return false;
            if (!inDPDC(0)) return false;
        } while (inDPDC.FrameID() != this->startFrameID);
        this->startPoints.resize(inDPDC.GetParticleListCount());
        for (unsigned int pl = 0; pl < inDPDC.GetParticleListCount(); pl++) {
            this->serialiseParticleList(inDPDC.AccessParticles(pl), this->startPoints[pl]);
        }
        if (suppMPDC != nullptr) {
            suppMPDC->SetFrameID(this->startFrameID, true);
            do {
                if (!(*suppMPDC)(1)) return false;
                if (!(*suppMPDC)(0)) return false;
            } while (suppMPDC->FrameID() != this->startFrameID);
            this->startSupp.resize(suppMPDC->GetParticleListCount());
            for (unsigned int pl = 0; pl < suppMPDC->GetParticleListCount(); pl++) {
                this->serialiseParticleList(suppMPDC->AccessParticles(pl), this->startSupp[pl]);
            }

            this->mergeInputs(this->startPoints, this->startSupp, this->startMap);
        }

        inDPDC.SetFrameID(this->startFrameID + 1, true);
        do {
            if (!inDPDC(1)) return false;
            if (!inDPDC(0)) return false;
        } while (inDPDC.FrameID() != (this->startFrameID + 1));
        nextFrame.resize(inDPDC.GetParticleListCount());
        nextFrame.shrink_to_fit();
        for (unsigned int pl = 0; pl < inDPDC.GetParticleListCount(); pl++) {
            this->serialiseParticleList(inDPDC.AccessParticles(pl), nextFrame[pl]);
        }
        if (suppMPDC != nullptr) {
            suppMPDC->SetFrameID(this->startFrameID + 1, true);
            do {
                if (!(*suppMPDC)(1)) return false;
                if (!(*suppMPDC)(0)) return false;
            } while (suppMPDC->FrameID() != this->startFrameID + 1);
            nextSupp.resize(suppMPDC->GetParticleListCount());
            for (unsigned int pl = 0; pl < suppMPDC->GetParticleListCount(); pl++) {
                this->serialiseParticleList(suppMPDC->AccessParticles(pl), nextSupp[pl]);
            }

            this->mergeInputs(nextFrame, nextSupp, this->nextMap);
        }

        /*printf("PFI: PL0 %d %d %d\n", (int)(this->startPoints[0].size()) - (int)(nextFrame[0].size()),
        this->startPoints[0].size(), nextFrame[0].size()); printf("PFI: PL1 %d %d %d\n",
        (int)(this->startPoints[1].size()) - (int)(nextFrame[1].size()), this->startPoints[1].size(),
        nextFrame[1].size()); printf("PFI: PL2 %d %d %d\n", (int)(this->startPoints[2].size()) -
        (int)(nextFrame[2].size()), this->startPoints[2].size(), nextFrame[2].size()); printf("PFI: PL3 %d %d %d\n",
        (int)(this->startPoints[3].size()) - (int)(nextFrame[3].size()), this->startPoints[3].size(),
        nextFrame[3].size());*/
    }

    // calculate trajectories
    if (suppMPDC == nullptr) {
        this->findTrajectories(nextFrame);
    } else {
        this->findTrajectories(nextFrame, this->startMap, this->nextMap);
    }

    // interpolate
    this->interpolateOnTrajectories(requestedTime - static_cast<float>(this->startFrameID), nextFrame);
    this->serialiseForOutput();

    this->currentTimeStamp = requestedTime;
    this->initialized = true;

    printf("PFI: Rebuilding\n");

    return true;
}

/*
 * ParticleFrameInterpolator::slerp
 */
vislib::math::Vector<float, 3> ParticleFrameInterpolator::slerp(
    const float inter, const vislib::math::Vector<float, 3>& a, const vislib::math::Vector<float, 3>& b) {
    float omega = std::acosf(a.Dot(b));

    vislib::math::Vector<float, 3> ret =
        std::sinf((1.0f - inter) * omega) / std::sinf(omega) * a + std::sinf(inter * omega) / std::sinf(omega) * b;

    return ret;
}

/*
 * ParticleFrameInterpolator::mergeInputs
 */
void ParticleFrameInterpolator::mergeInputs(
    std::vector<std::vector<particle_t>>& a, std::vector<std::vector<particle_t>>& b, particleIDMap_t& m) {
    m.clear();

    // build map
    m.resize(a.size());
    for (unsigned int pl = 0; pl < a.size(); pl++) {
        for (size_t p = 0; p < a[pl].size(); p++) {
            m[pl].insert(std::make_pair(a[pl][p][4], p));
            a[pl][p][4] = b[pl][p][4];
            a[pl][p][5] = b[pl][p][5];
            a[pl][p][6] = b[pl][p][6];
            a[pl][p][7] = b[pl][p][7];
        }
    }
}
