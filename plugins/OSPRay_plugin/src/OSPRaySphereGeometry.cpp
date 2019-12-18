/*
* OSPRaySphereGeometry.cpp
* Copyright (C) 2009-2017 by MegaMol Team
* Alle Rechte vorbehalten.
*/

#include "stdafx.h"
#include "OSPRaySphereGeometry.h"
#include "vislib/forceinline.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/EnumParam.h"
#include "vislib/sys/Log.h"
#include "mmcore/Call.h"

#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallClipPlane.h"

using namespace megamol::ospray;


VISLIB_FORCEINLINE float floatFromVoidArray(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index) {
    //const float* parts = static_cast<const float*>(p.GetVertexData());
    //return parts[index * stride + offset];
    return static_cast<const float*>(p.GetVertexData())[index];
}

VISLIB_FORCEINLINE unsigned char byteFromVoidArray(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index) {
    return static_cast<const unsigned char*>(p.GetVertexData())[index];
}

typedef float(*floatFromArrayFunc)(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index);
typedef unsigned char(*byteFromArrayFunc)(const megamol::core::moldyn::MultiParticleDataCall::Particles& p, size_t index);


OSPRaySphereGeometry::OSPRaySphereGeometry(void) :
    AbstractOSPRayStructure(),
    getDataSlot("getdata", "Connects to the data source"),
    getTFSlot("gettransferfunction", "Connects to the transfer function module"),
    getClipPlaneSlot("getclipplane", "Connects to a clipping plane module"),

    particleList("ParticleList", "Switches between particle lists")
{

    this->getDataSlot.SetCompatibleCall<core::moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getTFSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

    this->getClipPlaneSlot.SetCompatibleCall<core::view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->getClipPlaneSlot);

    this->particleList << new core::param::IntParam(0);
    this->MakeSlotAvailable(&this->particleList);
}


bool OSPRaySphereGeometry::readData(megamol::core::Call &call) {

    // fill material container
    this->processMaterial();

    // fill transformation container
    this->processTransformation();

    // read Data, calculate  shape parameters, fill data vectors
    CallOSPRayStructure *os = dynamic_cast<CallOSPRayStructure*>(&call);
    megamol::core::moldyn::MultiParticleDataCall *cd = this->getDataSlot.CallAs<megamol::core::moldyn::MultiParticleDataCall>();

    this->structureContainer.dataChanged = false;
    if (cd == NULL) return false;
    cd->SetTimeStamp(os->getTime());
    cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
    if (this->datahash != cd->DataHash() || this->time != os->getTime() || this->InterfaceIsDirty()) {
        this->datahash = cd->DataHash();
        this->time = os->getTime();
        this->structureContainer.dataChanged = true;
    } else {
        return true;
    }

    if (!(*cd)(1)) return false;
    if (!(*cd)(0)) return false;
    if (cd->GetParticleListCount() == 0) return false;

    if (this->particleList.Param<core::param::IntParam>()->Value() > (cd->GetParticleListCount() - 1)) {
        this->particleList.Param<core::param::IntParam>()->SetValue(0);
    }

    core::moldyn::MultiParticleDataCall::Particles &parts = cd->AccessParticles(this->particleList.Param<core::param::IntParam>()->Value());


    unsigned int partCount = parts.GetCount();
    float globalRadius = parts.GetGlobalRadius();

    size_t vertexLength;
    size_t colorLength;

    // Vertex data type check
    if (parts.GetVertexDataType() == core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ) {
        vertexLength = 3;
    } else if (parts.GetVertexDataType() == core::moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR) {
        vertexLength = 4;
    } else {
        vislib::sys::Log::DefaultLog.WriteError("[OSPRaySphereGeometry] Vertex type not supported.");
        return false;
    }
    // reserve space for vertex data object
    vd.reserve(parts.GetCount() * vertexLength);

    // Color data type check
    if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA) {
        colorLength = 4;
        //convertedColorType = OSP_FLOAT4;
        cd_rgba.reserve(parts.GetCount() * colorLength);

        floatFromArrayFunc ffaf;
        ffaf = floatFromVoidArray;

        for (size_t loop = 0; loop < (parts.GetCount() * parts.GetVertexDataStride() / sizeof(float)); loop++) {
            if (loop % (vertexLength + colorLength) >= vertexLength) {
                cd_rgba.push_back(ffaf(parts, loop));
            } else {
                vd.push_back(ffaf(parts, loop));
            }
        }
    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
        // this colorType will be transformed to:
        //convertedColorType = OSP_FLOAT4;
        colorLength = 4;
        cd_rgba.reserve(parts.GetCount() * colorLength);

        floatFromArrayFunc ffaf;
        ffaf = floatFromVoidArray;
        std::vector<float> cd;



        if (parts.GetVertexDataStride() > 3 * sizeof(float) && parts.GetVertexDataStride() == parts.GetColourDataStride()) {
            for (size_t loop = 0; loop < (parts.GetCount() * parts.GetVertexDataStride() / sizeof(float)); loop++) {
                if (loop % (vertexLength + 1) >= vertexLength) {
                    cd.push_back(ffaf(parts, loop));
                } else {
                    vd.push_back(ffaf(parts, loop));
                }
            }
        } else {
            auto vptr = (float*)parts.GetVertexData();
            auto cptr = (float*)parts.GetColourData();
            vd.assign(vptr, vptr + parts.GetCount() * 3);
            cd.assign(cptr, cptr + parts.GetCount());
        }

        // Color transfer call and calculation
        core::view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<core::view::CallGetTransferFunction>();
        if (cgtf != NULL && ((*cgtf)())) {
            float const* tf_tex = cgtf->GetTextureData();
            tex_size = cgtf->TextureSize();
            this->colorTransferGray(cd, tf_tex, tex_size, cd_rgba);
        } else {
            this->colorTransferGray(cd, NULL, 0, cd_rgba);
        }


    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB) {
        colorLength = 3;
        //convertedColorType = OSP_FLOAT3;
        cd_rgba.reserve(parts.GetCount() * (colorLength + 1));


        floatFromArrayFunc ffaf;
        ffaf = floatFromVoidArray;
        unsigned int iter = 0;
        for (size_t loop = 0; loop < (parts.GetCount() * parts.GetVertexDataStride() / sizeof(float)); loop++) {
            if (loop % (vertexLength + colorLength) >= vertexLength) {
                cd_rgba.push_back(std::abs(ffaf(parts, loop)));
                iter++;
                if (iter == 3) {
                    cd_rgba.push_back(1.0f);
                    iter = 0;
                }
            } else {
                vd.push_back(ffaf(parts, loop));
            }
        }
        colorLength = 4;

    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA) {
        colorLength = 4;
        //convertedColorType = OSP_FLOAT4;
        cd_rgba.reserve(parts.GetCount() * colorLength);

        float alpha = 1.0f;
        byteFromArrayFunc bfaf;
        bfaf = byteFromVoidArray;

        vd.resize(parts.GetCount() * vertexLength);
        auto data = static_cast<const float*>(parts.GetVertexData());

        auto posSize = parts.VertexDataSize[parts.GetVertexDataType()];
        auto vertSize = parts.GetVertexDataStride();
        //if (vertSize == 0)

        auto const& cRAcc = parts.GetParticleStore().GetCRAcc();
        auto const& cGAcc = parts.GetParticleStore().GetCGAcc();
        auto const& cBAcc = parts.GetParticleStore().GetCBAcc();
        auto const& cAAcc = parts.GetParticleStore().GetCAAcc();

        for (size_t i = 0; i < parts.GetCount(); i++) {
            std::copy_n(reinterpret_cast<float const*>(reinterpret_cast<char const*>(data) + (i * vertSize)),
                vertexLength,
                vd.begin() + (i * vertexLength));
            cd_rgba.push_back(cRAcc->Get_f(i) / 255.0);
            cd_rgba.push_back(cGAcc->Get_f(i) / 255.0);
            cd_rgba.push_back(cBAcc->Get_f(i) / 255.0);
            cd_rgba.push_back(cAAcc->Get_f(i) / 255.0);
            /*for (size_t j = 0; j < colorLength; j++) {
                cd_rgba.push_back((float)bfaf(parts, i * parts.GetVertexDataStride() + posSize + j) / 255.0f);
            }*/
        }
    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB) {
        //vislib::sys::Log::DefaultLog.WriteError("File format deprecated. Convert your data.");
        colorLength = 4;
        //convertedColorType = OSP_FLOAT4;
        cd_rgba.reserve(parts.GetCount() * colorLength);

        float alpha = 1.0f;
        byteFromArrayFunc bfaf;
        bfaf = byteFromVoidArray;

        vd.resize(parts.GetCount() * vertexLength);
        auto data = static_cast<const float*>(parts.GetVertexData());

        auto posSize = parts.VertexDataSize[parts.GetVertexDataType()];
        auto vertSize = parts.GetVertexDataStride();
        //if (vertSize == 0)

        auto const& cRAcc = parts.GetParticleStore().GetCRAcc();
        auto const& cGAcc = parts.GetParticleStore().GetCGAcc();
        auto const& cBAcc = parts.GetParticleStore().GetCBAcc();
        //auto const& cAAcc = parts.GetParticleStore().GetCAAcc();

        for (size_t i = 0; i < parts.GetCount(); i++) {
            std::copy_n(reinterpret_cast<float const*>(reinterpret_cast<char const*>(data) + (i * vertSize)),
                vertexLength,
                vd.begin() + (i * vertexLength));
            cd_rgba.push_back(cRAcc->Get_f(i) / 255.0);
            cd_rgba.push_back(cGAcc->Get_f(i) / 255.0);
            cd_rgba.push_back(cBAcc->Get_f(i) / 255.0);
            cd_rgba.push_back(1.0f);
            /*for (size_t j = 0; j < colorLength; j++) {
                cd_rgba.push_back((float)bfaf(parts, i * parts.GetVertexDataStride() + posSize + j) / 255.0f);
            }*/
        }
    } else if (parts.GetColourDataType() == core::moldyn::MultiParticleDataCall::Particles::COLDATA_NONE) {
        colorLength = 4;
        cd_rgba.reserve(parts.GetCount() * colorLength);
        auto pdata = (float*)parts.GetVertexData();
        vd.assign(pdata, pdata + parts.GetCount() * vertexLength);
        auto globalColor = parts.GetGlobalColour();
        for (size_t i = 0; i < parts.GetCount(); i++) {
            for (size_t j = 0; j < colorLength; j++) {
                cd_rgba.push_back((float)globalColor[j] / 255.0f);
            }
        }
    }


    // clipPlane setup
    std::vector<float> clipDat(4);
    std::vector<float> clipCol(4);
    this->getClipData(clipDat.data(), clipCol.data());


    // Write stuff into the structureContainer
    this->structureContainer.type = structureTypeEnum::GEOMETRY;
    this->structureContainer.geometryType = geometryTypeEnum::SPHERES;
    this->structureContainer.vertexData = std::make_shared<std::vector<float>>(std::move(vd));
    this->structureContainer.colorData = std::make_shared<std::vector<float>>(std::move(cd_rgba));
    this->structureContainer.vertexLength = vertexLength;
    this->structureContainer.colorLength = colorLength;
    this->structureContainer.partCount = partCount;
    this->structureContainer.globalRadius = globalRadius;
    this->structureContainer.clipPlaneData = std::make_shared<std::vector<float>>(std::move(clipDat));
    this->structureContainer.clipPlaneColor = std::make_shared<std::vector<float>>(std::move(clipCol));

    return true;
}


void OSPRaySphereGeometry::colorTransferGray(std::vector<float> &grayArray, float const* transferTable, unsigned int tableSize, std::vector<float> &rgbaArray) {

    float gray_max = *std::max_element(grayArray.begin(), grayArray.end());
    float gray_min = *std::min_element(grayArray.begin(), grayArray.end());

    for (auto &gray : grayArray) {
        float scaled_gray;
        if ((gray_max - gray_min) <= 1e-4f) {
            scaled_gray = 0;
        } else {
            scaled_gray = (gray - gray_min) / (gray_max - gray_min);
        }
        if (transferTable == NULL && tableSize == 0) {
            for (int i = 0; i < 3; i++) {
                rgbaArray.push_back((0.3f + scaled_gray) / 1.3f);
            }
            rgbaArray.push_back(1.0f);
        } else {
            float exact_tf = (tableSize - 1) * scaled_gray;
            int floor = std::floor(exact_tf);
            float tail = exact_tf - (float)floor;
            floor *= 4;
            for (int i = 0; i < 4; i++) {
                float colorFloor = transferTable[floor + i];
                float colorCeil = transferTable[floor + i + 4];
                float finalColor = colorFloor + (colorCeil - colorFloor)*(tail);
                rgbaArray.push_back(finalColor);
            }
        }
    }
}


OSPRaySphereGeometry::~OSPRaySphereGeometry() {
    this->Release();
}

bool OSPRaySphereGeometry::create() {
    return true;
}

void OSPRaySphereGeometry::release() {

}

/*
ospray::OSPRaySphereGeometry::InterfaceIsDirty()
*/
bool OSPRaySphereGeometry::InterfaceIsDirty() {
    if (
        this->particleList.IsDirty()
        ) {
        this->particleList.ResetDirty();
        return true;
    } else {
        return false;
    }
}

/*
* ospray::OSPRaySphereGeometry::getClipData
*/
void OSPRaySphereGeometry::getClipData(float *clipDat, float *clipCol) {
    megamol::core::view::CallClipPlane *ccp = this->getClipPlaneSlot.CallAs<megamol::core::view::CallClipPlane>();
    if ((ccp != NULL) && (*ccp)()) {
        clipDat[0] = ccp->GetPlane().Normal().X();
        clipDat[1] = ccp->GetPlane().Normal().Y();
        clipDat[2] = ccp->GetPlane().Normal().Z();
        vislib::math::Vector<float, 3> grr(ccp->GetPlane().Point().PeekCoordinates());
        clipDat[3] = grr.Dot(ccp->GetPlane().Normal());
        clipCol[0] = static_cast<float>(ccp->GetColour()[0]) / 255.0f;
        clipCol[1] = static_cast<float>(ccp->GetColour()[1]) / 255.0f;
        clipCol[2] = static_cast<float>(ccp->GetColour()[2]) / 255.0f;
        clipCol[3] = static_cast<float>(ccp->GetColour()[3]) / 255.0f;

    } else {
        clipDat[0] = clipDat[1] = clipDat[2] = clipDat[3] = 0.0f;
        clipCol[0] = clipCol[1] = clipCol[2] = 0.75f;
        clipCol[3] = 1.0f;
    }
}

bool OSPRaySphereGeometry::getExtends(megamol::core::Call &call) {
    CallOSPRayStructure *os = dynamic_cast<CallOSPRayStructure*>(&call);
    megamol::core::moldyn::MultiParticleDataCall *cd = this->getDataSlot.CallAs<megamol::core::moldyn::MultiParticleDataCall>();
    
    if (cd == NULL) return false;
    cd->SetFrameID(os->getTime(), true); // isTimeForced flag set to true
    // if (!(*cd)(1)) return false; // table returns flase at first attempt and breaks everything
    (*cd)(1);
    this->extendContainer.boundingBox = std::make_shared<megamol::core::BoundingBoxes_2>();
    this->extendContainer.boundingBox->SetBoundingBox(cd->AccessBoundingBoxes().ObjectSpaceBBox());
    this->extendContainer.timeFramesCount = cd->FrameCount();
    this->extendContainer.isValid = true;

    return true;
}