/*
 * ParticleTranslateRotateScaleScale.h
 *
 * Copyright (C) 2018 MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "ParticleTranslateRotateScale.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"
#include "vislib/math/Matrix4.h"
#include "mmcore/view/CallGetTransferFunction.h"

using namespace megamol;
using namespace megamol::stdplugin;


/*
 * datatools::ParticleTranslateRotateScale::ParticleTranslateRotateScale
 */
datatools::ParticleTranslateRotateScale::ParticleTranslateRotateScale(void)
    : AbstractParticleManipulator("outData", "indata")
    , translateSlot("translation", "Translates the particles in x, y, z direction")
    , quaternionSlot("quaternion", "Rotates the particles around x, y, z axes")
    , scaleSlot("scale", "Scales the particle data")
    , getTFSlot("gettransferfunction", "Connects to the transfer function module")
, finalData(NULL){
    this->translateSlot.SetParameter(new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0, 0, 0)));
    this->MakeSlotAvailable(&this->translateSlot);

    this->quaternionSlot.SetParameter(new core::param::Vector4fParam(vislib::math::Vector<float, 4>(0, 0, 0, 1)));
    this->MakeSlotAvailable(&this->quaternionSlot);

    this->scaleSlot.SetParameter(new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1, 1, 1)));
    this->MakeSlotAvailable(&this->scaleSlot);

    this->getTFSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);
}


/*
 * datatools::ParticleTranslateRotateScale::~ParticleTranslateRotateScale
 */
datatools::ParticleTranslateRotateScale::~ParticleTranslateRotateScale(void) {
    this->Release();
    delete[] finalData;
}


/*
 * datatools::ParticleTranslateRotateScale::InterfaceIsDirty
 */
bool datatools::ParticleTranslateRotateScale::InterfaceIsDirty() const {
    return (translateSlot.IsDirty() || quaternionSlot.IsDirty() || scaleSlot.IsDirty());
}


/*
 * datatools::ParticleTranslateRotateScale::InterfaceResetDirty
 */
void datatools::ParticleTranslateRotateScale::InterfaceResetDirty() {
    translateSlot.ResetDirty();
    quaternionSlot.ResetDirty();
    scaleSlot.ResetDirty();
}


/*
 * datatools::ParticleTranslateRotateScale::manipulateData
 */
bool datatools::ParticleTranslateRotateScale::manipulateData(
    megamol::core::moldyn::MultiParticleDataCall& outData, megamol::core::moldyn::MultiParticleDataCall& inData) {
    using megamol::core::moldyn::MultiParticleDataCall;


    outData = inData; // also transfers the unlocker to 'outData'

    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData

    const float transX = translateSlot.Param<core::param::Vector3fParam>()->Value().GetX();
    const float transY = translateSlot.Param<core::param::Vector3fParam>()->Value().GetY();
    const float transZ = translateSlot.Param<core::param::Vector3fParam>()->Value().GetZ();

    const float scaleX = scaleSlot.Param<core::param::Vector3fParam>()->Value().GetX();
    const float scaleY = scaleSlot.Param<core::param::Vector3fParam>()->Value().GetY();
    const float scaleZ = scaleSlot.Param<core::param::Vector3fParam>()->Value().GetZ();

    // Calculate matrices
    vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> transMX(
        1, 0, 0, transX, 0, 1, 0, transY, 0, 0, 1, transZ, 0, 0, 0, 1);

    vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> scaleMX(
        scaleX, 0, 0, 0, 0, scaleY, 0, 0, 0, 0, scaleZ, 0, 0, 0, 0, 1);

    /*
    const float rotX = 2 * M_PI * rotateSlot.Param<core::param::Vector3fParam>()->Value().GetX() / 360.0f;
    const float rotY = 2 * M_PI * rotateSlot.Param<core::param::Vector3fParam>()->Value().GetY() / 360.0f;
    const float rotZ = 2 * M_PI * rotateSlot.Param<core::param::Vector3fParam>()->Value().GetZ() / 360.0f;


    vislib::math::Quaternion<float> qx(rotX, vislib::math::Vector<float, 3>(1.0f, 0.0f, 0.0f));
    vislib::math::Quaternion<float> qy(rotY, vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f));
    vislib::math::Quaternion<float> qz(rotZ, vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f));

    vislib::math::Matrix<float, 4, vislib::math::MatrixLayout::COLUMN_MAJOR> totRotMX;
    totRotMX = qx * qy * qz;
    */

    vislib::math::Quaternion<float> q(quaternionSlot.Param<core::param::Vector4fParam>()->Value().GetX(),
        quaternionSlot.Param<core::param::Vector4fParam>()->Value().GetY(),
        quaternionSlot.Param<core::param::Vector4fParam>()->Value().GetZ(),
        quaternionSlot.Param<core::param::Vector4fParam>()->Value().GetW());

    auto qr = q.GetK();
    auto qi = q.GetI();
    auto qj = q.GetJ();
    auto qk = q.GetR();
    auto s = 1.0f / (qr*qr + qi*qi + qj*qj +  qk*qk);

    vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> rotMX(
        1-2*s*(qj*qj + qk*qk), 2*s*(qi*qj - qk*qr), 2*s*(qi*qk - qj*qr), 0,
        2*s*(qi*qj + qk*qr), 1-2*s*(qi*qi + qk*qk), 2*s*(qj*qk - qi*qr), 0,
        2*s*(qi*qk - qj*qr), 2*s*(qj*qk + qi*qr), 1-2*s*(qi*qi + qj*qj), 0,
        0, 0, 0, 1);

    auto bboxCenterX = inData.GetBoundingBoxes().ObjectSpaceBBox().CalcCenter().GetX();
    auto bboxCenterY = inData.GetBoundingBoxes().ObjectSpaceBBox().CalcCenter().GetY();
    auto bboxCenterZ = inData.GetBoundingBoxes().ObjectSpaceBBox().CalcCenter().GetZ();

    vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> origTransMX(
        1, 0, 0, -bboxCenterX, 0, 1, 0, -bboxCenterY, 0, 0, 1, -bboxCenterZ, 0, 0, 0, 1);

    vislib::math::Matrix<float, 4, vislib::math::COLUMN_MAJOR> invOrigTransMX(
        1, 0, 0, bboxCenterX, 0, 1, 0, bboxCenterY, 0, 0, 1, bboxCenterZ, 0, 0, 0, 1);

    auto const totMX = transMX * invOrigTransMX * scaleMX * rotMX * origTransMX;
    
    //auto lbb = static_cast<vislib::math::Vector<float, 3>>(inData.GetBoundingBoxes().ObjectSpaceBBox().GetLeftBottomBack());
    //auto rtf = static_cast<vislib::math::Vector<float, 3>>(inData.GetBoundingBoxes().ObjectSpaceBBox().GetRightTopFront());
    //lbb = totMX * lbb;rotMX
    //rtf = totMX * rtf;
    //vislib::math::Cuboid<float> newBox;
    //newBox.Set(lbb.GetX(), lbb.GetY(), lbb.GetZ(), rtf.GetX(), rtf.GetY(), rtf.GetZ());

    //auto scale = 10.0f / newBox.LongestEdge();

    //outData.AccessBoundingBoxes().SetObjectSpaceBBox(newBox);
    //outData.AccessBoundingBoxes().MakeScaledWorld(scale);

    unsigned int plc = outData.GetParticleListCount();
    vislib::math::Vector<float, 4> pos;


    finalData = new float*[plc];
    for (unsigned int i = 0; i < plc; i++) {
        MultiParticleDataCall::Particles& p = outData.AccessParticles(i);

        uint64_t cnt = p.GetCount();

        const void* cd = p.GetColourData();
        unsigned int cds = p.GetColourDataStride();
        MultiParticleDataCall::Particles::ColourDataType cdt = p.GetColourDataType();

        const void* vd = p.GetVertexData();
        unsigned int vds = p.GetVertexDataStride();
        MultiParticleDataCall::Particles::VertexDataType vdt = p.GetVertexDataType();


        // Color transfer call and calculation
        std::vector<float> rgba;
        rgba.reserve(cnt * 4);
        unsigned int texSize = 0;
        megamol::core::view::CallGetTransferFunction* cgtf =
            this->getTFSlot.CallAs<core::view::CallGetTransferFunction>();
        if (cgtf != NULL && ((*cgtf)())) {
            float const* tfTex = cgtf->GetTextureData();
            texSize = cgtf->TextureSize();
            this->colorTransferGray(p, tfTex, texSize, rgba);
        } else {
            this->colorTransferGray(p, NULL, 0, rgba);
        }

        auto const& parStore = p.GetParticleStore();
        auto const& xAcc = parStore.GetXAcc();
        auto const& yAcc = parStore.GetYAcc();
        auto const& zAcc = parStore.GetZAcc();

        finalData[i] = new float[cnt * 7];
        for (size_t loop = 0; loop < cnt; loop++) {

            pos.SetX(xAcc->Get_f(loop));
            pos.SetY(yAcc->Get_f(loop));
            pos.SetZ(zAcc->Get_f(loop));
            pos.SetW(1.0f);

            pos = totMX * pos;

            finalData[i][7*loop+0] = pos.GetX();
            finalData[i][7*loop+1] = pos.GetY();
            finalData[i][7*loop+2] = pos.GetZ();
            finalData[i][7*loop+3] = rgba[4*loop+0];
            finalData[i][7*loop+4] = rgba[4*loop+1];
            finalData[i][7*loop+5] = rgba[4*loop+2];
            finalData[i][7*loop+6] = rgba[4*loop+3];


            // Does not work ??
            // finalData.emplace_back(pos.GetX(), pos.GetY(), pos.GetZ(), part.col.GetRf(), part.col.GetGf(),
            //    part.col.GetBf(), part.col.GetAf());
        }

        auto lbbLocal = static_cast<vislib::math::Vector<float, 3>>(p.GetBBox().GetLeftBottomBack());
        auto rtfLocal = static_cast<vislib::math::Vector<float, 3>>(p.GetBBox().GetRightTopFront());
        lbbLocal = totMX * lbbLocal;
        rtfLocal = totMX * rtfLocal;
        vislib::math::Cuboid<float> newBoxLocal;
        newBoxLocal.Set(
            lbbLocal.GetX(), lbbLocal.GetY(), lbbLocal.GetZ(), rtfLocal.GetX(), rtfLocal.GetY(), rtfLocal.GetZ());
        
        p.SetBBox(newBoxLocal);
        p.SetCount(cnt);
        p.SetVertexData(MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ, finalData[i], 7 * sizeof(float));
        p.SetColourData(MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA, finalData[i], 7 * sizeof(float));
        p.SetGlobalRadius(p.GetGlobalRadius() * scaleX);
    }

    return true;
}

void datatools::ParticleTranslateRotateScale::colorTransferGray(core::moldyn::MultiParticleDataCall::Particles& p,
    float const* transferTable, unsigned int tableSize, std::vector<float>& rgbaArray) {

    auto const& parStore = p.GetParticleStore();
    auto const& iAcc = parStore.GetCRAcc();

    std::vector<float> grayArray(p.GetCount());
    for (size_t i = 0; i < p.GetCount(); i++) {
        grayArray[i] = iAcc->Get_f(i);
    }
    
    float gray_max = *std::max_element(grayArray.begin(), grayArray.end());
    float gray_min = *std::min_element(grayArray.begin(), grayArray.end());

    for (auto& gray : grayArray) {
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
                float finalColor = colorFloor + (colorCeil - colorFloor) * (tail);
                rgbaArray.push_back(finalColor);
            }
        }
    }
}