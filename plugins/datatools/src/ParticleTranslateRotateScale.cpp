/*
 * ParticleTranslateRotateScaleScale.h
 *
 * Copyright (C) 2018 MegaMol Team
 * Alle Rechte vorbehalten.
 */
#include "ParticleTranslateRotateScale.h"

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>


#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"

using namespace megamol;


/*
 * datatools::ParticleTranslateRotateScale::ParticleTranslateRotateScale
 */
datatools::ParticleTranslateRotateScale::ParticleTranslateRotateScale(void)
        : AbstractParticleManipulator("outData", "indata")
        , translateSlot("translation", "Translates the particles in x, y, z direction")
        , quaternionSlot("quaternion", "Rotates the particles around x, y, z axes")
        , scaleSlot("scale", "Scales the particle data")
        , finalData(NULL) {
    this->translateSlot.SetParameter(new core::param::Vector3fParam(vislib::math::Vector<float, 3>(0, 0, 0)));
    this->MakeSlotAvailable(&this->translateSlot);

    this->quaternionSlot.SetParameter(new core::param::Vector4fParam(vislib::math::Vector<float, 4>(0, 0, 0, 1)));
    this->MakeSlotAvailable(&this->quaternionSlot);
    this->quaternionSlot.Parameter()->SetGUIPresentation(
        core::param::AbstractParamPresentation::Presentation::Rotation);

    this->scaleSlot.SetParameter(new core::param::Vector3fParam(vislib::math::Vector<float, 3>(1, 1, 1)));
    this->MakeSlotAvailable(&this->scaleSlot);
}


/*
 * datatools::ParticleTranslateRotateScale::~ParticleTranslateRotateScale
 */
datatools::ParticleTranslateRotateScale::~ParticleTranslateRotateScale(void) {
    this->Release();
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
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {
    using geocalls::MultiParticleDataCall;


    const float transX = translateSlot.Param<core::param::Vector3fParam>()->Value().GetX();
    const float transY = translateSlot.Param<core::param::Vector3fParam>()->Value().GetY();
    const float transZ = translateSlot.Param<core::param::Vector3fParam>()->Value().GetZ();

    const float scaleX = scaleSlot.Param<core::param::Vector3fParam>()->Value().GetX();
    const float scaleY = scaleSlot.Param<core::param::Vector3fParam>()->Value().GetY();
    const float scaleZ = scaleSlot.Param<core::param::Vector3fParam>()->Value().GetZ();

    auto bboxCenterX = inData.GetBoundingBoxes().ObjectSpaceBBox().CalcCenter().GetX();
    auto bboxCenterY = inData.GetBoundingBoxes().ObjectSpaceBBox().CalcCenter().GetY();
    auto bboxCenterZ = inData.GetBoundingBoxes().ObjectSpaceBBox().CalcCenter().GetZ();

    auto trafo = glm::mat4(1.0);
    trafo = glm::translate(trafo, glm::vec3(-bboxCenterX, -bboxCenterY, -bboxCenterZ));
    trafo = glm::scale(trafo, glm::vec3(scaleX, scaleY, scaleZ));
    auto& qs = quaternionSlot.Param<core::param::Vector4fParam>()->Value();
    auto glmq = glm::quat(qs.GetW(), qs.GetX(), qs.GetY(), qs.GetZ());
    trafo = glm::toMat4(glmq) * trafo;
    trafo = glm::translate(trafo, glm::vec3(bboxCenterX, bboxCenterY, bboxCenterZ));
    trafo = glm::translate(trafo, glm::vec3(transX, transY, transZ));

    if (InterfaceIsDirty() || (hash != inData.DataHash()) || (inData.DataHash() == 0) ||
        (frameID != inData.FrameID())) {
        // Update data
        hash = inData.DataHash();
        frameID = inData.FrameID();
        InterfaceResetDirty();

        unsigned int plc = inData.GetParticleListCount();
        outData.SetParticleListCount(plc);

        outData = inData;                   // also transfers the unlocker to 'outData'
        inData.SetUnlocker(nullptr, false); // keep original data locked
                                            // original data will be unlocked through outData

        finalData.resize(plc);
        for (unsigned int i = 0; i < plc; i++) {
            MultiParticleDataCall::Particles& p = inData.AccessParticles(i);

            uint64_t cnt = p.GetCount();

            const void* cd = p.GetColourData();
            unsigned int cds = p.GetColourDataStride();
            MultiParticleDataCall::Particles::ColourDataType cdt = p.GetColourDataType();

            const void* vd = p.GetVertexData();
            unsigned int vds = p.GetVertexDataStride();
            MultiParticleDataCall::Particles::VertexDataType vdt = p.GetVertexDataType();


            // Color transfer call and calculation
            //std::vector<float> rgba;
            //rgba.reserve(cnt * 4);
            //unsigned int texSize = 0;
            //megamol::core_gl::view::CallGetTransferFunctionGL* cgtf =
            //    this->getTFSlot.CallAs<core_gl::view::CallGetTransferFunctionGL>();
            //if (cgtf != NULL && ((*cgtf)())) {
            //    float const* tfTex = cgtf->GetTextureData();
            //    texSize = cgtf->TextureSize();
            //    this->colorTransferGray(p, tfTex, texSize, rgba);
            //} else {
            //    this->colorTransferGray(p, NULL, 0, rgba);
            //}

            auto const& parStore = p.GetParticleStore();
            auto const& xAcc = parStore.GetXAcc();
            auto const& yAcc = parStore.GetYAcc();
            auto const& zAcc = parStore.GetZAcc();
            auto const& rAcc = parStore.GetCRAcc();
            auto const& gAcc = parStore.GetCGAcc();
            auto const& bAcc = parStore.GetCBAcc();
            auto const& aAcc = parStore.GetCAAcc();

            finalData[i].resize(cnt * 7, 0.0f);
            //#pragma omp parallel for
            for (int64_t loop = 0; loop < cnt; loop++) {

                glm::vec4 glmpos = trafo * glm::vec4(xAcc->Get_f(loop), yAcc->Get_f(loop), zAcc->Get_f(loop), 1.0);

                finalData[i][7 * loop + 0] = glmpos.x;          // pos.GetX();
                finalData[i][7 * loop + 1] = glmpos.y;          // pos.GetY();
                finalData[i][7 * loop + 2] = glmpos.z;          //pos.GetZ();
                finalData[i][7 * loop + 3] = rAcc->Get_f(loop); // rgba[4 * loop + 0];
                finalData[i][7 * loop + 4] = gAcc->Get_f(loop); // rgba[4 * loop + 1];
                finalData[i][7 * loop + 5] = bAcc->Get_f(loop); // rgba[4 * loop + 2];
                finalData[i][7 * loop + 6] = aAcc->Get_f(loop); // rgba[4 * loop + 3];
            }

            auto lbb_local = glm::vec3(finalData[i][0], finalData[i][1], finalData[i][2]);
            auto rtf_local = lbb_local;
            for (int64_t loop = 1; loop < cnt; loop++) {
                lbb_local = glm::min(lbb_local,
                    glm::vec3(finalData[i][7 * loop + 0], finalData[i][7 * loop + 1], finalData[i][7 * loop + 2]));
                rtf_local = glm::max(rtf_local,
                    glm::vec3(finalData[i][7 * loop + 0], finalData[i][7 * loop + 1], finalData[i][7 * loop + 2]));
            }

            vislib::math::Cuboid<float> newBoxLocal;
            newBoxLocal.Set(lbb_local.x, lbb_local.y, lbb_local.z, rtf_local.x, rtf_local.y, rtf_local.z);
            _global_box = newBoxLocal;
            MultiParticleDataCall::Particles& outp = outData.AccessParticles(i);
            outp.SetBBox(newBoxLocal);
            outp.SetCount(cnt);
            outp.SetVertexData(
                MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ, finalData[i].data(), 7 * sizeof(float));
            outp.SetColourData(
                MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA, finalData[i].data() + 3, 7 * sizeof(float));
            outp.SetGlobalRadius(p.GetGlobalRadius() * scaleX);
        }

        if (plc > 0) {
            auto bbox = outData.AccessParticles(0).GetBBox();
            auto lbb = glm::vec3(bbox.Left(), bbox.Bottom(), bbox.Back());
            auto rtf = glm::vec3(bbox.Right(), bbox.Top(), bbox.Front());
            for (unsigned int i = 1; i < plc; i++) {
                bbox = outData.AccessParticles(i).GetBBox();
                lbb = glm::min(lbb, glm::vec3(bbox.Left(), bbox.Bottom(), bbox.Back()));
                rtf = glm::max(rtf, glm::vec3(bbox.Right(), bbox.Top(), bbox.Front()));
            }

            _global_box.Set(lbb.x, lbb.y, lbb.z, rtf.x, rtf.y, rtf.z);
            outData.AccessBoundingBoxes().SetObjectSpaceBBox(_global_box);
            outData.AccessBoundingBoxes().SetObjectSpaceClipBox(_global_box);
        }
    }
    outData.SetDataHash(this->hash);
    outData.SetFrameID(this->frameID);

    return true;
}

bool datatools::ParticleTranslateRotateScale::manipulateExtent(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {

    outData = inData;

    if (!_global_box.IsEmpty()) {
        outData.AccessBoundingBoxes().SetObjectSpaceBBox(_global_box);
        outData.AccessBoundingBoxes().SetObjectSpaceClipBox(_global_box);
    }

    return true;
}
