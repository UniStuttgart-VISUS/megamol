/*
 * MoleculeBallifier.cpp
 *
 * Copyright (C) 2012 by TU Dresden
 * All rights reserved.
 */

#include "MoleculeBallifier.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "protein_calls/MolecularDataCall.h"
#include "stdafx.h"


using namespace megamol;
using namespace megamol::protein;
using namespace megamol::protein_calls;
using namespace megamol::geocalls;
using namespace megamol::protein_calls;

MoleculeBallifier::MoleculeBallifier(void)
        : core::Module()
        , outDataSlot("outData", "Sends MultiParticleDataCall data out into the world")
        , inDataSlot("inData", "Fetches MolecularDataCall data")
        , colorTableFileParam_("colorTableFilename", "")
        , coloringModeParam0_("coloringMode0", "")
        , coloringModeParam1_("coloringMode1", "")
        , cmWeightParam_("colorWeight", "")
        , minGradColorParam_("minGradColor", "")
        , midGradColorParam_("midGradColor", "")
        , maxGradColorParam_("maxGradColor", "")
        , specialColorParam_("specialColor", "")
        , inHash(0)
        , outHash(0)
        , data()
        , frameOld(-1) {

    this->inDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);

    this->outDataSlot.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(0), &MoleculeBallifier::getData);
    this->outDataSlot.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(1), &MoleculeBallifier::getExt);
    this->MakeSlotAvailable(&this->outDataSlot);

    std::string filename("colors.txt");
    ProteinColor::ReadColorTableFromFile(filename, fileLookupTable_);
    colorTableFileParam_.SetParameter(
        new core::param::FilePathParam(filename, core::param::FilePathParam::FilePathFlags_::Flag_File_ToBeCreated));
    this->MakeSlotAvailable(&colorTableFileParam_);

    curColoringMode0_ = ProteinColor::ColoringMode::ELEMENT;
    curColoringMode1_ = ProteinColor::ColoringMode::ELEMENT;
    core::param::EnumParam* cm0 = new core::param::EnumParam(static_cast<int>(curColoringMode0_));
    core::param::EnumParam* cm1 = new core::param::EnumParam(static_cast<int>(curColoringMode1_));
    MolecularDataCall* mol = new MolecularDataCall();
    ProteinColor::ColoringMode cMode;
    for (uint32_t cCnt = 0; cCnt < static_cast<uint32_t>(ProteinColor::ColoringMode::MODE_COUNT); ++cCnt) {
        cMode = static_cast<ProteinColor::ColoringMode>(cCnt);
        cm0->SetTypePair(static_cast<int>(cMode), ProteinColor::GetName(cMode).c_str());
        cm1->SetTypePair(static_cast<int>(cMode), ProteinColor::GetName(cMode).c_str());
    }
    delete mol;
    coloringModeParam0_ << cm0;
    coloringModeParam1_ << cm1;
    this->MakeSlotAvailable(&coloringModeParam0_);
    this->MakeSlotAvailable(&coloringModeParam1_);

    cmWeightParam_.SetParameter(new core::param::FloatParam(0.5f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&cmWeightParam_);

    minGradColorParam_.SetParameter(new core::param::ColorParam("#146496"));
    this->MakeSlotAvailable(&minGradColorParam_);

    midGradColorParam_.SetParameter(new core::param::ColorParam("#f0f0f0"));
    this->MakeSlotAvailable(&midGradColorParam_);

    maxGradColorParam_.SetParameter(new core::param::ColorParam("#ae3b32"));
    this->MakeSlotAvailable(&maxGradColorParam_);

    specialColorParam_.SetParameter(new core::param::ColorParam("#228B22"));
    this->MakeSlotAvailable(&specialColorParam_);

    ProteinColor::MakeRainbowColorTable(100, rainbowColors_);
}


/*
 *
 */
MoleculeBallifier::~MoleculeBallifier(void) {
    this->Release();
}


/*
 *
 */
bool MoleculeBallifier::create(void) {
    // intentionally empty
    return true;
}


/*
 *
 */
void MoleculeBallifier::release(void) {
    // intentionally empty
}


/*
 *
 */
bool MoleculeBallifier::getData(core::Call& c) {
    using geocalls::MultiParticleDataCall;
    MultiParticleDataCall* ic = dynamic_cast<MultiParticleDataCall*>(&c);
    if (ic == NULL)
        return false;

    MolecularDataCall* oc = this->inDataSlot.CallAs<MolecularDataCall>();
    if (oc == NULL)
        return false;

    // Transfer frame ID plus force flag
    oc->SetFrameID(ic->FrameID(), ic->IsFrameForced());
    curColoringMode0_ =
        static_cast<ProteinColor::ColoringMode>(coloringModeParam0_.Param<core::param::EnumParam>()->Value());
    curColoringMode1_ =
        static_cast<ProteinColor::ColoringMode>(coloringModeParam1_.Param<core::param::EnumParam>()->Value());

    bool updatedColorTable = false;
    if (colorTableFileParam_.IsDirty()) {
        ProteinColor::ReadColorTableFromFile(
            colorTableFileParam_.Param<core::param::FilePathParam>()->Value(), fileLookupTable_);
        colorTableFileParam_.ResetDirty();
        updatedColorTable = true;
    }

    if (coloringModeParam0_.IsDirty() || coloringModeParam1_.IsDirty() || cmWeightParam_.IsDirty() ||
        minGradColorParam_.IsDirty() || midGradColorParam_.IsDirty() || maxGradColorParam_.IsDirty()) {
        coloringModeParam0_.ResetDirty();
        coloringModeParam1_.ResetDirty();
        cmWeightParam_.ResetDirty();
        minGradColorParam_.ResetDirty();
        midGradColorParam_.ResetDirty();
        maxGradColorParam_.ResetDirty();
        updatedColorTable = true;
    }

    if ((*oc)(0)) {
        // Rewrite data if the frame number OR the datahash has changed
        if ((this->inHash != oc->DataHash()) || (this->frameOld != static_cast<int>(oc->FrameID())) ||
            updatedColorTable) {
            this->inHash = oc->DataHash();
            this->frameOld = static_cast<int>(oc->FrameID());
            this->outHash++;

            colorArray_.clear();

            unsigned int cnt = oc->AtomCount();
            colorArray_.resize(cnt);
            this->data.AssertSize(sizeof(float) * 7 * cnt);
            float* fData = this->data.As<float>();

            this->colorLookupTable_ = {
                glm::make_vec3(this->minGradColorParam_.Param<core::param::ColorParam>()->Value().data()),
                glm::make_vec3(this->midGradColorParam_.Param<core::param::ColorParam>()->Value().data()),
                glm::make_vec3(this->maxGradColorParam_.Param<core::param::ColorParam>()->Value().data())};

            ProteinColor::MakeWeightedColorTable(*oc, curColoringMode0_, curColoringMode1_,
                cmWeightParam_.Param<core::param::FloatParam>()->Value(),
                1.0f - cmWeightParam_.Param<core::param::FloatParam>()->Value(), colorArray_, colorLookupTable_,
                fileLookupTable_, rainbowColors_, nullptr, nullptr, true);

            for (unsigned int i = 0; i < cnt; i++, fData += 7) {
                fData[0] = oc->AtomPositions()[i * 3 + 0];
                fData[1] = oc->AtomPositions()[i * 3 + 1];
                fData[2] = oc->AtomPositions()[i * 3 + 2];
                fData[3] = oc->AtomTypes()[oc->AtomTypeIndices()[i]].Radius();

                fData[4] = colorArray_[i].x;
                fData[5] = colorArray_[i].y;
                fData[6] = colorArray_[i].z;
            }
        }

        ic->SetDataHash(this->outHash);
        ic->SetParticleListCount(1);
        MultiParticleDataCall::Particles& p = ic->AccessParticles(0);
        p.SetCount(this->data.GetSize() / (sizeof(float) * 7));
        if (p.GetCount() > 0) {
            p.SetVertexData(MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR, this->data.At(0), sizeof(float) * 7);
            p.SetColourData(MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB, this->data.At(sizeof(float) * 4),
                sizeof(float) * 7);
        }
    }

    return true;
}


/*
 *
 */
bool MoleculeBallifier::getExt(core::Call& c) {
    using geocalls::MultiParticleDataCall;
    MultiParticleDataCall* ic = dynamic_cast<MultiParticleDataCall*>(&c);
    if (ic == NULL)
        return false;

    MolecularDataCall* oc = this->inDataSlot.CallAs<MolecularDataCall>();
    if (oc == NULL)
        return false;

    if ((*oc)(1)) {
        ic->SetFrameCount(oc->FrameCount());
        ic->AccessBoundingBoxes() = oc->AccessBoundingBoxes();
        return true;
    }

    return false;
}
