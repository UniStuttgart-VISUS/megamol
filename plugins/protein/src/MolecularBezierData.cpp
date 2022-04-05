/*
 * MolecularBezierData.cpp
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */
#include "MolecularBezierData.h"
#include "stdafx.h"

#include "geometry_calls/BezierCurvesListDataCall.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "protein_calls/ProteinColor.h"
#include "vislib/RawStorage.h"
#include "vislib/RawStorageWriter.h"
#include "vislib/graphics/NamedColours.h"
#include "vislib/math/BezierCurve.h"
#include "vislib/math/Point.h"
#include <vector>

using namespace megamol;
using namespace megamol::protein;
using namespace megamol::protein_calls;


/*
 * MolecularBezierData::MolecularBezierData
 */
MolecularBezierData::MolecularBezierData(void)
        : Module()
        , outDataSlot("outData", "presents data as BezierCurvesListDataCall")
        , inDataSlot("inData", "fetches data either as ExtBezierDataCall or v1.BezierDataCall")
        , hash(0)
        , outhash(0)
        , timeCode(0)
        , data()
        , colorLookupTable()
        , atomColorTable()
        , rainbowColors()
        , color1Slot("color1", "The primary color mode")
        , color2Slot("color2", "The secondary color mode")
        , minGradColorSlot("gradCol.min", "The color for minimum values")
        , mixGradColorSlot("gradCol.normal", "The color for normal values")
        , maxGradColorSlot("gradCol.max", "The color for maximum values")
        , colorMixSlot("colorMix", "Mixing value for the two color modes") {

    this->outDataSlot.SetCallback(
        geocalls::BezierCurvesListDataCall::ClassName(), "GetData", &MolecularBezierData::getDataCallback);
    this->outDataSlot.SetCallback(
        geocalls::BezierCurvesListDataCall::ClassName(), "GetExtent", &MolecularBezierData::getExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->inDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);

    core::param::EnumParam* colMode1 =
        new core::param::EnumParam(static_cast<int>(ProteinColor::ColoringMode::SECONDARY_STRUCTURE));
    core::param::EnumParam* colMode2 =
        new core::param::EnumParam(static_cast<int>(ProteinColor::ColoringMode::BFACTOR));

    MolecularDataCall* mol = new MolecularDataCall();
    BindingSiteCall* bs = new BindingSiteCall();
    for (unsigned int cCnt = 0; cCnt < static_cast<uint32_t>(ProteinColor::ColoringMode::MODE_COUNT); ++cCnt) {
        ProteinColor::ColoringMode cMode = static_cast<ProteinColor::ColoringMode>(cCnt);
        colMode1->SetTypePair(static_cast<int>(cMode), ProteinColor::GetName(cMode).c_str());
        colMode2->SetTypePair(static_cast<int>(cMode), ProteinColor::GetName(cMode).c_str());
    }

    this->color1Slot << colMode1;
    this->MakeSlotAvailable(&this->color1Slot);

    this->color2Slot << colMode2;
    this->MakeSlotAvailable(&this->color2Slot);

    this->colorMixSlot << new core::param::FloatParam(0.5f, 0.0f, 1.0f);
    this->MakeSlotAvailable(&this->colorMixSlot);

    this->minGradColorSlot << new core::param::ColorParam("#146496");
    this->MakeSlotAvailable(&this->minGradColorSlot);

    this->mixGradColorSlot << new core::param::ColorParam("#f0f0f0");
    this->MakeSlotAvailable(&this->mixGradColorSlot);

    this->maxGradColorSlot << new core::param::ColorParam("#ae3b32");
    this->MakeSlotAvailable(&this->maxGradColorSlot);
}


/*
 * MolecularBezierData::~MolecularBezierData
 */
MolecularBezierData::~MolecularBezierData(void) {
    this->Release();
}


/*
 * MolecularBezierData::create
 */
bool MolecularBezierData::create(void) {
    ProteinColor::ReadColorTableFromFile(std::string(), this->fileLookupTable);
    ProteinColor::MakeRainbowColorTable(100, this->rainbowColors);
    return true;
}


/*
 * MolecularBezierData::release
 */
void MolecularBezierData::release(void) {
    // intentionally empty
}


/*
 * MolecularBezierData::getDataCallback
 */
bool MolecularBezierData::getDataCallback(core::Call& caller) {
    geocalls::BezierCurvesListDataCall* bcldc = dynamic_cast<geocalls::BezierCurvesListDataCall*>(&caller);
    if (bcldc == nullptr)
        return false;

    core::AbstractGetData3DCall* agd3dc = this->inDataSlot.CallAs<core::AbstractGetData3DCall>();
    if (agd3dc == nullptr)
        return false;

    // check hash via cheap ext test
    agd3dc->SetFrameID(bcldc->FrameID(), bcldc->IsFrameForced());
    if (!(*agd3dc)(1u))
        return false;
    agd3dc->Unlock();

    if (this->color1Slot.IsDirty() || this->color2Slot.IsDirty() || this->minGradColorSlot.IsDirty() ||
        this->mixGradColorSlot.IsDirty() || this->maxGradColorSlot.IsDirty() || this->colorMixSlot.IsDirty()) {
        this->color1Slot.ResetDirty();
        this->color2Slot.ResetDirty();
        this->minGradColorSlot.ResetDirty();
        this->mixGradColorSlot.ResetDirty();
        this->maxGradColorSlot.ResetDirty();
        this->colorMixSlot.ResetDirty();

        this->hash = static_cast<size_t>(-1);
    }

    if ((bcldc->FrameID() != this->timeCode) || (this->hash == 0) || (this->hash != agd3dc->DataHash())) {
        agd3dc->SetFrameID(bcldc->FrameID(), bcldc->IsFrameForced());

        // fetch actual data
        if (!(*agd3dc)(0u))
            return false;

        if ((this->hash != agd3dc->DataHash()) || (this->timeCode != agd3dc->FrameID())) {
            // new data: recompute bezier data
            this->hash = agd3dc->DataHash();
            this->timeCode = agd3dc->FrameID();
            this->outhash++;

            MolecularDataCall* mdc = dynamic_cast<MolecularDataCall*>(agd3dc);
            if (mdc != nullptr) {
                this->update(*mdc);

            } else {
                this->data.Clear();
                this->data.SetGlobalColour(127, 127, 127);
                this->data.SetGlobalRadius(0.1f);
            }
        }
    }

    bcldc->SetDataHash(this->outhash);
    bcldc->SetFrameID(this->timeCode);
    bcldc->SetUnlocker(nullptr); // HAZARD
    bcldc->SetData(&this->data, 1);

    return true;
}


/*
 * MolecularBezierData::getExtentCallback
 */
bool MolecularBezierData::getExtentCallback(core::Call& caller) {
    geocalls::BezierCurvesListDataCall* bcldc = dynamic_cast<geocalls::BezierCurvesListDataCall*>(&caller);
    if (bcldc == nullptr)
        return false;

    core::AbstractGetData3DCall* agd3dc = this->inDataSlot.CallAs<core::AbstractGetData3DCall>();
    if (agd3dc == nullptr)
        return false;

    agd3dc->SetFrameID(bcldc->FrameID(), bcldc->IsFrameForced());
    if (!(*agd3dc)(1u))
        return false;

    bcldc->AccessBoundingBoxes() = agd3dc->AccessBoundingBoxes();
    bcldc->SetFrameID(agd3dc->FrameID(), agd3dc->IsFrameForced());
    bcldc->SetFrameCount(agd3dc->FrameCount());
    bcldc->SetDataHash(this->outhash);
    bcldc->SetUnlocker(nullptr);

    agd3dc->Unlock();

    return true;
}


/**
 * local utility function for linear interpolation
 *
 * @param v1 Value 1
 * @param v2 Value 2
 * @param a Interpolation value
 *
 * @return The interpolated value
 */
static float interpolate(float v1, float v2, float a) {
    if (a < 0.0f)
        a = 0.0f;
    if (a > 1.0f)
        a = 1.0f;
    return v1 * (1.0f - a) + v2 * a;
}


/*
 * MolecularBezierData::update
 */
void MolecularBezierData::update(MolecularDataCall& dat) {
    this->data.Clear();
    vislib::RawStorage pt_blob;
    vislib::RawStorageWriter pt(pt_blob);
    vislib::RawStorage idx_blob;
    vislib::RawStorageWriter idx(idx_blob);
    size_t cnt = 0;

    const float col_mix = this->colorMixSlot.Param<core::param::FloatParam>()->Value();

    this->colorLookupTable = {glm::make_vec3(this->minGradColorSlot.Param<core::param::ColorParam>()->Value().data()),
        glm::make_vec3(this->mixGradColorSlot.Param<core::param::ColorParam>()->Value().data()),
        glm::make_vec3(this->maxGradColorSlot.Param<core::param::ColorParam>()->Value().data())};

    ProteinColor::MakeWeightedColorTable(dat,
        static_cast<ProteinColor::ColoringMode>(this->color1Slot.Param<core::param::EnumParam>()->Value()),
        static_cast<ProteinColor::ColoringMode>(this->color2Slot.Param<core::param::EnumParam>()->Value()), col_mix,
        1.0f - col_mix, this->atomColorTable, this->colorLookupTable, this->fileLookupTable, this->rainbowColors,
        nullptr, nullptr, true);

    for (unsigned int mi = 0; mi < dat.MoleculeCount(); mi++) {
        std::vector<const float*> poss;
        std::vector<float> rads;
        std::vector<const float*> cols;

        unsigned ssi = dat.Molecules()[mi].FirstSecStructIndex();
        for (unsigned int ssit = 0; ssit < dat.Molecules()[mi].SecStructCount(); ssi++, ssit++) {

            unsigned int aai = dat.SecondaryStructures()[ssi].FirstAminoAcidIndex();
            for (unsigned int aait = 0; aait < dat.SecondaryStructures()[ssi].AminoAcidCount(); aait++, aai++) {
                if (dat.Residues()[aai]->Identifier() != MolecularDataCall::Residue::AMINOACID)
                    continue;
                unsigned int ca = static_cast<const MolecularDataCall::AminoAcid*>(dat.Residues()[aai])->CAlphaIndex();

                float rad = 0.3f; // Ångström
                switch (dat.SecondaryStructures()[ssi].Type()) {
                case MolecularDataCall::SecStructure::TYPE_HELIX:
                    rad = 0.4f; // Ångström
                    break;
                case MolecularDataCall::SecStructure::TYPE_SHEET:
                    rad = 0.4f; // Ångström
                    break;
                case MolecularDataCall::SecStructure::TYPE_TURN:
                    rad = 0.2f; // Ångström
                    break;
                case MolecularDataCall::SecStructure::TYPE_COIL: // fall through
                default:
                    rad = 0.3f; // Ångström
                }

                poss.push_back(dat.AtomPositions() + (ca * 3));
                rads.push_back(rad);
                cols.push_back(&atomColorTable[0].x + (ca * 3));
            }
        }

        ASSERT(poss.size() == rads.size());
        ASSERT(poss.size() == cols.size());
        if (poss.size() < 2) {
            continue; // skip this empty data
        } else if (poss.size() == 2) {
            // simple line:
            pt.Write(poss[0][0]);
            pt.Write(poss[0][1]);
            pt.Write(poss[0][2]);
            pt.Write(rads[0]);
            pt.Write(static_cast<unsigned char>(vislib::math::Clamp(static_cast<int>(cols[0][0] * 255.0f), 0, 255)));
            pt.Write(static_cast<unsigned char>(vislib::math::Clamp(static_cast<int>(cols[0][1] * 255.0f), 0, 255)));
            pt.Write(static_cast<unsigned char>(vislib::math::Clamp(static_cast<int>(cols[0][2] * 255.0f), 0, 255)));
            cnt++;
            pt.Write(poss[1][0]);
            pt.Write(poss[1][1]);
            pt.Write(poss[1][2]);
            pt.Write(rads[1]);
            pt.Write(static_cast<unsigned char>(vislib::math::Clamp(static_cast<int>(cols[1][0] * 255.0f), 0, 255)));
            pt.Write(static_cast<unsigned char>(vislib::math::Clamp(static_cast<int>(cols[1][1] * 255.0f), 0, 255)));
            pt.Write(static_cast<unsigned char>(vislib::math::Clamp(static_cast<int>(cols[1][2] * 255.0f), 0, 255)));
            cnt++;
            idx.Write(static_cast<unsigned int>(cnt - 2));
            idx.Write(static_cast<unsigned int>(cnt - 2));
            idx.Write(static_cast<unsigned int>(cnt - 1));
            idx.Write(static_cast<unsigned int>(cnt - 1));
        } else {
            // more than three points (spline-siff-code from old core module)

            // first curve
            pt.Write(poss[0][0]);
            pt.Write(poss[0][1]);
            pt.Write(poss[0][2]);
            pt.Write(rads[0]);
            pt.Write(static_cast<unsigned char>(vislib::math::Clamp(static_cast<int>(cols[0][0] * 255.0f), 0, 255)));
            pt.Write(static_cast<unsigned char>(vislib::math::Clamp(static_cast<int>(cols[0][1] * 255.0f), 0, 255)));
            pt.Write(static_cast<unsigned char>(vislib::math::Clamp(static_cast<int>(cols[0][2] * 255.0f), 0, 255)));
            idx.Write(static_cast<unsigned int>(cnt));
            cnt++;

            pt.Write(interpolate(poss[0][0], poss[1][0], 0.75f));
            pt.Write(interpolate(poss[0][1], poss[1][1], 0.75f));
            pt.Write(interpolate(poss[0][2], poss[1][2], 0.75f));
            pt.Write(interpolate(rads[0], rads[1], 0.75f));
            pt.Write(static_cast<unsigned char>(
                vislib::math::Clamp(static_cast<int>(interpolate(cols[0][0], cols[1][0], 0.75f) * 255.0f), 0, 255)));
            pt.Write(static_cast<unsigned char>(
                vislib::math::Clamp(static_cast<int>(interpolate(cols[0][1], cols[1][1], 0.75f) * 255.0f), 0, 255)));
            pt.Write(static_cast<unsigned char>(
                vislib::math::Clamp(static_cast<int>(interpolate(cols[0][2], cols[1][2], 0.75f) * 255.0f), 0, 255)));
            idx.Write(static_cast<unsigned int>(cnt));
            cnt++;

            // inner curves
            for (unsigned int i = 1; i < poss.size() - 1; i++) {

                pt.Write(interpolate(poss[i][0], poss[i + 1][0], 0.25f));
                pt.Write(interpolate(poss[i][1], poss[i + 1][1], 0.25f));
                pt.Write(interpolate(poss[i][2], poss[i + 1][2], 0.25f));
                pt.Write(interpolate(rads[i], rads[i + 1], 0.25f));
                pt.Write(static_cast<unsigned char>(vislib::math::Clamp(
                    static_cast<int>(interpolate(cols[i][0], cols[i + 1][0], 0.25f) * 255.0f), 0, 255)));
                pt.Write(static_cast<unsigned char>(vislib::math::Clamp(
                    static_cast<int>(interpolate(cols[i][1], cols[i + 1][1], 0.25f) * 255.0f), 0, 255)));
                pt.Write(static_cast<unsigned char>(vislib::math::Clamp(
                    static_cast<int>(interpolate(cols[i][2], cols[i + 1][2], 0.25f) * 255.0f), 0, 255)));
                idx.Write(static_cast<unsigned int>(cnt));
                cnt++;

                pt.Write(interpolate(poss[i][0], poss[i + 1][0], 0.5f));
                pt.Write(interpolate(poss[i][1], poss[i + 1][1], 0.5f));
                pt.Write(interpolate(poss[i][2], poss[i + 1][2], 0.5f));
                pt.Write(interpolate(rads[i], rads[i + 1], 0.5f));
                pt.Write(static_cast<unsigned char>(vislib::math::Clamp(
                    static_cast<int>(interpolate(cols[i][0], cols[i + 1][0], 0.5f) * 255.0f), 0, 255)));
                pt.Write(static_cast<unsigned char>(vislib::math::Clamp(
                    static_cast<int>(interpolate(cols[i][1], cols[i + 1][1], 0.5f) * 255.0f), 0, 255)));
                pt.Write(static_cast<unsigned char>(vislib::math::Clamp(
                    static_cast<int>(interpolate(cols[i][2], cols[i + 1][2], 0.5f) * 255.0f), 0, 255)));
                idx.Write(static_cast<unsigned int>(cnt));
                idx.Write(static_cast<unsigned int>(cnt)); // use this point twice
                cnt++;

                pt.Write(interpolate(poss[i][0], poss[i + 1][0], 0.75f));
                pt.Write(interpolate(poss[i][1], poss[i + 1][1], 0.75f));
                pt.Write(interpolate(poss[i][2], poss[i + 1][2], 0.75f));
                pt.Write(interpolate(rads[i], rads[i + 1], 0.75f));
                pt.Write(static_cast<unsigned char>(vislib::math::Clamp(
                    static_cast<int>(interpolate(cols[i][0], cols[i + 1][0], 0.75f) * 255.0f), 0, 255)));
                pt.Write(static_cast<unsigned char>(vislib::math::Clamp(
                    static_cast<int>(interpolate(cols[i][1], cols[i + 1][1], 0.75f) * 255.0f), 0, 255)));
                pt.Write(static_cast<unsigned char>(vislib::math::Clamp(
                    static_cast<int>(interpolate(cols[i][2], cols[i + 1][2], 0.75f) * 255.0f), 0, 255)));
                idx.Write(static_cast<unsigned int>(cnt));
                cnt++;
            }

            // last curve
            pt.Write(interpolate(poss[poss.size() - 2][0], poss[poss.size() - 1][0], 0.25f));
            pt.Write(interpolate(poss[poss.size() - 2][1], poss[poss.size() - 1][1], 0.25f));
            pt.Write(interpolate(poss[poss.size() - 2][2], poss[poss.size() - 1][2], 0.25f));
            pt.Write(interpolate(rads[poss.size() - 2], rads[poss.size() - 1], 0.25f));
            pt.Write(static_cast<unsigned char>(vislib::math::Clamp(
                static_cast<int>(interpolate(cols[poss.size() - 2][0], cols[poss.size() - 1][0], 0.25f) * 255.0f), 0,
                255)));
            pt.Write(static_cast<unsigned char>(vislib::math::Clamp(
                static_cast<int>(interpolate(cols[poss.size() - 2][1], cols[poss.size() - 1][1], 0.25f) * 255.0f), 0,
                255)));
            pt.Write(static_cast<unsigned char>(vislib::math::Clamp(
                static_cast<int>(interpolate(cols[poss.size() - 2][2], cols[poss.size() - 1][2], 0.25f) * 255.0f), 0,
                255)));
            idx.Write(static_cast<unsigned int>(cnt));
            cnt++;

            pt.Write(poss[poss.size() - 1][0]);
            pt.Write(poss[poss.size() - 1][1]);
            pt.Write(poss[poss.size() - 1][2]);
            pt.Write(rads[poss.size() - 1]);
            pt.Write(static_cast<unsigned char>(
                vislib::math::Clamp(static_cast<int>(cols[poss.size() - 1][0] * 255.0f), 0, 255)));
            pt.Write(static_cast<unsigned char>(
                vislib::math::Clamp(static_cast<int>(cols[poss.size() - 1][1] * 255.0f), 0, 255)));
            pt.Write(static_cast<unsigned char>(
                vislib::math::Clamp(static_cast<int>(cols[poss.size() - 1][2] * 255.0f), 0, 255)));
            idx.Write(static_cast<unsigned int>(cnt));
            cnt++;
        }
    }

    if (pt.Position() > 0) {
        unsigned char* pt_dat = new unsigned char[pt.Position()];
        ::memcpy(pt_dat, pt_blob, pt.Position());
        unsigned int* idx_dat = new unsigned int[idx.Position() / sizeof(unsigned int)];
        ::memcpy(idx_dat, idx_blob, idx.Position());

        this->data.Set(geocalls::BezierCurvesListDataCall::DATALAYOUT_XYZR_F_RGB_B, pt_dat, cnt, true, idx_dat,
            idx.Position() / sizeof(unsigned int), true, 0.5f, 127, 127, 127);
    } else {
        this->data.Set(geocalls::BezierCurvesListDataCall::DATALAYOUT_NONE, nullptr, 0, nullptr, 0);
    }
}
