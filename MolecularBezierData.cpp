/*
 * MolecularBezierData.cpp
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "MolecularBezierData.h"
#include "vislib/BezierCurve.h"
#include "vislib/RawStorage.h"
#include "vislib/RawStorageWriter.h"
#include "vislib/Point.h"
#include <list>
#include "Color.h"
#include "vislib/NamedColours.h"

using namespace megamol;
using namespace megamol::protein;


/*
 * MolecularBezierData::MolecularBezierData
 */
MolecularBezierData::MolecularBezierData(void) : Module(),
        outDataSlot("outData", "presents data as BezierCurvesListDataCall"),
        inDataSlot("inData", "fetches data either as ExtBezierDataCall or v1.BezierDataCall"),
        hash(0), timeCode(0) {

    this->outDataSlot.SetCallback(core::misc::BezierCurvesListDataCall::ClassName(), "GetData", &MolecularBezierData::getDataCallback);
    this->outDataSlot.SetCallback(core::misc::BezierCurvesListDataCall::ClassName(), "GetExtent", &MolecularBezierData::getExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->inDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);
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
    Color::ReadColorTableFromFile(nullptr, this->colorLookupTable);
    Color::MakeRainbowColorTable(100, this->rainbowColors);
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
    core::misc::BezierCurvesListDataCall *bcldc = dynamic_cast<core::misc::BezierCurvesListDataCall *>(&caller);
    if (bcldc == nullptr) return false;

    core::AbstractGetData3DCall *agd3dc = this->inDataSlot.CallAs<core::AbstractGetData3DCall>();
    if (agd3dc == nullptr) return false;

    // check hash via cheap ext test
    agd3dc->SetFrameID(bcldc->FrameID(), bcldc->IsFrameForced());
    if (!(*agd3dc)(1u)) return false;
    agd3dc->Unlock();

    if ((bcldc->FrameID() != this->timeCode) || (this->hash == 0) || (this->hash != agd3dc->DataHash())) {
        agd3dc->SetFrameID(bcldc->FrameID(), bcldc->IsFrameForced());

        // fetch actual data
        if (!(*agd3dc)(0u)) return false;

        if ((this->hash != agd3dc->DataHash()) || (this->timeCode != agd3dc->FrameID())) {
            // new data: recompute bezier data
            this->hash = agd3dc->DataHash();
            this->timeCode = agd3dc->FrameID();

            MolecularDataCall *mdc = dynamic_cast<MolecularDataCall*>(agd3dc);
            if (mdc != nullptr) {
                this->update(*mdc);

            } else {
                this->data.Clear();
                this->data.SetGlobalColour(127, 127, 127);
                this->data.SetGlobalRadius(0.1f);

            }
        }
    }

    bcldc->SetDataHash(this->hash);
    bcldc->SetFrameID(this->timeCode);
    bcldc->SetUnlocker(nullptr); // HAZARD
    bcldc->SetData(&this->data, 1);

    return true;
}


/*
 * MolecularBezierData::getExtentCallback
 */
bool MolecularBezierData::getExtentCallback(core::Call& caller) {
    core::misc::BezierCurvesListDataCall *bcldc = dynamic_cast<core::misc::BezierCurvesListDataCall *>(&caller);
    if (bcldc == nullptr) return false;

    core::AbstractGetData3DCall *agd3dc = this->inDataSlot.CallAs<core::AbstractGetData3DCall>();
    if (agd3dc == nullptr) return false;

    agd3dc->SetFrameID(bcldc->FrameID(), bcldc->IsFrameForced());
    if (!(*agd3dc)(1u)) return false;

    bcldc->AccessBoundingBoxes() = agd3dc->AccessBoundingBoxes();
    bcldc->SetFrameID(agd3dc->FrameID(), agd3dc->IsFrameForced());
    bcldc->SetFrameCount(agd3dc->FrameCount());
    bcldc->SetDataHash(agd3dc->DataHash());
    bcldc->SetUnlocker(nullptr);

    agd3dc->Unlock();

    return true;
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

    const float col_mix = 1.0f;
    Color::MakeColorTable(&dat,
        Color::STRUCTURE,
        Color::BFACTOR,
        col_mix,       // blending factor between both color maps
        1.0 - col_mix, // 
        this->atomColorTable, this->colorLookupTable, this->rainbowColors,
        _T("blue"), _T("white"), _T("red"), true);

    for (unsigned int mi = 0; mi < dat.MoleculeCount(); mi++) {
        std::list<vislib::math::Point<float, 3> > poss;
        std::list<float> rads;
        std::list<vislib::math::Vector<float, 3> > cols;

        unsigned ssi = dat.Molecules()[mi].FirstSecStructIndex();
        for (unsigned int ssit = 0; ssit < dat.Molecules()[mi].SecStructCount(); ssi++, ssit++) {

            unsigned int aai = dat.SecondaryStructures()[ssi].FirstAminoAcidIndex();
            for (unsigned int aait = 0; aait < dat.SecondaryStructures()[ssi].AminoAcidCount(); aait++, aai++) {
                if (dat.Residues()[aai]->Identifier() != MolecularDataCall::Residue::AMINOACID) continue;
                unsigned int ca = static_cast<const MolecularDataCall::AminoAcid*>(dat.Residues()[aai])->CAlphaIndex();

                vislib::math::Point<float, 3> pos(dat.AtomPositions() + (ca * 3));
                float rad = 0.3f;
                vislib::math::Vector<float, 3> col(atomColorTable.PeekElements() + (ca * 3));

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
                default: rad = 0.3f; // Ångström
                }

                poss.push_back(pos);
                rads.push_back(rad);
                cols.push_back(col);
            }
        }

        // TODO: Fix me

        auto possE = poss.end();
        auto posI = poss.begin();
        auto radI = rads.begin();
        auto colI = cols.begin();
        for (; posI != possE; posI++, radI++, colI++) {
            pt.Write(posI->X());
            pt.Write(posI->Y());
            pt.Write(posI->Z());
            pt.Write(*radI);
            pt.Write(static_cast<unsigned char>(vislib::math::Clamp(static_cast<int>(colI->X() * 255.0f), 0, 255)));
            pt.Write(static_cast<unsigned char>(vislib::math::Clamp(static_cast<int>(colI->Y() * 255.0f), 0, 255)));
            pt.Write(static_cast<unsigned char>(vislib::math::Clamp(static_cast<int>(colI->Z() * 255.0f), 0, 255)));
            if (cnt > 1) {
                idx.Write(static_cast<unsigned int>(cnt - 1));
                idx.Write(static_cast<unsigned int>(cnt - 1));
                idx.Write(static_cast<unsigned int>(cnt));
                idx.Write(static_cast<unsigned int>(cnt));
            }
            cnt++;
        }

    }

    if (pt.Position() > 0) {
        unsigned char *pt_dat = new unsigned char[pt.Position()];
        ::memcpy(pt_dat, pt_blob, pt.Position());
        unsigned int *idx_dat = new unsigned int[idx.Position() / sizeof(unsigned int)];
        ::memcpy(idx_dat, idx_blob, idx.Position());

        this->data.Set(core::misc::BezierCurvesListDataCall::DATALAYOUT_XYZR_F_RGB_B, pt_dat, cnt, true, idx_dat, idx.Position() / sizeof(unsigned int), true, 0.5f, 127, 127, 127);
    } else {
        this->data.Set(core::misc::BezierCurvesListDataCall::DATALAYOUT_NONE, nullptr, 0, nullptr, 0);
    }
}
