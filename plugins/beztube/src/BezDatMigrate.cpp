/*
 * BezDatMigrate.cpp
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "BezDatMigrate.h"
#include "vislib/math/BezierCurve.h"
#include "vislib/RawStorage.h"
#include "vislib/RawStorageWriter.h"

using namespace megamol;
using namespace megamol::beztube;


/*
 * BezDatMigrate::BezDatMigrate
 */
BezDatMigrate::BezDatMigrate(void) : Module(),
        outDataSlot("outData", "presents data as BezierCurvesListDataCall"),
        inDataSlot("inData", "fetches data either as ExtBezierDataCall or v1.BezierDataCall"),
        hash(0), timeCode(0) {

    this->outDataSlot.SetCallback(core::misc::BezierCurvesListDataCall::ClassName(), "GetData", &BezDatMigrate::getDataCallback);
    this->outDataSlot.SetCallback(core::misc::BezierCurvesListDataCall::ClassName(), "GetExtent", &BezDatMigrate::getExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->inDataSlot.SetCompatibleCall<ext::ExtBezierDataCallDescription>();
    this->inDataSlot.SetCompatibleCall<v1::BezierDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);
}


/*
 * BezDatMigrate::~BezDatMigrate
 */
BezDatMigrate::~BezDatMigrate(void) {
    this->Release();
}


/*
 * BezDatMigrate::create
 */
bool BezDatMigrate::create(void) {
    // intentionally empty
    return true;
}


/*
 * BezDatMigrate::release
 */
void BezDatMigrate::release(void) {
    // intentionally empty
}


/*
 * BezDatMigrate::getDataCallback
 */
bool BezDatMigrate::getDataCallback(core::Call& caller) {
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

            v1::BezierDataCall *bdc = dynamic_cast<v1::BezierDataCall*>(agd3dc);
            ext::ExtBezierDataCall *ebdc = dynamic_cast<ext::ExtBezierDataCall*>(agd3dc);
            if (bdc != nullptr) {
                this->update(*bdc);

            } else if (ebdc != nullptr) {
                this->update(*ebdc);

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
 * BezDatMigrate::getExtentCallback
 */
bool BezDatMigrate::getExtentCallback(core::Call& caller) {
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
 * BezDatMigrate::update
 */
void BezDatMigrate::update(ext::ExtBezierDataCall& dat) {
    this->data.Clear();
    vislib::RawStorage pt_blob;
    vislib::RawStorageWriter pt(pt_blob);
    vislib::RawStorage idx_blob;
    vislib::RawStorageWriter idx(idx_blob);
    size_t cnt = 0;

#ifdef _WIN32
#pragma warning(disable:4996)
#endif

    for (unsigned int i = 0; i < dat.CountElliptic(); i++) {
        const vislib::math::BezierCurve<ext::ExtBezierDataCall::Point, 3> &cur = dat.EllipticCurves()[i];
        for (unsigned int j = 0; j < 4; j++) {
            pt.Write(cur[j].GetPosition().X());
            pt.Write(cur[j].GetPosition().Y());
            pt.Write(cur[j].GetPosition().Z());
            pt.Write((cur[j].GetRadiusY() + cur[j].GetRadiusZ()) * 0.5f);
            pt.Write(cur[j].GetColour().R());
            pt.Write(cur[j].GetColour().G());
            pt.Write(cur[j].GetColour().B());
            idx.Write(static_cast<unsigned int>(cnt));
            cnt++;
        }
    }

    for (unsigned int i = 0; i < dat.CountRectangular(); i++) {
        const vislib::math::BezierCurve<ext::ExtBezierDataCall::Point, 3> &cur = dat.RectangularCurves()[i];
        for (unsigned int j = 0; j < 4; j++) {
            pt.Write(cur[j].GetPosition().X());
            pt.Write(cur[j].GetPosition().Y());
            pt.Write(cur[j].GetPosition().Z());
            pt.Write((cur[j].GetRadiusY() + cur[j].GetRadiusZ()) * 0.5f);
            pt.Write(cur[j].GetColour().R());
            pt.Write(cur[j].GetColour().G());
            pt.Write(cur[j].GetColour().B());
            idx.Write(static_cast<unsigned int>(cnt));
            cnt++;
        }
    }

#ifdef _WIN32
#pragma warning(default:4996)
#endif

    unsigned char *pt_dat = new unsigned char[pt.Position()];
    ::memcpy(pt_dat, pt_blob, pt.Position());
    unsigned int *idx_dat = new unsigned int[idx.Position() / sizeof(unsigned int)];
    ::memcpy(idx_dat, idx_blob, idx.Position());
    this->data.Set(core::misc::BezierCurvesListDataCall::DATALAYOUT_XYZR_F_RGB_B, pt_dat, cnt, true, idx_dat, cnt, true, 0.5f, 127, 127, 127);
}


/*
 * BezDatMigrate::update
 */
void BezDatMigrate::update(v1::BezierDataCall& dat) {
    this->data.Clear();
    vislib::RawStorage pt_blob;
    vislib::RawStorageWriter pt(pt_blob);
    vislib::RawStorage idx_blob;
    vislib::RawStorageWriter idx(idx_blob);
    size_t cnt = 0;

#ifdef _WIN32
#pragma warning(disable:4996)
#endif

    for (unsigned int i = 0; i < dat.Count(); i++) {
        const vislib::math::BezierCurve<v1::BezierDataCall::BezierPoint, 3> &cur = dat.Curves()[i];
        for (unsigned int j = 0; j < 4; j++) {
            pt.Write(cur[j].X());
            pt.Write(cur[j].Y());
            pt.Write(cur[j].Z());
            pt.Write(cur[j].Radius());
            pt.Write(cur[j].R());
            pt.Write(cur[j].G());
            pt.Write(cur[j].B());
            idx.Write(static_cast<unsigned int>(cnt));
            cnt++;
        }
    }

#ifdef _WIN32
#pragma warning(default:4996)
#endif

    unsigned char *pt_dat = new unsigned char[pt.Position()];
    ::memcpy(pt_dat, pt_blob, pt.Position());
    unsigned int *idx_dat = new unsigned int[idx.Position() / sizeof(unsigned int)];
    ::memcpy(idx_dat, idx_blob, idx.Position());
    this->data.Set(core::misc::BezierCurvesListDataCall::DATALAYOUT_XYZR_F_RGB_B, pt_dat, cnt, true, idx_dat, cnt, true, 0.5f, 127, 127, 127);
}
