/*
 * ExtBezierDataSource.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ExtBezierDataSource.h"
#include "ExtBezierDataCall.h"
#include "param/FilePathParam.h"
#include "vislib/ASCIIFileBuffer.h"
#include "vislib/BezierCurve.h"
#include "vislib/Exception.h"
#include "vislib/Log.h"
#include "vislib/Point.h"
#include "vislib/VersionNumber.h"

using namespace megamol::core;


/*
 * misc::ExtBezierDataSource::ExtBezierDataSource
 */
misc::ExtBezierDataSource::ExtBezierDataSource(void) : Module(),
        filenameSlot("filename", "The path of the IMD file to read"),
        getDataSlot("getdata", "The slot exposing the loaded data"),
        bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f), curves(), datahash(0) {

    this->filenameSlot << new param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);

    this->getDataSlot.SetCallback("ExtBezierDataCall", "GetData",
        &ExtBezierDataSource::getDataCallback);
    this->getDataSlot.SetCallback("ExtBezierDataCall", "GetExtent",
        &ExtBezierDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);
}


/*
 * misc::ExtBezierDataSource::~ExtBezierDataSource
 */
misc::ExtBezierDataSource::~ExtBezierDataSource(void) {
    this->Release();
}


/*
 * misc::ExtBezierDataSource::create
 */
bool misc::ExtBezierDataSource::create(void) {
    // intentionally empty
    return true;
}


/*
 * misc::ExtBezierDataSource::release
 */
void misc::ExtBezierDataSource::release(void) {
    this->curves.Clear();
    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    this->datahash = 0;
}


/*
 * misc::ExtBezierDataSource::getDataCallback
 */
bool misc::ExtBezierDataSource::getDataCallback(Call& caller) {
    ExtBezierDataCall *bdc = dynamic_cast<ExtBezierDataCall*>(&caller);
    if (bdc == NULL) return false;

    this->assertData();

    bdc->SetData(static_cast<unsigned int>(this->curves.Count()),
        this->curves.PeekElements());
    bdc->SetDataHash(this->datahash);
    bdc->AccessBoundingBoxes().Clear();
    bdc->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
    bdc->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox);
    bdc->SetFrameCount(1);
    bdc->SetFrameID(0);
    bdc->SetUnlocker(NULL);

    return true;
}


/*
 * misc::ExtBezierDataSource::getExtentCallback
 */
bool misc::ExtBezierDataSource::getExtentCallback(Call& caller) {
    ExtBezierDataCall *bdc = dynamic_cast<ExtBezierDataCall*>(&caller);
    if (bdc == NULL) return false;

    this->assertData();

    bdc->SetDataHash(this->datahash);
    bdc->AccessBoundingBoxes().Clear();
    bdc->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
    bdc->AccessBoundingBoxes().SetObjectSpaceClipBox(this->bbox);
    bdc->SetFrameCount(1);

    return true;
}


/*
 * misc::ExtBezierDataSource::assertData
 */
void misc::ExtBezierDataSource::assertData(void) {
    if (!this->filenameSlot.IsDirty()) return;
    this->filenameSlot.ResetDirty();

    this->curves.Clear();
    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    this->datahash = 0;

    this->curves.Add(vislib::math::BezierCurve<misc::ExtBezierDataCall::Point, 3>());
    this->curves.Last().ControlPoint(0).Set(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.2f, 0.1f, 255, 0, 0);
    this->curves.Last().ControlPoint(1).Set(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.2f, 0.1f, 255, 255, 0);
    this->curves.Last().ControlPoint(2).Set(1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.2f, 0.1f, 0, 255, 255);
    this->curves.Last().ControlPoint(3).Set(0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.2f, 0.1f, 0, 0, 255);

    this->curves.Add(vislib::math::BezierCurve<misc::ExtBezierDataCall::Point, 3>());
    this->curves.Last().ControlPoint(0).Set(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.2f, 0.1f, 255, 0, 0);
    this->curves.Last().ControlPoint(1).Set(-1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.2f, 0.1f, 255, 255, 0);
    this->curves.Last().ControlPoint(2).Set(-1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.2f, 0.1f, 0, 255, 255);
    this->curves.Last().ControlPoint(3).Set(0.0f, -1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.2f, 0.1f, 0, 0, 255);

    this->datahash = 1;

    // calc bounding box (and datahash)
    if (this->curves.Count() > 0) {
        float mr = vislib::math::Max(
            this->curves[0].ControlPoint(0).GetRadiusY(),
            this->curves[0].ControlPoint(0).GetRadiusZ());
        vislib::math::Cuboid<float> obox;
        this->bbox.Set(
            this->curves[0].ControlPoint(0).GetPosition().X() - mr,
            this->curves[0].ControlPoint(0).GetPosition().Y() - mr,
            this->curves[0].ControlPoint(0).GetPosition().Z() - mr,
            this->curves[0].ControlPoint(0).GetPosition().X() + mr,
            this->curves[0].ControlPoint(0).GetPosition().Y() + mr,
            this->curves[0].ControlPoint(0).GetPosition().Z() + mr);

        for (SIZE_T idx = this->curves.Count(); idx > 0;) {
            idx--;
            for (SIZE_T cpi = 0; cpi < 4; cpi++) {
                const misc::ExtBezierDataCall::Point& pt
                    = this->curves[idx].ControlPoint(
                        static_cast<unsigned int>(cpi));
                mr = vislib::math::Max(pt.GetRadiusY(), pt.GetRadiusZ());
                obox.Set(
                    pt.GetPosition().X() - mr,
                    pt.GetPosition().Y() - mr,
                    pt.GetPosition().Z() - mr,
                    pt.GetPosition().X() + mr,
                    pt.GetPosition().Y() + mr,
                    pt.GetPosition().Z() + mr);
                this->bbox.Union(obox);
            }
        }
    }

}
