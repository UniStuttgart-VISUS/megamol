/*
 * ExtBezierDataSource.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define VLDEPRECATED
#define VISLIB_DEPRECATED_H_INCLUDED
#include "ExtBezierDataSource.h"
#include "ExtBezierDataCall.h"
#include "mmcore/param/FilePathParam.h"
#include "vislib/sys/ASCIIFileBuffer.h"
#include "vislib/assert.h"
#include "vislib/math/BezierCurve.h"
#include "vislib/Exception.h"
#include "vislib/sys/Log.h"
#include "vislib/math/Point.h"
#include "vislib/VersionNumber.h"

using namespace megamol::beztube;


/*
 * ext::ExtBezierDataSource::ExtBezierDataSource
 */
ext::ExtBezierDataSource::ExtBezierDataSource(void) : Module(),
        //filenameSlot("filename", "The path of the IMD file to read"),
        getDataSlot("getdata", "The slot exposing the loaded data"),
        bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f), ellipCurves(),
        rectCurves(), datahash(0) {

    //this->filenameSlot << new core::param::FilePathParam("");
    //this->MakeSlotAvailable(&this->filenameSlot);
    //this->filenameSlot.ForceSetDirty();

    this->getDataSlot.SetCallback("ExtBezierDataCall", "GetData",
        &ExtBezierDataSource::getDataCallback);
    this->getDataSlot.SetCallback("ExtBezierDataCall", "GetExtent",
        &ExtBezierDataSource::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);
}


/*
 * ext::ExtBezierDataSource::~ExtBezierDataSource
 */
ext::ExtBezierDataSource::~ExtBezierDataSource(void) {
    this->Release();
}


/*
 * ext::ExtBezierDataSource::create
 */
bool ext::ExtBezierDataSource::create(void) {
    // intentionally empty
    return true;
}


/*
 * ext::ExtBezierDataSource::release
 */
void ext::ExtBezierDataSource::release(void) {
    this->ellipCurves.Clear();
    this->rectCurves.Clear();
    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    this->datahash = 0;
}


/*
 * ext::ExtBezierDataSource::getDataCallback
 */
bool ext::ExtBezierDataSource::getDataCallback(megamol::core::Call& caller) {
    ExtBezierDataCall *bdc = dynamic_cast<ExtBezierDataCall*>(&caller);
    if (bdc == NULL) return false;

    this->assertData();

    bdc->SetData(static_cast<unsigned int>(this->ellipCurves.Count()),
        static_cast<unsigned int>(this->rectCurves.Count()),
        this->ellipCurves.PeekElements(),
        this->rectCurves.PeekElements());
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
 * ext::ExtBezierDataSource::getExtentCallback
 */
bool ext::ExtBezierDataSource::getExtentCallback(megamol::core::Call& caller) {
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
 * ext::ExtBezierDataSource::assertData
 */
void ext::ExtBezierDataSource::assertData(void) {
    //if (!this->filenameSlot.IsDirty()) return;
    //this->filenameSlot.ResetDirty();

    // There is no file format ...
    if (!this->ellipCurves.IsEmpty()) return;

    this->ellipCurves.Clear();
    this->rectCurves.Clear();
    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    this->datahash = 0;

    this->ellipCurves.Add(vislib::math::BezierCurve<ext::ExtBezierDataCall::Point, 3>());
    this->ellipCurves.Last()[0].Set(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.2f, 0.1f, 255, 0, 0);
    this->ellipCurves.Last()[1].Set(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.2f, 0.1f, 255, 255, 0);
    this->ellipCurves.Last()[2].Set(1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.1f, 0.1f, 0, 255, 255);
    this->ellipCurves.Last()[3].Set(0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.1f, 0.1f, 0, 0, 255);

    this->ellipCurves.Add(vislib::math::BezierCurve<ext::ExtBezierDataCall::Point, 3>());
    this->ellipCurves.Last()[0].Set(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.2f, 0.1f, 255, 0, 0);
    this->ellipCurves.Last()[1].Set(-1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.2f, 0.1f, 255, 255, 0);
    this->ellipCurves.Last()[2].Set(-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.4f, 0.1f, 0, 255, 255);
    this->ellipCurves.Last()[3].Set(0.0f, -1.0f, 1.0f, 0.0f, -1.0f, 0.0f, 0.4f, 0.1f, 0, 0, 255);

    this->rectCurves.Add(vislib::math::BezierCurve<ext::ExtBezierDataCall::Point, 3>());
    this->rectCurves.Last()[0].Set(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.2f, 0.1f, 255, 0, 0);
    this->rectCurves.Last()[1].Set(-1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.2f, 0.1f, 255, 255, 0);
    this->rectCurves.Last()[2].Set(-1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.1f, 0.1f, 0, 255, 255);
    this->rectCurves.Last()[3].Set(0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.1f, 0.1f, 0, 0, 255);

    this->rectCurves.Add(vislib::math::BezierCurve<ext::ExtBezierDataCall::Point, 3>());
    this->rectCurves.Last()[0].Set(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.2f, 0.1f, 255, 0, 0);
    this->rectCurves.Last()[1].Set(1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.2f, 0.1f, 255, 255, 0);
    this->rectCurves.Last()[2].Set(1.0f, -1.0f, 2.0f, 1.0f, -1.0f, 0.0f, 0.4f, 0.1f, 0, 255, 255);
    this->rectCurves.Last()[3].Set(0.0f, -1.0f, 2.0f, 0.0f, -1.0f, 0.0f, 0.4f, 0.1f, 0, 0, 255);

    this->datahash = 1;

    // calc bounding box (and datahash)
    if (this->rectCurves.Count() + this->ellipCurves.Count() > 0) {
        float mr;
        vislib::math::Cuboid<float> obox;

        if (this->rectCurves.Count() > 0) {
            mr = vislib::math::Max(
                this->rectCurves[0].ControlPoint(0).GetRadiusY(),
                this->rectCurves[0].ControlPoint(0).GetRadiusZ());
            this->bbox.Set(
                this->rectCurves[0].ControlPoint(0).GetPosition().X() - mr,
                this->rectCurves[0].ControlPoint(0).GetPosition().Y() - mr,
                this->rectCurves[0].ControlPoint(0).GetPosition().Z() - mr,
                this->rectCurves[0].ControlPoint(0).GetPosition().X() + mr,
                this->rectCurves[0].ControlPoint(0).GetPosition().Y() + mr,
                this->rectCurves[0].ControlPoint(0).GetPosition().Z() + mr);
        } else {
            ASSERT(this->ellipCurves.Count());
            mr = vislib::math::Max(
                this->ellipCurves[0].ControlPoint(0).GetRadiusY(),
                this->ellipCurves[0].ControlPoint(0).GetRadiusZ());
            this->bbox.Set(
                this->ellipCurves[0].ControlPoint(0).GetPosition().X() - mr,
                this->ellipCurves[0].ControlPoint(0).GetPosition().Y() - mr,
                this->ellipCurves[0].ControlPoint(0).GetPosition().Z() - mr,
                this->ellipCurves[0].ControlPoint(0).GetPosition().X() + mr,
                this->ellipCurves[0].ControlPoint(0).GetPosition().Y() + mr,
                this->ellipCurves[0].ControlPoint(0).GetPosition().Z() + mr);
        }

        for (SIZE_T idx = this->ellipCurves.Count(); idx > 0;) {
            idx--;
            for (SIZE_T cpi = 0; cpi < 4; cpi++) {
                const ext::ExtBezierDataCall::Point& pt
                    = this->ellipCurves[idx].ControlPoint(
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

        for (SIZE_T idx = this->rectCurves.Count(); idx > 0;) {
            idx--;
            for (SIZE_T cpi = 0; cpi < 4; cpi++) {
                const ext::ExtBezierDataCall::Point& pt
                    = this->rectCurves[idx].ControlPoint(
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
