/*
 * IsoSurface.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "volumetrics/IsoSurface.h"
#include "CallVolumeData.h"
#include "CallTriMeshData.h"
#include "param/StringParam.h"
#include "param/FloatParam.h"
#include <climits>
#include <cfloat>
#include <cmath>
#include "vislib/Log.h"
#include "vislib/RawStorageWriter.h"

using namespace megamol;
using namespace megamol::trisoup;
using namespace megamol::trisoup::volumetrics;


/*
 * IsoSurface::IsoSurface
 */
IsoSurface::IsoSurface(void) : 
        inDataSlot("inData", "The slot for requesting input data"),
        outDataSlot("outData", "Gets the data"), 
        attributeSlot("attr", "The attribute to show"),
        isoValueSlot("isoval", "The iso value"),
        dataHash(0), frameIdx(0), index(), vertex(), normal(), mesh() {

    this->inDataSlot.SetCompatibleCall<core::CallVolumeDataDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);

    this->outDataSlot.SetCallback("CallTriMeshData", "GetData", &IsoSurface::outDataCallback);
    this->outDataSlot.SetCallback("CallTriMeshData", "GetExtent", &IsoSurface::outExtentCallback);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->attributeSlot << new core::param::StringParam("0");
    this->MakeSlotAvailable(&this->attributeSlot);

    this->isoValueSlot << new core::param::FloatParam(0.5f);
    this->MakeSlotAvailable(&this->isoValueSlot);

}


/*
 * IsoSurface::~IsoSurface
 */
IsoSurface::~IsoSurface(void) {
    this->Release();
}


/*
 * IsoSurface::create
 */
bool IsoSurface::create(void) {
    // intentionally empty
    return true;
}


/*
 * IsoSurface::release
 */
void IsoSurface::release(void) {
    this->index.EnforceSize(0);
    this->vertex.EnforceSize(0);
    this->normal.EnforceSize(0);
}


/*
 * IsoSurface::outDataCallback
 */
bool IsoSurface::outDataCallback(core::Call& caller) {
    CallTriMeshData *tmd = dynamic_cast<CallTriMeshData*>(&caller);
    if (tmd == NULL) return false;

    core::CallVolumeData *cvd = this->inDataSlot.CallAs<core::CallVolumeData>();
    if (cvd != NULL) {

        bool recalc = false;

        if (this->isoValueSlot.IsDirty()) {
            this->isoValueSlot.ResetDirty();
            recalc = true;
        }

        if (this->attributeSlot.IsDirty()) {
            this->attributeSlot.ResetDirty();
            recalc = true;
        }

        cvd->SetFrameID(tmd->FrameID(), tmd->IsFrameForced());
        if (!(*cvd)(0)) {
            recalc = false;
        } else {
            if ((this->dataHash != cvd->DataHash()) || (this->frameIdx != cvd->FrameID())) {
                recalc = true;
            }
        }

        unsigned int attrIdx = UINT_MAX;
        if (recalc) {
            vislib::StringA attrName(this->attributeSlot.Param<core::param::StringParam>()->Value());
            attrIdx = cvd->FindAttribute(attrName);
            if (attrIdx == UINT_MAX) {
                try {
                    attrIdx = static_cast<unsigned int>(vislib::CharTraitsA::ParseInt(attrName));
                } catch(...) {
                    attrIdx = UINT_MAX;
                }
            }
            if (attrIdx >= cvd->AttributeCount()) {
                recalc = false;
            } else if (cvd->Attribute(attrIdx).Type() != core::CallVolumeData::TYPE_FLOAT) {
                vislib::sys::Log::DefaultLog.WriteError("Only float volumes are supported ATM");
                recalc = false;
            }
        }

        if (recalc) {
            float isoVal = this->isoValueSlot.Param<core::param::FloatParam>()->Value();

            this->index.EnforceSize(0);
            this->vertex.EnforceSize(0);
            this->normal.EnforceSize(0);

            vislib::RawStorageWriter i(this->index);
            vislib::RawStorageWriter v(this->vertex);
            vislib::RawStorageWriter n(this->normal);
            i.SetIncrement(1024 * 1024);
            v.SetIncrement(1024 * 1024);
            n.SetIncrement(1024 * 1024);

            // Rebuild mesh data
            this->buildMesh(i, v, n, isoVal, cvd->Attribute(attrIdx).Floats(), cvd->XSize(), cvd->YSize(), cvd->ZSize());

            this->index.EnforceSize(i.End(), true);
            this->vertex.EnforceSize(v.End(), true);
            this->normal.EnforceSize(n.End(), true);

            this->mesh.SetMaterial(NULL);
            this->mesh.SetVertexData(this->vertex.GetSize() / static_cast<unsigned int>(3 * sizeof(float)), this->vertex.As<float>(), this->normal.As<float>(), NULL, NULL, false);
            this->mesh.SetTriangleData(this->index.GetSize() / static_cast<unsigned int>(3 * sizeof(unsigned int)), this->index.As<unsigned int>(), false);

            this->dataHash = cvd->DataHash();
            this->frameIdx = cvd->FrameID();
        }
    }

    tmd->SetDataHash(this->dataHash);
    tmd->SetFrameID(this->frameIdx);
    tmd->SetObjects(1, &this->mesh);
    tmd->SetUnlocker(NULL);

    return true;

}


/*
 * IsoSurface::outExtentCallback
 */
bool IsoSurface::outExtentCallback(megamol::core::Call& caller) {
    CallTriMeshData *tmd = dynamic_cast<CallTriMeshData*>(&caller);
    if (tmd == NULL) return false;

    tmd->AccessBoundingBoxes().Clear();
    core::CallVolumeData *cvd = this->inDataSlot.CallAs<core::CallVolumeData>();
    if ((cvd == NULL) || (!(*cvd)(1))) {
        // no input data
        tmd->SetDataHash(0);
        tmd->SetFrameCount(1);

    } else {
        // input data in cvd
        tmd->SetDataHash(cvd->DataHash());
        tmd->SetExtent(cvd->FrameCount(), cvd->AccessBoundingBoxes());
        this->osbb = cvd->AccessBoundingBoxes().ObjectSpaceBBox();
    }
    tmd->SetUnlocker(NULL);

    return true;
}


/*
 * IsoSurface::buildMesh
 */
void IsoSurface::buildMesh(
        vislib::RawStorageWriter& i, vislib::RawStorageWriter& v, vislib::RawStorageWriter& n,
        float val, const float *vol, unsigned int sx, unsigned int sy, unsigned int sz) {

    // DEBUG: though all voxel
    for (unsigned int z = 0; z < sz; z++) {
        float pZ = static_cast<float>(z) / static_cast<float>(sz - 1);
        pZ = pZ * this->osbb.Depth() + this->osbb.Back();

        for (unsigned int y = 0; y < sy; y++) {
            float pY = static_cast<float>(y) / static_cast<float>(sy - 1);
            pY = pY * this->osbb.Height() + this->osbb.Bottom();

            for (unsigned int x = 0; x < sx; x++) {
                float pX = static_cast<float>(x) / static_cast<float>(sx - 1);
                pX = pX * this->osbb.Width() + this->osbb.Left();

                if (vol[x + sx * (y + sy * z)] < val) {
                    v.Write(pX);
                    v.Write(pY);
                    v.Write(pZ);
                }

            }

        }

    }

    // TODO: Implement

}
