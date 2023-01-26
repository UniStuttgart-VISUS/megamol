/*
 * QuartzDataGridder.cpp
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#define _USE_MATH_DEFINES
#include "QuartzDataGridder.h"
#include "QuartzCrystalDataCall.h"
#include "QuartzParticleDataCall.h"
#include "mmcore/param/IntParam.h"
#include <climits>
#include <cmath>
//#include "mmcore/utility/log/Log.h"
//#include "vislib/sys/MemmappedFile.h"
//#include "vislib/memutils.h"
//#include "vislib/math/ShallowVector.h"
//#include "vislib/String.h"
//#include "vislib/utils.h"
//#include "vislib/math/Point.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol {
namespace demos_gl {


/*
 * DataGridder::DataGridder
 */
DataGridder::DataGridder()
        : core::Module()
        , dataOutSlot("dataout", "The slot providing the gridded data")
        , dataInSlot("datain", "The slot fetching the flat data")
        , crysInSlot("crysin", "The slot fetching the crystalite data")
        , gridSizeXSlot("gridsizex", "The number of grid cells in x direction")
        , gridSizeYSlot("gridsizey", "The number of grid cells in y direction")
        , gridSizeZSlot("gridsizez", "The number of grid cells in z direction")
        , partHash(0)
        , crysHash(0)
        , dataHash(0)
        , cells(NULL)
        , bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f)
        , cbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f)
        , pdata()
        , lists(NULL) {

    this->dataOutSlot.SetCallback(ParticleGridDataCall::ClassName(),
        ParticleGridDataCall::FunctionName(ParticleGridDataCall::CallForGetData), &DataGridder::getData);
    this->dataOutSlot.SetCallback(ParticleGridDataCall::ClassName(),
        ParticleGridDataCall::FunctionName(ParticleGridDataCall::CallForGetExtent), &DataGridder::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);

    this->dataInSlot.SetCompatibleCall<core::factories::CallAutoDescription<ParticleDataCall>>();
    this->MakeSlotAvailable(&this->dataInSlot);

    this->crysInSlot.SetCompatibleCall<core::factories::CallAutoDescription<CrystalDataCall>>();
    this->MakeSlotAvailable(&this->crysInSlot);

    this->gridSizeXSlot << new core::param::IntParam(5, 1);
    this->MakeSlotAvailable(&this->gridSizeXSlot);

    this->gridSizeYSlot << new core::param::IntParam(5, 1);
    this->MakeSlotAvailable(&this->gridSizeYSlot);

    this->gridSizeZSlot << new core::param::IntParam(5, 1);
    this->MakeSlotAvailable(&this->gridSizeZSlot);
}


/*
 * DataGridder::~DataGridder
 */
DataGridder::~DataGridder() {
    this->Release();
}


/*
 * DataGridder::create
 */
bool DataGridder::create() {
    // intentionally empty
    return true;
}


/*
 * DataGridder::getData
 */
bool DataGridder::getData(core::Call& c) {
    ParticleGridDataCall* pgdc = dynamic_cast<ParticleGridDataCall*>(&c);
    if (pgdc == NULL)
        return false;
    if (this->needClearData())
        this->clearData();
    if (this->needMakeData())
        this->makeData();
    pgdc->Set(static_cast<unsigned int>(this->gridSizeXSlot.Param<core::param::IntParam>()->Value()),
        static_cast<unsigned int>(this->gridSizeYSlot.Param<core::param::IntParam>()->Value()),
        static_cast<unsigned int>(this->gridSizeZSlot.Param<core::param::IntParam>()->Value()), this->cells);
    pgdc->SetDataHash(this->dataHash);
    pgdc->SetUnlocker(NULL);
    return true;
}


/*
 * DataGridder::getExtent
 */
bool DataGridder::getExtent(core::Call& c) {
    ParticleGridDataCall* pgdc = dynamic_cast<ParticleGridDataCall*>(&c);
    if (pgdc == NULL)
        return false;
    if (this->needClearData())
        this->clearData();
    if (this->needMakeData())
        this->makeData();
    pgdc->SetFrameCount(1);
    pgdc->AccessBoundingBoxes().Clear();
    pgdc->AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
    pgdc->AccessBoundingBoxes().SetObjectSpaceClipBox(this->cbox);
    pgdc->SetDataHash(this->dataHash);
    pgdc->SetUnlocker(NULL);
    return true;
}


/*
 * DataGridder::release
 */
void DataGridder::release() {
    this->clearData();
    this->pdata.Resize(0);
    ASSERT(this->cells == NULL);
    ASSERT(this->lists == NULL);
    ASSERT(this->pdata.Count() == 0);
}


/*
 * DataGridder::clearData
 */
void DataGridder::clearData() {
    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    this->cbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    ARY_SAFE_DELETE(this->cells);
    ARY_SAFE_DELETE(this->lists);
    this->pdata.Clear(); // the raw particle data pointers
}


/*
 * DataGridder::makeData
 */
void DataGridder::makeData() {
    ParticleDataCall* di = this->dataInSlot.CallAs<ParticleDataCall>();
    CrystalDataCall* ci = this->crysInSlot.CallAs<CrystalDataCall>();
    ASSERT((di != NULL) && (ci != NULL));

    this->clearData();
    if (!(*di)(ParticleDataCall::CallForGetExtent)) {
        return;
    }

    ASSERT(this->cells == NULL);
    ASSERT(this->lists == NULL);
    ASSERT(this->pdata.Count() == 0);

    if (di->AccessBoundingBoxes().IsObjectSpaceBBoxValid() && di->AccessBoundingBoxes().IsObjectSpaceClipBoxValid()) {
        this->bbox = di->AccessBoundingBoxes().ObjectSpaceBBox();
        this->cbox = di->AccessBoundingBoxes().ObjectSpaceClipBox();
    } else {
        this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
        this->cbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    }

    if (!(*ci)(CrystalDataCall::CallForGetData)) {
        this->clearData();
        return;
    }
    if (!(*di)(ParticleDataCall::CallForGetData)) {
        this->clearData();
        return;
    }
    this->partHash = di->DataHash();
    this->crysHash = ci->DataHash();
    this->gridSizeXSlot.ResetDirty();
    this->gridSizeYSlot.ResetDirty();
    this->gridSizeZSlot.ResetDirty();
    unsigned int sx = static_cast<unsigned int>(this->gridSizeXSlot.Param<core::param::IntParam>()->Value());
    unsigned int sy = static_cast<unsigned int>(this->gridSizeYSlot.Param<core::param::IntParam>()->Value());
    unsigned int sz = static_cast<unsigned int>(this->gridSizeZSlot.Param<core::param::IntParam>()->Value());
    this->cells = new ParticleGridDataCall::Cell[sx * sy * sz];
    this->bbox.EnforcePositiveSize();
    for (unsigned int z = 0; z < sz; z++) {
        float zlow = static_cast<float>(z) / static_cast<float>(sz);
        float zhigh = static_cast<float>(z + 1) / static_cast<float>(sz);
        for (unsigned int y = 0; y < sy; y++) {
            float ylow = static_cast<float>(y) / static_cast<float>(sy);
            float yhigh = static_cast<float>(y + 1) / static_cast<float>(sy);
            for (unsigned int x = 0; x < sx; x++) {
                unsigned int i = x + sx * (y + sy * z);
                float xlow = static_cast<float>(x) / static_cast<float>(sx);
                float xhigh = static_cast<float>(x + 1) / static_cast<float>(sx);
                this->cells[i].SetBBox(vislib::math::Cuboid<float>(this->bbox.Left() + xlow * this->bbox.Width(),
                    this->bbox.Bottom() + ylow * this->bbox.Height(), this->bbox.Back() + zlow * this->bbox.Depth(),
                    this->bbox.Left() + xhigh * this->bbox.Width(), this->bbox.Bottom() + yhigh * this->bbox.Height(),
                    this->bbox.Back() + zhigh * this->bbox.Depth()));
            }
        }
    }

    unsigned int* pccnt = new unsigned int[sx * sy * sz]; // # particles / (group * cell)
    unsigned int* pcidx = new unsigned int[sx * sy * sz]; // # particles / (group * cell)
    vislib::Array<ParticleGridDataCall::List>* lists = new vislib::Array<ParticleGridDataCall::List>[sx * sy * sz];

    for (unsigned int g = 0; g < di->GetGroupCount(); g++) {
        ::memset(pccnt, 0, sizeof(unsigned int) * sx * sy * sz);
        unsigned int ti = di->GetCrystalType(g) % ci->GetCount();
        const float* pd = di->GetParticleData(g);
        for (unsigned int p = 0; p < di->GetParticleCount(g); p++, pd += 8) {
            int ix = static_cast<int>(static_cast<float>(sx) * (pd[0] - this->bbox.Left()) / this->bbox.Width());
            if (ix < 0)
                ix = 0;
            else if (ix >= static_cast<int>(sx))
                ix = sx - 1;
            int iy = static_cast<int>(static_cast<float>(sy) * (pd[1] - this->bbox.Bottom()) / this->bbox.Height());
            if (iy < 0)
                iy = 0;
            else if (iy >= static_cast<int>(sy))
                iy = sy - 1;
            int iz = static_cast<int>(static_cast<float>(sz) * (pd[2] - this->bbox.Back()) / this->bbox.Depth());
            if (iz < 0)
                iz = 0;
            else if (iz >= static_cast<int>(sz))
                iz = sz - 1;
            pccnt[ix + sx * (iy + sy * iz)]++;
        }
        for (unsigned int z = 0; z < sz; z++) {
            for (unsigned int y = 0; y < sy; y++) {
                for (unsigned int x = 0; x < sx; x++) {
                    unsigned int i = x + sx * (y + sy * z);
                    if (pccnt[i] == 0) {
                        pcidx[i] = UINT_MAX;
                        continue;
                    }
                    pcidx[i] = static_cast<unsigned int>(this->pdata.Count());
                    this->pdata.Add(new float[pccnt[i] * 8]);
                    lists[i].Add(ParticleGridDataCall::List());
                    lists[i].Last().Set(pccnt[i], this->pdata.Last(), ti);
                    pccnt[i] = 0;
                }
            }
        }
        pd = di->GetParticleData(g);
        for (unsigned int p = 0; p < di->GetParticleCount(g); p++, pd += 8) {
            int ix = static_cast<int>(static_cast<float>(sx) * (pd[0] - this->bbox.Left()) / this->bbox.Width());
            if (ix < 0)
                ix = 0;
            else if (ix >= static_cast<int>(sx))
                ix = sx - 1;
            int iy = static_cast<int>(static_cast<float>(sy) * (pd[1] - this->bbox.Bottom()) / this->bbox.Height());
            if (iy < 0)
                iy = 0;
            else if (iy >= static_cast<int>(sy))
                iy = sy - 1;
            int iz = static_cast<int>(static_cast<float>(sz) * (pd[2] - this->bbox.Back()) / this->bbox.Depth());
            if (iz < 0)
                iz = 0;
            else if (iz >= static_cast<int>(sz))
                iz = sz - 1;
            unsigned int i = ix + sx * (iy + sy * iz);
            ASSERT(pcidx[i] != UINT_MAX);
            ::memcpy(this->pdata[pcidx[i]] + (pccnt[i] * 8), pd, sizeof(float) * 8);
            pccnt[i]++;
        }
    }
    delete[] pcidx;
    delete[] pccnt;

    SIZE_T listscnt = 0;
    for (unsigned int i = 0; i < sx * sy * sz; i++) {
        listscnt += lists[i].Count();
    }
    this->lists = new ParticleGridDataCall::List[listscnt];
    listscnt = 0;
    for (unsigned int i = 0; i < sx * sy * sz; i++) {
        float cboxGrow = 0.0f;
        SIZE_T lcnt = lists[i].Count();
        this->cells[i].Set(this->cells[i].BoundingBox(), this->cells[i].ClipBox(), static_cast<unsigned int>(lcnt),
            this->lists + listscnt);
        for (SIZE_T j = 0; j < lcnt; j++) {
            this->lists[listscnt++] = lists[i][j];
        }
        for (unsigned int l = 0; l < this->cells[i].Count(); l++) {
            unsigned int t = this->cells[i].Lists()[l].Type();
            unsigned int pc = this->cells[i].Lists()[l].Count();
            ASSERT(t < ci->GetCount());
            float rp = 0.0f;
            const float* pd = this->cells[i].Lists()[l].Data();
            for (unsigned int p = 0; p < pc; p++, pd += 8) {
                if (rp < pd[3])
                    rp = pd[3];
            }
            float rb = rp * ci->GetCrystals()[t].GetBoundingRadius();
            if (rb > cboxGrow)
                cboxGrow = rb;
        }
        this->cells[i].SetCBoxRelative(cboxGrow);
    }
    delete[] lists;

    di->Unlock();
    ci->Unlock();
}


/*
 * DataGridder::needClearData
 */
bool DataGridder::needClearData() {
    return ((this->dataHash != 0) && ((this->dataInSlot.CallAs<ParticleDataCall>() == NULL) ||
                                         (this->crysInSlot.CallAs<CrystalDataCall>() == NULL)));
}


/*
 * DataGridder::needMakeData
 */
bool DataGridder::needMakeData() {
    ParticleDataCall* di = this->dataInSlot.CallAs<ParticleDataCall>();
    CrystalDataCall* ci = this->crysInSlot.CallAs<CrystalDataCall>();
    if ((di == NULL) || (ci == NULL))
        return false;
    if (!(*di)(ParticleDataCall::CallForGetExtent))
        return false;
    if (!(*ci)(CrystalDataCall::CallForGetExtent))
        return false;
    return (di->DataHash() == 0) || (ci->DataHash() == 0) || (di->DataHash() != this->partHash) ||
           (ci->DataHash() != this->crysHash) || this->gridSizeXSlot.IsDirty() || this->gridSizeYSlot.IsDirty() ||
           this->gridSizeZSlot.IsDirty();
}

} // namespace demos_gl
} /* end namespace megamol */
