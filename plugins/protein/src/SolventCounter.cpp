/*
 * SolventCounter.cpp
 *
 * Copyright (C) 2015 by Michael Krone
 * Copyright (C) 2015 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "SolventCounter.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/utility/log/Log.h"
#include "protein_calls/PerAtomFloatCall.h"
#include "vislib/assert.h"
#include <cfloat>
#include <climits>
#include <omp.h>


using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;


/*
 * SolventCounter::SolventCounter
 */
SolventCounter::SolventCounter()
        : core::Module()
        , getDataSlot("getdata", "The slot publishing the loaded data")
        , molDataSlot("moldata", "The slot requesting molecular data")
        , solDataSlot("soldata", "The slot requesting solvent data")
        , radiusParam("radius", "The search radius for solvent molecules")
        , minValue(0.0f)
        , midValue(0.0f)
        , maxValue(0.0f) {
    // the data out slot
    this->getDataSlot.SetCallback(PerAtomFloatCall::ClassName(),
        PerAtomFloatCall::FunctionName(PerAtomFloatCall::CallForGetFloat), &SolventCounter::getDataCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

    // the data in slots
    this->molDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->molDataSlot);
    this->solDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->solDataSlot);

    // Radius parameter
    this->radiusParam.SetParameter(new param::FloatParam(3.0f, 0.1f));
    this->MakeSlotAvailable(&this->radiusParam);
}


/*
 * SolventCounter::~SolventCounter
 */
SolventCounter::~SolventCounter() {
    this->Release();
}


/*
 * SolventCounter::create
 */
bool SolventCounter::create() {
    // intentionally empty
    return true;
}


/*
 * SolventCounter::release
 */
void SolventCounter::release() {
    // TODO clear data
}


/*
 * SolventCounter::getDataCallback
 */
bool SolventCounter::getDataCallback(core::Call& call) {
    using megamol::core::utility::log::Log;

    PerAtomFloatCall* dc = dynamic_cast<PerAtomFloatCall*>(&call);
    if (dc == NULL)
        return false;

    MolecularDataCall* mol = this->molDataSlot.CallAs<MolecularDataCall>();
    MolecularDataCall* sol = this->solDataSlot.CallAs<MolecularDataCall>();
    if (mol == NULL)
        return false;
    if (sol == NULL)
        return false;
    if (!(*mol)(MolecularDataCall::CallForGetExtent))
        return false;
    if (!(*sol)(MolecularDataCall::CallForGetExtent))
        return false;
#if GET_ONE_TIMESTEP
    if (sol->FrameCount() != mol->FrameCount())
        return false;
    mol->SetFrameID(dc->FrameID());
    if (!(*mol)(MolecularDataCall::CallForGetData))
        return false;
    sol->SetFrameID(dc->FrameID());
    if (!(*sol)(MolecularDataCall::CallForGetData))
        return false;

    if (solvent.Count() != mol->AtomCount() || this->datahash != mol->DataHash()) {
        this->solvent.Clear();
        this->solvent.SetCount(mol->AtomCount());
        vislib::math::Vector<float, 3> molAtomPos, solAtomPos;
        for (unsigned int i = 0; i < mol->AtomCount(); i++) {
            molAtomPos.Set(
                mol->AtomPositions()[3 * i], mol->AtomPositions()[3 * i + 1], mol->AtomPositions()[3 * i + 2]);
            this->solvent[i] = 0.0f;
            for (unsigned int j = 0; j < sol->AtomCount(); j++) {
                solAtomPos.Set(
                    sol->AtomPositions()[3 * j], sol->AtomPositions()[3 * j + 1], sol->AtomPositions()[3 * j + 2]);
                // increase counter if the current solvent atom is within the given radius
                if ((molAtomPos - solAtomPos).Length() <= this->radiusParam.Param<param::FloatParam>()->Value()) {
                    this->solvent[i] = 1.0f;
                }
            }
        }
        this->datahash = mol->DataHash();
    }
    mol->Unlock();
    sol->Unlock();
#else
    // sol and mol must have the same number of frames
    if (sol->FrameCount() != mol->FrameCount())
        return false;
    unsigned int frameCount = mol->FrameCount();
    // only recompute everything if this is necessary
    if (solvent.Count() != mol->AtomCount() || this->datahash != mol->DataHash()) {
        // load data once...
        mol->SetFrameID(0);
        if (!(*mol)(MolecularDataCall::CallForGetData))
            return false;
        sol->SetFrameID(0);
        if (!(*sol)(MolecularDataCall::CallForGetData))
            return false;
        megamol::core::utility::log::Log::DefaultLog.WriteInfo(
            "Start recomputing solvent neighborhood information per atom...");
        this->solvent.Clear();
        this->solvent.SetCount(mol->AtomCount());
        // set all values to zero
        for (unsigned int i = 0; i < mol->AtomCount(); i++) {
            this->solvent[i] = 0.0f;
        }
        this->minValue = FLT_MAX;
        this->maxValue = FLT_MIN;
        // loop over all frames
        vislib::math::Vector<float, 3> molAtomPos, solAtomPos;
        for (unsigned int fID = 0; fID < frameCount; fID++) {
            if (fID % 100 == 0)
                megamol::core::utility::log::Log::DefaultLog.WriteInfo("Computing Frame %i", fID);
            mol->SetFrameID(fID);
            if (!(*mol)(MolecularDataCall::CallForGetData))
                return false;
            sol->SetFrameID(fID);
            if (!(*sol)(MolecularDataCall::CallForGetData))
                return false;
            // loop over all molecule atoms and check for neighboring solvent atoms
            for (unsigned int i = 0; i < mol->AtomCount(); i++) {
                molAtomPos.Set(
                    mol->AtomPositions()[3 * i], mol->AtomPositions()[3 * i + 1], mol->AtomPositions()[3 * i + 2]);
                for (unsigned int j = 0; j < sol->AtomCount(); j++) {
                    solAtomPos.Set(
                        sol->AtomPositions()[3 * j], sol->AtomPositions()[3 * j + 1], sol->AtomPositions()[3 * j + 2]);
                    // increase counter if the current solvent atom is within the given radius
                    if ((molAtomPos - solAtomPos).Length() <= this->radiusParam.Param<param::FloatParam>()->Value()) {
                        this->solvent[i] += 1.0f;
                        break;
                    }
                }
            }
            this->datahash = mol->DataHash();
            mol->Unlock();
            sol->Unlock();
        }
        // normalize values
        for (unsigned int i = 0; i < mol->AtomCount(); i++) {
            this->solvent[i] /= static_cast<float>(frameCount);
            this->minValue = vislib::math::Min(this->minValue, this->solvent[i]);
            this->maxValue = vislib::math::Max(this->maxValue, this->solvent[i]);
        }
        this->midValue = (this->maxValue - this->minValue) * 0.8f + this->minValue;
        megamol::core::utility::log::Log::DefaultLog.WriteInfo(
            "Finished recomputing solvent neighborhood information per atom (%.3f, %.3f, %.3f).", this->minValue,
            this->midValue, this->maxValue);
    }
#endif // GET_ONE_TIMESTEP

    dc->SetData(this->solvent);
    dc->SetMinValue(this->minValue);
    dc->SetMidValue(this->midValue);
    dc->SetMaxValue(this->maxValue);

    return true;
}
