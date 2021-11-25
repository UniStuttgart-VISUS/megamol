#include "PDBInterpolator.h"
#include "stdafx.h"

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;


PDBInterpolator::PDBInterpolator()
        : getDataSlot("getData", "Calls pdb data.")
        , dataOutSlot("dataout", "The slot providing the interpolated data") {
    this->getDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&getDataSlot);

    this->dataOutSlot.SetCallback(MolecularDataCall::ClassName(),
        MolecularDataCall::FunctionName(MolecularDataCall::CallForGetData), &PDBInterpolator::getData);
    this->dataOutSlot.SetCallback(MolecularDataCall::ClassName(),
        MolecularDataCall::FunctionName(MolecularDataCall::CallForGetExtent), &PDBInterpolator::getExtent);
    this->MakeSlotAvailable(&this->dataOutSlot);
}


PDBInterpolator::~PDBInterpolator() {
    this->Release();
}

/*
 * PDBInterpolator::create
 */
bool PDBInterpolator::create(void) {
    return true;
}

/*
 * PDBInterpolator::release
 */
void PDBInterpolator::release(void) {}

/*
 * PDBInterpolator::getData
 */
bool PDBInterpolator::getData(core::Call& call) {
    MolecularDataCall* mdc_in = dynamic_cast<MolecularDataCall*>(&call);
    if (mdc_in == NULL)
        return false;
    MolecularDataCall* mdc = getDataSlot.CallAs<MolecularDataCall>();
    if (mdc == NULL)
        return false;

    float call_time = mdc->Calltime();

    int call_time_one = (int)call_time;
    int call_time_two = call_time_one + 1;
    float x = call_time - (float)call_time_one;

    mdc->SetCalltime((float)call_time_one);
    mdc->SetFrameID(static_cast<int>(call_time_one));

    if (!(*mdc)(MolecularDataCall::CallForGetData))
        return false;
    if (mdc->AtomCount() == 0)
        return true;

    float* pos0 = new float[mdc->AtomCount() * 3];
    memcpy(pos0, mdc->AtomPositions(), mdc->AtomCount() * 3 * sizeof(float));

    mdc->SetCalltime((float)call_time_one);
    mdc->SetFrameID(static_cast<int>(call_time_one));

    if (!(*mdc)(MolecularDataCall::CallForGetData))
        return false;
    if (mdc->AtomCount() == 0)
        return true;

    float* pos1 = new float[mdc->AtomCount() * 3];
    memcpy(pos1, mdc->AtomPositions(), mdc->AtomCount() * 3 * sizeof(float));

    float* interpolated = new float[mdc->AtomCount() * 3];
    for (unsigned int i = 0; i < mdc->AtomCount() * 3; i++) {
        interpolated[i] = (1.0f - x) * pos0[i] + x * pos1[i];
    }
    mdc_in->SetAtomPositions(interpolated);

    return true;
}

/*
 * PDBInterpolator::getExtent
 */
bool PDBInterpolator::getExtent(core::Call& call) {
    MolecularDataCall* mdc_in = dynamic_cast<MolecularDataCall*>(&call);
    if (mdc_in == NULL)
        return false;
    MolecularDataCall* mdc = getDataSlot.CallAs<MolecularDataCall>();
    if (mdc == NULL)
        return false;

    if (!(*mdc)(MolecularDataCall::CallForGetExtent))
        return false;

    float scale;
    if (!vislib::math::IsEqual(mdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge(), 0.0f)) {
        scale = 2.0f / mdc->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
    } else {
        scale = 1.0f;
    }

    mdc_in->AccessBoundingBoxes() = mdc->AccessBoundingBoxes();
    mdc_in->AccessBoundingBoxes().MakeScaledWorld(scale);

    return true;
}
