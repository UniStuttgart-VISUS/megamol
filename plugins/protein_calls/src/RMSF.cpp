#include "protein_calls/RMSF.h"
#include "mmcore/utility/log/Log.h"
#include "stdafx.h"
#include "vislib/Array.h"
#include "vislib/math/ShallowVector.h"
#include <cfloat>
#include <climits>
#include <cmath>
#include <fstream>

using namespace megamol;
using namespace megamol::protein_calls;

bool megamol::protein_calls::computeRMSF(protein_calls::MolecularDataCall* mol) {
    if (mol == NULL)
        return false;

    // store current frame and calltime
    float currentCallTime = mol->Calltime();
    unsigned int currentFrame = mol->FrameID();

    // load first frame
    mol->SetFrameID(0, true);
    if (!(*mol)(MolecularDataCall::CallForGetData))
        return false;
    // check if atom count is zero
    if (mol->AtomCount() < 1)
        return false;

    // no frames available -> false
    if (mol->FrameCount() < 2)
        return false;

    // allocate mem
    vislib::Array<float> meanPos;
    meanPos.SetCount(mol->AtomCount() * 3);
    float* rmsf;
    rmsf = new float[mol->AtomCount()];
    // initialize rmsf to zero
    for (unsigned int i = 0; i < mol->AtomCount(); i++) {
        rmsf[i] = 0;
    }

    // get atom pos of first frame
    for (unsigned int atomIdx = 0; atomIdx < mol->AtomCount(); atomIdx++) {
        meanPos[atomIdx * 3 + 0] = mol->AtomPositions()[atomIdx * 3 + 0];
        meanPos[atomIdx * 3 + 1] = mol->AtomPositions()[atomIdx * 3 + 1];
        meanPos[atomIdx * 3 + 2] = mol->AtomPositions()[atomIdx * 3 + 2];
    }

    // sum up all atom positions
    for (unsigned int i = 1; i < mol->FrameCount(); i++) {
        // load frame
        mol->SetFrameID(i, true);
        if (!(*mol)(MolecularDataCall::CallForGetData))
            return false;
        // add atom pos
        for (unsigned int atomIdx = 0; atomIdx < mol->AtomCount(); atomIdx++) {
            meanPos[atomIdx * 3 + 0] += mol->AtomPositions()[atomIdx * 3 + 0];
            meanPos[atomIdx * 3 + 1] += mol->AtomPositions()[atomIdx * 3 + 1];
            meanPos[atomIdx * 3 + 2] += mol->AtomPositions()[atomIdx * 3 + 2];
        }
    }
    // compute average pos
    for (unsigned int i = 0; i < meanPos.Count(); i++) {
        meanPos[i] /= static_cast<float>(mol->FrameCount());
    }

    // compute RMSF
    vislib::math::Vector<float, 3> tmpVec;
    float len;
    for (unsigned int i = 0; i < mol->FrameCount(); i++) {
        // load frame
        mol->SetFrameID(i, true);
        if (!(*mol)(MolecularDataCall::CallForGetData))
            return false;
        // get deviation from mean pos for current atom pos
        for (unsigned int atomIdx = 0; atomIdx < mol->AtomCount(); atomIdx++) {
            tmpVec.SetX(mol->AtomPositions()[atomIdx * 3 + 0] - meanPos[atomIdx * 3 + 0]);
            tmpVec.SetY(mol->AtomPositions()[atomIdx * 3 + 1] - meanPos[atomIdx * 3 + 1]);
            tmpVec.SetZ(mol->AtomPositions()[atomIdx * 3 + 2] - meanPos[atomIdx * 3 + 2]);
            len = tmpVec.Length();
            rmsf[atomIdx] += len * len;
        }
    }

    // compute average pos
    float minRMSF = FLT_MAX, maxRMSF = 0.0f;
    for (unsigned int i = 0; i < mol->AtomCount(); i++) {
        rmsf[i] = sqrtf(rmsf[i] / static_cast<float>(mol->FrameCount()));
        minRMSF = rmsf[i] < minRMSF ? rmsf[i] : minRMSF;
        maxRMSF = rmsf[i] > maxRMSF ? rmsf[i] : maxRMSF;
    }

    // restore current frame
    mol->SetCalltime(currentCallTime);
    mol->SetFrameID(currentFrame);
    if (!(*mol)(MolecularDataCall::CallForGetData))
        return false;

    // write RMSF to B-Factor
    mol->SetAtomBFactors(rmsf, true);
    mol->SetBFactorRange(minRMSF, maxRMSF);

    return true;
}
