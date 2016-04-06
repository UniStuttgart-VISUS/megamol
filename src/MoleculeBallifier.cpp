/*
 * MoleculeBallifier.cpp
 *
 * Copyright (C) 2012 by TU Dresden
 * All rights reserved.
 */

#include "stdafx.h"
#include "MoleculeBallifier.h"
#include "mmcore/moldyn/MolecularDataCall.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"


using namespace megamol;
using namespace megamol::protein_cuda;
using namespace megamol::core::moldyn;


/*
 * 
 */
MoleculeBallifier::MoleculeBallifier(void) : core::Module(), 
        outDataSlot("outData", "Sends MultiParticleDataCall data out into the world"),
        inDataSlot("inData", "Fetches MolecularDataCall data"),
        inHash(0), outHash(0), data(), colMin(0.0f), colMax(1.0f), frameOld(-1) {

    this->inDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);

    this->outDataSlot.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &MoleculeBallifier::getData);
    this->outDataSlot.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &MoleculeBallifier::getExt);
    this->MakeSlotAvailable(&this->outDataSlot);

}


/*
 * 
 */
MoleculeBallifier::~MoleculeBallifier(void) {
    this->Release();
}


/*
 * 
 */
bool MoleculeBallifier::create(void) {
    // intentionally empty
    return true;
}


/*
 * 
 */
void MoleculeBallifier::release(void) {
    // intentionally empty
}


/*
 * 
 */
bool MoleculeBallifier::getData(core::Call& c) {
    using core::moldyn::MultiParticleDataCall;
    MultiParticleDataCall *ic = dynamic_cast<MultiParticleDataCall*>(&c);
    if (ic == NULL) return false;

    MolecularDataCall *oc = this->inDataSlot.CallAs<MolecularDataCall>();
    if (oc == NULL) return false;

    // Transfer frame ID plus force flag
    oc->SetFrameID(ic->FrameID(), ic->IsFrameForced());

    if ((*oc)(0)) {
        // Rewrite data if the frame number OR the datahash has changed
        if ((this->inHash != oc->DataHash())||(this->frameOld = static_cast<int>(oc->FrameID()))) {
            this->inHash = oc->DataHash();
            this->frameOld = static_cast<int>(oc->FrameID());
            this->outHash++;

/*
da kannst Du die atom-position und den B-Faktor auslesen
AtomCount, AtomPositions, AtomBFactors
ueber den Atom-Type kannst Du auf das Type-Array zugreifen, das den Radius drin hat
*/

            unsigned int cnt = oc->AtomCount();
            this->data.AssertSize(sizeof(float) * 5 * cnt);
            float *fData = this->data.As<float>();

            for (unsigned int i = 0; i < cnt; i++, fData += 5) {
                fData[0] = oc->AtomPositions()[i * 3 + 0];
                fData[1] = oc->AtomPositions()[i * 3 + 1];
                fData[2] = oc->AtomPositions()[i * 3 + 2];

                fData[3] = oc->AtomTypes()[oc->AtomTypeIndices()[i]].Radius();

                fData[4] = oc->AtomBFactors()[i];
                if ((i == 0) || (this->colMin > fData[4])) this->colMin = fData[4];
                if ((i == 0) || (this->colMax < fData[4])) this->colMax = fData[4];
            }

        }

        ic->SetDataHash(this->outHash);
        ic->SetParticleListCount(1);
        MultiParticleDataCall::Particles& p = ic->AccessParticles(0);
        p.SetCount(this->data.GetSize() / (sizeof(float) * 5));
        if (p.GetCount() > 0) {
            p.SetVertexData(MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR, this->data.At(0), sizeof(float) * 5);
            p.SetColourData(MultiParticleDataCall::Particles::COLDATA_FLOAT_I, this->data.At(sizeof(float) * 4), sizeof(float) * 5);
            p.SetColourMapIndexValues(this->colMin, this->colMax);
        }

    }

    return true;
}


/*
 * 
 */
bool MoleculeBallifier::getExt(core::Call& c) {
    using core::moldyn::MultiParticleDataCall;
    MultiParticleDataCall *ic = dynamic_cast<MultiParticleDataCall*>(&c);
    if (ic == NULL) return false;

    MolecularDataCall *oc = this->inDataSlot.CallAs<MolecularDataCall>();
    if (oc == NULL) return false;

    if ((*oc)(1)) {
        ic->SetFrameCount(oc->FrameCount());
        ic->AccessBoundingBoxes() = oc->AccessBoundingBoxes();
        return true;
    }

    return false;
}
