/*
 * ParticleRelaxationModule.h
 *
 * Copyright (C) 2014 by S. Grottel
 * Alle Rechte vorbehalten.
 */
#include "ParticleRelaxationModule.h"
#include <algorithm>

using namespace megamol;


/*
 * datatools::ParticleRelaxationModule::ParticleRelaxationModule
 */
datatools::ParticleRelaxationModule::ParticleRelaxationModule()
        : AbstractParticleManipulator("outData", "indata")
        , dataHash(0)
        , frameId(0)
        , outDataHash(0)
        , bbox()
        , cbox() {}


/*
 * datatools::ParticleRelaxationModule::~ParticleRelaxationModule
 */
datatools::ParticleRelaxationModule::~ParticleRelaxationModule() {
    this->Release();
}


/*
 * datatools::ParticleRelaxationModule::manipulateExtent
 */
bool datatools::ParticleRelaxationModule::manipulateExtent(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {

    if ((this->frameId != inData.FrameID()) || (this->dataHash != inData.DataHash()) || (inData.DataHash() == 0)) {
        // will be updates next frame

        // spoiler input boxes, because there seems to be a problem with the view3d
        this->outDataHash++;
        this->bbox = inData.AccessBoundingBoxes().ObjectSpaceBBox();
        this->cbox = inData.AccessBoundingBoxes().ObjectSpaceClipBox();
    }

    outData = inData;
    inData.SetUnlocker(nullptr, false);

    outData.SetDataHash(this->outDataHash);
    outData.AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
    outData.AccessBoundingBoxes().SetObjectSpaceClipBox(this->cbox);
    return true;
}


/*
 * datatools::ParticleRelaxationModule::manipulateData
 */
bool datatools::ParticleRelaxationModule::manipulateData(
    geocalls::MultiParticleDataCall& outData, geocalls::MultiParticleDataCall& inData) {
    using geocalls::MultiParticleDataCall;

    if ((this->frameId != inData.FrameID()) || (this->dataHash != inData.DataHash()) || (inData.DataHash() == 0)) {
        this->frameId = inData.FrameID();
        this->dataHash = inData.DataHash();
        this->outDataHash++;
        // Data updated. Need refresh.

        // first simply copy the data
        uint64_t cnt = 0;
        for (unsigned int li = 0; li < inData.GetParticleListCount(); li++) {
            if (inData.AccessParticles(li).GetVertexDataType() == MultiParticleDataCall::Particles::VERTDATA_NONE)
                continue;
            cnt += inData.AccessParticles(li).GetCount();
        }
        this->data.EnforceSize(
            static_cast<SIZE_T>(cnt * 4 * sizeof(float))); // new particle position data (including radii)
        float* vert = this->data.As<float>();
        for (unsigned int li = 0; li < inData.GetParticleListCount(); li++) {
            MultiParticleDataCall::Particles& p = inData.AccessParticles(li);
            const uint8_t* vd = static_cast<const uint8_t*>(p.GetVertexData());
            size_t vds = p.GetVertexDataStride();
            bool isFloat = true;
            bool hasRadius = false;
            switch (p.GetVertexDataType()) {
            case MultiParticleDataCall::Particles::VERTDATA_NONE:
                continue;
            case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                hasRadius = true;
                vds = std::max<size_t>(vds, 16);
                break;
            case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                vds = std::max<size_t>(vds, 12);
                break;
            case MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
                isFloat = false;
                vds = std::max<size_t>(vds, 6);
                break;
            default:
                throw std::exception();
            }
            float gr = p.GetGlobalRadius();

            ASSERT((hasRadius && isFloat) || !hasRadius); // radius is only used with floats
            if (isFloat) {
                for (uint64_t pi = 0; pi < p.GetCount(); pi++, vert += 4, vd += vds) {
                    vert[3] = gr;
                    ::memcpy(vert, vd, sizeof(float) * (hasRadius ? 4 : 3));
                }
            } else {
                for (uint64_t pi = 0; pi < p.GetCount(); pi++, vert += 4, vd += vds) {
                    const uint16_t* v = reinterpret_cast<const uint16_t*>(vd);
                    vert[0] = static_cast<float>(v[0]);
                    vert[1] = static_cast<float>(v[1]);
                    vert[2] = static_cast<float>(v[2]);
                    vert[3] = gr;
                }
            }
        }

        // now run relaxation code

        // TODO: Implement!

        // finally compute new bounding boxes
        if (cnt > 0) {
            float* vert = this->data.As<float>();
            this->bbox.Set(vert[0], vert[1], vert[2], vert[0], vert[1], vert[2]);
            float r = vert[3];
            for (uint64_t pi = 1; pi < cnt; pi++, vert += 4) {
                this->bbox.GrowToPoint(vert[0], vert[1], vert[2]);
                r = std::max<float>(r, vert[3]);
            }
            this->cbox = this->bbox;
            this->cbox.Grow(r);

        } else {
            this->bbox = inData.AccessBoundingBoxes().ObjectSpaceBBox();
            this->cbox = inData.AccessBoundingBoxes().ObjectSpaceClipBox();
        }
    }

    outData = inData;                   // also transfers the unlocker to 'outData'
    inData.SetUnlocker(nullptr, false); // keep original data locked
                                        // original data will be unlocked through outData

    outData.SetDataHash(this->outDataHash);
    outData.AccessBoundingBoxes().SetObjectSpaceBBox(this->bbox);
    outData.AccessBoundingBoxes().SetObjectSpaceClipBox(this->cbox);
    uint64_t cnt = 0;
    for (unsigned int li = 0; li < outData.GetParticleListCount(); li++) {
        MultiParticleDataCall::Particles& p = outData.AccessParticles(li);
        if (p.GetVertexDataType() == MultiParticleDataCall::Particles::VERTDATA_NONE)
            continue;
        p.SetVertexData(MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR,
            this->data.At(static_cast<SIZE_T>(cnt * sizeof(float) * 4)));
        cnt += p.GetCount();
    }

    return true;
}
