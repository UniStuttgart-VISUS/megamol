/*
 * ADIOStoMultiParticle.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ADIOStoMultiParticle.h"
#include "CallADIOSData.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "vislib/sys/Log.h"

namespace megamol {
namespace adios {

ADIOStoMultiParticle::ADIOStoMultiParticle(void)
    : core::Module()
    , mpSlot("mpSlot", "Slot to send multi particle data.")
    , adiosSlot("adiosSlot", "Slot to request ADIOS IO") {

    this->mpSlot.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(0), &ADIOStoMultiParticle::getDataCallback);
    this->mpSlot.SetCallback(core::moldyn::MultiParticleDataCall::ClassName(),
        core::moldyn::MultiParticleDataCall::FunctionName(1), &ADIOStoMultiParticle::getExtentCallback);
    this->MakeSlotAvailable(&this->mpSlot);

    this->adiosSlot.SetCompatibleCall<CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->adiosSlot);
}

ADIOStoMultiParticle::~ADIOStoMultiParticle(void) { this->Release(); }

bool ADIOStoMultiParticle::create(void) { return true; }

void ADIOStoMultiParticle::release(void) {}

bool ADIOStoMultiParticle::getDataCallback(core::Call& call) {
    core::moldyn::MultiParticleDataCall* mpdc = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&call);
    if (mpdc == nullptr) return false;

    CallADIOSData* cad = this->adiosSlot.CallAs<CallADIOSData>();
    if (cad == nullptr) return false;

    if (!(*cad)(1)) {
        vislib::sys::Log::DefaultLog.WriteError("ADIOStoMultiParticle: Error during GetHeader");
        return false;
    }
    bool dathashChanged = (mpdc->DataHash() != cad->getDataHash());
    if ((mpdc->FrameID() != currentFrame) || dathashChanged) {

        cad->setFrameIDtoLoad(mpdc->FrameID());


        auto availVars = cad->getAvailableVars();
        if (cad->isInVars("xyz")) {
            cad->inquire("xyz");
        } else if (cad->isInVars("x") && cad->isInVars("y") && cad->isInVars("z")) {
            cad->inquire("x");
            cad->inquire("y");
            cad->inquire("z");
        } else {
            vislib::sys::Log::DefaultLog.WriteError("ADIOStoMultiParticle: No particle positions found");
            return false;
        }

        cad->inquire("box");
        cad->inquire("p_count");
        // Radius
        if (cad->isInVars("radius")) {
            cad->inquire("radius");
        } else if (cad->isInVars("global_radius")) {
            cad->inquire("global_radius");
        }
        // Colors
        if (cad->isInVars("r")) {
            cad->inquire("r");
            cad->inquire("g");
            cad->inquire("b");
            cad->inquire("a");
        } else if (cad->isInVars("global_r")) {
            cad->inquire("global_r");
            cad->inquire("global_g");
            cad->inquire("global_b");
            cad->inquire("global_a");
        }
        // Intensity
        else if (cad->isInVars("i")) {
            cad->inquire("i");
        }
        // ID
        if (cad->isInVars("id")) {
            cad->inquire("id");
        }
        // plist_offset
        if (cad->isInVars("plist_offset")) {
            cad->inquire("plist_offset");
        } else {
            vislib::sys::Log::DefaultLog.WriteError("Expected plist_offset variable to be set");
            return false;
        }


        if (!(*cad)(0)) {
            vislib::sys::Log::DefaultLog.WriteError("ADIOStoMultiParticle: Error during GetData");
            return false;
        }

        std::vector<float> X;
        std::vector<float> Y;
        std::vector<float> Z;

        size_t stride = 0;
        if (cad->isInVars("xyz")) {
            X = cad->getData("xyz")->GetAsFloat();
            stride += 3 * cad->getData("xyz")->getTypeSize();
        } else if (cad->isInVars("x") && cad->isInVars("y") && cad->isInVars("z")) {
            X = cad->getData("x")->GetAsFloat();
            Y = cad->getData("y")->GetAsFloat();
            Z = cad->getData("z")->GetAsFloat();
            stride += 3 * cad->getData("x")->getTypeSize();
        } else {
            vislib::sys::Log::DefaultLog.WriteError("ADIOStoMultiParticle: No particle positions found");
            return false;
        }
        auto box = cad->getData("box")->GetAsFloat();
        auto p_count = cad->getData("p_count")->GetAsInt();
        std::vector<float> radius;
        std::vector<float> r;
        std::vector<float> g;
        std::vector<float> b;
        std::vector<float> a;
        std::vector<char> id;
        std::vector<float> intensity;

        // Radius
        if (cad->isInVars("radius")) {
            radius = cad->getData("radius")->GetAsFloat();
            stride += 3 * cad->getData("radius")->getTypeSize();
        } else if (cad->isInVars("global_radius")) {
            radius = cad->getData("global_radius")->GetAsFloat();
        }
        // Colors
        if (cad->isInVars("r")) {
            r = cad->getData("r")->GetAsFloat();
            g = cad->getData("g")->GetAsFloat();
            b = cad->getData("b")->GetAsFloat();
            a = cad->getData("a")->GetAsFloat();
            stride += 4 * cad->getData("r")->getTypeSize();
        } else if (cad->isInVars("global_r")) {
            r = cad->getData("global_r")->GetAsFloat();
            g = cad->getData("global_g")->GetAsFloat();
            b = cad->getData("global_b")->GetAsFloat();
            a = cad->getData("global_a")->GetAsFloat();
        } else if (cad->isInVars("i")) {
            intensity = cad->getData("i")->GetAsFloat();
            stride += cad->getData("i")->getTypeSize();
            // normalizing intentsity to [0,1]
            // std::vector<float>::iterator minIt = std::min_element(std::begin(intensity), std::end(intensity));
            // std::vector<float>::iterator maxIt = std::max_element(std::begin(intensity), std::end(intensity));
            // float minIntensity = intensity[std::distance(std::begin(intensity), minIt)];
            // float maxIntensity = intensity[std::distance(std::begin(intensity), maxIt)];
            // for (auto i = 0; i < intensity.size(); i++) {
            //	intensity[i] = (intensity[i] - minIntensity)/(maxIntensity - minIntensity);
            //}
        }
        // ID
        if (cad->isInVars("id")) {
            id = cad->getData("id")->GetAsChar();
            stride += cad->getData("id")->getTypeSize();
        }

        // Set bounding box
        const vislib::math::Cuboid<float> cubo(box[0], box[1], box[2], box[3], box[4], box[5]);
        mpdc->AccessBoundingBoxes().SetObjectSpaceBBox(cubo);
        mpdc->AccessBoundingBoxes().SetObjectSpaceClipBox(cubo);



        // ParticeList offset
        std::vector<unsigned long long int> plist_offset = cad->getData("plist_offset")->GetAsUInt64();

        // Set particle list count
        mpdc->SetParticleListCount(plist_offset.size());
        mix.resize(plist_offset.size());
        for (auto k = 0; k < plist_offset.size(); k++) {

            // Set particles
            const size_t particleCount = plist_offset[k];

            // Set types
            auto colType = core::moldyn::SimpleSphericalParticles::COLDATA_NONE;
            auto vertType = core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ;
            auto idType = core::moldyn::SimpleSphericalParticles::IDDATA_NONE;

            if (cad->isInVars("global_radius")) {
                mpdc->AccessParticles(k).SetGlobalRadius(radius[0]);
            } else if (cad->isInVars("radius")) {
                vertType = core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZR;
            } else {
                mpdc->AccessParticles(k).SetGlobalRadius(1.0f);
            }
            if (cad->isInVars("global_r")) {
                mpdc->AccessParticles(k).SetGlobalColour(r[0] * 255, g[0] * 255, b[0] * 255, a[0] * 255);
            } else if (cad->isInVars("r")) {
                if (cad->getData("r")->getType() == "float") {
                    colType = core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGBA;
                } else {
                    colType = core::moldyn::SimpleSphericalParticles::COLDATA_UINT8_RGBA;
                }
            } else if (cad->isInVars("i")) {
                colType = core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I;
            } else {
                mpdc->AccessParticles(k).SetGlobalColour(0.8 * 255, 0.8 * 255, 0.8 * 255, 1.0 * 255);
            }
            if (cad->isInVars("id")) {
                if (cad->getData("id")->getType() == "unsigned long long int") {
                    idType = core::moldyn::SimpleSphericalParticles::IDDATA_UINT64;
                } else if (cad->getData("id")->getType() == "unsigned int") {
                    idType = core::moldyn::SimpleSphericalParticles::IDDATA_UINT32;
                }
            }

            // Fill mmpld byte array
            mix[k].clear();
            size_t start = k == 0 ? 0 : plist_offset[k-1];
            for (size_t i = start; i < (start+particleCount); i++) {

                if (cad->isInVars("xyz")) {
                    std::vector<char> tmp_xyz = reinterpret_cast<std::vector<char>&>(X);
                    const size_t sot = cad->getData("xyz")->getTypeSize();
                    mix[k].insert(mix[k].end(), tmp_xyz.begin() + 3 * sot * i, tmp_xyz.begin() + 3 * sot * i + 3 * sot);
                } else {
                    std::vector<char> tmp_x = reinterpret_cast<std::vector<char>&>(X);
                    std::vector<char> tmp_y = reinterpret_cast<std::vector<char>&>(Y);
                    std::vector<char> tmp_z = reinterpret_cast<std::vector<char>&>(Z);
                    const size_t sot = cad->getData("x")->getTypeSize();
                    mix[k].insert(mix[k].end(), tmp_x.begin() + sot * i, tmp_x.begin() + sot * i + sot);
                    mix[k].insert(mix[k].end(), tmp_y.begin() + sot * i, tmp_y.begin() + sot * i + sot);
                    mix[k].insert(mix[k].end(), tmp_z.begin() + sot * i, tmp_z.begin() + sot * i + sot);
                }
                if (cad->isInVars("radius")) {
                    std::vector<char> tmp_radius = reinterpret_cast<std::vector<char>&>(radius);
                    const size_t sot = cad->getData("radius")->getTypeSize();
                    mix[k].insert(mix[k].end(), tmp_radius.begin() + sot * i, tmp_radius.begin() + sot * i + sot);
                }
                if (cad->isInVars("r")) {
                    std::vector<char> tmp_r = reinterpret_cast<std::vector<char>&>(r);
                    std::vector<char> tmp_g = reinterpret_cast<std::vector<char>&>(g);
                    std::vector<char> tmp_b = reinterpret_cast<std::vector<char>&>(b);
                    std::vector<char> tmp_a = reinterpret_cast<std::vector<char>&>(a);
                    const size_t sot = cad->getData("r")->getTypeSize();
                    mix[k].insert(mix[k].end(), tmp_r.begin() + sot * i, tmp_r.begin() + sot * i + sot);
                    mix[k].insert(mix[k].end(), tmp_g.begin() + sot * i, tmp_g.begin() + sot * i + sot);
                    mix[k].insert(mix[k].end(), tmp_b.begin() + sot * i, tmp_b.begin() + sot * i + sot);
                    mix[k].insert(mix[k].end(), tmp_a.begin() + sot * i, tmp_a.begin() + sot * i + sot);
                } else if (cad->isInVars("i")) {
                    std::vector<char> tmp_i = reinterpret_cast<std::vector<char>&>(intensity);
                    const size_t sot = cad->getData("i")->getTypeSize();
                    mix[k].insert(mix[k].end(), tmp_i.begin() + sot * i, tmp_i.begin() + sot * i + sot);
                }
                if (cad->isInVars("id")) {
                    const size_t sot = cad->getData("id")->getTypeSize();
                    mix[k].insert(mix[k].end(), id.begin() + sot * i, id.begin() + sot * i + sot);
                }
            }


            mpdc->AccessParticles(k).SetCount(particleCount);

            mpdc->AccessParticles(k).SetVertexData(vertType, mix[k].data(), stride);
            mpdc->AccessParticles(k).SetColourData(
                colType, mix[k].data() + core::moldyn::SimpleSphericalParticles::VertexDataSize[vertType], stride);
            mpdc->AccessParticles(k).SetIDData(idType,
                mix[k].data() + core::moldyn::SimpleSphericalParticles::VertexDataSize[vertType] +
                    core::moldyn::SimpleSphericalParticles::ColorDataSize[colType],
                stride);
        }
        mpdc->SetFrameCount(cad->getFrameCount());
        mpdc->SetDataHash(cad->getDataHash());
        currentFrame = mpdc->FrameID();
    }
    return true;
}

bool ADIOStoMultiParticle::getExtentCallback(core::Call& call) {

    core::moldyn::MultiParticleDataCall* mpdc = dynamic_cast<core::moldyn::MultiParticleDataCall*>(&call);
    if (mpdc == nullptr) return false;

    CallADIOSData* cad = this->adiosSlot.CallAs<CallADIOSData>();
    if (cad == nullptr) return false;

    if (!this->getDataCallback(call)) return false;

    return true;
}

} // end namespace adios
} // end namespace megamol