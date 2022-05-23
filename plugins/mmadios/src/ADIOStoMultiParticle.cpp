/*
 * ADIOStoMultiParticle.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "ADIOStoMultiParticle.h"
#include "geometry_calls/MultiParticleDataCall.h"
#include "mmadios/CallADIOSData.h"
#include "mmcore/utility/log/Log.h"
#include <numeric>


namespace megamol {
namespace adios {

ADIOStoMultiParticle::ADIOStoMultiParticle(void)
        : core::Module()
        , mpSlot("mpSlot", "Slot to send multi particle data.")
        , adiosSlot("adiosSlot", "Slot to request ADIOS IO") {

    this->mpSlot.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(0), &ADIOStoMultiParticle::getDataCallback);
    this->mpSlot.SetCallback(geocalls::MultiParticleDataCall::ClassName(),
        geocalls::MultiParticleDataCall::FunctionName(1), &ADIOStoMultiParticle::getExtentCallback);
    this->MakeSlotAvailable(&this->mpSlot);

    this->adiosSlot.SetCompatibleCall<CallADIOSDataDescription>();
    this->MakeSlotAvailable(&this->adiosSlot);
}

ADIOStoMultiParticle::~ADIOStoMultiParticle(void) {
    this->Release();
}

bool ADIOStoMultiParticle::create(void) {
    return true;
}

void ADIOStoMultiParticle::release(void) {}

bool ADIOStoMultiParticle::getDataCallback(core::Call& call) {
    geocalls::MultiParticleDataCall* mpdc = dynamic_cast<geocalls::MultiParticleDataCall*>(&call);
    if (mpdc == nullptr)
        return false;

    CallADIOSData* cad = this->adiosSlot.CallAs<CallADIOSData>();
    if (cad == nullptr)
        return false;

    if (!(*cad)(1)) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("ADIOStoMultiParticle: Error during GetHeader");
        return false;
    }
    bool dathashChanged = (mpdc->DataHash() != cad->getDataHash());
    if ((mpdc->FrameID() != currentFrame) || dathashChanged) {

        cad->setFrameIDtoLoad(mpdc->FrameID());

        try {
            auto availVars = cad->getAvailableVars();
            if (cad->isInVars("xyz")) {
                cad->inquireVar("xyz");
            } else if (cad->isInVars("x") && cad->isInVars("y") && cad->isInVars("z")) {
                cad->inquireVar("x");
                cad->inquireVar("y");
                cad->inquireVar("z");
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "ADIOStoMultiParticle: No particle positions found");
                return false;
            }

            cad->inquireVar("global_box");
            cad->inquireVar("count");
            // Radius
            if (cad->isInVars("radius")) {
                cad->inquireVar("radius");
            } else if (cad->isInVars("global_radius")) {
                cad->inquireVar("global_radius");
            }
            // Colors
            if (cad->isInVars("r")) {
                cad->inquireVar("r");
                cad->inquireVar("g");
                cad->inquireVar("b");
                cad->inquireVar("a");
            } else if (cad->isInVars("global_r")) {
                cad->inquireVar("global_r");
                cad->inquireVar("global_g");
                cad->inquireVar("global_b");
                cad->inquireVar("global_a");
            }
            // Intensity
            else if (cad->isInVars("i")) {
                cad->inquireVar("i");
            }
            // ID
            if (cad->isInVars("id")) {
                cad->inquireVar("id");
            }
            // list_offset
            if (cad->isInVars("list_offset")) {
                cad->inquireVar("list_offset");
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError("Expected list_offset variable to be set");
                return false;
            }
            // list_box
            if (cad->isInVars("list_box")) {
                cad->inquireVar("list_box");
            }


            if (!(*cad)(0)) {
                megamol::core::utility::log::Log::DefaultLog.WriteError("ADIOStoMultiParticle: Error during GetData");
                return false;
            }

            std::vector<unsigned char> X;
            std::vector<unsigned char> Y;
            std::vector<unsigned char> Z;

            stride = 0;
            if (cad->isInVars("xyz")) {
                X = cad->getData("xyz")->GetAsUChar();
                stride += 3 * cad->getData("xyz")->getTypeSize();
                if (cad->getData("xyz")->getTypeSize() == 4) {
                    vertType = geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ;
                } else {
                    vertType = geocalls::SimpleSphericalParticles::VERTDATA_DOUBLE_XYZ;
                }
            } else if (cad->isInVars("x") && cad->isInVars("y") && cad->isInVars("z")) {
                X = cad->getData("x")->GetAsUChar();
                Y = cad->getData("y")->GetAsUChar();
                Z = cad->getData("z")->GetAsUChar();
                stride += 3 * cad->getData("x")->getTypeSize();
                if (cad->getData("x")->getTypeSize() == 4) {
                    vertType = geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ;
                } else {
                    vertType = geocalls::SimpleSphericalParticles::VERTDATA_DOUBLE_XYZ;
                }
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "ADIOStoMultiParticle: No particle positions found");
                return false;
            }
            std::vector<float> box = cad->getData("global_box")->GetAsFloat();

            auto p_count = cad->getData("count")->GetAsUInt64();
            std::vector<unsigned char> radius;
            std::vector<unsigned char> r;
            std::vector<unsigned char> g;
            std::vector<unsigned char> b;
            std::vector<unsigned char> a;
            std::vector<unsigned char> id;
            std::vector<unsigned char> intensity;

            // list_box
            if (cad->isInVars("list_box")) {
                list_box = cad->getData("list_box")->GetAsFloat();
            }
            // Radius
            if (cad->isInVars("radius")) {
                radius = cad->getData("radius")->GetAsUChar();
                stride += 3 * cad->getData("radius")->getTypeSize();
            }
            // Colors
            if (cad->isInVars("r")) {
                r = cad->getData("r")->GetAsUChar();
                g = cad->getData("g")->GetAsUChar();
                b = cad->getData("b")->GetAsUChar();
                a = cad->getData("a")->GetAsUChar();
                stride += 4 * cad->getData("r")->getTypeSize();
            } else if (cad->isInVars("global_r")) {
                r = cad->getData("global_r")->GetAsUChar();
                g = cad->getData("global_g")->GetAsUChar();
                b = cad->getData("global_b")->GetAsUChar();
                a = cad->getData("global_a")->GetAsUChar();
            } else if (cad->isInVars("i")) {
                intensity = cad->getData("i")->GetAsUChar();
                stride += cad->getData("i")->getTypeSize();
                // normalizing intentsity to [0,1]
                // std::vector<float>::iterator minIt = std::min_element(std::begin(intensity), std::end(intensity));
                // std::vector<float>::iterator maxIt = std::max_element(std::begin(intensity), std::end(intensity));
                // float minIntensity = intensity[std::distance(std::begin(intensity), minIt)];
                // float maxIntensity = intensity[std::distance(std::begin(intensity), maxIt)];
                // for (auto i = 0; i < intensity.size(); i++) {
                //     intensity[i] = (intensity[i] - minIntensity)/(maxIntensity - minIntensity);
                //}
            }
            // ID
            if (cad->isInVars("id")) {
                id = cad->getData("id")->GetAsUChar();
                stride += cad->getData("id")->getTypeSize();
            }

            // Set bounding box
            const vislib::math::Cuboid<float> cubo(box[0], box[1], box[2], box[3], box[4], box[5]);
            mpdc->AccessBoundingBoxes().SetObjectSpaceBBox(cubo);
            mpdc->AccessBoundingBoxes().SetObjectSpaceClipBox(cubo);

            // ParticeList offset
            plist_offset = cad->getData("list_offset")->GetAsUInt64();

            // merge node offsets
            size_t count_index = 0;
            for (auto k = 0; k < plist_offset.size(); k++) {

                // Set particles
                unsigned long long int particleCount;
                if (plist_offset[k] == 0 && count_index != 0) {
                    ++count_index;
                }

                if (count_index > 0) {
                    plist_offset[k] += p_count[count_index - 1];
                }
            }

            // Set particle list count
            plist_count.reserve(plist_offset.size());
            mpdc->SetParticleListCount(plist_offset.size());
            mix.resize(plist_offset.size());
            for (auto k = 0; k < plist_offset.size(); k++) {

                unsigned long long int particleCount;

                if (k == plist_offset.size() - 1) {
                    auto const tot_count = std::accumulate(p_count.begin(), p_count.end(), uint64_t(0));
                    particleCount = tot_count - plist_offset[k];
                } else {
                    particleCount = plist_offset[k + 1] - plist_offset[k];
                }
                plist_count.emplace_back(particleCount);

                // Set types
                colType = geocalls::SimpleSphericalParticles::COLDATA_NONE;
                idType = geocalls::SimpleSphericalParticles::IDDATA_NONE;

                if (cad->isInVars("global_radius")) {
                    auto flt_radius = cad->getData("global_radius")->GetAsFloat();
                    mpdc->AccessParticles(k).SetGlobalRadius(flt_radius[0]);
                } else if (cad->isInVars("radius")) {
                    vertType = geocalls::SimpleSphericalParticles::VERTDATA_FLOAT_XYZR;
                } else {
                    mpdc->AccessParticles(k).SetGlobalRadius(1.0f);
                }
                if (cad->isInVars("global_r")) {
                    auto flt_r = reinterpret_cast<std::vector<float>&>(r);
                    auto flt_g = reinterpret_cast<std::vector<float>&>(g);
                    auto flt_b = reinterpret_cast<std::vector<float>&>(b);
                    auto flt_a = reinterpret_cast<std::vector<float>&>(a);
                    mpdc->AccessParticles(k).SetGlobalColour(r[0] * 255, g[0] * 255, b[0] * 255, a[0] * 255);
                } else if (cad->isInVars("r")) {
                    if (cad->getData("r")->getType() == "float") {
                        colType = geocalls::SimpleSphericalParticles::COLDATA_FLOAT_RGBA;
                    } else {
                        colType = geocalls::SimpleSphericalParticles::COLDATA_UINT8_RGBA;
                    }
                } else if (cad->isInVars("i")) {
                    if (cad->getData("i")->getType() == "float") {
                        colType = geocalls::SimpleSphericalParticles::COLDATA_FLOAT_I;
                    } else {
                        colType = geocalls::SimpleSphericalParticles::COLDATA_DOUBLE_I;
                    }
                } else {
                    mpdc->AccessParticles(k).SetGlobalColour(0.8 * 255, 0.8 * 255, 0.8 * 255, 1.0 * 255);
                }
                if (cad->isInVars("id")) {
                    if (cad->getData("id")->getType() == "uint64_t") {
                        idType = geocalls::SimpleSphericalParticles::IDDATA_UINT64;
                    } else if (cad->getData("id")->getType() == "uint32_t") {
                        idType = geocalls::SimpleSphericalParticles::IDDATA_UINT32;
                    }
                }

                // Fill mmpld byte array
                mix[k].clear();
                mix[k].shrink_to_fit();
                mix[k].reserve(stride * p_count[k]);

                const bool have_interleaved_pos = cad->isInVars("xyz");
                const bool have_radius = cad->isInVars("radius");
                const bool have_colors = cad->isInVars("r");
                const bool have_intensity = cad->isInVars("i");
                const bool have_ids = cad->isInVars("id");
                const size_t interleaved_pos_size = have_interleaved_pos ? cad->getData("xyz")->getTypeSize() : 0;
                const size_t pos_size = have_interleaved_pos ? 0 : cad->getData("x")->getTypeSize();
                const size_t radius_size = have_radius ? cad->getData("radius")->getTypeSize() : 0;
                const size_t col_size = have_colors ? cad->getData("r")->getTypeSize() : 0;
                const size_t intensity_size = have_intensity ? cad->getData("i")->getTypeSize() : 0;
                const size_t id_size = have_ids ? cad->getData("id")->getTypeSize() : 0;

                for (size_t i = plist_offset[k]; i < (plist_offset[k] + particleCount); i++) {

                    if (have_interleaved_pos) {
                        mix[k].insert(mix[k].end(), X.begin() + 3 * interleaved_pos_size * i,
                            X.begin() + 3 * interleaved_pos_size * (i + 1));
                    } else {
                        mix[k].insert(mix[k].end(), X.begin() + pos_size * i, X.begin() + pos_size * (i + 1));
                        mix[k].insert(mix[k].end(), Y.begin() + pos_size * i, Y.begin() + pos_size * (i + 1));
                        mix[k].insert(mix[k].end(), Z.begin() + pos_size * i, Z.begin() + pos_size * (i + 1));
                    }
                    if (have_radius) {
                        std::vector<unsigned char> tmp_radius = reinterpret_cast<std::vector<unsigned char>&>(radius);
                        mix[k].insert(mix[k].end(), tmp_radius.begin() + radius_size * i,
                            tmp_radius.begin() + radius_size * (i + 1));
                    }
                    if (have_colors) {
                        mix[k].insert(mix[k].end(), r.begin() + col_size * i, r.begin() + col_size * (i + 1));
                        mix[k].insert(mix[k].end(), g.begin() + col_size * i, g.begin() + col_size * (i + 1));
                        mix[k].insert(mix[k].end(), b.begin() + col_size * i, b.begin() + col_size * (i + 1));
                        mix[k].insert(mix[k].end(), a.begin() + col_size * i, a.begin() + col_size * (i + 1));
                    } else if (have_intensity) {
                        mix[k].insert(mix[k].end(), intensity.begin() + intensity_size * i,
                            intensity.begin() + intensity_size * (i + 1));
                    }
                    if (have_ids) {
                        mix[k].insert(mix[k].end(), id.begin() + id_size * i, id.begin() + id_size * (i + 1));
                    }
                }
            }
        } catch (std::exception ex) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "ADIOStoMultiParticle: exception while trying to use data: %s", ex.what());
        }
    }

    for (auto k = 0; k < mix.size(); k++) {
        // Set particles
        mpdc->AccessParticles(k).SetCount(plist_count[k]);

        mpdc->AccessParticles(k).SetVertexData(vertType, mix[k].data(), stride);
        mpdc->AccessParticles(k).SetColourData(
            colType, mix[k].data() + geocalls::SimpleSphericalParticles::VertexDataSize[vertType], stride);
        mpdc->AccessParticles(k).SetIDData(idType,
            mix[k].data() + geocalls::SimpleSphericalParticles::VertexDataSize[vertType] +
                geocalls::SimpleSphericalParticles::ColorDataSize[colType],
            stride);
        if (cad->isInVars("list_box")) {
            vislib::math::Cuboid<float> lbox(list_box[6 * k + 0], list_box[6 * k + 1],
                std::min(list_box[6 * k + 2], list_box[6 * k + 5]), list_box[6 * k + 3], list_box[6 * k + 4],
                std::max(list_box[6 * k + 2], list_box[6 * k + 5]));
            mpdc->AccessParticles(k).SetBBox(lbox);
        }
    }

    mpdc->SetFrameCount(cad->getFrameCount());
    mpdc->SetDataHash(cad->getDataHash());
    currentFrame = mpdc->FrameID();

    return true;
}

bool ADIOStoMultiParticle::getExtentCallback(core::Call& call) {

    geocalls::MultiParticleDataCall* mpdc = dynamic_cast<geocalls::MultiParticleDataCall*>(&call);
    if (mpdc == nullptr)
        return false;

    CallADIOSData* cad = this->adiosSlot.CallAs<CallADIOSData>();
    if (cad == nullptr)
        return false;

    if (!this->getDataCallback(call))
        return false;

    return true;
}

} // end namespace adios
} // end namespace megamol
