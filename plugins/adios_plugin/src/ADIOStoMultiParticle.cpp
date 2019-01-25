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
        int stride = 3;
        cad->inquire("box");
        cad->inquire("p_count");
        // Radius
        if (cad->isInVars("radius")) {
            cad->inquire("radius");
            stride += 1;
        } else if (cad->isInVars("global_radius")) {
            cad->inquire("global_radius");
        }
        // Colors
        if (cad->isInVars("r")) {
            cad->inquire("r");
            cad->inquire("g");
            cad->inquire("b");
            cad->inquire("a");
            stride += 4;
        } else if (cad->isInVars("global_r")) {
            cad->inquire("global_r");
            cad->inquire("global_g");
            cad->inquire("global_b");
            cad->inquire("global_a");
        }		
		// Intensity
		else if (cad->isInVars("i")) {
			cad->inquire("i");
			stride += 1;
		}
        // ID
        if (cad->isInVars("id")) {
            cad->inquire("id");
            // TODO stride with int cannot work properly
            stride += 1;
        }


        if (!(*cad)(0)) {
            vislib::sys::Log::DefaultLog.WriteError("ADIOStoMultiParticle: Error during GetData");
            return false;
        }

        std::vector<float> X;
        std::vector<float> Y;
        std::vector<float> Z;

        if (cad->isInVars("xyz")) {
            X = cad->getData("xyz")->GetAsFloat();
        } else if (cad->isInVars("x") && cad->isInVars("y") && cad->isInVars("z")) {
            X = cad->getData("x")->GetAsFloat();
            Y = cad->getData("y")->GetAsFloat();
            Z = cad->getData("z")->GetAsFloat();
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
        std::vector<int> id;
		std::vector<float> intensity;

        // Radius
        if (cad->isInVars("radius")) {
            radius = cad->getData("radius")->GetAsFloat();
        } else if (cad->isInVars("global_radius")) {
            radius = cad->getData("global_radius")->GetAsFloat();
        }
        // Colors
        if (cad->isInVars("r")) {
            r = cad->getData("r")->GetAsFloat();
            g = cad->getData("g")->GetAsFloat();
            b = cad->getData("b")->GetAsFloat();
            a = cad->getData("a")->GetAsFloat();
        } else if (cad->isInVars("global_r")) {
            r = cad->getData("global_r")->GetAsFloat();
            g = cad->getData("global_g")->GetAsFloat();
            b = cad->getData("global_b")->GetAsFloat();
            a = cad->getData("global_a")->GetAsFloat();
		} else if (cad->isInVars("i")) {
			intensity = cad->getData("i")->GetAsFloat();
			// normalizing intentsity to [0,1]    
			std::vector<float>::iterator minIt = std::min_element(std::begin(intensity), std::end(intensity));
			std::vector<float>::iterator maxIt = std::max_element(std::begin(intensity), std::end(intensity));
			float minIntensity = intensity[std::distance(std::begin(intensity), minIt)];
			float maxIntensity = intensity[std::distance(std::begin(intensity), maxIt)];
			for (auto i = 0; i < intensity.size(); i++) {
				intensity[i] = (intensity[i] - minIntensity)/(maxIntensity - minIntensity);
			}
		}
        // ID
        if (cad->isInVars("id")) {
            id = cad->getData("id")->GetAsInt();
        }


        // Set bounding box
        const vislib::math::Cuboid<float> cubo(box[0], box[1], box[2], box[3], box[4], box[5]);
        mpdc->AccessBoundingBoxes().SetObjectSpaceBBox(cubo);
        mpdc->AccessBoundingBoxes().SetObjectSpaceClipBox(cubo);

        // Set particles
        const size_t particleCount = p_count[0];

        mix.clear();
        mix.resize(particleCount * stride);

        for (auto i = 0; i < particleCount; i++) {
            int pos = 0;
            if (cad->isInVars("xyz")) {
                mix[stride * i + 0] = X[3 * i + 0];
                mix[stride * i + 1] = X[3 * i + 1];
                mix[stride * i + 2] = X[3 * i + 2];
            } else {
                mix[stride * i + 0] = X[i];
                mix[stride * i + 1] = Y[i];
                mix[stride * i + 2] = Z[i];
			}
            pos += 3;
            if (cad->isInVars("radius")) {
                mix[stride * i + pos] = radius[i];
                pos += 1;
            }
            if (cad->isInVars("r")) {
                mix[stride * i + pos + 0] = r[i];
                mix[stride * i + pos + 1] = g[i];
                mix[stride * i + pos + 2] = b[i];
                mix[stride * i + pos + 3] = a[i];
			} else if (cad->isInVars("i")) {
				mix[stride * i + pos] = intensity[i];
			}
            // TODO
            // if (cad->isInVars("id")) {
        }

        mpdc->SetFrameCount(cad->getFrameCount());
        mpdc->SetDataHash(cad->getDataHash());
        mpdc->SetParticleListCount(1);
        mpdc->AccessParticles(0).SetCount(particleCount);

        auto colType = core::moldyn::SimpleSphericalParticles::COLDATA_NONE;
        auto vertType = core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ;
        auto idType = core::moldyn::SimpleSphericalParticles::IDDATA_NONE;

        if (cad->isInVars("global_radius")) {
            mpdc->AccessParticles(0).SetGlobalRadius(radius[0]);
        } else if (cad->isInVars("radius")) {
            vertType = core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZR;
		} else {
			mpdc->AccessParticles(0).SetGlobalRadius(1.0f);
		}
        if (cad->isInVars("global_r")) {
            mpdc->AccessParticles(0).SetGlobalColour(r[0] * 255, g[0] * 255, b[0] * 255, a[0] * 255);
        } else if (cad->isInVars("r")) {
            colType = core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_RGBA;
		} else if (cad->isInVars("i")) {
			colType = core::moldyn::SimpleSphericalParticles::COLDATA_FLOAT_I; 
		} else {
			mpdc->AccessParticles(0).SetGlobalColour(0.8 * 255, 0.8 * 255, 0.8 * 255, 1.0 * 255);
		}
        if (cad->isInVars("id")) {
            idType = core::moldyn::SimpleSphericalParticles::IDDATA_UINT32;
        }


        mpdc->AccessParticles(0).SetVertexData(vertType, mix.data(), stride * sizeof(float));
        mpdc->AccessParticles(0).SetColourData(colType,
            mix.data() + (vertType == core::moldyn::SimpleSphericalParticles::VERTDATA_FLOAT_XYZ ? 3 : 4),
            stride * sizeof(float));
        // mpdc->AccessParticles(0).SetIDData(idType, mix.data(), stride * sizeof(float));

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