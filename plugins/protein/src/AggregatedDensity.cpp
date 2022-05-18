/*
 * AggregatedDensity.cpp
 *
 * Copyright (C) 2011 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "AggregatedDensity.h"
#include "geometry_calls/VolumetricDataCall.h"
#include "mmcore/AbstractGetData3DCall.h"
#include "stdafx.h"
#include <cfloat>
#include <climits>
#include <cmath>
#include <float.h>
#include <iostream>
#include <math.h>
#include <omp.h>

#define _USE_MATH_DEFINES 1
/*
 * megamol::protein::AggregatedDensity::AggregatedDensity
 */
megamol::protein::AggregatedDensity::AggregatedDensity(void)
        : getDensitySlot("sendAggregatedDensity", "Sends the aggrated density data")
        , getZvelocitySlot("sendAggregatedZvelocity", "Sends the aggrated velocity data")
        , is_aggregated(false)
        , framecounter(0)
        , molDataCallerSlot("getMolecularData", "Connects the aggregation with molecule data storage") {

    this->getDensitySlot.SetCallback("VolumetricDataCall", "getData", &AggregatedDensity::getDensityCallback);
    this->getDensitySlot.SetCallback("VolumetricDataCall", "getExtent", &AggregatedDensity::getExtentCallback);
    this->MakeSlotAvailable(&this->getDensitySlot);


    this->getZvelocitySlot.SetCallback("VolumetricDataCall", "getData", &AggregatedDensity::getZvelocityCallback);
    this->getZvelocitySlot.SetCallback("VolumetricDataCall", "getExtent", &AggregatedDensity::getExtentCallback);
    this->MakeSlotAvailable(&this->getZvelocitySlot);

    this->molDataCallerSlot.SetCompatibleCall<megamol::protein_calls::MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->molDataCallerSlot);

    pdbfilename = "K.pdb";
    xtcfilenames.push_back("K.xtc");

    origin_x = 50;
    origin_y = 50;
    origin_z = 00;
    box_x = 50.;
    box_y = 50.;
    box_z = 50.;
    res = 1.;


    xbins = static_cast<unsigned int>(ceil(box_x / res));
    ybins = static_cast<unsigned int>(ceil(box_y / res));
    zbins = static_cast<unsigned int>(ceil(box_z / res));


    density = new float[xbins * ybins * zbins];
    velocity = new float[3 * xbins * ybins * zbins];
    memset(density, 0, xbins * ybins * zbins * sizeof(float));
    memset(velocity, 0, 3 * xbins * ybins * zbins * sizeof(float));
}


/*
 * megamol::protein::AggregatedDensity::~AggregatedDensity
 */
megamol::protein::AggregatedDensity::~AggregatedDensity(void) {}


/*
 * megamol::protein::AggregatedDensity::create
 */
bool megamol::protein::AggregatedDensity::create(void) {
    return true;
}


/*
 * megamol::protein::AggregatedDensity::release
 */
void megamol::protein::AggregatedDensity::release(void) {
    ARY_SAFE_DELETE(this->vol);
}


/*
 * megamol::protein::AggregatedDensity::getDataCallback
 */
bool megamol::protein::AggregatedDensity::getDensityCallback(megamol::core::Call& caller) {
    geocalls::VolumetricDataCall* cvd = dynamic_cast<geocalls::VolumetricDataCall*>(&caller);
    if (cvd == NULL)
        return false;

    if (!this->is_aggregated)
        if (!this->aggregate())
            return false;
    //#pragma omp parallel for
    cvd->SetDataHash(1);
    cvd->SetFrameID(0);
    auto metadata = std::make_shared<geocalls::VolumetricDataCall::Metadata>();
    metadata->Resolution[0] = xbins;
    metadata->Resolution[1] = ybins;
    metadata->Resolution[2] = zbins;
    metadata->ScalarType = geocalls::VolumetricDataCall::ScalarType::FLOATING_POINT;
    cvd->SetMetadata(metadata.get());
    cvd->SetData(this->density);

    return true;
}

/*
 * megamol::protein::AggregatedDensity::getDataCallback
 */
bool megamol::protein::AggregatedDensity::getZvelocityCallback(megamol::core::Call& caller) {
    geocalls::VolumetricDataCall* cvd = dynamic_cast<geocalls::VolumetricDataCall*>(&caller);
    if (cvd == NULL)
        return false;

    if (!this->is_aggregated)
        if (!this->aggregate())
            return false;
    //#pragma omp parallel for

    cvd->SetDataHash(1);
    cvd->SetFrameID(0);
    auto metadata = std::make_shared<geocalls::VolumetricDataCall::Metadata>();
    metadata->Resolution[0] = xbins;
    metadata->Resolution[1] = ybins;
    metadata->Resolution[2] = zbins;
    metadata->ScalarType = geocalls::VolumetricDataCall::ScalarType::FLOATING_POINT;
    cvd->SetMetadata(metadata.get());
    cvd->SetData(this->velocity);

    return true;
}

/*
 * megamol::protein::AggregatedDensity::getExtentCallback
 */
bool megamol::protein::AggregatedDensity::getExtentCallback(megamol::core::Call& caller) {
    geocalls::VolumetricDataCall* cvd = dynamic_cast<geocalls::VolumetricDataCall*>(&caller);
    if (cvd == NULL)
        return false;

    cvd->AccessBoundingBoxes().Clear();
    cvd->AccessBoundingBoxes().SetObjectSpaceBBox(
        origin_x, origin_y, origin_z, origin_x + box_x, origin_y + box_y, origin_z + box_z);
    cvd->SetDataHash(1);
    cvd->SetFrameCount(1);

    return true;
}

bool megamol::protein::AggregatedDensity::aggregate() {
    megamol::protein_calls::MolecularDataCall* mol =
        this->molDataCallerSlot.CallAs<megamol::protein_calls::MolecularDataCall>();
    if (!mol) {
        return false;
    }

    // set call time
    mol->SetCalltime(0);
    // set frame ID and call data
    mol->SetFrameID(0);

    if (!(*mol)(megamol::protein_calls::MolecularDataCall::CallForGetData))
        return false;

    // this number must remain constant!
    unsigned int n_atoms = mol->AtomCount();

    float* pos0 = new float[mol->AtomCount() * 3];
    float* vel = new float[mol->AtomCount() * 3];
    memcpy(pos0, mol->AtomPositions(), mol->AtomCount() * 3 * sizeof(float));

    for (unsigned int frame = 0; frame < mol->FrameCount(); frame++) {
        mol->SetFrameID(frame);
        if (!(*mol)(megamol::protein_calls::MolecularDataCall::CallForGetData))
            return false;
        if (mol->AtomCount() != n_atoms) {
            delete[] pos0, vel;
            return false;
        }
        const float* pos_new = mol->AtomPositions();

        for (unsigned int i = 0; i < 3 * n_atoms; i++) {
            vel[i] = pos_new[i] - pos0[i];
        }
        memcpy(pos0, mol->AtomPositions(), mol->AtomCount() * 3 * sizeof(float));
        this->aggregate_frame(pos0, vel, n_atoms);
        framecounter++;
    }

    delete[] pos0, vel;
    is_aggregated = true;
    float maxdensity = 0;
    float minvelocity = FLT_MAX;
    float maxvelocity = FLT_MIN;
    for (unsigned int i = 0; i < xbins * ybins * zbins; i++) {
        density[i] *= 1.0f / framecounter / res / res / res * 1000.0f;
        if (density[i] > 0) {
            velocity[i] *= 1.0f / density[i] / framecounter;
        } else {
            velocity[i] = 0;
        }
        maxdensity = vislib::math::Max(density[i], maxdensity);
        maxvelocity = vislib::math::Max(velocity[i], maxvelocity);
        minvelocity = vislib::math::Min(velocity[i], minvelocity);
    }
    std::cout << "The max is " << maxdensity << std::endl;
    std::cout << "The max vel is " << maxvelocity << std::endl;
    std::cout << "The min vel is " << minvelocity << std::endl;
    return true;
}

bool megamol::protein::AggregatedDensity::aggregate_frame(float* pos, float* vel, unsigned int n_atoms) {
    float x, y, z, dx, dy, dz;
    unsigned int X, Y, Z;
    float weight;
    unsigned int linear_index;
    for (unsigned int i = 0; i < 3 * n_atoms; i++) {
        x = (pos[3 * i + 0] - origin_x) / res; // in lattice constants
        X = static_cast<unsigned int>(floor(x));
        dx = x - X;
        y = (pos[3 * i + 1] - origin_y) / res; // in lattice constants
        Y = static_cast<unsigned int>(floor(y));
        dy = y - Y;
        z = (pos[3 * i + 2] - origin_z) / res; // in lattice constants
        Z = static_cast<unsigned int>(floor(z));
        dz = z - Z;

        if (X > 0 && X < xbins - 1 && Y > 0 && Y < ybins - 1 && Z > 0 && Z < zbins - 1) {
            //weight=1;
            //density[X+xbins*Y+xbins*ybins*Z]+=weight;
            weight = (1 - dx) * (1 - dy) * (1 - dz);
            linear_index = (X + 0) + (Y + 0) * xbins + (Z + 0) * xbins * ybins;
            density[linear_index] += weight;
            velocity[3 * linear_index + 0] += weight * vel[3 * i + 0];
            velocity[3 * linear_index + 1] += weight * vel[3 * i + 1];
            velocity[3 * linear_index + 2] += weight * vel[3 * i + 2];

            weight = (1 - dx) * (1 - dy) * (dz);
            linear_index = (X + 0) + (Y + 0) * xbins + (Z + 1) * xbins * ybins;
            density[linear_index] += weight;
            velocity[3 * linear_index + 0] += weight * vel[3 * i + 0];
            velocity[3 * linear_index + 1] += weight * vel[3 * i + 1];
            velocity[3 * linear_index + 2] += weight * vel[3 * i + 2];

            weight = (1 - dx) * (dy) * (1 - dz);
            linear_index = (X + 0) + (Y + 1) * xbins + (Z + 0) * xbins * ybins;
            density[linear_index] += weight;
            velocity[3 * linear_index + 0] += weight * vel[3 * i + 0];
            velocity[3 * linear_index + 1] += weight * vel[3 * i + 1];
            velocity[3 * linear_index + 2] += weight * vel[3 * i + 2];

            weight = (1 - dx) * (dy) * (dz);
            linear_index = (X + 0) + (Y + 1) * xbins + (Z + 1) * xbins * ybins;
            density[linear_index] += weight;
            velocity[3 * linear_index + 0] += weight * vel[3 * i + 0];
            velocity[3 * linear_index + 1] += weight * vel[3 * i + 1];
            velocity[3 * linear_index + 2] += weight * vel[3 * i + 2];

            weight = (dx) * (1 - dy) * (1 - dz);
            linear_index = (X + 1) + (Y + 0) * xbins + (Z + 0) * xbins * ybins;
            density[linear_index] += weight;
            velocity[3 * linear_index + 0] += weight * vel[3 * i + 0];
            velocity[3 * linear_index + 1] += weight * vel[3 * i + 1];
            velocity[3 * linear_index + 2] += weight * vel[3 * i + 2];

            weight = (dx) * (1 - dy) * (dz);
            linear_index = (X + 1) + (Y + 0) * xbins + (Z + 1) * xbins * ybins;
            density[linear_index] += weight;
            velocity[3 * linear_index + 0] += weight * vel[3 * i + 0];
            velocity[3 * linear_index + 1] += weight * vel[3 * i + 1];
            velocity[3 * linear_index + 2] += weight * vel[3 * i + 2];

            weight = (dx) * (dy) * (1 - dz);
            linear_index = (X + 1) + (Y + 1) * xbins + (Z + 0) * xbins * ybins;
            density[linear_index] += weight;
            velocity[3 * linear_index + 0] += weight * vel[3 * i + 0];
            velocity[3 * linear_index + 1] += weight * vel[3 * i + 1];
            velocity[3 * linear_index + 2] += weight * vel[3 * i + 2];

            weight = (dx) * (dy) * (dz);
            linear_index = (X + 1) + (Y + 1) * xbins + (Z + 1) * xbins * ybins;
            density[linear_index] += weight;
            velocity[3 * linear_index + 0] += weight * vel[3 * i + 0];
            velocity[3 * linear_index + 1] += weight * vel[3 * i + 1];
            velocity[3 * linear_index + 2] += weight * vel[3 * i + 2];
        }
    }
    return true;
}
