/*
 * ReducedSurface.cpp
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */


#include "protein/ReducedSurface.h"
#include "stdafx.h"
#include "vislib/OutOfRangeException.h"
#include "vislib/String.h"
#include "vislib/Trace.h"
#include "vislib/assert.h"
#include "vislib/sys/File.h"
#include <ctime>
#include <iostream>
#include <math.h>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;
using namespace megamol::protein_calls;

#pragma push_macro("min")
#undef min
#pragma push_macro("max")
#undef max

/*
 * ReducedSurface::ReducedSurface
 */
ReducedSurface::ReducedSurface(MolecularDataCall* mol, float probeRad)
        : molecule(mol)
        , globalRS(true)
        , zeroVec3(0, 0, 0) {
    // set the first atom index to 0
    this->firstAtomIdx = 0;
    // set the number of atoms to the total number of protein atoms
    this->numberOfAtoms = this->molecule->AtomCount();

    // set epsilon value for float-comparison
    this->epsilon = vislib::math::FLOAT_EPSILON;
    // set probe radius
    this->probeRadius = probeRad;
    // set number of cut edges to 0
    this->countCutEdges = 0;

    // copy the atom data
    if (this->molecule && this->numberOfAtoms > 0) {
        this->atoms.resize(this->molecule->AtomCount() * 4);
        for (unsigned int i = 0; i < this->molecule->AtomCount(); ++i) {
            this->atoms[4 * i + 0] = this->molecule->AtomPositions()[3 * i + 0];
            this->atoms[4 * i + 1] = this->molecule->AtomPositions()[3 * i + 1];
            this->atoms[4 * i + 2] = this->molecule->AtomPositions()[3 * i + 2];
            this->atoms[4 * i + 3] = this->molecule->AtomTypes()[this->molecule->AtomTypeIndices()[i]].Radius();
        }
    }
}

ReducedSurface::ReducedSurface(unsigned int molId, MolecularDataCall* mol, float probeRad)
        : molecule(mol)
        , globalRS(false)
        , zeroVec3(0, 0, 0) {
    // check if the chain exists
    if (molId < this->molecule->MoleculeCount()) {
        // set the first atom index
        this->firstAtomIdx = molecule->Residues()[molecule->Molecules()[molId].FirstResidueIndex()]->FirstAtomIndex();
        // set the number of atoms to the total number of protein atoms
        if ((molId + 1) < this->molecule->MoleculeCount()) {
            this->numberOfAtoms =
                molecule->Residues()[molecule->Molecules()[molId + 1].FirstResidueIndex()]->FirstAtomIndex() -
                this->firstAtomIdx;
        } else {
            this->numberOfAtoms = this->molecule->AtomCount() - this->firstAtomIdx;
        }
    } else {
        // chain index too high!
        // --> set first atom and number of atoms to zero
        this->firstAtomIdx = 0;
        this->numberOfAtoms = 0;
    }

    // set epsilon value for float-comparison
    this->epsilon = vislib::math::FLOAT_EPSILON;
    // set probe radius
    this->probeRadius = probeRad;
    // set number of cut edges to 0
    this->countCutEdges = 0;

    // compute the reduced surface
    if (this->molecule && this->numberOfAtoms > 0) {
        // this->ComputeReducedSurfaceMolecule();

        this->atoms.resize(this->molecule->AtomCount() * 4);
        for (unsigned int i = 0; i < this->molecule->AtomCount(); ++i) {
            this->atoms[4 * i + 0] = this->molecule->AtomPositions()[3 * i + 0];
            this->atoms[4 * i + 1] = this->molecule->AtomPositions()[3 * i + 1];
            this->atoms[4 * i + 2] = this->molecule->AtomPositions()[3 * i + 2];
            this->atoms[4 * i + 3] = this->molecule->AtomTypes()[this->molecule->AtomTypeIndices()[i]].Radius();
        }
    }
}


/*
 * ReducedSurface::~ReducedSurface
 */
ReducedSurface::~ReducedSurface(void) {
    // set molecular data interface to NULL
    this->molecule = NULL;
}


/*
 * Compute the reduced surface of a molecule
 */
void ReducedSurface::ComputeReducedSurface() {

    if (this->atoms.empty())
        return;

    time_t t = clock();
    time_t t_total = clock();

    // counter variables
    unsigned int cnt1, cnt2;
    // clear the RS-vertices, -edges and -faces
    for (cnt1 = 0; cnt1 < this->rsVertex.size(); ++cnt1) {
        delete this->rsVertex[cnt1];
    }
    this->rsVertex.clear();
    for (cnt1 = 0; cnt1 < this->rsEdge.size(); ++cnt1) {
        delete this->rsEdge[cnt1];
    }
    this->rsEdge.clear();
    for (cnt1 = 0; cnt1 < this->rsFace.size(); ++cnt1) {
        delete this->rsFace[cnt1];
    }
    this->rsFace.clear();
    // temporary position vector
    vislib::math::Vector<float, 3> tmpVec1;
    // temporary radius
    float radius;
    // index of the atom with the smallest x-value
    unsigned int xIdx = 0;
    // index of the atom with the smallest y-value
    unsigned int yIdx = 0;
    // index of the atom with the smallest z-value
    unsigned int zIdx = 0;
    // create the voxel map
    this->bBox = this->molecule->AccessBoundingBoxes().ObjectSpaceBBox();
    // set voxel lenght --> diameter of the probe + maximum atom diameter
    this->voxelLength = 2 * this->probeRadius + 2 * 3.0f;
    unsigned int tmpSize = (unsigned int)ceilf(this->bBox.Width() / this->voxelLength);
    this->voxelMap.clear();
    this->voxelMapProbes.clear();
    this->voxelMap.resize(tmpSize);
    this->voxelMapProbes.resize(tmpSize);
    for (cnt1 = 0; cnt1 < this->voxelMap.size(); ++cnt1) {
        this->voxelMap[cnt1].resize((unsigned int)ceilf(this->bBox.Height() / this->voxelLength));
        this->voxelMapProbes[cnt1].resize((unsigned int)ceilf(this->bBox.Height() / this->voxelLength));
        for (cnt2 = 0; cnt2 < this->voxelMap[cnt1].size(); ++cnt2) {
            this->voxelMap[cnt1][cnt2].resize((unsigned int)ceilf(this->bBox.Depth() / this->voxelLength));
            this->voxelMapProbes[cnt1][cnt2].resize((unsigned int)ceilf(this->bBox.Depth() / this->voxelLength));
        }
    }
    std::cout << "time for resizing voxel maps:  " << (double(clock() - t) / double(CLOCKS_PER_SEC)) << std::endl;
    t = clock();

    // get all molecule atom positions
    for (cnt1 = firstAtomIdx; cnt1 < (firstAtomIdx + numberOfAtoms); ++cnt1) {
        // get position of current atom
        tmpVec1.SetX(this->atoms[4 * cnt1 + 0]);
        tmpVec1.SetY(this->atoms[4 * cnt1 + 1]);
        tmpVec1.SetZ(this->atoms[4 * cnt1 + 2]);

        // get the radius of current atom
        radius = this->atoms[4 * cnt1 + 3];

        // add new RS-vertex to the list
        this->rsVertex.push_back(new RSVertex(tmpVec1, radius, cnt1));

        // add RS-vertex to voxel map cell
        this->voxelMap[(unsigned int)std::min((unsigned int)this->voxelMap.size() - 1,
            (unsigned int)std::max(0, (int)floorf((tmpVec1.GetX() - bBox.Left()) / voxelLength)))]
                      [(unsigned int)std::min((unsigned int)this->voxelMap[0].size() - 1,
                          (unsigned int)std::max(0, (int)floorf((tmpVec1.GetY() - bBox.Bottom()) / voxelLength)))]
                      [(unsigned int)std::min((unsigned int)this->voxelMap[0][0].size() - 1,
                           (unsigned int)std::max(0, (int)floorf((tmpVec1.GetZ() - bBox.Back()) / voxelLength)))]
                          .push_back(this->rsVertex.back());
        // if this is the first atom OR the x-value is larger than the current smallest x
        // --> store cnt as xIdx
        if (this->rsVertex.size() > 0 ||
            (this->rsVertex[xIdx]->GetPosition().GetX() - this->rsVertex[xIdx]->GetRadius()) >
                (this->rsVertex.back()->GetPosition().GetX() - this->rsVertex.back()->GetRadius())) {
            xIdx = (unsigned int)this->rsVertex.size() - 1;
        }
        // if this is the first atom OR the y-value is larger than the current smallest y
        // --> store cnt as yIdx
        if (this->rsVertex.size() > 0 ||
            (this->rsVertex[yIdx]->GetPosition().GetY() - this->rsVertex[yIdx]->GetRadius()) >
                (this->rsVertex.back()->GetPosition().GetY() - this->rsVertex.back()->GetRadius())) {
            yIdx = (unsigned int)this->rsVertex.size() - 1;
        }
        // if this is the first atom OR the z-value is larger than the current smallest z
        // --> store cnt as zIdx
        if (this->rsVertex.size() > 0 ||
            (this->rsVertex[zIdx]->GetPosition().GetZ() - this->rsVertex[zIdx]->GetRadius()) >
                (this->rsVertex.back()->GetPosition().GetZ() - this->rsVertex.back()->GetRadius())) {
            zIdx = (unsigned int)this->rsVertex.size() - 1;
        }
    }

    std::cout << "time for reading all atoms: " << (double(clock() - t) / double(CLOCKS_PER_SEC)) << std::endl;
    t = clock();

    // DEBUG
    /*
    for( cnt1 = 0; cnt1 < this->rsVertex.size(); ++cnt1 ) {
        this->ComputeVicinityVertex( this->rsVertex[cnt1]);
        if( this->vicinity.size() > 100 )
            std::cout << "atom " << cnt1 << " vicinity size ------> " << this->vicinity.size() << std::endl;
        else if( this->vicinity.size() > 70 )
            std::cout << "atom " << cnt1 << " vicinity size ----> " << this->vicinity.size() << std::endl;
    }
    */

    // try to find the initial RS-face
    if (!this->FindFirstRSFace(this->rsVertex[xIdx]))         // --> on the x-axis
        if (!this->FindFirstRSFace(this->rsVertex[yIdx]))     // --> on the y-axis
            if (!this->FindFirstRSFace(this->rsVertex[zIdx])) // --> on the z-axis
            {
                std::cout << "no first face found!" << std::endl;
                return; // --> if no face was found: return
            }

    std::cout << "finding initial RS-face: " << (double(clock() - t) / double(CLOCKS_PER_SEC)) << std::endl;
    t = clock();

    // for each edge of the first RS-face: find neighbours
    cnt1 = 0;
    while (cnt1 < this->rsEdge.size()) {
        this->ComputeRSFace(cnt1);
        cnt1++;
    }

    // remove all RS-edges with only one face from the list of RS-edges
    std::vector<RSEdge*> tmpRSEdge;
    for (cnt1 = 0; cnt1 < this->rsEdge.size(); ++cnt1) {
        if (this->rsEdge[cnt1]->GetFace2() != NULL) {
            tmpRSEdge.push_back(this->rsEdge[cnt1]);
        } else {
            // remove RS-edge from vertices
            this->rsEdge[cnt1]->GetVertex1()->RemoveEdge(this->rsEdge[cnt1]);
            this->rsEdge[cnt1]->GetVertex2()->RemoveEdge(this->rsEdge[cnt1]);
            // delete RS-edge
            delete this->rsEdge[cnt1];
        }
    }
    this->rsEdge = tmpRSEdge;

    std::cout << "time for treating RS-edges: " << (double(clock() - t) / double(CLOCKS_PER_SEC)) << std::endl;
    t = clock();

    // create singularity texture
    this->ComputeSingularities();

    std::cout << "find cutting probes per edge and create singularity texture... "
              << (double(clock() - t) / double(CLOCKS_PER_SEC)) << std::endl;

    std::cout << "computation of RS-surface finished." << (double(clock() - t_total) / double(CLOCKS_PER_SEC))
              << std::endl;
}


/*
 * Creates the texture for singularity handling.
 */
void ReducedSurface::ComputeSingularities() {
    /*
            time_t t = clock();
    */
    unsigned int cnt1;
    // check number of cutting probes per edge
    countCutEdges = 0;
    for (cnt1 = 0; cnt1 < this->rsEdge.size(); ++cnt1) {
        // check cutting probes only for spindle tori
        if (this->rsEdge[cnt1]->GetTorusRadius() < this->probeRadius) {
            WriteProbesCutEdge(this->rsEdge[cnt1]);
            if (this->rsEdge[cnt1]->cuttingProbes.size() > 0) {
                countCutEdges++;
            }
        } else {
            this->rsEdge[cnt1]->cuttingProbes.clear();
        }
    }

    /*
            std::cout << "Number of cutted edges: " << countCutEdges << " / " << this->rsEdge.size() << " " <<
                    ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << std::endl;
            t = clock();
    */
}


/*
 * find next face for the given edge
 */
void ReducedSurface::ComputeRSFace(unsigned int edgeIdx) {
    // check, if edgeIdx is within the bounds of the RS-edge list
    if (edgeIdx >= this->rsEdge.size())
        return;
    // get the edge from the RS-edge list
    RSEdge* edge = this->rsEdge[edgeIdx];
    // do nothing if the edge has both faces already set or if this is a free edge
    if (edge->GetFace2() != NULL || edge->GetFace1() == NULL)
        return;
    unsigned int cnt;
    int result = -1;
    // the angle between two faces
    float angle = (float)vislib::math::PI_DOUBLE * 5.0f;
    float alpha;
    // names of the variables according to: Connolly "Analytical Molecular Surface Calculation", 1983
    vislib::math::Vector<float, 3> ai = edge->GetVertex1()->GetPosition();
    vislib::math::Vector<float, 3> aj = edge->GetVertex2()->GetPosition();
    vislib::math::Vector<float, 3> pijk0 = edge->GetFace1()->GetProbeCenter();
    vislib::math::Vector<float, 3> ak, uik, tik, tjk, uijk, utb, bijk, pijk1;
    RSVertex* ak0Vertex;
    vislib::math::Vector<float, 3> ak0, uijk0, bijk0;
    float rk0;
    float dik, djk, rk, rik, rjk, wijk, hijk;
    // store the face's vertex which does not belong to the edge as vertex ak0
    if (edge->GetFace1()->GetVertex1() != edge->GetVertex1() && edge->GetFace1()->GetVertex1() != edge->GetVertex2())
        ak0Vertex = edge->GetFace1()->GetVertex1();
    else if (edge->GetFace1()->GetVertex2() != edge->GetVertex1() &&
             edge->GetFace1()->GetVertex2() != edge->GetVertex2())
        ak0Vertex = edge->GetFace1()->GetVertex2();
    else
        ak0Vertex = edge->GetFace1()->GetVertex3();
    ak0 = ak0Vertex->GetPosition();
    rk0 = ak0Vertex->GetRadius();

    float ri = edge->GetVertex1()->GetRadius();
    float rj = edge->GetVertex2()->GetRadius();
    float rp = this->probeRadius;
    float dij = (aj - ai).Length();
    vislib::math::Vector<float, 3> uij = (aj - ai) / dij;
    vislib::math::Vector<float, 3> tij = edge->GetTorusCenter();
    // float rij = edge->GetTorusRadius();
    uijk0 = edge->GetFace1()->GetFaceNormal();
    float factor, tmpFac;
    factor = 1.0;
    float gamma, beta1, beta2, dir1, dir2;
    vislib::math::Vector<float, 3> bijk0Dir, bijkDir, ak0Dir, akDir;

    // search all atoms that are in the vicinity of this edge
    this->ComputeVicinityEdge(edge);
    // do nothing if the edge has no vicinity
    if (this->vicinity.empty())
        return;

    // d of plane defined by uijk0, ai
    float d1 = ai.Dot(uijk0);
    // distance between the point ai + uijk0 and the plane definded by ai and uijk0
    float dist1;
    float dist2 = (ai + uijk0).Dot(uijk0) - d1;
    // compute bijk0 of the old face
    bijk0 = pijk0 - uijk0 * (fabs(pijk0.Dot(uijk0) - d1));
    // compute directions
    bijk0Dir = bijk0 - tij;
    ak0Dir = ak0 - tij;
    // compute the angle beta1
    beta1 = bijk0Dir.Dot(pijk0 - tij) / (bijk0Dir.Length() * (pijk0 - tij).Length());
    beta1 = acos(beta1);
    if ((bijk0Dir - ak0Dir).Length() > (bijk0Dir + ak0Dir).Length()) {
        beta1 = float(vislib::math::PI_DOUBLE) - beta1;
        dir1 = -1.0f;
    } else
        dir1 = 1.0f;

    // loop over all atoms which are in the vicinty
    for (cnt = 0; cnt < this->vicinity.size(); ++cnt) {
        ak = this->vicinity[cnt]->GetPosition();
        rk = this->vicinity[cnt]->GetRadius();
        dik = (ak - ai).Length();
        djk = (ak - aj).Length();
        // continue, if one or more of the distances are too large
        if (dij > (ri + rj + rp * 2.0f) || dik > (ri + rk + rp * 2.0f) || djk > (rj + rk + rp * 2.0f))
            continue;
        if (((ri - rj) * (ri - rj) > dij * dij) || ((ri - rk) * (ri - rk) > dik * dik) ||
            ((rj - rk) * (rj - rk) > djk * djk))
            continue;
        uik = (ak - ai) / dik;
        tik = 0.5f * (ai + ak) + 0.5f * (ak - ai) * ((ri + rp) * (ri + rp) - (rk + rp) * (rk + rp)) / (dik * dik);
        wijk = acos(uij.Dot(uik));
        uijk = uij.Cross(uik) / sin(wijk);
        utb = uijk.Cross(uij);
        // bijk = tij + utb * ( uik.Dot( tik - tij)) * pow( sin( wijk), -1.0f);
        bijk = tij + utb * (uik.Dot(tik - tij) / sin(wijk));
        if ((ri + rp) * (ri + rp) < ((bijk - ai).Length() * (bijk - ai).Length()))
            continue;
        // hijk = pow( pow( ri + rp, 2.0f) - pow( ( bijk - ai).Length(), 2.0f), 0.5f);
        hijk = pow((ri + rp) * (ri + rp) - ((bijk - ai).Length()) * ((bijk - ai).Length()), 0.5f);
        pijk1 = bijk + uijk * hijk;

        // distance between the point ak of tiangle 2 and the plane definded by ai and uijk0
        dist1 = ak.Dot(uijk0) - d1;
        // compute directions
        bijkDir = bijk - tij;
        akDir = ak - tij;

        // if the face is dual to the existing face of the edge:
        if (ak0Vertex == this->vicinity[cnt]) {
            // check if the normal is the inverted normal of ak0
            if ((uijk + uijk0).Length() < (uijk - uijk0).Length())
                tmpFac = 1.0f;
            else
                tmpFac = -1.0f;
            // compute the angle gamma
            gamma = float(2.0 * vislib::math::PI_DOUBLE);
            // compute the angle alpha
            alpha = gamma - (beta1 + beta1);
        }
        // if p2 lies on the same side of the plane as n1:
        else if ((dist1 < 0.0f && dist2 < 0.0f) || (dist1 > 0.0f && dist2 > 0.0f)) {
            // compute the potential position of the probe
            pijk1 = bijk + uijk * hijk;
            // compute the angle beta2
            beta2 = bijkDir.Dot(pijk1 - tij) / (bijkDir.Length() * (pijk1 - tij).Length());
            beta2 = acos(beta2);
            if ((bijkDir - akDir).Length() > (bijkDir + akDir).Length()) {
                beta2 = float(vislib::math::PI_DOUBLE) - beta2;
                dir2 = -1.0f;
            } else
                dir2 = 1.0f;
            gamma = ((bijk0Dir * dir1).Dot(bijkDir * dir2)) / (bijk0Dir.Length() * bijkDir.Length());
            // if gamma > 90�: | n1 - n2 | < | n1 - n2 * (-1) | --> n2, otherwise n2 * (-1)
            if ((uijk0 - uijk).Length() < (uijk0 + uijk).Length()) {
                if (gamma < 0.0f)
                    tmpFac = 1.0f;
                else
                    tmpFac = -1.0f;
            } else {
                if (gamma < 0.0f)
                    tmpFac = -1.0f;
                else
                    tmpFac = 1.0f;
            }
            // compute the angle gamma
            gamma = acos(gamma);
            // compute the angle alpha
            alpha = gamma - (beta1 + beta2);
        }
        // if p2 lies on the other side of the plane as n1:
        else // if( ( dist1 < 0 && dist2 > 0 ) || ( dist1 < 0 && dist2 > 0 ) )
        {
            // compute the potential position of the probe
            pijk1 = bijk + uijk * hijk;
            // compute the angle beta2
            beta2 = bijkDir.Dot(pijk1 - tij) / (bijkDir.Length() * (pijk1 - tij).Length());
            beta2 = acos(beta2);
            if ((bijkDir - akDir).Length() > (bijkDir + akDir).Length()) {
                beta2 = float(vislib::math::PI_DOUBLE) - beta2;
                dir2 = -1.0f;
            } else
                dir2 = 1.0f;
            gamma = ((bijk0Dir * dir1).Dot(bijkDir * dir2)) / (bijk0Dir.Length() * bijkDir.Length());
            // if gamma > 90�: | n1 - n2 | < | n1 - n2 * (-1) | --> n2, otherwise n2 * (-1)
            if ((uijk0 - uijk).Length() < (uijk0 + uijk).Length()) {
                if (gamma < 0.0f)
                    tmpFac = 1.0f;
                else
                    tmpFac = -1.0f;
            } else {
                if (gamma < 0.0f)
                    tmpFac = -1.0f;
                else
                    tmpFac = 1.0f;
            }
            // compute the angle gamma
            gamma = float(2.0 * vislib::math::PI_DOUBLE) - acos(gamma);
            // compute the angle alpha
            alpha = gamma - (beta1 + beta2);
        }
        // compute the potential position of the probe
        pijk1 = bijk + uijk * hijk * tmpFac;

        // alpha must be greater than 0
        if (alpha < epsilon) {
            // set atom with greater angle as the current angle as buried
            this->vicinity[cnt]->SetAtomBuried(true);
        } else if (alpha < angle) {
            if (result > -1) {
                // set former atom with the smallest angle as buried
                this->vicinity[result]->SetAtomBuried(true);
            }
            // set atom with the current smallest angle as not burried
            this->vicinity[cnt]->SetAtomBuried(false);
            angle = alpha;
            factor = tmpFac;
            result = cnt;
        } else {
            // set atom with greater angle as the current angle as buried
            this->vicinity[cnt]->SetAtomBuried(true);
        }
        // set vicinity atom as treated
        this->vicinity[cnt]->SetTreated();
    }

    // compute values for the result
    if (result >= 0) {
        edge->SetRotationAngle(angle * factor);
        ak = this->vicinity[result]->GetPosition();
        rk = this->vicinity[result]->GetRadius();
        dik = (ak - ai).Length();
        djk = (ak - aj).Length();
        uik = (ak - ai) / dik;
        // tik = 0.5f*( ai + ak) + 0.5f*( ak - ai) * ( pow( ri + rp, 2.0f) - pow( rk + rp, 2.0f))/pow( dik, 2.0f);
        tik = 0.5f * (ai + ak) + 0.5f * (ak - ai) * ((ri + rp) * (ri + rp) - (rk + rp) * (rk + rp)) / (dik * dik);
        // tjk = 0.5f*( aj + ak) + 0.5f*( ak - aj) * ( pow( rj + rp, 2.0f) - pow( rk + rp, 2.0f))/pow( djk, 2.0f);
        tjk = 0.5f * (aj + ak) + 0.5f * (ak - aj) * ((rj + rp) * (rj + rp) - (rk + rp) * (rk + rp)) / (djk * djk);
        // rik = 0.5f*pow( pow(ri + rk + 2.0f*rp, 2.0f) - pow( dik, 2.0f), 0.5f) * ( pow( pow( dik, 2.0f) - pow( ri -
        // rk, 2.0f), 0.5f) / dik);
        rik = 0.5f * pow((ri + rk + 2.0f * rp) * (ri + rk + 2.0f * rp) - dik * dik, 0.5f) *
              (pow(dik * dik - (ri - rk) * (ri - rk), 0.5f) / dik);
        // rjk = 0.5f*pow( pow(rj + rk + 2.0f*rp, 2.0f) - pow( djk, 2.0f), 0.5f) * ( pow( pow( djk, 2.0f) - pow( rj -
        // rk, 2.0f), 0.5f) / djk);
        rjk = 0.5f * pow((rj + rk + 2.0f * rp) * (rj + rk + 2.0f * rp) - djk * djk, 0.5f) *
              (pow(djk * djk - (rj - rk) * (rj - rk), 0.5f) / djk);
        wijk = acos(uij.Dot(uik));
        uijk = uij.Cross(uik) / sin(wijk);
        utb = uijk.Cross(uij);
        // bijk = tij + utb * ( uik.Dot( tik - tij)) * pow( sin( wijk), -1.0f);
        bijk = tij + utb * (uik.Dot(tik - tij) / sin(wijk));
        // hijk = pow( pow( ri + rp, 2.0f) - pow( ( bijk - ai).Length(), 2.0f), 0.5f);
        hijk = pow((ri + rp) * (ri + rp) - ((bijk - ai).Length()) * ((bijk - ai).Length()), 0.5f);
        pijk1 = bijk + uijk * hijk * factor;

        // pointer to a dual face of the new face
        RSFace* dualFace = NULL;
        // store the attributes of the new face
        std::vector<RSVertex*> vertsNewFace;
        vertsNewFace.push_back(edge->GetVertex1());
        if (edge->GetVertex2()->GetIndex() < edge->GetVertex1()->GetIndex())
            vertsNewFace.insert(vertsNewFace.begin(), edge->GetVertex2());
        else
            vertsNewFace.push_back(edge->GetVertex2());
        if (this->vicinity[result]->GetIndex() < vertsNewFace[0]->GetIndex()) {
            vertsNewFace.insert(vertsNewFace.begin(), this->vicinity[result]);
        } else {
            if (this->vicinity[result]->GetIndex() < vertsNewFace[1]->GetIndex())
                vertsNewFace.insert(vertsNewFace.begin() + 1, this->vicinity[result]);
            else
                vertsNewFace.push_back(this->vicinity[result]);
        }
        vislib::math::Vector<float, 3> normalNewFace = uijk * factor;
        vislib::math::Vector<float, 3> probeCenterNewFace = pijk1;
        // create first RS-edge
        RSEdge* tmpEdge1 = new RSEdge(edge->GetVertex1(), this->vicinity[result], tik, rik);
        std::vector<RSEdge*> index1, index2;
        RSFace* face = NULL;
        for (cnt = 0; cnt < this->vicinity[result]->GetEdgeCount(); ++cnt) {
            if (*(this->vicinity[result]->GetEdge(cnt)) == *tmpEdge1) {
                index1.push_back(this->vicinity[result]->GetEdge(cnt));
            }
        }

        // check, if this face already exists for edge 1
        for (cnt = 0; cnt < index1.size(); ++cnt) {
            if (index1[cnt]->GetFace1()->GetVertex1() == vertsNewFace[0] &&
                index1[cnt]->GetFace1()->GetVertex2() == vertsNewFace[1] &&
                index1[cnt]->GetFace1()->GetVertex3() == vertsNewFace[2]) {
                if ((index1[cnt]->GetFace1()->GetFaceNormal() - normalNewFace).Length() < this->epsilon)
                    face = index1[cnt]->GetFace1();
                else
                    dualFace = index1[cnt]->GetFace1();
            } else if (index1[cnt]->GetFace2() != NULL) {
                if (index1[cnt]->GetFace2()->GetVertex1() == vertsNewFace[0] &&
                    index1[cnt]->GetFace2()->GetVertex2() == vertsNewFace[1] &&
                    index1[cnt]->GetFace2()->GetVertex3() == vertsNewFace[2]) {
                    if ((index1[cnt]->GetFace2()->GetFaceNormal() - normalNewFace).Length() < this->epsilon)
                        face = index1[cnt]->GetFace2();
                    else
                        dualFace = index1[cnt]->GetFace2();
                }
            }
        }
        // create second RS-edge
        RSEdge* tmpEdge2 = new RSEdge(edge->GetVertex2(), this->vicinity[result], tjk, rjk);
        for (cnt = 0; cnt < this->vicinity[result]->GetEdgeCount(); ++cnt) {
            if (*(this->vicinity[result]->GetEdge(cnt)) == *tmpEdge2) {
                index2.push_back(this->vicinity[result]->GetEdge(cnt));
            }
        }
        // check, if this face already exists for edge 2
        for (cnt = 0; cnt < index2.size(); ++cnt) {
            if (index2[cnt]->GetFace1()->GetVertex1() == vertsNewFace[0] &&
                index2[cnt]->GetFace1()->GetVertex2() == vertsNewFace[1] &&
                index2[cnt]->GetFace1()->GetVertex3() == vertsNewFace[2]) {
                if ((index2[cnt]->GetFace1()->GetFaceNormal() - normalNewFace).Length() < this->epsilon)
                    face = index2[cnt]->GetFace1();
                else
                    dualFace = index2[cnt]->GetFace1();
            } else if (index2[cnt]->GetFace2() != NULL) {
                if (index2[cnt]->GetFace2()->GetVertex1() == vertsNewFace[0] &&
                    index2[cnt]->GetFace2()->GetVertex2() == vertsNewFace[1] &&
                    index2[cnt]->GetFace2()->GetVertex3() == vertsNewFace[2]) {
                    if ((index2[cnt]->GetFace2()->GetFaceNormal() - normalNewFace).Length() < this->epsilon)
                        face = index2[cnt]->GetFace2();
                    else
                        dualFace = index2[cnt]->GetFace2();
                }
            }
        }

        // the new face is NOT already existing:
        if (face == NULL) {
            // add the first temporary edge to the edge list and to its vertices
            this->rsEdge.push_back(tmpEdge1);
            tmpEdge1->GetVertex1()->AddEdge(tmpEdge1);
            tmpEdge1->GetVertex2()->AddEdge(tmpEdge1);
            // add the second temporary edge to the edge list and to its vertices
            this->rsEdge.push_back(tmpEdge2);
            tmpEdge2->GetVertex1()->AddEdge(tmpEdge2);
            tmpEdge2->GetVertex2()->AddEdge(tmpEdge2);
            // create new RS-face
            face = new RSFace(vertsNewFace[0], vertsNewFace[1], vertsNewFace[2], edge, tmpEdge1, tmpEdge2,
                normalNewFace, probeCenterNewFace);
            this->rsFace.push_back(face);
            // add new RS-face to its edges
            edge->SetRSFace(face);
            tmpEdge1->SetRSFace(face);
            tmpEdge2->SetRSFace(face);
            if (dualFace != NULL) {
                dualFace->SetDualFace(face);
                face->SetDualFace(dualFace);
            }
            // add probe position to voxel map cell
            face->SetProbeIndex(
                std::min((unsigned int)this->voxelMapProbes.size() - 1,
                    (unsigned int)std::max(0, (int)floorf((probeCenterNewFace.GetX() - bBox.Left()) / voxelLength))),
                std::min((unsigned int)this->voxelMapProbes[0].size() - 1,
                    (unsigned int)std::max(0, (int)floorf((probeCenterNewFace.GetY() - bBox.Bottom()) / voxelLength))),
                std::min((unsigned int)this->voxelMapProbes[0][0].size() - 1,
                    (unsigned int)std::max(0, (int)floorf((probeCenterNewFace.GetZ() - bBox.Back()) / voxelLength))));
            this->voxelMapProbes[face->GetProbeIndex().GetX()][face->GetProbeIndex().GetY()]
                                [face->GetProbeIndex().GetZ()]
                                    .push_back(face);
        } else {
            // delete temporary edges
            delete tmpEdge1;
            delete tmpEdge2;
            // set the first face of the current edge as adjacent face to the already existing face
            if (*(face->GetEdge1()) == *edge) {
                if (face->GetEdge1()->SetRSFace(edge->GetFace1())) {
                    face->GetEdge1()->SetRotationAngle(angle * (-factor));
                    if (edge->GetFace1()->GetEdge1() == edge)
                        edge->GetFace1()->SetEdge1(face->GetEdge1());
                    else if (edge->GetFace1()->GetEdge2() == edge)
                        edge->GetFace1()->SetEdge2(face->GetEdge1());
                    else
                        edge->GetFace1()->SetEdge3(face->GetEdge1());
                } else {
                    edge->SetRSFace(face);
                    // std::cout << "error1 " << std::endl;
                }
            } else if (*(face->GetEdge2()) == *edge) {
                if (face->GetEdge2()->SetRSFace(edge->GetFace1())) {
                    face->GetEdge2()->SetRotationAngle(angle * (-factor));
                    if (edge->GetFace1()->GetEdge1() == edge)
                        edge->GetFace1()->SetEdge1(face->GetEdge2());
                    else if (edge->GetFace1()->GetEdge2() == edge)
                        edge->GetFace1()->SetEdge2(face->GetEdge2());
                    else
                        edge->GetFace1()->SetEdge3(face->GetEdge2());
                } else {
                    edge->SetRSFace(face);
                    // std::cout << "error2 " << std::endl;
                }
            } else {
                if (face->GetEdge3()->SetRSFace(edge->GetFace1())) {
                    face->GetEdge3()->SetRotationAngle(angle * (-factor));
                    if (edge->GetFace1()->GetEdge1() == edge)
                        edge->GetFace1()->SetEdge1(face->GetEdge3());
                    else if (edge->GetFace1()->GetEdge2() == edge)
                        edge->GetFace1()->SetEdge2(face->GetEdge3());
                    else
                        edge->GetFace1()->SetEdge3(face->GetEdge3());
                } else {
                    edge->SetRSFace(face);
                    // std::cout << "error3 " << std::endl;
                }
            }
            if (dualFace != NULL) {
                dualFace->SetDualFace(face);
                face->SetDualFace(dualFace);
            }
        }
    }
}


/*
 * Compute the rotation angle for two probes
 */
float ReducedSurface::ComputeAngleBetweenProbes(vislib::math::Vector<float, 3> tCenter,
    vislib::math::Vector<float, 3> n1, vislib::math::Vector<float, 3> pPos, vislib::math::Vector<float, 3> pPosOld) {
    float angle;
    // the plane defined by normal n1 going though point tCenter
    float d = tCenter.Dot(n1);
    // the distance & direction of the plane normal n1
    float dist1 = (tCenter + n1).Dot(n1) - d;
    // the distance & direction of the new probe position pPos
    float dist2 = pPos.Dot(n1) - d;
    // compute the angle between the old and new probe position
    float cosinus =
        (pPosOld - tCenter).Dot(pPos - tCenter) / ((pPosOld - tCenter).Length() * (pPos - tCenter).Length());
    angle = acos(cosinus);
    // correct angle if the new probe lies in the back of the plane
    if ((dist1 < 0.0f && dist2 > 0.0f) || (dist1 > 0.0f && dist2 < 0.0f))
        angle = 2.0f * (float)vislib::math::PI_DOUBLE - angle;
    return angle;
}


/*
 * Compute the possible position of the probe in contact with three atoms
 */
bool ReducedSurface::ComputeFirstFixedProbePos(RSVertex* vI, RSVertex* vJ, RSVertex* vK) {
    // names of the variables according to: Connolly "Analytical Molecular Surface Calculation", 1983
    vislib::math::Vector<float, 3> ai = vI->GetPosition();
    vislib::math::Vector<float, 3> aj = vJ->GetPosition();
    vislib::math::Vector<float, 3> ak = vK->GetPosition();
    float ri = vI->GetRadius();
    float rj = vJ->GetRadius();
    float rk = vK->GetRadius();
    float rp = this->probeRadius;
    float dij = (aj - ai).Length();
    float dik = (ak - ai).Length();
    float djk = (ak - aj).Length();
    // return false, if one or more of the distances are too large
    if (dij > (ri + rj + rp * 2.0f) || dik > (ri + rk + rp * 2.0f) || djk > (rj + rk + rp * 2.0f))
        return false;
    if (pow(ri - rj, 2.0f) > pow(dij, 2.0f))
        return false;
    vislib::math::Vector<float, 3> uij = (aj - ai) / dij;
    vislib::math::Vector<float, 3> uik = (ak - ai) / dik;
    vislib::math::Vector<float, 3> tij =
        0.5f * (ai + aj) + 0.5f * (aj - ai) * (pow(ri + rp, 2.0f) - pow(rj + rp, 2.0f)) / pow(dij, 2.0f);
    vislib::math::Vector<float, 3> tik =
        0.5f * (ai + ak) + 0.5f * (ak - ai) * (pow(ri + rp, 2.0f) - pow(rk + rp, 2.0f)) / pow(dik, 2.0f);
    vislib::math::Vector<float, 3> tjk =
        0.5f * (aj + ak) + 0.5f * (ak - aj) * (pow(rj + rp, 2.0f) - pow(rk + rp, 2.0f)) / pow(djk, 2.0f);
    float rij = 0.5f * pow(pow(ri + rj + 2.0f * rp, 2.0f) - pow(dij, 2.0f), 0.5f) *
                (pow(pow(dij, 2.0f) - pow(ri - rj, 2.0f), 0.5f) / dij);
    float rik = 0.5f * pow(pow(ri + rk + 2.0f * rp, 2.0f) - pow(dik, 2.0f), 0.5f) *
                (pow(pow(dik, 2.0f) - pow(ri - rk, 2.0f), 0.5f) / dik);
    float rjk = 0.5f * pow(pow(rj + rk + 2.0f * rp, 2.0f) - pow(djk, 2.0f), 0.5f) *
                (pow(pow(djk, 2.0f) - pow(rj - rk, 2.0f), 0.5f) / djk);
    float wijk = acos(uij.Dot(uik));
    vislib::math::Vector<float, 3> uijk = uij.Cross(uik) / sin(wijk);
    vislib::math::Vector<float, 3> utb = uijk.Cross(uij);
    vislib::math::Vector<float, 3> bijk = tij + utb * (uik.Dot(tik - tij)) * pow(sin(wijk), -1.0f);
    if (pow(ri + rp, 2.0f) < pow((bijk - ai).Length(), 2.0f))
        return false;
    float hijk = pow(pow(ri + rp, 2.0f) - pow((bijk - ai).Length(), 2.0f), 0.5f);
    vislib::math::Vector<float, 3> pijk1 = bijk + uijk * hijk;
    vislib::math::Vector<float, 3> pijk2 = bijk - uijk * hijk;

    // check, if the probe hits a vicinity sphere
    unsigned int cnt;
    bool pos1Inter, pos2Inter;
    pos1Inter = pos2Inter = false;
    for (cnt = 0; cnt < this->vicinity.size(); ++cnt) {
        if (SphereSphereIntersection(vicinity[cnt]->GetPosition(), vicinity[cnt]->GetRadius(), pijk1, probeRadius)) {
            pos1Inter = true;
        }
        if (SphereSphereIntersection(vicinity[cnt]->GetPosition(), vicinity[cnt]->GetRadius(), pijk2, probeRadius)) {
            pos2Inter = true;
        }
    }
    if (!pos1Inter) {
        // create first RS-edge and add it to its vertices
        this->rsEdge.push_back(new RSEdge(vI, vJ, tij, rij));
        vI->AddEdge(this->rsEdge.back());
        vJ->AddEdge(this->rsEdge.back());
        // create second RS-edge and add it to its vertices
        this->rsEdge.push_back(new RSEdge(vI, vK, tik, rik));
        vI->AddEdge(this->rsEdge.back());
        vK->AddEdge(this->rsEdge.back());
        // create third RS-edge and add it to its vertices
        this->rsEdge.push_back(new RSEdge(vJ, vK, tjk, rjk));
        vJ->AddEdge(this->rsEdge.back());
        vK->AddEdge(this->rsEdge.back());

        // create first RS-face and add it to its edges
        this->rsFace.push_back(new RSFace(vI, vJ, vK, this->rsEdge[this->rsEdge.size() - 3],
            this->rsEdge[this->rsEdge.size() - 2], this->rsEdge[this->rsEdge.size() - 1], uijk, pijk1));
        // add probe position to voxel map cell
        this->rsFace.back()->SetProbeIndex(
            std::min((unsigned int)this->voxelMapProbes.size() - 1,
                (unsigned int)std::max(0, (int)floorf((pijk1.GetX() - bBox.Left()) / voxelLength))),
            std::min((unsigned int)this->voxelMapProbes[0].size() - 1,
                (unsigned int)std::max(0, (int)floorf((pijk1.GetY() - bBox.Bottom()) / voxelLength))),
            std::min((unsigned int)this->voxelMapProbes[0][0].size() - 1,
                (unsigned int)std::max(0, (int)floorf((pijk1.GetZ() - bBox.Back()) / voxelLength))));
        this->voxelMapProbes[this->rsFace.back()->GetProbeIndex().GetX()][this->rsFace.back()->GetProbeIndex().GetY()]
                            [this->rsFace.back()->GetProbeIndex().GetZ()]
                                .push_back(this->rsFace.back());

        this->rsEdge[this->rsEdge.size() - 3]->SetRSFace(this->rsFace.back());
        this->rsEdge[this->rsEdge.size() - 2]->SetRSFace(this->rsFace.back());
        this->rsEdge[this->rsEdge.size() - 1]->SetRSFace(this->rsFace.back());
        return true;
    }
    if (!pos2Inter) {
        // create first RS-edge and add it to its vertices
        this->rsEdge.push_back(new RSEdge(vI, vJ, tij, rij));
        vI->AddEdge(this->rsEdge.back());
        vJ->AddEdge(this->rsEdge.back());
        // create second RS-edge and add it to its vertices
        this->rsEdge.push_back(new RSEdge(vI, vK, tik, rik));
        vI->AddEdge(this->rsEdge.back());
        vK->AddEdge(this->rsEdge.back());
        // create third RS-edge and add it to its vertices
        this->rsEdge.push_back(new RSEdge(vJ, vK, tjk, rjk));
        vJ->AddEdge(this->rsEdge.back());
        vK->AddEdge(this->rsEdge.back());

        // create first RS-face and add it to its edges
        this->rsFace.push_back(new RSFace(vI, vJ, vK, this->rsEdge[this->rsEdge.size() - 3],
            this->rsEdge[this->rsEdge.size() - 2], this->rsEdge[this->rsEdge.size() - 1], uijk * (-1.0f), pijk2));
        // add probe position to voxel map cell
        this->rsFace.back()->SetProbeIndex(
            std::min((unsigned int)this->voxelMapProbes.size() - 1,
                (unsigned int)std::max(0, (int)floorf((pijk2.GetX() - bBox.Left()) / voxelLength))),
            std::min((unsigned int)this->voxelMapProbes[0].size() - 1,
                (unsigned int)std::max(0, (int)floorf((pijk2.GetY() - bBox.Bottom()) / voxelLength))),
            std::min((unsigned int)this->voxelMapProbes[0][0].size() - 1,
                (unsigned int)std::max(0, (int)floorf((pijk2.GetZ() - bBox.Back()) / voxelLength))));
        this->voxelMapProbes[this->rsFace.back()->GetProbeIndex().GetX()][this->rsFace.back()->GetProbeIndex().GetY()]
                            [this->rsFace.back()->GetProbeIndex().GetZ()]
                                .push_back(this->rsFace.back());

        this->rsEdge[this->rsEdge.size() - 3]->SetRSFace(this->rsFace.back());
        this->rsEdge[this->rsEdge.size() - 2]->SetRSFace(this->rsFace.back());
        this->rsEdge[this->rsEdge.size() - 1]->SetRSFace(this->rsFace.back());
        return true;
    }

    // return false if no intersection-free fixed probe position was found
    return false;
}


/*
 * Compute vicinity for an atom at position 'm' with radius 'rad'
 * TODO: this could be implemented faster!!!
 */
void ReducedSurface::ComputeVicinity(vislib::math::Vector<float, 3> m, float rad) {
    unsigned int cnt, xId, yId, zId, maxXId, maxYId, maxZId;
    // maxXId = (unsigned int)floorf( this->bBox.Width() / this->voxelLength);
    // maxYId = (unsigned int)floorf( this->bBox.Height() / this->voxelLength);
    // maxZId = (unsigned int)floorf( this->bBox.Depth() / this->voxelLength);
    maxXId = (unsigned int)this->voxelMap.size() - 1;
    maxYId = (unsigned int)this->voxelMap[0].size() - 1;
    maxZId = (unsigned int)this->voxelMap[0][0].size() - 1;
    int cntX, cntY, cntZ;

    xId = (unsigned int)std::max(0, (int)floorf((m.GetX() - bBox.Left()) / voxelLength));
    xId = (unsigned int)std::min(maxXId, xId);
    yId = (unsigned int)std::max(0, (int)floorf((m.GetY() - bBox.Bottom()) / voxelLength));
    yId = (unsigned int)std::min(maxYId, yId);
    zId = (unsigned int)std::max(0, (int)floorf((m.GetZ() - bBox.Back()) / voxelLength));
    zId = (unsigned int)std::min(maxZId, zId);

    float distance;
    // float threshold;
    // clear old vicinity indices
    this->vicinity.clear();
    // loop over all atoms to find vicinity
    for (cntX = ((xId > 0) ? (-1) : 0); cntX < ((xId < maxXId) ? 2 : 1); ++cntX) {
        for (cntY = ((yId > 0) ? (-1) : 0); cntY < ((yId < maxYId) ? 2 : 1); ++cntY) {
            for (cntZ = ((zId > 0) ? (-1) : 0); cntZ < ((zId < maxZId) ? 2 : 1); ++cntZ) {
                for (cnt = 0; cnt < this->voxelMap[xId + cntX][yId + cntY][zId + cntZ].size(); ++cnt) {
                    // compute distance
                    distance = (this->voxelMap[xId + cntX][yId + cntY][zId + cntZ][cnt]->GetPosition() - m).Length();
                    // don't check self --> continue if distance is zero
                    if (distance < epsilon)
                        continue;
                    // compute threshold
                    // threshold = this->voxelMap[xId+cntX][yId+cntY][zId+cntZ][cnt]->GetRadius() + rad
                    // + 2.0f*this->probeRadius;
                    // if distance < threshold --> add atom 'cnt' to vicinity
                    // if( distance <= threshold )
                    //{
                    //    this->vicinity.push_back( this->voxelMap[xId+cntX][yId+cntY][zId+cntZ][cnt]);
                    //}
                    this->vicinity.push_back(this->voxelMap[xId + cntX][yId + cntY][zId + cntZ][cnt]);
                }
            }
        }
    }
    /*
    unsigned int cnt;
    float distance, threshold;
    // clear old vicinity indices
    this->vicinity.clear();
    // loop over all atoms to find vicinity
    for( cnt = 0; cnt < this->rsVertex.size(); cnt++ )
    {
            // compute distance
            distance = (this->rsVertex[cnt]->GetPosition() - m).Length();
            // don't check self --> continue if distance is zero
            if( distance < epsilon ) continue;
            // compute threshold
            threshold = this->rsVertex[cnt]->GetRadius() + rad + 2.0f*this->probeRadius;
            // if distance < threshold --> add atom 'cnt' to vicinity
            if( distance <= threshold )
            {
                    this->vicinity.push_back( cnt);
            }
    }
    */
}


/*
 * Compute vicinity for the torus around edge 'idx'
 * TODO: this could be implemented faster!!!
 */
void ReducedSurface::ComputeVicinityEdge(RSEdge* edge) {
    unsigned int cnt, xId, yId, zId, maxXId, maxYId, maxZId;

    maxXId = (unsigned int)this->voxelMap.size() - 1;
    maxYId = (unsigned int)this->voxelMap[0].size() - 1;
    maxZId = (unsigned int)this->voxelMap[0][0].size() - 1;
    int cntX, cntY, cntZ;

    xId = std::min(
        maxXId, (unsigned int)std::max(0, (int)floorf((edge->GetTorusCenter().GetX() - bBox.Left()) / voxelLength)));
    yId = std::min(
        maxYId, (unsigned int)std::max(0, (int)floorf((edge->GetTorusCenter().GetY() - bBox.Bottom()) / voxelLength)));
    zId = std::min(
        maxZId, (unsigned int)std::max(0, (int)floorf((edge->GetTorusCenter().GetZ() - bBox.Back()) / voxelLength)));

    float distance, threshold;
    // clear old vicinity indices
    this->vicinity.clear();
    // loop over all atoms to find vicinity
    for (cntX = ((xId > 0) ? (-1) : 0); cntX < ((xId < maxXId) ? 2 : 1); ++cntX) {
        for (cntY = ((yId > 0) ? (-1) : 0); cntY < ((yId < maxYId) ? 2 : 1); ++cntY) {
            for (cntZ = ((zId > 0) ? (-1) : 0); cntZ < ((zId < maxZId) ? 2 : 1); ++cntZ) {
                for (cnt = 0; cnt < this->voxelMap[xId + cntX][yId + cntY][zId + cntZ].size(); ++cnt) {
                    // don't check vertices of the edge --> continue
                    if (*(this->voxelMap[xId + cntX][yId + cntY][zId + cntZ][cnt]) == *(edge->GetVertex1()) ||
                        *(this->voxelMap[xId + cntX][yId + cntY][zId + cntZ][cnt]) == *(edge->GetVertex2()))
                        continue;
                    // --> the following is not necessary, because real vicinity is checked when RS-face is computed
                    // --> but it results in a considerable speedup!
                    // compute distance
                    distance = (this->voxelMap[xId + cntX][yId + cntY][zId + cntZ][cnt]->GetPosition() -
                                edge->GetTorusCenter())
                                   .Length();
                    // compute threshold
                    threshold = this->voxelMap[xId + cntX][yId + cntY][zId + cntZ][cnt]->GetRadius() +
                                edge->GetTorusRadius() + this->probeRadius;
                    // if distance < threshold --> add atom 'cnt' to vicinity
                    if (distance <= threshold) {
                        this->vicinity.push_back(this->voxelMap[xId + cntX][yId + cntY][zId + cntZ][cnt]);
                    }
                }
            }
        }
    }
}


/*
 * Compute vicinity for atom 'idx'
 * TODO: this could be implemented faster!!!
 */
void ReducedSurface::ComputeVicinityVertex(RSVertex* vertex) {
    unsigned int cnt, xId, yId, zId, maxXId, maxYId, maxZId;
    // maxXId = (unsigned int)floorf( this->bBox.Width() / this->voxelLength);
    // maxYId = (unsigned int)floorf( this->bBox.Height() / this->voxelLength);
    // maxZId = (unsigned int)floorf( this->bBox.Depth() / this->voxelLength);
    maxXId = (unsigned int)this->voxelMap.size() - 1;
    maxYId = (unsigned int)this->voxelMap[0].size() - 1;
    maxZId = (unsigned int)this->voxelMap[0][0].size() - 1;
    int cntX, cntY, cntZ;

    xId = (unsigned int)std::max(0, (int)floorf((vertex->GetPosition().GetX() - bBox.Left()) / voxelLength));
    xId = (unsigned int)std::min(maxXId, xId);
    yId = (unsigned int)std::max(0, (int)floorf((vertex->GetPosition().GetY() - bBox.Bottom()) / voxelLength));
    yId = (unsigned int)std::min(maxYId, yId);
    zId = (unsigned int)std::max(0, (int)floorf((vertex->GetPosition().GetZ() - bBox.Back()) / voxelLength));
    zId = (unsigned int)std::min(maxZId, zId);

    float distance, threshold;
    // clear old vicinity indices
    this->vicinity.clear();
    // loop over all atoms to find vicinity
    for (cntX = ((xId > 0) ? (-1) : 0); cntX < ((xId < maxXId) ? 2 : 1); ++cntX) {
        for (cntY = ((yId > 0) ? (-1) : 0); cntY < ((yId < maxYId) ? 2 : 1); ++cntY) {
            for (cntZ = ((zId > 0) ? (-1) : 0); cntZ < ((zId < maxZId) ? 2 : 1); ++cntZ) {
                for (cnt = 0; cnt < this->voxelMap[xId + cntX][yId + cntY][zId + cntZ].size(); ++cnt) {
                    // don't check vertices of the edge --> continue
                    if (this->voxelMap[xId + cntX][yId + cntY][zId + cntZ][cnt]->GetIndex() == vertex->GetIndex())
                        continue;
                    // compute distance
                    distance =
                        (this->voxelMap[xId + cntX][yId + cntY][zId + cntZ][cnt]->GetPosition() - vertex->GetPosition())
                            .Length();
                    // compute threshold
                    threshold = this->voxelMap[xId + cntX][yId + cntY][zId + cntZ][cnt]->GetRadius() +
                                vertex->GetRadius() + 2.0f * this->probeRadius;
                    // if distance < threshold --> add atom 'cnt' to vicinity
                    if (distance <= threshold) {
                        this->vicinity.push_back(this->voxelMap[xId + cntX][yId + cntY][zId + cntZ][cnt]);
                    }
                }
            }
        }
    }
}


/*
 * Get the positions of all probes which cut a specific RS-edge.
 */
std::vector<ReducedSurface::RSFace*> ReducedSurface::GetProbesCutEdge(RSEdge* edge) {
    unsigned int cnt, xId, yId, zId, maxXId, maxYId, maxZId;
    // maxXId = (unsigned int)floorf( this->bBox.Width() / this->voxelLength);
    // maxYId = (unsigned int)floorf( this->bBox.Height() / this->voxelLength);
    // maxZId = (unsigned int)floorf( this->bBox.Depth() / this->voxelLength);
    maxXId = (unsigned int)this->voxelMapProbes.size() - 1;
    maxYId = (unsigned int)this->voxelMapProbes[0].size() - 1;
    maxZId = (unsigned int)this->voxelMapProbes[0][0].size() - 1;
    int cntX, cntY, cntZ;

    vislib::math::Vector<float, 3> v1, v2, center, probe, dir21;
    // first vertex of the edge
    v1 = edge->GetVertex1()->GetPosition();
    // second vertex of the edge
    v2 = edge->GetVertex2()->GetPosition();
    // center of the edge
    dir21 = v1 - v2;
    center = dir21 / 2.0f + v2;
    // normalize dir21
    dir21.Normalise();
    // compute voxel indices for edge center
    xId = std::min((unsigned int)std::max(0, (int)floorf((center.GetX() - bBox.Left()) / voxelLength)), maxXId);
    yId = std::min((unsigned int)std::max(0, (int)floorf((center.GetY() - bBox.Bottom()) / voxelLength)), maxYId);
    zId = std::min((unsigned int)std::max(0, (int)floorf((center.GetZ() - bBox.Back()) / voxelLength)), maxZId);

    float dist1, dist2, edgeLen, lenH;
    edgeLen = (v1 - v2).Length();
    // clear old vicinity indices
    std::vector<RSFace*> cuttingProbes;
    cuttingProbes.clear();
    // loop over all atoms to find vicinity
    for (cntX = ((xId > 0) ? (-1) : 0); cntX < ((xId < maxXId) ? 2 : 1); ++cntX) {
        for (cntY = ((yId > 0) ? (-1) : 0); cntY < ((yId < maxYId) ? 2 : 1); ++cntY) {
            for (cntZ = ((zId > 0) ? (-1) : 0); cntZ < ((zId < maxZId) ? 2 : 1); ++cntZ) {
                for (cnt = 0; cnt < this->voxelMapProbes[xId + cntX][yId + cntY][zId + cntZ].size(); ++cnt) {
                    probe = this->voxelMapProbes[xId + cntX][yId + cntY][zId + cntZ][cnt]->GetProbeCenter();
                    // distances between probe and edges
                    dist1 = (probe - v1).Length();
                    dist2 = (probe - v2).Length();
                    // compute the height
                    lenH = (probe - v2).Dot(dir21);
                    // if base of height is not on edge, the probe does not cut the edge
                    if (lenH >= edgeLen || lenH <= 0.0f)
                        continue;
                    // if the height is larger than the probe radius, the probe does not cut the edge
                    if ((dir21 * lenH + v2 - probe).Length() > this->probeRadius)
                        continue;
                    // add probe to the list of cutting probes
                    cuttingProbes.push_back(this->voxelMapProbes[xId + cntX][yId + cntY][zId + cntZ][cnt]);
                }
            }
        }
    }
    return cuttingProbes;
}


/*
 * Write the positions of all probes which cut a specific RS-edge.
 */
void ReducedSurface::WriteProbesCutEdge(RSEdge* edge) {
    unsigned int cnt, xId, yId, zId, maxXId, maxYId, maxZId;
    // maxXId = (unsigned int)floorf( this->bBox.Width() / this->voxelLength);
    // maxYId = (unsigned int)floorf( this->bBox.Height() / this->voxelLength);
    // maxZId = (unsigned int)floorf( this->bBox.Depth() / this->voxelLength);
    maxXId = (unsigned int)this->voxelMapProbes.size() - 1;
    maxYId = (unsigned int)this->voxelMapProbes[0].size() - 1;
    maxZId = (unsigned int)this->voxelMapProbes[0][0].size() - 1;
    int cntX, cntY, cntZ;

    vislib::math::Vector<float, 3> v1, v2, center, probe, dir21;
    // first vertex of the edge
    v1 = edge->GetVertex1()->GetPosition();
    // second vertex of the edge
    v2 = edge->GetVertex2()->GetPosition();
    // center of the edge
    dir21 = v1 - v2;
    center = dir21 / 2.0f + v2;
    // normalize dir21
    dir21.Normalise();
    // compute voxel indices for edge center
    xId = std::min((unsigned int)std::max(0, (int)floorf((center.GetX() - bBox.Left()) / voxelLength)), maxXId);
    yId = std::min((unsigned int)std::max(0, (int)floorf((center.GetY() - bBox.Bottom()) / voxelLength)), maxYId);
    zId = std::min((unsigned int)std::max(0, (int)floorf((center.GetZ() - bBox.Back()) / voxelLength)), maxZId);

    float dist1, dist2, edgeLen, lenH;
    edgeLen = (v1 - v2).Length();
    // clear old vicinity indices
    edge->cuttingProbes.clear();
    // loop over all atoms to find vicinity
    for (cntX = ((xId > 0) ? (-1) : 0); cntX < ((xId < maxXId) ? 2 : 1); ++cntX) {
        for (cntY = ((yId > 0) ? (-1) : 0); cntY < ((yId < maxYId) ? 2 : 1); ++cntY) {
            for (cntZ = ((zId > 0) ? (-1) : 0); cntZ < ((zId < maxZId) ? 2 : 1); ++cntZ) {
                for (cnt = 0; cnt < this->voxelMapProbes[xId + cntX][yId + cntY][zId + cntZ].size(); ++cnt) {
                    probe = this->voxelMapProbes[xId + cntX][yId + cntY][zId + cntZ][cnt]->GetProbeCenter();
                    // distances between probe and edges
                    dist1 = (probe - v1).Length();
                    dist2 = (probe - v2).Length();
                    // compute the height
                    lenH = (probe - v2).Dot(dir21);
                    // if base of height is not on edge, the probe does not cut the edge
                    if (lenH >= edgeLen || lenH <= 0.0f)
                        continue;
                    // if the height is larger than the probe radius, the probe does not cut the edge
                    if ((dir21 * lenH + v2 - probe).Length() > this->probeRadius)
                        continue;
                    // add probe to the list of cutting probes
                    edge->cuttingProbes.push_back(this->voxelMapProbes[xId + cntX][yId + cntY][zId + cntZ][cnt]);
                }
            }
        }
    }
}


/*
 * Search all RS-faces whose probe is cut by the given RS-vertex.
 */
void ReducedSurface::ComputeProbeCutVertex(RSVertex* vertex) {
    unsigned int cnt, xId, yId, zId, maxXId, maxYId, maxZId;
    // maxXId = (unsigned int)floorf( this->bBox.Width() / this->voxelLength);
    // maxYId = (unsigned int)floorf( this->bBox.Height() / this->voxelLength);
    // maxZId = (unsigned int)floorf( this->bBox.Depth() / this->voxelLength);
    maxXId = (unsigned int)this->voxelMap.size() - 1;
    maxYId = (unsigned int)this->voxelMap[0].size() - 1;
    maxZId = (unsigned int)this->voxelMap[0][0].size() - 1;
    int cntX, cntY, cntZ;

    vislib::math::Vector<float, 3> v1, probe;
    // first vertex of the edge
    v1 = vertex->GetPosition();
    // compute voxel indices for edge center
    xId = std::min((unsigned int)std::max(0, (int)floorf((v1.GetX() - bBox.Left()) / voxelLength)), maxXId);
    yId = std::min((unsigned int)std::max(0, (int)floorf((v1.GetY() - bBox.Bottom()) / voxelLength)), maxYId);
    zId = std::min((unsigned int)std::max(0, (int)floorf((v1.GetZ() - bBox.Back()) / voxelLength)), maxZId);

    float dist;
    // clear old face list
    this->cutFaces.clear();
    // loop over all atoms to find vicinity
    for (cntX = ((xId > 0) ? (-1) : 0); cntX < ((xId < maxXId) ? 2 : 1); ++cntX) {
        for (cntY = ((yId > 0) ? (-1) : 0); cntY < ((yId < maxYId) ? 2 : 1); ++cntY) {
            for (cntZ = ((zId > 0) ? (-1) : 0); cntZ < ((zId < maxZId) ? 2 : 1); ++cntZ) {
                for (cnt = 0; cnt < this->voxelMapProbes[xId + cntX][yId + cntY][zId + cntZ].size(); ++cnt) {
                    // store probe center
                    probe = this->voxelMapProbes[xId + cntX][yId + cntY][zId + cntZ][cnt]->GetProbeCenter();
                    // compute distance between probe and vertex
                    dist = (probe - v1).Length();
                    // if the distance is smaller than the two radii, the probe is cut
                    if (dist < (vertex->GetRadius() + probeRadius - epsilon)) {
                        // add RS-face to the list of cut faces
                        cutFaces.push_back(this->voxelMapProbes[xId + cntX][yId + cntY][zId + cntZ][cnt]);
                    }
                }
            }
        }
    }
}


/*
 * Test if two spheres (m1,rad1) and (m2,rad2) are intersecting.
 */
bool ReducedSurface::SphereSphereIntersection(
    vislib::math::Vector<float, 3> m1, float rad1, vislib::math::Vector<float, 3> m2, float rad2) {
    if ((m1 - m2).Length() < (rad1 + rad2 - epsilon)) {
        return true;
    }
    return false;
}


/*
 * Find the first RS-face
 */
bool ReducedSurface::FindFirstRSFace(RSVertex* vertex) {
    unsigned int cnt1, cnt2;
    // get vicinity for atom with the given index
    this->ComputeVicinityVertex(vertex);
    vertex->SetTreated();
    // search for possible RS-vertices
    for (cnt1 = 0; cnt1 < this->vicinity.size(); ++cnt1) {
        this->vicinity[cnt1]->SetTreated();
        for (cnt2 = cnt1 + 1; cnt2 < this->vicinity.size(); ++cnt2) {
            this->vicinity[cnt2]->SetTreated();
            if (this->ComputeFirstFixedProbePos(vertex, vicinity[cnt1], vicinity[cnt2])) {
                return true;
            }
        }
    }
    return false;
}


/*
 * Read the next timestep and check for differences between the atoms.
 */
bool ReducedSurface::UpdateData(const float lowerThreshold, const float upperThreshold) {
    ///////////////////////////////////////////////////////////////////
    // update changed parts
    ///////////////////////////////////////////////////////////////////

    if (this->numberOfAtoms < 0)
        return false;

    if (this->rsEdge.size() > (this->rsVertex.size() + this->rsVertex.size() + 1000)) {
        std::cout << "ERROR: too many RS-edges! " << this->rsEdge.size() << std::endl;
        return false;
    }
    // counter variables
    unsigned int cnt1, cnt2, cnt3;
    // boolean variables: store exceedance of thresholds
    bool lowerThresholdExceeded = false;
    bool upperThresholdExceeded = false;
    // set of pointers to RS-vertices whose atom positions differ more than 'lowerThreshold'
    std::set<RSVertex*> changedRSVertices;
    // set of pointers to RS-edges which has to be removed
    std::set<RSEdge*> changedRSEdges;
    // set of pointer to RS-faces which has to be removed
    std::set<RSFace*> changedRSFaces;
    // temporary vector (for intermediate result)
    vislib::math::Vector<float, 3> tmpVec1;
    // indices of the atoms with the smallest x-, y- and z-value
    unsigned int xIdx = 0;
    unsigned int yIdx = 0;
    unsigned int zIdx = 0;
    // indices of voxel map entries
    unsigned int oldVoxelMapIdxX, oldVoxelMapIdxY, oldVoxelMapIdxZ, newVoxelMapIdxX, newVoxelMapIdxY, newVoxelMapIdxZ;
    // difference between the current and the subsequent atom position
    float difference;
    // temporary vector for RS-edges
    std::vector<RSEdge*> tmpRSEdge;

    // do nothing if the number of atoms differs
    if (this->molecule->AtomCount() < (firstAtomIdx + numberOfAtoms)) {
        std::cout << "ERROR: too few atoms!" << std::endl;
        return false;
    }

    // check each atom position and update if necessary
    for (cnt1 = firstAtomIdx; cnt1 < (firstAtomIdx + numberOfAtoms); ++cnt1) {
        cnt3 = cnt1 - firstAtomIdx;
        // get position of current atom
        tmpVec1.SetX(this->molecule->AtomPositions()[cnt1 * 3 + 0]);
        tmpVec1.SetY(this->molecule->AtomPositions()[cnt1 * 3 + 1]);
        tmpVec1.SetZ(this->molecule->AtomPositions()[cnt1 * 3 + 2]);
        // compute the difference
        difference = (this->rsVertex[cnt3]->GetPosition() - tmpVec1).Length();

        // check, if difference exceeds the upper threshold
        if (difference > upperThreshold) {
            // std::cout << "INFO: upper threshold exceeded --> recompute everything new!" << std::endl;
            upperThresholdExceeded = true;
            /*
            // recompute the reduced surface
            this->ComputeReducedSurface( molecule);

            // recompute geometry
            if( this->currentRendermode == GPU_RAYCASTING )
                    this->ComputeRaycastingArrays();
            else if( this->currentRendermode == POLYGONAL_GPU )
                    this->ComputePolygonalArraysGPU( 0.5f);
            else if( this->currentRendermode == POLYGONAL )
                    this->ComputePolygonalArrays( 0.5f, 1.0f);

            // exit update function
            return true;
            */
        }

        // check, if difference exceeds the lower threshold
        if (difference > lowerThreshold) {
            // the lower threshold is exceeded
            lowerThresholdExceeded = true;
            // compute old voxel map index
            oldVoxelMapIdxX = (unsigned int)std::max(
                0, (int)floorf((this->rsVertex[cnt3]->GetPosition().GetX() - bBox.Left()) / voxelLength));
            oldVoxelMapIdxX = std::min(oldVoxelMapIdxX, (unsigned int)this->voxelMap.size() - 1);
            oldVoxelMapIdxY = (unsigned int)std::max(
                0, (int)floorf((this->rsVertex[cnt3]->GetPosition().GetY() - bBox.Bottom()) / voxelLength));
            oldVoxelMapIdxY = std::min(oldVoxelMapIdxY, (unsigned int)this->voxelMap[oldVoxelMapIdxX].size() - 1);
            oldVoxelMapIdxZ = (unsigned int)std::max(
                0, (int)floorf((this->rsVertex[cnt3]->GetPosition().GetZ() - bBox.Back()) / voxelLength));
            oldVoxelMapIdxZ =
                std::min(oldVoxelMapIdxZ, (unsigned int)this->voxelMap[oldVoxelMapIdxX][oldVoxelMapIdxY].size() - 1);
            // compute new voxel map index --> make sure the index is within bounds
            newVoxelMapIdxX =
                (unsigned int)std::max(0, (int)floorf((tmpVec1.GetX() - bBox.Left()) / this->voxelLength));
            newVoxelMapIdxX = std::min(newVoxelMapIdxX, (unsigned int)this->voxelMap.size() - 1);
            newVoxelMapIdxY =
                (unsigned int)std::max(0, (int)floorf((tmpVec1.GetY() - bBox.Bottom()) / this->voxelLength));
            newVoxelMapIdxY = std::min(newVoxelMapIdxY, (unsigned int)this->voxelMap[newVoxelMapIdxX].size() - 1);
            newVoxelMapIdxZ =
                (unsigned int)std::max(0, (int)floorf((tmpVec1.GetZ() - bBox.Back()) / this->voxelLength));
            newVoxelMapIdxZ =
                std::min(newVoxelMapIdxZ, (unsigned int)this->voxelMap[newVoxelMapIdxX][newVoxelMapIdxY].size() - 1);
            // if the new atom position lies in another voxel --> remove old and add new position
            if ((oldVoxelMapIdxX != newVoxelMapIdxX) || (oldVoxelMapIdxY != newVoxelMapIdxY) ||
                (oldVoxelMapIdxZ != newVoxelMapIdxZ)) {
                // add the rsVertex-pointer to the new voxel
                this->voxelMap[newVoxelMapIdxX][newVoxelMapIdxY][newVoxelMapIdxZ].push_back(this->rsVertex[cnt3]);
                // remove the rsVertex-pointer from the old voxel
                this->voxelMap[oldVoxelMapIdxX][oldVoxelMapIdxY][oldVoxelMapIdxZ].erase(
                    std::find(this->voxelMap[oldVoxelMapIdxX][oldVoxelMapIdxY][oldVoxelMapIdxZ].begin(),
                        this->voxelMap[oldVoxelMapIdxX][oldVoxelMapIdxY][oldVoxelMapIdxZ].end(), this->rsVertex[cnt3]));
            }
            // set new atom position
            this->rsVertex[cnt3]->SetPosition(tmpVec1);
            // set RS-vertex as not buried
            this->rsVertex[cnt3]->SetAtomBuried(false);
            // set RS-vertex as not treated
            this->rsVertex[cnt3]->SetNotTreated();
            // add the rsVertex-pointer to the list of changed atoms
            changedRSVertices.insert(this->rsVertex[cnt3]);
        }

        // if this is the first atom OR the x-value is larger than the current smallest x --> store cnt as xIdx
        if (cnt1 == 0 || (this->rsVertex[xIdx]->GetPosition().GetX() - this->rsVertex[xIdx]->GetRadius()) >
                             (this->rsVertex[cnt3]->GetPosition().GetX() - this->rsVertex[cnt3]->GetRadius())) {
            xIdx = cnt3;
        }
        // if this is the first atom OR the y-value is larger than the current smallest y --> store cnt as yIdx
        if (cnt1 == 0 || (this->rsVertex[yIdx]->GetPosition().GetY() - this->rsVertex[yIdx]->GetRadius()) >
                             (this->rsVertex[cnt3]->GetPosition().GetY() - this->rsVertex[cnt3]->GetRadius())) {
            yIdx = cnt3;
        }
        // if this is the first atom OR the z-value is larger than the current smallest z --> store cnt as zIdx
        if (cnt1 == 0 || (this->rsVertex[zIdx]->GetPosition().GetZ() - this->rsVertex[zIdx]->GetRadius()) >
                             (this->rsVertex[cnt3]->GetPosition().GetZ() - this->rsVertex[cnt3]->GetRadius())) {
            zIdx = cnt3;
        }
    }

    // std::cout << "INFO: found all changed RS-vertices (" << changedRSVertices.size() << ")" << std::endl;
    if (changedRSVertices.empty()) {
        return false;
    }

    // find all RS-edges and -faces that have contact to at least one changed RS-vertex
    std::set<RSVertex*>::iterator itVertex;
    changedRSFaces.clear();
    changedRSEdges.clear();
    for (itVertex = changedRSVertices.begin(); itVertex != changedRSVertices.end(); ++itVertex) {
        for (cnt1 = 0; cnt1 < (*itVertex)->GetEdgeCount(); ++cnt1) {
            changedRSFaces.insert((*itVertex)->GetEdge(cnt1)->GetFace1());
            changedRSFaces.insert((*itVertex)->GetEdge(cnt1)->GetFace2());
            changedRSEdges.insert((*itVertex)->GetEdge(cnt1));
        }
    }

    // find all RS-faces, whose probe is cut by a moved atom
    // TODO: this could be done in the loop above (and would be faster that way)
    for (itVertex = changedRSVertices.begin(); itVertex != changedRSVertices.end(); ++itVertex) {
        this->ComputeProbeCutVertex(*itVertex);
        // add all found RS-faces to the list of changed faces
        if (this->cutFaces.size() > 0) {
            for (cnt1 = 0; cnt1 < this->cutFaces.size(); ++cnt1) {
                changedRSFaces.insert(this->cutFaces[cnt1]);
            }
            // std::cout << "INFO: found RS-faces whose probes are cut by moved atoms (" << this->cutFaces.size() << ")"
            // << std::endl;
        }
    }
    this->cutFaces.clear();

    // std::cout << "INFO: marked RS-faces (" << changedRSFaces.size() << ") and RS-edges (" << changedRSEdges.size() <<
    // ")" << std::endl; std::cout << "INFO: total number of RS-faces (" << this->rsFace.size() << ") and RS-edges (" <<
    // this->rsEdge.size() << ")" << std::endl;

    // delete all marked RS-faces
    std::set<RSFace*>::iterator itFace;
    std::vector<RSFace*>::iterator itProbe;
    RSFace* face;
    for (itFace = changedRSFaces.begin(); itFace != changedRSFaces.end(); ++itFace) {
        face = *itFace;
        // remove RS-face from probe voxel map
        oldVoxelMapIdxX = face->GetProbeIndex().GetX();
        oldVoxelMapIdxY = face->GetProbeIndex().GetY();
        oldVoxelMapIdxZ = face->GetProbeIndex().GetZ();
        // find and remove old probe position
        itProbe = std::find(this->voxelMapProbes[oldVoxelMapIdxX][oldVoxelMapIdxY][oldVoxelMapIdxZ].begin(),
            this->voxelMapProbes[oldVoxelMapIdxX][oldVoxelMapIdxY][oldVoxelMapIdxZ].end(), face);
        if (itProbe != this->voxelMapProbes[oldVoxelMapIdxX][oldVoxelMapIdxY][oldVoxelMapIdxZ].end()) {
            this->voxelMapProbes[oldVoxelMapIdxX][oldVoxelMapIdxY][oldVoxelMapIdxZ].erase(itProbe);
            ////std::cout << "SUCCESS: probe found in voxel map! ["<< oldVoxelMapIdxX << "][" << oldVoxelMapIdxY << "]["
            ///<< oldVoxelMapIdxZ << "]" << std::endl;
        } else {
            std::cout << "ERROR: probe not found in voxel map! [" << oldVoxelMapIdxX << "][" << oldVoxelMapIdxY << "]["
                      << oldVoxelMapIdxZ << "]" << std::endl;
        }

        // remove RS-face from all RS-edges which belong to one of its three RS-vertices
        for (cnt2 = 0; cnt2 < (*itFace)->GetVertex1()->GetEdgeCount(); ++cnt2) {
            if ((*itFace)->GetVertex1()->GetEdge(cnt2)->GetFace2() == (*itFace)) {
                (*itFace)->GetVertex1()->GetEdge(cnt2)->SetRSFaces(
                    (*itFace)->GetVertex1()->GetEdge(cnt2)->GetFace1(), NULL);
            }
            if ((*itFace)->GetVertex1()->GetEdge(cnt2)->GetFace1() == (*itFace)) {
                (*itFace)->GetVertex1()->GetEdge(cnt2)->SetRSFaces(
                    (*itFace)->GetVertex1()->GetEdge(cnt2)->GetFace2(), NULL);
            }
            if ((*itFace)->GetVertex1()->GetEdge(cnt2)->GetFace1() == NULL &&
                (*itFace)->GetVertex1()->GetEdge(cnt2)->GetFace2() == NULL) {
                changedRSEdges.insert((*itFace)->GetVertex1()->GetEdge(cnt2));
            }
        }
        for (cnt2 = 0; cnt2 < (*itFace)->GetVertex2()->GetEdgeCount(); ++cnt2) {
            if ((*itFace)->GetVertex2()->GetEdge(cnt2)->GetFace2() == (*itFace)) {
                (*itFace)->GetVertex2()->GetEdge(cnt2)->SetRSFaces(
                    (*itFace)->GetVertex2()->GetEdge(cnt2)->GetFace1(), NULL);
            }
            if ((*itFace)->GetVertex2()->GetEdge(cnt2)->GetFace1() == (*itFace)) {
                (*itFace)->GetVertex2()->GetEdge(cnt2)->SetRSFaces(
                    (*itFace)->GetVertex2()->GetEdge(cnt2)->GetFace2(), NULL);
            }
            if ((*itFace)->GetVertex2()->GetEdge(cnt2)->GetFace1() == NULL &&
                (*itFace)->GetVertex2()->GetEdge(cnt2)->GetFace2() == NULL) {
                changedRSEdges.insert((*itFace)->GetVertex2()->GetEdge(cnt2));
            }
        }
        for (cnt2 = 0; cnt2 < (*itFace)->GetVertex3()->GetEdgeCount(); ++cnt2) {
            if ((*itFace)->GetVertex3()->GetEdge(cnt2)->GetFace2() == (*itFace)) {
                (*itFace)->GetVertex3()->GetEdge(cnt2)->SetRSFaces(
                    (*itFace)->GetVertex3()->GetEdge(cnt2)->GetFace1(), NULL);
            }
            if ((*itFace)->GetVertex3()->GetEdge(cnt2)->GetFace1() == (*itFace)) {
                (*itFace)->GetVertex3()->GetEdge(cnt2)->SetRSFaces(
                    (*itFace)->GetVertex3()->GetEdge(cnt2)->GetFace2(), NULL);
            }
            if ((*itFace)->GetVertex3()->GetEdge(cnt2)->GetFace1() == NULL &&
                (*itFace)->GetVertex3()->GetEdge(cnt2)->GetFace2() == NULL) {
                changedRSEdges.insert((*itFace)->GetVertex3()->GetEdge(cnt2));
            }
        }

        // delete RS-face
        itProbe = std::find(this->rsFace.begin(), this->rsFace.end(), (*itFace));
        if (itProbe != this->rsFace.end()) {
            delete (*itProbe);
            this->rsFace.erase(itProbe);
        } else {
            std::cout << "ERROR: RS-Face not found in list of RS-faces!" << std::endl;
        }
    }

    // std::cout << "INFO: deleted RS-faces" << std::endl;

    // remove all changed RS-edges from the list of RS-edges
    std::set<RSEdge*>::iterator itEdge;
    std::vector<RSEdge*>::iterator itDelEdge;
    for (itEdge = changedRSEdges.begin(); itEdge != changedRSEdges.end(); ++itEdge) {
        // remove RS-edge from its two RS-vertices
        (*itEdge)->GetVertex1()->RemoveEdge((*itEdge));
        (*itEdge)->GetVertex2()->RemoveEdge((*itEdge));
        // delete RS-edge
        itDelEdge = std::find(this->rsEdge.begin(), this->rsEdge.end(), (*itEdge));
        if (itDelEdge != this->rsEdge.end()) {
            delete (*itDelEdge);
            this->rsEdge.erase(itDelEdge);
        } else {
            // std::cout << "ERROR: RS-edge not found in list of RS-edges!" << std::endl;
        }
    }
    // std::cout << "INFO: number of RS-edges after deletion: " << this->rsEdge.size() << std::endl;

    // std::cout << "INFO: new number of RS-faces (" << this->rsFace.size() << ") and RS-edges (" << this->rsEdge.size()
    // << ")" << std::endl;

    // std::cout << "INFO: time for updating data: " << ( double( clock() - t) / double( CLOCKS_PER_SEC) ) << std::endl;
    // t = clock();

    //////////////////////////////////////////////////////////////////////////////
    // deleted all changes partes, ready for recomputation
    //////////////////////////////////////////////////////////////////////////////
    if (lowerThresholdExceeded) {
        // check, if rsEdge is empty --> if true, find first face
        if (this->rsEdge.empty()) {
            // try to find the initial RS-face
            if (!this->FindFirstRSFace(this->rsVertex[xIdx]))         // --> on the x-axis
                if (!this->FindFirstRSFace(this->rsVertex[yIdx]))     // --> on the y-axis
                    if (!this->FindFirstRSFace(this->rsVertex[zIdx])) // --> on the z-axis
                        return false;                                 // --> if no face was found: return
        }

        // std::cout << "INFO: computing new RS-faces from old RS-edges..." << std::endl;

        // for each edge: find neighbours
        cnt1 = 0;
        while (cnt1 < this->rsEdge.size()) {
            this->ComputeRSFace(cnt1);
            cnt1++;
        }

        // std::cout << "INFO: computed new RS-faces from old RS-edges" << std::endl;

        // remove all RS-edges with only one face from the list of RS-edges
        std::vector<RSEdge*> tmpRSEdge;
        for (cnt1 = 0; cnt1 < this->rsEdge.size(); ++cnt1) {
            if (this->rsEdge[cnt1]->GetFace2() != NULL)
                tmpRSEdge.push_back(this->rsEdge[cnt1]);
            else {
                // remove RS-edge from vertices
                this->rsEdge[cnt1]->GetVertex1()->RemoveEdge(this->rsEdge[cnt1]);
                this->rsEdge[cnt1]->GetVertex2()->RemoveEdge(this->rsEdge[cnt1]);
                // delete RS-edge
                delete this->rsEdge[cnt1];
            }
        }
        this->rsEdge = tmpRSEdge;
        // std::cout << "INFO: deleted RS-edges with only one face" << std::endl;

        // recompute cutting probes and singularity texture
        this->ComputeSingularities();

        return true;
    }

    // std::cout << "INFO: update data finished" << std::endl;

    return false;
}


///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Data Structure Classes                                                    //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////


ReducedSurface::RSVertex::RSVertex(vislib::math::Vector<float, 3> pos, float rad, unsigned int atomIdx) {
    position = pos;
    radius = rad;
    idx = atomIdx;
    buried = true;
    edgeList.clear();
    treated = false;
}


ReducedSurface::RSVertex& ReducedSurface::RSVertex::operator=(const RSVertex& rhs) {
    position = rhs.position;
    radius = rhs.radius;
    idx = rhs.idx;
    edgeList = rhs.edgeList;
    treated = rhs.treated;
    buried = rhs.buried;

    return *this;
}


/*
 * Remove RS-edge from edge list of RS-vertex
 */
void ReducedSurface::RSVertex::RemoveEdge(RSEdge* edge) {
    std::vector<RSEdge*>::iterator edgeIt = std::find(this->edgeList.begin(), this->edgeList.end(), edge);
    if (edgeIt != this->edgeList.end()) {
        this->edgeList.erase(edgeIt);
    }
    if (this->edgeList.empty()) {
        this->SetAtomBuried(false);
    }
}


/*
 * RSEdge ctor
 */
ReducedSurface::RSEdge::RSEdge(RSVertex* v1, RSVertex* v2, vislib::math::Vector<float, 3> tCenter, float tRad) {
    if (v1->GetIndex() < v2->GetIndex()) {
        this->vert1 = v1;
        this->vert2 = v2;
    } else {
        this->vert1 = v2;
        this->vert2 = v1;
    }
    this->face1 = NULL;
    this->face2 = NULL;
    this->torusCenter = tCenter;
    this->torusRadius = tRad;
    this->rotationAngle = 0.0f;
    this->cuttingProbes.clear();
}


/*
 * RSEdge dtor
 */
ReducedSurface::RSEdge::~RSEdge() {}


ReducedSurface::RSEdge& ReducedSurface::RSEdge::operator=(const RSEdge& rhs) {
    this->vert1 = rhs.vert1;
    this->vert2 = rhs.vert2;
    this->face1 = rhs.face1;
    this->face2 = rhs.face2;
    this->torusCenter = rhs.torusCenter;
    this->torusRadius = rhs.torusRadius;
    this->rotationAngle = rhs.rotationAngle;
    this->cuttingProbes = rhs.cuttingProbes;
    this->texCoordX = rhs.texCoordX;
    this->texCoordY = rhs.texCoordY;

    return *this;
}


bool ReducedSurface::RSEdge::operator==(const RSEdge& rhs) const {
    return (*(this->vert1) == *(rhs.vert1)) && (*(this->vert2) == *(rhs.vert2));
}


bool ReducedSurface::RSEdge::SetRSFace(RSFace* f) {
    // a edge may not belong to more than two faces
    if (this->face1 == NULL)
        this->face1 = f;
    else if (this->face2 == NULL)
        this->face2 = f;
    else if (this->face2 != f && this->face1 != f)
        return false;
    return true;
}


/*
 * RSFace ctor
 */
ReducedSurface::RSFace::RSFace(RSVertex* v1, RSVertex* v2, RSVertex* v3, RSEdge* e1, RSEdge* e2, RSEdge* e3,
    vislib::math::Vector<float, 3> norm, vislib::math::Vector<float, 3> pCenter) {
    // vector used for sorting the RS-vertices
    std::vector<RSVertex*> verts;
    verts.push_back(v1);
    if (v2->GetIndex() < v1->GetIndex())
        verts.insert(verts.begin(), v2);
    else
        verts.push_back(v2);
    if (v3->GetIndex() < verts[0]->GetIndex()) {
        verts.insert(verts.begin(), v3);
    } else {
        if (v3->GetIndex() < verts[1]->GetIndex())
            verts.insert(verts.begin() + 1, v3);
        else
            verts.push_back(v3);
    }
    // assign RS-vertices
    this->vert1 = verts[0];
    this->vert2 = verts[1];
    this->vert3 = verts[2];
    // assign RS-edges
    this->edge1 = e1;
    this->edge2 = e2;
    this->edge3 = e3;
    // assign face normal
    this->normal = norm;
    // assign probe center
    this->probeCenter = pCenter;
    // set the dual face to NULL
    this->dualFace = NULL;

    toDelete = false;
}


/*
 * RSFace dtor
 */
ReducedSurface::RSFace::~RSFace() {}


/*
 * RSFace operator=
 */
ReducedSurface::RSFace& ReducedSurface::RSFace::operator=(const RSFace& rhs) {
    this->vert1 = rhs.vert1;
    this->vert2 = rhs.vert2;
    this->vert3 = rhs.vert3;
    this->edge1 = rhs.edge1;
    this->edge2 = rhs.edge2;
    this->edge3 = rhs.edge3;
    this->normal = rhs.normal;
    this->probeCenter = rhs.probeCenter;
    this->dualFace = rhs.dualFace;
    this->probeIdx = rhs.probeIdx;
    this->toDelete = false;

    return *this;
}


/*
 * RSFace operator==
 */
bool ReducedSurface::RSFace::operator==(const RSFace& rhs) const {
    return (*(this->vert1) == *(rhs.vert1)) && (*(this->vert2) == *(rhs.vert2)) && (*(this->vert3) == *(rhs.vert3));
}
