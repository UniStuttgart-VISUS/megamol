/*
 * MSMSGenus0Generator.cpp
 *
 * Copyright (C) 2015 by Michael Krone
 * Copyright (C) 2015 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "MSMSGenus0Generator.h"
#include <fstream>
#include "Color.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "protein_calls/MolecularDataCall.h"
#include "protein_calls/PerAtomFloatCall.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"
#include "vislib/assert.h"
#include "vislib/sys/ASCIIFileBuffer.h"
#include "vislib/sys/Log.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::geocalls;
using namespace megamol::protein_calls;
using namespace megamol::protein;


/*
 * MSMSGenus0Generator::MSMSGenus0Generator
 */
MSMSGenus0Generator::MSMSGenus0Generator(void)
    : core::Module()
    , bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f)
    , datahash(0)
    , getDataSlot("getdata", "The slot publishing the loaded data")
    , molDataSlot("moldata", "The slot requesting molecular data")
    , bsDataSlot("getBindingSites", "The slot requesting binding site data")
    , perAtomDataSlot("getPerAtomData", "The slot requesting per-atom data")
    , filenameSlot("filename", "The path to the file to load (without file extension)")
    , msmsDetailParam("MSMS_detail", "The detail level for the (optional) MSMS computation")
    , msmsStartingRadiusParam("MSMS_startingRadius", "The probe radius for the (optional) MSMS computation")
    , msmsMaxTryNumParam("MSMS_maxTryNum", "The maximum number of trys to get a proper genus 0 mesh")
    , msmsStepSizeParam("MSMS_stepSize", "The size of the increment of the radius to reach a genus 0 mesh")
    , colorTableFileParam("color::colorTableFilename", "The filename of the color table.")
    , coloringModeParam0("color::coloringMode0", "The first coloring mode.")
    , coloringModeParam1("color::coloringMode1", "The second coloring mode.")
    , colorWeightParam("color::colorWeighting", "The weighting of the two coloring modes.")
    , minGradColorParam("color::minGradColor", "Color for min value for gradient coloring")
    , midGradColorParam("color::midGradColor", "Color for mid value for gradient coloring")
    , maxGradColorParam("color::maxGradColor", "Color for max value for gradient coloring")
    , prevTime(-1) {
    // the data out slot
    this->getDataSlot.SetCallback(CallTriMeshData::ClassName(), "GetData", &MSMSGenus0Generator::getDataCallback);
    this->getDataSlot.SetCallback(CallTriMeshData::ClassName(), "GetExtent", &MSMSGenus0Generator::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

    // the data in slots
    this->molDataSlot.SetCompatibleCall<MolecularDataCallDescription>();
    this->MakeSlotAvailable(&this->molDataSlot);
    this->bsDataSlot.SetCompatibleCall<BindingSiteCallDescription>();
    this->MakeSlotAvailable(&this->bsDataSlot);
    this->perAtomDataSlot.SetCompatibleCall<PerAtomFloatCallDescription>();
    this->MakeSlotAvailable(&this->perAtomDataSlot);

    // the path of the input file (without extension)
    this->filenameSlot << new core::param::FilePathParam("");
    this->MakeSlotAvailable(&this->filenameSlot);

    // fill color table with default values and set the filename param
    vislib::StringA filename("colors.txt");
    Color::ReadColorTableFromFile(filename, this->colorLookupTable);
    this->colorTableFileParam.SetParameter(new param::StringParam(A2T(filename)));
    this->MakeSlotAvailable(&this->colorTableFileParam);

    // MSMS detail parameter
    this->msmsDetailParam.SetParameter(new param::FloatParam(3.0f, 0.1f));
    this->MakeSlotAvailable(&this->msmsDetailParam);

    // MSMS probe radius parameter
    this->msmsStartingRadiusParam.SetParameter(new param::FloatParam(2.3f, 0.5f, 10.0f));
    this->MakeSlotAvailable(&this->msmsStartingRadiusParam);

    this->msmsStepSizeParam.SetParameter(new param::FloatParam(0.2f, 0.05f, 10.0f));
    this->MakeSlotAvailable(&this->msmsStepSizeParam);

    this->msmsMaxTryNumParam.SetParameter(new param::IntParam(20, 1, 100));
    this->MakeSlotAvailable(&this->msmsMaxTryNumParam);

    // make the rainbow color table
    Color::MakeRainbowColorTable(100, this->rainbowColors);

    // coloring modes
    param::EnumParam* cm0 = new param::EnumParam(int(Color::CHAIN));
    param::EnumParam* cm1 = new param::EnumParam(int(Color::ELEMENT));
    MolecularDataCall* mol = new MolecularDataCall();
    BindingSiteCall* bs = new BindingSiteCall();
    PerAtomFloatCall* pa = new PerAtomFloatCall();
    unsigned int cCnt;
    Color::ColoringMode cMode;
    for (cCnt = 0; cCnt < Color::GetNumOfColoringModes(mol, bs, pa); ++cCnt) {
        cMode = Color::GetModeByIndex(mol, bs, pa, cCnt);
        cm0->SetTypePair(cMode, Color::GetName(cMode).c_str());
        cm1->SetTypePair(cMode, Color::GetName(cMode).c_str());
    }
    delete mol;
    delete bs;
    delete pa;
    this->coloringModeParam0 << cm0;
    this->coloringModeParam1 << cm1;
    this->MakeSlotAvailable(&this->coloringModeParam0);
    this->MakeSlotAvailable(&this->coloringModeParam1);

    // Color weighting parameter
    this->colorWeightParam.SetParameter(new param::FloatParam(0.5f, 0.0f, 1.0f));
    this->MakeSlotAvailable(&this->colorWeightParam);

    // the color for the minimum value (gradient coloring
    this->minGradColorParam.SetParameter(new param::StringParam("#146496"));
    this->MakeSlotAvailable(&this->minGradColorParam);

    // the color for the middle value (gradient coloring
    this->midGradColorParam.SetParameter(new param::StringParam("#f0f0f0"));
    this->MakeSlotAvailable(&this->midGradColorParam);

    // the color for the maximum value (gradient coloring
    this->maxGradColorParam.SetParameter(new param::StringParam("#ae3b32"));
    this->MakeSlotAvailable(&this->maxGradColorParam);
}


/*
 * MSMSGenus0Generator::~MSMSGenus0Generator
 */
MSMSGenus0Generator::~MSMSGenus0Generator(void) { this->Release(); }


/*
 * MSMSGenus0Generator::create
 */
bool MSMSGenus0Generator::create(void) {
    // intentionally empty
    return true;
}


/*
 * MSMSGenus0Generator::release
 */
void MSMSGenus0Generator::release(void) {
    // TODO clear data
}


/*
 * MSMSGenus0Generator::getDataCallback
 */
bool MSMSGenus0Generator::getDataCallback(core::Call& caller) {
    CallTriMeshData* ctmd = dynamic_cast<CallTriMeshData*>(&caller);
    if (ctmd == NULL) return false;

    float probeRadius = this->msmsStartingRadiusParam.Param<param::FloatParam>()->Value();
    float stepSize = this->msmsStepSizeParam.Param<param::FloatParam>()->Value();
    uint32_t maxSteps = static_cast<uint32_t>(this->msmsMaxTryNumParam.Param<param::IntParam>()->Value());

    // load data on demand
    if (this->filenameSlot.IsDirty()) {
        this->filenameSlot.ResetDirty();

        uint32_t genus = 1;
        uint32_t step = 1;
        while (genus != 0 && step <= maxSteps) {
            this->load(this->filenameSlot.Param<core::param::FilePathParam>()->Value(), probeRadius);
            this->isGenus0(0, &genus);
            vislib::sys::Log::DefaultLog.WriteInfo(
                "Step %u: Mesh with radius %f has genus %u", step, probeRadius, genus);
            probeRadius += stepSize;
            step++;
        }
        if (genus != 0) return false;
    }
    if (this->filenameSlot.Param<core::param::FilePathParam>()->Value().IsEmpty() &&
        // this->prevTime != int(ctmd->FrameID())) {
        this->obj.Count() == ctmd->FrameCount() && this->obj[ctmd->FrameID()]->GetVertexCount() == 0) {

        uint32_t genus = 1;
        uint32_t step = 1;

        while (genus != 0 && step <= maxSteps) {
            this->load(vislib::StringA(""), probeRadius, ctmd->FrameID());
            this->isGenus0(ctmd->FrameID(), &genus);
            vislib::sys::Log::DefaultLog.WriteInfo(
                "Step %u: Mesh with radius %f has genus %u", step, probeRadius, genus);
            probeRadius += stepSize;
            step++;
        }
        if (genus != 0) return false;
    }

    PerAtomFloatCall* pa = this->perAtomDataSlot.CallAs<PerAtomFloatCall>();
    if (pa) {
        pa->SetFrameID(ctmd->FrameID());
        // try to call data
        if ((*pa)(PerAtomFloatCall::CallForGetFloat)) {
        }
    }

    // try to call molecular data and compute colors
    MolecularDataCall* mol = this->molDataSlot.CallAs<MolecularDataCall>();
    if (mol) {
        // get pointer to BindingSiteCall
        BindingSiteCall* bs = this->bsDataSlot.CallAs<BindingSiteCall>();
        if (bs) {
            (*bs)(BindingSiteCall::CallForGetData);
        }
        // set call time
        mol->SetCalltime(float(ctmd->FrameID()));
        // set frame ID
        mol->SetFrameID(ctmd->FrameID());
        // try to call data
        if ((*mol)(MolecularDataCall::CallForGetData)) {
            // recompute color table, if necessary
            if (this->coloringModeParam0.IsDirty() || this->coloringModeParam1.IsDirty() ||
                this->colorWeightParam.IsDirty() || this->minGradColorParam.IsDirty() ||
                this->midGradColorParam.IsDirty() || this->maxGradColorParam.IsDirty() ||
                this->prevTime != int(ctmd->FrameID())) {

                Color::ColoringMode currentColoringMode0 =
                    static_cast<Color::ColoringMode>(this->coloringModeParam0.Param<param::EnumParam>()->Value());
                Color::ColoringMode currentColoringMode1 =
                    static_cast<Color::ColoringMode>(this->coloringModeParam1.Param<param::EnumParam>()->Value());
                vislib::Array<float> atomColorTable;
                vislib::Array<float> atomColorTable2;

                Color::MakeColorTable(mol, currentColoringMode0, atomColorTable, this->colorLookupTable,
                    this->rainbowColors, this->minGradColorParam.Param<param::StringParam>()->Value(),
                    this->midGradColorParam.Param<param::StringParam>()->Value(),
                    this->maxGradColorParam.Param<param::StringParam>()->Value(), true, bs, false, pa);

                Color::MakeColorTable(mol, currentColoringMode1, atomColorTable2, this->colorLookupTable,
                    this->rainbowColors, this->minGradColorParam.Param<param::StringParam>()->Value(),
                    this->midGradColorParam.Param<param::StringParam>()->Value(),
                    this->maxGradColorParam.Param<param::StringParam>()->Value(), true, bs, false, pa);

                // loop over atoms and compute color
                float* vertex = new float[this->obj[ctmd->FrameID()]->GetVertexCount() * 3];
                float* normal = new float[this->obj[ctmd->FrameID()]->GetVertexCount() * 3];
                unsigned char* color = new unsigned char[this->obj[ctmd->FrameID()]->GetVertexCount() * 3];
                unsigned int* atomIndex = new unsigned int[this->obj[ctmd->FrameID()]->GetVertexCount()];

                // calculate centroid
                float* centroid = new float[3];
                centroid[0] = 0.0f;
                centroid[1] = 0.0f;
                centroid[2] = 0.0f;
                for (unsigned int i = 0; i < this->obj[ctmd->FrameID()]->GetVertexCount(); i++) {
                    centroid[0] += this->obj[ctmd->FrameID()]->GetVertexPointerFloat()[i * 3 + 0];
                    centroid[1] += this->obj[ctmd->FrameID()]->GetVertexPointerFloat()[i * 3 + 1];
                    centroid[2] += this->obj[ctmd->FrameID()]->GetVertexPointerFloat()[i * 3 + 2];
                }
                centroid[0] /= float(this->obj[ctmd->FrameID()]->GetVertexCount());
                centroid[1] /= float(this->obj[ctmd->FrameID()]->GetVertexCount());
                centroid[2] /= float(this->obj[ctmd->FrameID()]->GetVertexCount());

                // calculate max distance
                float max_dist = std::numeric_limits<float>::min();
                float dist, steps;
                unsigned int anz_cols = 15;
                for (unsigned int i = 0; i < this->obj[ctmd->FrameID()]->GetVertexCount(); i++) {
                    dist = sqrt(pow(centroid[0] - this->obj[ctmd->FrameID()]->GetVertexPointerFloat()[i * 3 + 0], 2) +
                                pow(centroid[1] - this->obj[ctmd->FrameID()]->GetVertexPointerFloat()[i * 3 + 1], 2) +
                                pow(centroid[2] - this->obj[ctmd->FrameID()]->GetVertexPointerFloat()[i * 3 + 2], 2));
                    if (dist > max_dist) max_dist = dist;
                }
                steps = max_dist / float(anz_cols);

                std::vector<std::vector<unsigned char>> colours =
                    std::vector<std::vector<unsigned char>>(anz_cols + 2, std::vector<unsigned char>(3));
                colours[0][0] = 160;
                colours[1][0] = 173;
                colours[2][0] = 185;
                colours[3][0] = 196;
                colours[4][0] = 206;
                colours[5][0] = 218;
                colours[0][1] = 194;
                colours[1][1] = 203;
                colours[2][1] = 213;
                colours[3][1] = 223;
                colours[4][1] = 231;
                colours[5][1] = 240;
                colours[0][2] = 222;
                colours[1][2] = 230;
                colours[2][2] = 237;
                colours[3][2] = 244;
                colours[4][2] = 249;
                colours[5][2] = 253;

                colours[6][0] = 172;
                colours[7][0] = 148;
                colours[8][0] = 168;
                colours[9][0] = 209;
                colours[10][0] = 239;
                colours[11][0] = 222;
                colours[6][1] = 208;
                colours[7][1] = 191;
                colours[8][1] = 198;
                colours[9][1] = 215;
                colours[10][1] = 235;
                colours[11][1] = 214;
                colours[6][2] = 165;
                colours[7][2] = 139;
                colours[8][2] = 143;
                colours[9][2] = 171;
                colours[10][2] = 192;
                colours[11][2] = 163;

                colours[12][0] = 211;
                colours[13][0] = 202;
                colours[14][0] = 195;
                colours[15][0] = 195;
                colours[16][0] = 195;
                colours[12][1] = 202;
                colours[13][1] = 185;
                colours[14][1] = 167;
                colours[15][1] = 167;
                colours[16][1] = 167;
                colours[12][2] = 157;
                colours[13][2] = 130;
                colours[14][2] = 107;
                colours[15][2] = 107;
                colours[16][2] = 107;

                std::vector<unsigned char> col;
                unsigned int bin;
                float alpha, upper_bound, lower_bound;
                // search for the correct attrib index (the first attrib with uint32 type):
                auto atCnt = this->obj[ctmd->FrameID()]->GetVertexAttribCount();
                bool found = false;
                if (atCnt != 0) {
                    for (attIdx = 0; attIdx < atCnt; attIdx++) {
                        if (this->obj[ctmd->FrameID()]->GetVertexAttribDataType(attIdx) ==
                            CallTriMeshData::Mesh::DataType::DT_UINT32) {
                            found = true;
                            break;
                        }
                    }
                }
                // add an attribute or set the existing one
                if (atCnt == 0 || !found) {
                    this->attIdx = this->obj[ctmd->FrameID()]->AddVertexAttribPointer(atomIndex);
                }

                // weighting factors
                float weight0 = this->colorWeightParam.Param<param::FloatParam>()->Value();
                float weight1 = 1.0f - weight0;

                // Clamp weights to zero
                if (weight0 < 0.0) weight0 = 0.0;
                if (weight1 < 0.0) weight1 = 0.0;

                // Normalize weights
                weight0 = weight0 / (weight0 + weight1);
                weight1 = weight1 / (weight0 + weight1);

                for (unsigned int i = 0; i < this->obj[ctmd->FrameID()]->GetVertexCount(); i++) {
                    vertex[i * 3 + 0] = this->obj[ctmd->FrameID()]->GetVertexPointerFloat()[i * 3 + 0];
                    vertex[i * 3 + 1] = this->obj[ctmd->FrameID()]->GetVertexPointerFloat()[i * 3 + 1];
                    vertex[i * 3 + 2] = this->obj[ctmd->FrameID()]->GetVertexPointerFloat()[i * 3 + 2];
                    normal[i * 3 + 0] = this->obj[ctmd->FrameID()]->GetNormalPointerFloat()[i * 3 + 0];
                    normal[i * 3 + 1] = this->obj[ctmd->FrameID()]->GetNormalPointerFloat()[i * 3 + 1];
                    normal[i * 3 + 2] = this->obj[ctmd->FrameID()]->GetNormalPointerFloat()[i * 3 + 2];
                    color[i * 3 + 0] = 0;
                    color[i * 3 + 1] = 0;
                    color[i * 3 + 2] = 0;
                    atomIndex[i] = this->obj[ctmd->FrameID()]->GetVertexAttribPointerUInt32(attIdx)[i];

                    // create hightmap colours or read per atom colours
                    if (currentColoringMode0 == Color::HEIGHTMAP_COL || currentColoringMode0 == Color::HEIGHTMAP_VAL) {
                        col = std::vector<unsigned char>(3, 0);
                        dist =
                            sqrt(pow(centroid[0] - this->obj[ctmd->FrameID()]->GetVertexPointerFloat()[i * 3 + 0], 2) +
                                 pow(centroid[1] - this->obj[ctmd->FrameID()]->GetVertexPointerFloat()[i * 3 + 1], 2) +
                                 pow(centroid[2] - this->obj[ctmd->FrameID()]->GetVertexPointerFloat()[i * 3 + 2], 2));
                        bin = static_cast<unsigned int>(std::truncf(dist / steps));
                        lower_bound = float(bin) * steps;
                        upper_bound = float(bin + 1) * steps;
                        alpha = (dist - lower_bound) / (upper_bound - lower_bound);
                        dist /= max_dist;

                        if (currentColoringMode0 == Color::HEIGHTMAP_COL) {
                            col[0] = static_cast<unsigned char>(
                                (1.0f - alpha) * colours[bin][0] + alpha * colours[bin + 1][0]);
                            col[1] = static_cast<unsigned char>(
                                (1.0f - alpha) * colours[bin][1] + alpha * colours[bin + 1][1]);
                            col[2] = static_cast<unsigned char>(
                                (1.0f - alpha) * colours[bin][2] + alpha * colours[bin + 1][2]);
                        } else { // currentColoringMode0 == Color::HEIGHTMAP_VAL
                            unsigned char dist_uc = static_cast<unsigned char>(dist * 255.0f);
                            col[0] = dist_uc;
                            col[1] = dist_uc;
                            col[2] = dist_uc;
                        }
                        color[i * 3 + 0] += static_cast<unsigned char>(weight0 * col[0]);
                        color[i * 3 + 1] += static_cast<unsigned char>(weight0 * col[1]);
                        color[i * 3 + 2] += static_cast<unsigned char>(weight0 * col[2]);
                    } else {
                        color[i * 3 + 0] +=
                            static_cast<unsigned char>(weight0 * atomColorTable[atomIndex[i] * 3 + 0] * 255);
                        color[i * 3 + 1] +=
                            static_cast<unsigned char>(weight0 * atomColorTable[atomIndex[i] * 3 + 1] * 255);
                        color[i * 3 + 2] +=
                            static_cast<unsigned char>(weight0 * atomColorTable[atomIndex[i] * 3 + 2] * 255);
                    }

                    if (currentColoringMode1 == Color::HEIGHTMAP_COL || currentColoringMode1 == Color::HEIGHTMAP_VAL) {
                        col = std::vector<unsigned char>(3, 0);
                        dist =
                            sqrt(pow(centroid[0] - this->obj[ctmd->FrameID()]->GetVertexPointerFloat()[i * 3 + 0], 2) +
                                 pow(centroid[1] - this->obj[ctmd->FrameID()]->GetVertexPointerFloat()[i * 3 + 1], 2) +
                                 pow(centroid[2] - this->obj[ctmd->FrameID()]->GetVertexPointerFloat()[i * 3 + 2], 2));
                        bin = static_cast<unsigned int>(std::truncf(dist / steps));
                        lower_bound = float(bin) * steps;
                        upper_bound = float(bin + 1) * steps;
                        alpha = (dist - lower_bound) / (upper_bound - lower_bound);
                        dist /= max_dist;

                        if (currentColoringMode1 == Color::HEIGHTMAP_COL) {
                            col[0] = static_cast<unsigned char>(
                                (1.0f - alpha) * colours[bin][0] + alpha * colours[bin + 1][0]);
                            col[1] = static_cast<unsigned char>(
                                (1.0f - alpha) * colours[bin][1] + alpha * colours[bin + 1][1]);
                            col[2] = static_cast<unsigned char>(
                                (1.0f - alpha) * colours[bin][2] + alpha * colours[bin + 1][2]);
                        } else { // currentColoringMode1 == Color::HEIGHTMAP_VAL
                            unsigned char dist_uc = static_cast<unsigned char>(dist * 255.0f);
                            col[0] = dist_uc;
                            col[1] = dist_uc;
                            col[2] = dist_uc;
                        }
                        color[i * 3 + 0] += static_cast<unsigned char>(weight1 * col[0]);
                        color[i * 3 + 1] += static_cast<unsigned char>(weight1 * col[1]);
                        color[i * 3 + 2] += static_cast<unsigned char>(weight1 * col[2]);
                    } else {
                        color[i * 3 + 0] +=
                            static_cast<unsigned char>(weight1 * atomColorTable2[atomIndex[i] * 3 + 0] * 255);
                        color[i * 3 + 1] +=
                            static_cast<unsigned char>(weight1 * atomColorTable2[atomIndex[i] * 3 + 1] * 255);
                        color[i * 3 + 2] +=
                            static_cast<unsigned char>(weight1 * atomColorTable2[atomIndex[i] * 3 + 2] * 255);
                    }
                }

                this->obj[ctmd->FrameID()]->SetVertexData(
                    this->obj[ctmd->FrameID()]->GetVertexCount(), vertex, normal, color, NULL, true);
                // setVertexData clears all attributes, so we have to add a new one
                this->obj[ctmd->FrameID()]->AddVertexAttribPointer(atomIndex);
                this->datahash++;

                this->coloringModeParam0.ResetDirty();
                this->coloringModeParam1.ResetDirty();
                this->colorWeightParam.ResetDirty();
                this->minGradColorParam.ResetDirty();
                this->midGradColorParam.ResetDirty();
                this->maxGradColorParam.ResetDirty();
            }
        }
    }

    if (this->obj[ctmd->FrameID()]->GetVertexCount() > 0) {
        ctmd->SetDataHash(this->datahash);
        ctmd->SetObjects(1, this->obj[ctmd->FrameID()]);
        ctmd->SetUnlocker(NULL);
        this->prevTime = int(ctmd->FrameID());
        return true;
    } else {
        return false;
    }
}


/*
 * MSMSGenus0Generator::getExtentCallback
 */
bool MSMSGenus0Generator::getExtentCallback(core::Call& caller) {
    CallTriMeshData* ctmd = dynamic_cast<CallTriMeshData*>(&caller);
    if (ctmd == NULL) return false;

    unsigned int frameCnt = 1;

    // load data on demand
    if (this->filenameSlot.IsDirty()) {
        this->filenameSlot.ResetDirty();
        float probeRadius = this->msmsStartingRadiusParam.Param<param::FloatParam>()->Value();
        float stepSize = this->msmsStepSizeParam.Param<param::FloatParam>()->Value();
        uint32_t maxSteps = static_cast<uint32_t>(this->msmsMaxTryNumParam.Param<param::IntParam>()->Value());
        uint32_t genus = 1;
        uint32_t step = 1;
        while (genus != 0 && step <= maxSteps) {
            this->load(this->filenameSlot.Param<core::param::FilePathParam>()->Value(), probeRadius);
            this->isGenus0(0, &genus);
            vislib::sys::Log::DefaultLog.WriteInfo(
                "Step %u: Mesh with radius %f has genus %u", step, probeRadius, genus);
            probeRadius += stepSize;
            step++;
        }
        if (genus > 0) return false;
    }

    ctmd->SetDataHash(this->datahash);
    if (this->filenameSlot.Param<core::param::FilePathParam>()->Value().IsEmpty()) {
        MolecularDataCall* mol = this->molDataSlot.CallAs<MolecularDataCall>();
        if (mol) {
            // try to call for extent
            if ((*mol)(MolecularDataCall::CallForGetExtent)) {
                frameCnt = mol->FrameCount();
                this->bbox = mol->AccessBoundingBoxes().ObjectSpaceBBox();
            }
        }
    }
    ctmd->SetExtent(frameCnt, this->bbox.Left(), this->bbox.Bottom(), this->bbox.Back(), this->bbox.Right(),
        this->bbox.Top(), this->bbox.Front());
    if (this->obj.Count() != frameCnt) {
        for (unsigned int i = 0; i < this->obj.Count(); i++) {
            delete this->obj[i];
            this->obj[i] = NULL;
        }
        this->obj.SetCount(frameCnt);
        for (unsigned int i = 0; i < frameCnt; i++) {
            this->obj[i] = new CallTriMeshData::Mesh();
        }
    }
    return true;
}

/*
 * MSMSGenus0Generator::load
 */
bool MSMSGenus0Generator::load(const vislib::TString& filename, float probe_radius, unsigned int frameID) {
    using vislib::sys::Log;

    vislib::StringA vertFilename(filename);
    vertFilename.Append(".vert");
    vislib::StringA faceFilename(filename);
    faceFilename.Append(".face");
    vislib::sys::ASCIIFileBuffer vertFile;
    vislib::sys::ASCIIFileBuffer faceFile;

    if (filename.IsEmpty()) {
        // try to call molecular data and compute colors
        MolecularDataCall* mol = this->molDataSlot.CallAs<MolecularDataCall>();
        if (mol) {
            // set call time
            mol->SetCalltime(float(frameID));
            // set frame ID
            mol->SetFrameID(frameID);
            // try to call data
            if ((*mol)(MolecularDataCall::CallForGetData)) {
                std::ofstream xyzFile;
                xyzFile.open("msmstest.xyz", std::ofstream::out);
                if (xyzFile.is_open()) {
                    for (unsigned int i = 0; i < mol->AtomCount(); i++) {
                        xyzFile << mol->AtomPositions()[3 * i] << " " << mol->AtomPositions()[3 * i + 1] << " "
                                << mol->AtomPositions()[3 * i + 2] << " "
                                << mol->AtomTypes()[mol->AtomTypeIndices()[i]].Radius() << std::endl;
                    }
                    xyzFile.close();
                } else {
                    return false;
                }
                vislib::StringA msmsCmd;
#ifdef WIN32
                msmsCmd.Format("msms.exe -probe_radius %f -density %.1f -if msmstest.xyz -of msmstest", probe_radius,
                    this->msmsDetailParam.Param<param::FloatParam>()->Value());
#else
                msmsCmd.Format("./msms.x86_64Linux2.2.6.1 -probe_radius %f -density %.1f -if msmstest.xyz -of msmstest",
                    probe_radius, this->msmsDetailParam.Param<param::FloatParam>()->Value());
#endif
                system(msmsCmd.PeekBuffer());
                vertFilename = "msmstest.vert";
                faceFilename = "msmstest.face";
            } else
                return false;
        } else
            return false;
    }

    // temp variables
    vislib::StringA line;

    Log::DefaultLog.WriteInfo(
        "Loading MSMS files: %s and %s ...", vertFilename.PeekBuffer(), faceFilename.PeekBuffer());
    // try to load the file
    if (vertFile.LoadFile(vertFilename) && faceFile.LoadFile(faceFilename)) {
        // ----- load the vertices -----
        // do nothing if file is too short
        if (vertFile.Count() < 4) {
            Log::DefaultLog.WriteError("MSMS file %s too short.", vertFilename.PeekBuffer());
            return false;
        }
        // get number of vertices and spheres from third line
        line = vertFile.Line(2);
        vislib::Array<vislib::StringA> splitArray =
            vislib::StringTokeniser<vislib::CharTraitsA>::Split(line, ' ', true);
        if (splitArray.Count() != 4) {
            Log::DefaultLog.WriteError("Could not parse header or MSMS file %s.", vertFilename.PeekBuffer());
            return false;
        }
        this->vertexCount = static_cast<unsigned int>(atof(splitArray[0]));
        this->atomCount = static_cast<unsigned int>(atof(splitArray[1]));
        // resize vertex arrays
        float* vertex = new float[this->vertexCount * 3];
        float* normal = new float[this->vertexCount * 3];
        unsigned char* color = new unsigned char[this->vertexCount * 3];
        unsigned int* atomIndex = new unsigned int[this->vertexCount];

        // parse all vertex lines
        size_t j = 0;
        for (unsigned int i = 3; i < vertFile.Count(); i++) {
            line = vertFile.Line(i);
            if (line.IsEmpty()) continue;
            if (j >= this->vertexCount) {
                Log::DefaultLog.WriteError(
                    "Too many vertices in MSMS file %s (line %i).", vertFilename.PeekBuffer(), i);
                return false;
            }
            vertex[j * 3 + 0] = static_cast<float>(atof(line.Substring(0, 9)));
            vertex[j * 3 + 1] = static_cast<float>(atof(line.Substring(10, 9)));
            vertex[j * 3 + 2] = static_cast<float>(atof(line.Substring(20, 9)));
            normal[j * 3 + 0] = static_cast<float>(atof(line.Substring(30, 9)));
            normal[j * 3 + 1] = static_cast<float>(atof(line.Substring(40, 9)));
            normal[j * 3 + 2] = static_cast<float>(atof(line.Substring(50, 9)));
            color[j * 3 + 0] = 210;
            color[j * 3 + 1] = 170;
            color[j * 3 + 2] = 20;
            atomIndex[j] = static_cast<unsigned int>(atoi(line.Substring(68, 7))) - 1;
            if (atomIndex[j] >= this->atomCount) {
                Log::DefaultLog.WriteError(
                    "Invalid atom number in MSMS file %s (line %i).", vertFilename.PeekBuffer(), i);
                return false;
            }
            j++;
        }

        // ----- load the faces -----
        // do nothing if file is too short
        if (faceFile.Count() < 4) {
            Log::DefaultLog.WriteError("MSMS file %s too short.", faceFilename.PeekBuffer());
            return false;
        }
        // get number of faces and spheres from third line
        line = faceFile.Line(2);
        splitArray.Clear();
        splitArray = vislib::StringTokeniser<vislib::CharTraitsA>::Split(line, ' ', true);
        if (splitArray.Count() != 4) {
            Log::DefaultLog.WriteError("Could not parse header or MSMS file %s.", faceFilename.PeekBuffer());
            return false;
        }
        this->faceCount = static_cast<unsigned int>(atof(splitArray[0]));
        if (this->atomCount != static_cast<float>(atof(splitArray[1]))) {
            Log::DefaultLog.WriteError("Atom count mismatch between MSMS files %s and %s.", vertFilename.PeekBuffer(),
                faceFilename.PeekBuffer());
            return false;
        }
        // resize vertex arrays
        unsigned int* face = new unsigned int[this->faceCount * 3];
        // parse all face lines
        j = 0;
        for (unsigned int i = 3; i < faceFile.Count(); i++) {
            line = faceFile.Line(i);
            if (line.IsEmpty()) continue;
            if (j >= this->faceCount) {
                Log::DefaultLog.WriteError("Too many faces in MSMS file %s (line %i).", faceFilename.PeekBuffer(), i);
                return false;
            }
            face[j * 3 + 0] = static_cast<unsigned int>(atof(line.Substring(0, 6))) - 1;
            face[j * 3 + 1] = static_cast<unsigned int>(atof(line.Substring(7, 6))) - 1;
            face[j * 3 + 2] = static_cast<unsigned int>(atof(line.Substring(14, 6))) - 1;
            if (face[j * 3 + 0] >= this->faceCount || face[j * 3 + 1] >= this->faceCount ||
                face[j * 3 + 2] >= this->faceCount) {
                Log::DefaultLog.WriteError(
                    "Invalid vertex number in MSMS file %s (line %i).", faceFilename.PeekBuffer(), i);
                return false;
            }
            j++;
        }

        this->datahash++;
        Log::DefaultLog.WriteInfo(
            "Finished loading MSMS files %s and %s.", vertFilename.PeekBuffer(), faceFilename.PeekBuffer());

        // compute bounding box
        if (this->vertexCount > 1) {
            this->bbox.Set(vertex[0], vertex[1], vertex[2], vertex[0], vertex[1], vertex[2]);
            for (unsigned int i = 1; i < this->vertexCount; i++) {
                this->bbox.GrowToPoint(vertex[i * 3], vertex[i * 3 + 1], vertex[i * 3 + 2]);
            }
        }

        // try to call molecular data and compute colors
        MolecularDataCall* mol = this->molDataSlot.CallAs<MolecularDataCall>();
        if (mol) {
            // set call time
            mol->SetCalltime(static_cast<float>(frameID));
            // set frame ID
            mol->SetFrameID(frameID);
            // try to call data
            if ((*mol)(MolecularDataCall::CallForGetData)) {
                // check if atom count matches the MSMS file
                if (mol->AtomCount() == this->atomCount) {
                    Log::DefaultLog.WriteInfo("Get colors from PDB file...");
                    // loop over atoms and compute color
                    for (unsigned int i = 0; i < this->vertexCount; i++) {
                        color[i * 3 + 0] = mol->AtomTypes()[mol->AtomTypeIndices()[atomIndex[i]]].Colour()[0];
                        color[i * 3 + 1] = mol->AtomTypes()[mol->AtomTypeIndices()[atomIndex[i]]].Colour()[1];
                        color[i * 3 + 2] = mol->AtomTypes()[mol->AtomTypeIndices()[atomIndex[i]]].Colour()[2];
                    }
                }
            }
        }

        // set mesh object
        if (this->obj.Count() < frameID) {
            return false;
        }
        this->obj[frameID]->SetVertexData(this->vertexCount, vertex, normal, color, NULL, true);
        this->obj[frameID]->SetTriangleData(this->faceCount, face, true);
        this->obj[frameID]->SetMaterial(NULL);

        // check whether the mesh already has a vertex attribute with the name "atomID"
        auto atCnt = this->obj[frameID]->GetVertexAttribCount();
        bool found = false;
        if (atCnt != 0) {
            for (attIdx = 0; attIdx < atCnt; attIdx++) {
                if (this->obj[frameID]->GetVertexAttribDataType(attIdx) == CallTriMeshData::Mesh::DataType::DT_UINT32) {
                    found = true;
                    break;
                }
            }
        }
        // add an attribute or set the existing one
        if (atCnt == 0 || !found) {
            this->obj[frameID]->AddVertexAttribPointer(atomIndex);
        } else {
            this->obj[frameID]->SetVertexAttribData(atomIndex, attIdx);
        }
        return true;
    }
    Log::DefaultLog.WriteError(
        "Could not load MSMS files %s and %s.", vertFilename.PeekBuffer(), faceFilename.PeekBuffer());
    return false;
}

/*
 * MSMSGenus0Generator::computeGenus
 */
bool MSMSGenus0Generator::isGenus0(uint32_t frameID, uint32_t* out_genus) {
    if (this->obj.Count() < frameID) {
        vislib::sys::Log::DefaultLog.WriteError(
            "Requested non-existing frame %u out of %u frames", frameID, static_cast<uint32_t>(this->obj.Count()));
        return false;
    }
    auto ptr = this->obj[frameID];
    auto vertCnt = ptr->GetVertexCount();
    auto faceCnt = ptr->GetTriCount();
    uint32_t numEdges = (3 * faceCnt) - ((3 * faceCnt) / 2);
    int euler = vertCnt + faceCnt - numEdges;
    int genus = 1 - (euler / 2);
    if (out_genus != nullptr) {
        *out_genus = static_cast<uint32_t>(genus);
    }
    return genus == 0;
}
