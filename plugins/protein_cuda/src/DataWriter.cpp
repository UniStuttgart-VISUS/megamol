/*
 * DataWriter.cpp
 *
 * Copyright (C) 2012 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 *
 * $Id$
 */

#include "stdafx.h"
#include "DataWriter.h"

#define _USE_MATH_DEFINES 1

#include <fstream>
#include <sstream>

#include "vislib/graphics/gl/IncludeAllGL.h"
#include <GL/glu.h>

#include "vislib/sys/Log.h"
#include "CUDAQuickSurf.h"

#include <thrust/device_vector.h>
#include <thrust/version.h>
#include <iostream>
#include <vector_types.h>
#include <cuda_gl_interop.h>
#include "helper_cuda.h"

#include <thrust/sort.h>


using namespace megamol;


/*
 * protein_cuda::DataWriter::DataWriter
 */
protein_cuda::DataWriter::DataWriter() :
        AbstractJob(),
        Module(),
        dataCallerSlot("getdata", "Connects the module with the data source."),
        jobDone(false),
        frameData0(NULL),
        frameData1(NULL),
        frameDataDispl(NULL),
        addedPos(NULL),
        addedTiDispl(NULL){

	this->dataCallerSlot.SetCompatibleCall<protein_calls::CrystalStructureDataCallDescription>();
    this->MakeSlotAvailable(&this->dataCallerSlot);

    this->numvoxels[0] = 128;
    this->numvoxels[1] = 128;
    this->numvoxels[2] = 128;

    this->origin[0] = 0.0f;
    this->origin[1] = 0.0f;
    this->origin[2] = 0.0f;

    this->xaxis[0] = 1.0f;
    this->xaxis[1] = 0.0f;
    this->xaxis[2] = 0.0f;

    this->yaxis[0] = 0.0f;
    this->yaxis[1] = 1.0f;
    this->yaxis[2] = 0.0f;

    this->zaxis[0] = 0.0f;
    this->zaxis[1] = 0.0f;
    this->zaxis[2] = 1.0f;

    this->cudaqsurf = 0;
}


/*
 * protein_cuda::DataWriter::~DataWriter
 */
protein_cuda::DataWriter::~DataWriter() {
    this->Release();
}


/*
 * protein_cuda::DataWriter::IsRunning
 */
bool protein_cuda::DataWriter::IsRunning(void) const {
    return (!(this->jobDone));
}


/*
 * protein_cuda::DataWriter::Start
 */
bool protein_cuda::DataWriter::Start(void) {
    using namespace vislib::sys;

	protein_calls::CrystalStructureDataCall *dc = this->dataCallerSlot.CallAs<protein_calls::CrystalStructureDataCall>();
    if(dc == NULL) {
        this->jobDone = true;
        return false;
    }

    /*if(!this->PutCubeSize(0, 499, dc)) {
        this->jobDone = true;
        return false;
    }*/
    //this->WriteTiDispl(dc);
    //this->ReadTiDispl(dc);
    //this->WriteTiODipole(dc);
    //this->ReadTiODipole(dc);
    //this->PutVelocity();
    this->PutDisplacement(dc);
    //this->PutCubeVol(dc);
    //this->GetMaxCoords(dc);


    this->jobDone = true;
    return true;
}


/*
 * protein_cuda::DataWriter::PutStatistics
 */
bool protein_cuda::DataWriter::PutStatistics(unsigned int frameIdx0,
        unsigned int frameIdx1, unsigned int avgOffs) {

    using namespace vislib::sys;
    float gridspacing = 1.0f;

	protein_calls::CrystalStructureDataCall *dc = this->dataCallerSlot.CallAs<protein_calls::CrystalStructureDataCall>();
    if(dc == NULL) {
        this->jobDone = true;
        return false;
    }

    // Get extend
	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetExtent)) {
        this->jobDone = true;
        return false;
    }

    if(frameIdx1+avgOffs > dc->FrameCount()-1) {
        frameIdx1 -= (avgOffs-1);
    }

    // Compute grid dimensions
    float mincoord[3], maxcoord[3];
    mincoord[0] = dc->AccessBoundingBoxes().ObjectSpaceBBox().Left();
    mincoord[1] = dc->AccessBoundingBoxes().ObjectSpaceBBox().Bottom();
    mincoord[2] = dc->AccessBoundingBoxes().ObjectSpaceBBox().Back();
    maxcoord[0] = dc->AccessBoundingBoxes().ObjectSpaceBBox().Right();
    maxcoord[1] = dc->AccessBoundingBoxes().ObjectSpaceBBox().Top();
    maxcoord[2] = dc->AccessBoundingBoxes().ObjectSpaceBBox().Front();
    this->xaxis[0] = maxcoord[0]-mincoord[0];
    this->yaxis[1] = maxcoord[1]-mincoord[1];
    this->zaxis[2] = maxcoord[2]-mincoord[2];
    this->numvoxels[0] = (int) ceil(this->xaxis[0] / gridspacing);
    this->numvoxels[1] = (int) ceil(this->yaxis[1] / gridspacing);
    this->numvoxels[2] = (int) ceil(this->zaxis[2] / gridspacing);
    this->xaxis[0] = (this->numvoxels[0]-1) * gridspacing;
    this->yaxis[1] = (this->numvoxels[1]-1) * gridspacing;
    this->zaxis[2] = (this->numvoxels[2]-1) * gridspacing;
    maxcoord[0] = mincoord[0] + this->xaxis[0];
    maxcoord[1] = mincoord[1] + this->yaxis[1];
    maxcoord[2] = mincoord[2] + this->zaxis[2];
    this->origin[0] = mincoord[0];
    this->origin[1] = mincoord[1];
    this->origin[2] = mincoord[2];

    // Allocate memory
    float *griddata=NULL;
    griddata = new float[this->numvoxels[0]*this->numvoxels[1]*this->numvoxels[2]*3];

    for(unsigned int fr = frameIdx0; fr <= frameIdx1; fr++) {

        // Get data from data source
        dc->SetFrameID(fr, true);
        dc->SetCalltime(static_cast<float>(fr));
		if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetData)) {
            this->jobDone = true;
            return false;
        }

        // Calculate dipole moment for this frame
        /*if(!this->CalcMapDipoleAvg(dc, avgOffs, 1, 1.0f, 1.0f, 1.0f)) {
            this->jobDone = true;
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: Unable to calculate dipole map\n",
                    this->ClassName());
            return false;
        }*/

        // Compute ti atom displacement
        if(!this->CalcMapTiDisplAvg(dc, avgOffs, 1, 1.0f, 1.0f, 1.0f)) {
            this->jobDone = true;
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: Unable to calculate dipole map\n",
                    this->ClassName());
            return false;
        }

        // Copy data from device to host
        CUDAQuickSurf *cqs = (CUDAQuickSurf *) this->cudaqsurf;
        checkCudaErrors(cudaMemcpy(
                griddata,
                cqs->getColorMap(),
                this->numvoxels[0]*this->numvoxels[1]*this->numvoxels[2]*3*sizeof(float),
                cudaMemcpyDeviceToHost));

        ////// Calculate vector added dipole and avg dipole magnitude //////////
        /*float invCnt = 1.0f/static_cast<float>(this->numvoxels[0]*this->numvoxels[1]*this->numvoxels[2]);
        float Dipole[] = {0.0f, 0.0f, 0.0f};
        float avgMag = 0.0f;
        for(int cnt = 0; cnt < this->numvoxels[0]*this->numvoxels[1]*this->numvoxels[2]; cnt++) {
            Dipole[0] += (griddata[cnt*3+0]*invCnt);
            Dipole[1] += (griddata[cnt*3+1]*invCnt);
            Dipole[2] += (griddata[cnt*3+2]*invCnt);
            float mag = sqrt(griddata[cnt*3+0]*griddata[cnt*3+0]+
                             griddata[cnt*3+1]*griddata[cnt*3+1]+
                             griddata[cnt*3+2]*griddata[cnt*3+2]);
            avgMag += (mag*invCnt);
        }
        float DipoleMag = sqrt(Dipole[0]*Dipole[0]+
                Dipole[1]*Dipole[1]+
                Dipole[2]*Dipole[2]);
        //printf("FRAME %i | DIPOLE MAG %f | AVG MAG %f\n", fr, DipoleMag, avgMag);
        printf("%f %f\n", static_cast<float>(fr), avgMag);*/
        ////////////////////////////////////////////////////////////////////////

        ////// Print dipole and dipole mag of cell #cellIdx ////////////////////
        unsigned int cellIdx = 4500;
        float magIdx = sqrt(griddata[cellIdx*3+0]*griddata[cellIdx*3+0]+
                 griddata[cellIdx*3+1]*griddata[cellIdx*3+1]+
                 griddata[cellIdx*3+2]*griddata[cellIdx*3+2]);
        //printf("DIPOLE: (%f %f %f)\n", griddata[cellIdx*3+0], griddata[cellIdx*3+1], griddata[cellIdx*3+2]);
        printf("%u %f\n", fr, magIdx);
        ////////////////////////////////////////////////////////////////////////

    }

    if(griddata != NULL) delete[] griddata;

    this->jobDone = true;
    return true;
}


/*
 * protein_cuda::DataWriter::Terminate
 */
bool protein_cuda::DataWriter::Terminate(void) {
    return true;
}


/*
 * protein_cuda::DataWriter::WriteFrame2VTI
 */
bool protein_cuda::DataWriter::WriteFrame2VTI(std::string filePrefix,
                                         vislib::TString dataIdentifier,
                                         float org[3],
                                         float step[3],
                                         int dim[3],
                                         int cycle,
                                         float *data) {
    using namespace vislib::sys;

    // TODO
    // + Endianess?

    // Filename
    std::stringstream filenameStr;
    filenameStr << filePrefix << "_" << dataIdentifier<< ".";
    if(cycle < 100) {
        filenameStr << "0";
        if(cycle < 10)
            filenameStr << "0";
    }
    filenameStr << cycle << ".vti";
    std::string filename = filenameStr.str();

    std::ofstream outfile;
    outfile.open(filename.c_str(), std::ios::out | std::ios::binary);
    if(!outfile.good()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to open file %s\n",
                this->ClassName(),
                filename.data());
        return false;
    }


    /// Write header data ///

    outfile << "<?xml version=\"1.0\"?>" << std::endl;
    outfile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
    outfile << "  <ImageData WholeExtent=\"0 " << dim[0]-1 << " 0 " << dim[1]-1 << " 0 " << dim[2]-1;
    outfile << "\" Origin=\"" << org[0] << " " << org[1] << " " << org[2] << "\" Spacing=\"";
    outfile << step[0] << " " << step[1] << " " << step[2] << "\">" << std::endl;

    outfile << "    <FieldData>" << std::endl;
    outfile << "      <DataArray type=\"Int32\" Name=\"CYCLE\" NumberOfTuples=\"1\" format=\"ascii\">"; // TODO keyword CYCLE ?
    outfile << cycle;
    //outfile.write((char *)(&cycle), sizeof(int));
    outfile << "      </DataArray>" << std::endl;
    outfile << "    </FieldData>" << std::endl;

    outfile << "    <Piece Extent=\"0 " << dim[0]-1<< " 0 " << dim[1]-1 << " 0 " << dim[2]-1 << "\">" << std::endl;
    outfile << "      <PointData Vectors=\"" << dataIdentifier << "\">" << std::endl;
    //outfile << "        <DataArray type=\"Float32\" Name=\"" << dataIdentifier << "\" format=\"appended\" offset=\"0\" NumberOfComponents=\"3\" /> " << std::endl;

    // Write ascii
    outfile << "        <DataArray type=\"Float32\" Name=\"" << dataIdentifier << "\" format=\"ascii\" NumberOfComponents=\"3\"> " << std::endl;


    for(int cnt = 0; cnt < dim[0]*dim[1]*dim[2]; cnt++) {
        outfile << "          " << data[cnt*3+0] << " " << data[cnt*3+1] << " " << data[cnt*3+2] << std::endl;
    }

    /*std::stringstream dataOutfile;
    for(int cnt = 0; cnt < dim[0]*dim[1]*dim[2]; cnt++) {
        dataOutfile << "          " << data[cnt*3+0] << " " << data[cnt*3+1] << " " << data[cnt*3+2] << std::endl;
    }
    dataOutfile.flush();
    outfile.write(dataOutfile.str().data(), dataOutfile.str().size());*/

    outfile << std::endl;
    outfile << "        </DataArray> " << std::endl;


    outfile << "      </PointData>" << std::endl;
//    outfile << "      <CellData></CellData>
    outfile << "    </Piece>" << std::endl;
    outfile << "  </ImageData>" << std::endl;

    //outfile << " <AppendedData encoding=\"raw\">" << std::endl << "_";
    //outfile.write((char *)data, dim[0]*dim[1]*dim[2]*3*sizeof(float));
    //outfile << std::endl << "</AppendedData>" << std::endl;

    outfile << "</VTKFile>" << std::endl;

    outfile.close();
    return true;
}


/*
 * protein_cuda::DataWriter::WriteDipoleToVTI
 */
bool protein_cuda::DataWriter::WriteDipoleToVTI(unsigned int frameIdx0,
        unsigned int frameIdx1, unsigned int avgOffs) {
    using namespace vislib::sys;

	protein_calls::CrystalStructureDataCall *dc = this->dataCallerSlot.CallAs<protein_calls::CrystalStructureDataCall>();
    if(dc == NULL) {
        this->jobDone = true;
        return false;
    }

    // Get extend
	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetExtent)) {
        this->jobDone = true;
        return false;
    }

    // Prevent overflow when averaging over a time window
    if(frameIdx1+avgOffs > dc->FrameCount()-1) {
        frameIdx1 -= (avgOffs-1);
    }

    float gridspacing = 1.0f;
    // Compute grid dimensions
    float mincoord[3], maxcoord[3];
    mincoord[0] = dc->AccessBoundingBoxes().ObjectSpaceBBox().Left();
    mincoord[1] = dc->AccessBoundingBoxes().ObjectSpaceBBox().Bottom();
    mincoord[2] = dc->AccessBoundingBoxes().ObjectSpaceBBox().Back();
    maxcoord[0] = dc->AccessBoundingBoxes().ObjectSpaceBBox().Right();
    maxcoord[1] = dc->AccessBoundingBoxes().ObjectSpaceBBox().Top();
    maxcoord[2] = dc->AccessBoundingBoxes().ObjectSpaceBBox().Front();
    this->xaxis[0] = maxcoord[0]-mincoord[0];
    this->yaxis[1] = maxcoord[1]-mincoord[1];
    this->zaxis[2] = maxcoord[2]-mincoord[2];
    this->numvoxels[0] = (int) ceil(this->xaxis[0] / gridspacing);
    this->numvoxels[1] = (int) ceil(this->yaxis[1] / gridspacing);
    this->numvoxels[2] = (int) ceil(this->zaxis[2] / gridspacing);
    this->xaxis[0] = (this->numvoxels[0]-1) * gridspacing;
    this->yaxis[1] = (this->numvoxels[1]-1) * gridspacing;
    this->zaxis[2] = (this->numvoxels[2]-1) * gridspacing;
    maxcoord[0] = mincoord[0] + this->xaxis[0];
    maxcoord[1] = mincoord[1] + this->yaxis[1];
    maxcoord[2] = mincoord[2] + this->zaxis[2];
    this->origin[0] = mincoord[0];
    this->origin[1] = mincoord[1];
    this->origin[2] = mincoord[2];

    // Allocate memory for grid data
    float *griddata=NULL;
    griddata = new float[this->numvoxels[0]*this->numvoxels[1]*this->numvoxels[2]*3];

    for(unsigned int fr = frameIdx0; fr <= frameIdx1; fr++) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: FRAME %i | Computing grid\n", this->ClassName(), fr);

        // Calculate dipole moment for this frame
        dc->SetFrameID(fr, true);
        if(!this->CalcMapDipoleAvg(dc, avgOffs, 1, 1.0f, 1.0f, 1.0f)) {
            this->jobDone = true;
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: Unable to calculate dipole map\n",
                    this->ClassName());
            delete[] griddata;
            return false;
        }

        // Copy data from device to host

        CUDAQuickSurf *cqs = (CUDAQuickSurf *) this->cudaqsurf;
        checkCudaErrors(cudaMemcpy(
                griddata,
                cqs->getColorMap(),
                this->numvoxels[0]*this->numvoxels[1]*this->numvoxels[2]*3*sizeof(float),
                cudaMemcpyDeviceToHost));

        //printf("CUDA copy done. (%f %f %f)\n", griddata[0], griddata[1], griddata[2]); // DEBUG
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: FRAME %i | Writing to file\n", this->ClassName(), fr);
            // Write frame data to *.vti file
            float step[] = {1.0f, 1.0f, 1.0f};
            std::stringstream ss;
            ss << "BaTiO3_625000at_offs" << avgOffs;
            if(!this->WriteFrame2VTI(ss.str(),
                    vislib::TString("Dipole"),
                    this->origin,
                    step,
                    this->numvoxels,
                    fr,
                    griddata)) {
                this->jobDone = true;
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: Unable to write file\n",
                        this->ClassName());
                delete[] griddata;
                return false;
        }
    }

    if(griddata != NULL) delete[] griddata;
    return true;
}


/*
 * protein_cuda::DataWriter::WriteDipoleToVTI
 */
bool protein_cuda::DataWriter::WriteTiDisplVTI(unsigned int frameIdx0,
        unsigned int frameIdx1, unsigned int avgOffs) {
    using namespace vislib::sys;

	protein_calls::CrystalStructureDataCall *dc = this->dataCallerSlot.CallAs<protein_calls::CrystalStructureDataCall>();
    if(dc == NULL) {
        this->jobDone = true;
        return false;
    }

    // Get extend
	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetExtent)) {
        this->jobDone = true;
        return false;
    }

    float gridspacing = 1.0f;
    // Compute grid dimensions
    float mincoord[3], maxcoord[3];
    mincoord[0] = dc->AccessBoundingBoxes().ObjectSpaceBBox().Left();
    mincoord[1] = dc->AccessBoundingBoxes().ObjectSpaceBBox().Bottom();
    mincoord[2] = dc->AccessBoundingBoxes().ObjectSpaceBBox().Back();
    maxcoord[0] = dc->AccessBoundingBoxes().ObjectSpaceBBox().Right();
    maxcoord[1] = dc->AccessBoundingBoxes().ObjectSpaceBBox().Top();
    maxcoord[2] = dc->AccessBoundingBoxes().ObjectSpaceBBox().Front();
    this->xaxis[0] = maxcoord[0]-mincoord[0];
    this->yaxis[1] = maxcoord[1]-mincoord[1];
    this->zaxis[2] = maxcoord[2]-mincoord[2];
    this->numvoxels[0] = (int) ceil(this->xaxis[0] / gridspacing);
    this->numvoxels[1] = (int) ceil(this->yaxis[1] / gridspacing);
    this->numvoxels[2] = (int) ceil(this->zaxis[2] / gridspacing);
    this->xaxis[0] = (this->numvoxels[0]-1) * gridspacing;
    this->yaxis[1] = (this->numvoxels[1]-1) * gridspacing;
    this->zaxis[2] = (this->numvoxels[2]-1) * gridspacing;
    maxcoord[0] = mincoord[0] + this->xaxis[0];
    maxcoord[1] = mincoord[1] + this->yaxis[1];
    maxcoord[2] = mincoord[2] + this->zaxis[2];
    this->origin[0] = mincoord[0];
    this->origin[1] = mincoord[1];
    this->origin[2] = mincoord[2];

    // Allocate memory for grid data
    float *griddata=NULL;
    griddata = new float[this->numvoxels[0]*this->numvoxels[1]*this->numvoxels[2]*3];

    for(unsigned int fr = frameIdx0; fr <= frameIdx1; fr++) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: FRAME %i | Computing grid\n", this->ClassName(), fr);

        // Calculate displacement for this frame
        dc->SetFrameID(fr, true);
        if(!this->CalcMapTiDisplAvg(dc, avgOffs, 1, 1.0f, 1.0f, 1.0f)) {
            this->jobDone = true;
            Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                    "%s: Unable to calculate displacement map\n",
                    this->ClassName());
            delete[] griddata;
            return false;
        }

        // Copy data from device to host

        CUDAQuickSurf *cqs = (CUDAQuickSurf *) this->cudaqsurf;
        checkCudaErrors(cudaMemcpy(
                griddata,
                cqs->getColorMap(),
                this->numvoxels[0]*this->numvoxels[1]*this->numvoxels[2]*3*sizeof(float),
                cudaMemcpyDeviceToHost));

        //printf("CUDA copy done. (%f %f %f)\n", griddata[0], griddata[1], griddata[2]); // DEBUG
        Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "%s: FRAME %i | Writing to file\n", this->ClassName(), fr);
            // Write frame data to *.vti file
            float step[] = {1.0f, 1.0f, 1.0f};
            std::stringstream ss;
            ss << "BaTiO3_625000at_offs" << avgOffs;
            if(!this->WriteFrame2VTI(ss.str(),
                    vislib::TString("TiDispl"),
                    this->origin,
                    step,
                    this->numvoxels,
                    fr,
                    griddata)) {
                this->jobDone = true;
                Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                        "%s: Unable to write file\n",
                        this->ClassName());
                delete[] griddata;
                return false;
        }
    }

    if(griddata != NULL) delete[] griddata;
    return true;
}


/*
 * protein_cuda::DataWriter::calcMapDisplacement
 */
bool protein_cuda::DataWriter::CalcMapDipoleAvg(protein_calls::CrystalStructureDataCall *dc,
        int offset,
        int quality,
        float radscale,
        float gridspacing,
        float isoval) {

    float *xyzr = NULL;
    float *color = NULL;
    xyzr = (float *) malloc(dc->GetCellCnt() * sizeof(float) * 4);
    color = (float *) malloc(dc->GetCellCnt() * sizeof(float) * 4);
    if(this->frameData0 == NULL) {
        this->frameData0 = new float[dc->GetAtomCnt()*7];
    }
    if(this->addedPos == NULL) {
        this->addedPos = new float[3*dc->GetAtomCnt()];
    }

    float invOffs = 1.0f/static_cast<float>(offset);

    // Init arrays
#pragma omp parallel for
    for (int c = 0; c < static_cast<int>(dc->GetCellCnt()); c++) {
        color[c*4+0] = 0.0f;
        color[c*4+1] = 0.0f;
        color[c*4+2] = 0.0f;
        color[c*4+3] = 1.0f;
        xyzr[c*4+0] = 0.0f;
        xyzr[c*4+1] = 0.0f;
        xyzr[c*4+2] = 0.0f;
        xyzr[c*4+3] = 2.5f; // Assumed 'radius' for dipoles
    }
    // Add atom positions over first time window if necessary
    int idxBeg = static_cast<int>(dc->FrameID());


#pragma omp parallel for
    for (int c = 0; c < static_cast<int>(dc->GetAtomCnt()); c++) {
        this->addedPos[c*3+0] = 0.0f;
        this->addedPos[c*3+1] = 0.0f;
        this->addedPos[c*3+2] = 0.0f;
    }


    for(int fr = idxBeg; fr < idxBeg+offset; fr++) {

        // Get data from data source
        dc->SetFrameID(fr, true);
        dc->SetCalltime(static_cast<float>(fr));
		if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetData)) {
            this->jobDone = true;
            return false;
        }
        //memcpy(this->frameData0, dc->GetFrameData(), dc->GetAtomCnt()*7*sizeof(float));
        dc->Unlock();

#pragma omp parallel for
        for (int at = 0; at < static_cast<int>(dc->GetAtomCnt()); at++) {
            this->addedPos[at*3+0] += (this->frameData0[at*7+0]*invOffs);
            this->addedPos[at*3+1] += (this->frameData0[at*7+1]*invOffs);
            this->addedPos[at*3+2] += (this->frameData0[at*7+2]*invOffs);
        }
    }



    // Calculate dipole moment per cell
#pragma omp parallel for
    for (int c = 0; c < static_cast<int>(dc->GetCellCnt()); c++) {
        int idxTiAtom = dc->GetCells()[15*c+14];
        xyzr[c*4+0] = this->addedPos[idxTiAtom*3+0]-this->origin[0];
        xyzr[c*4+1] = this->addedPos[idxTiAtom*3+1]-this->origin[1];
        xyzr[c*4+2] = this->addedPos[idxTiAtom*3+2]-this->origin[2];

        // Check whether the cell is valid, if not go to next cell
        bool isValid = true;
        for (int cnt = 0; cnt < 15; cnt++) {
            if (dc->GetCells()[c*15+cnt] == -1) isValid = false;
        }

        if (!isValid) {
            continue;
        }

        vislib::math::Vector<float, 3> tmpVec1, tmpVec2, diffVec;

        // Calculate mass center of anions (= oxygen atoms)
        float anionCenter[] = {0.0, 0.0, 0.0};
        for(int oxy = 8; oxy < 14; oxy++) {
            anionCenter[0] += this->addedPos[3 * dc->GetCells()[15*c+oxy] + 0];
            anionCenter[1] += this->addedPos[3 * dc->GetCells()[15*c+oxy] + 1];
            anionCenter[2] += this->addedPos[3 * dc->GetCells()[15*c+oxy] + 2];
        }
        anionCenter[0] /= 6.0f;
        anionCenter[1] /= 6.0f;
        anionCenter[2] /= 6.0f;

        tmpVec1.Set(anionCenter[0], anionCenter[1], anionCenter[2]);

        // Calculate mass center of canions (= titanium and barium atoms)
        float cationCenter[] = {0.0f, 0.0f, 0.0f};
        for(int at = 0; at < 8; at++) {
            cationCenter[0] += this->addedPos[3 * dc->GetCells()[15*c+at] + 0];
            cationCenter[1] += this->addedPos[3 * dc->GetCells()[15*c+at] + 1];
            cationCenter[2] += this->addedPos[3 * dc->GetCells()[15*c+at] + 2];
        }
        cationCenter[0] += this->addedPos[3 * idxTiAtom + 0];
        cationCenter[1] += this->addedPos[3 * idxTiAtom + 1];
        cationCenter[2] += this->addedPos[3 * idxTiAtom + 2];
        cationCenter[0] /= 9.0f;
        cationCenter[1] /= 9.0f;
        cationCenter[2] /= 9.0f;

        tmpVec2.Set(cationCenter[0], cationCenter[1], cationCenter[2]);

        diffVec = tmpVec2 - tmpVec1;

        color[c*4+0] = diffVec.X();
        color[c*4+1] = diffVec.Y();
        color[c*4+2] = diffVec.Z();
    }


    // Set gaussian window size based on user-specified quality parameter
    float gausslim;
    switch (quality) {
    case 3: gausslim = 4.0f; break; // max quality
    case 2: gausslim = 3.0f; break; // high quality
    case 1: gausslim = 2.5f; break; // medium quality
    case 0:
    default: gausslim = 2.0f; // low quality
        break;
    }

    //for (int c = 0; c < static_cast<int>(dc->GetCellCnt()); c++) {
    //    printf("%i: %f %f %f %f\n", c, color[c*4+0], color[c*4+1], color[c*4+2], color[c*4+3]);
    //} // DEBUG

    // compute both density map and floating point color texture map
    CUDAQuickSurf *cqs = (CUDAQuickSurf *) this->cudaqsurf;
    int rc = cqs->calc_map(dc->GetCellCnt(),
                &xyzr[0],
                &color[0],
                true, this->origin,
                this->numvoxels,
                2.5f, // Max radius
                1.0f, // radius scaling
                gridspacing, isoval, gausslim);

    if (rc == 0) {
        free(xyzr);
        if (color) free(color);
        return true;
    } else {
        free(xyzr);
        if (color) free(color);
        return false;
    }
}


/*
 * protein_cuda::DataWriter::CalcMapTiDisplAvg
 */
bool protein_cuda::DataWriter::CalcMapTiDisplAvg(protein_calls::CrystalStructureDataCall *dc,
        int offset,
        int quality,
        float radscale,
        float gridspacing,
        float isoval) {

    float *xyzr = NULL;
    xyzr = (float *) malloc(dc->GetCellCnt() * sizeof(float) * 4);
    if(this->frameData0 == NULL) {
        this->frameData0 = new float[dc->GetAtomCnt()*7];
    }
    if(this->frameData1== NULL) {
        this->frameData1 = new float[dc->GetAtomCnt()*7];
    }
    if(this->addedTiDispl == NULL) {
        this->addedTiDispl = new float[4*dc->GetCellCnt()];
    }


    // Get data from data source
	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetExtent)) {
        this->jobDone = true;
        return false;
    }


#pragma omp parallel for
    for (int c = 0; c < static_cast<int>(dc->GetCellCnt()); c++) {
        this->addedTiDispl[c*4+0] = 0.0f;
        this->addedTiDispl[c*4+1] = 0.0f;
        this->addedTiDispl[c*4+2] = 0.0f;
        this->addedTiDispl[c*4+3] = 1.0f;
    }


    // Get data from data source
	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetData)) {
        this->jobDone = true;
        return false;
    }
    //memcpy(this->frameData0, dc->GetFrameData(), dc->GetAtomCnt()*7*sizeof(float));
    dc->Unlock();

    // Get data from data source
    if(dc->FrameID()-offset >= 0) {
        dc->SetFrameID(dc->FrameID()-offset, true);
    }
    dc->SetCalltime(static_cast<float>(dc->FrameID()));
	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetData)) {
        this->jobDone = true;
        return false;
    }
    //memcpy(this->frameData1, dc->GetFrameData(), dc->GetAtomCnt()*7*sizeof(float));
    dc->Unlock();

#pragma omp parallel for
    for (int c = 0; c < static_cast<int>(dc->GetCellCnt()); c++) {
        unsigned int at = dc->GetCells()[c*15+14]; // Get ti atom idx
        this->addedTiDispl[c*4+0] = this->frameData0[at*7+0]-this->frameData1[at*7+0];
        this->addedTiDispl[c*4+1] = this->frameData0[at*7+1]-this->frameData1[at*7+1];
        this->addedTiDispl[c*4+2] = this->frameData0[at*7+2]-this->frameData1[at*7+2];
    }

// Setup positions
#pragma omp parallel for
    for (int c = 0; c < static_cast<int>(dc->GetCellCnt()); c++) {
        unsigned int at = dc->GetCells()[c*15+14]; // Get ti atom idx
        xyzr[c*4+0] = this->frameData0[at*7+0] - this->origin[0];
        xyzr[c*4+1] = this->frameData0[at*7+1] - this->origin[1];
        xyzr[c*4+2] = this->frameData0[at*7+2] - this->origin[2];
        xyzr[c*4+3] = 2.5f; // Assumed radius
    }
    ////////////////////////////////////////////////////////////////////////////

    // Set gaussian window size based on user-specified quality parameter
    float gausslim;
    switch (quality) {
    case 3: gausslim = 4.0f; break; // max quality
    case 2: gausslim = 3.0f; break; // high quality
    case 1: gausslim = 2.5f; break; // medium quality
    case 0:
    default: gausslim = 2.0f; // low quality
        break;
    }



    /*for (int c = 0; c < static_cast<int>(dc->GetCellCnt()); c++) {
        printf("PT %i Pos(%f %f %f %f) Vec(%f %f %f %f)\n", c,
                xyzr[c*4+0], xyzr[c*4+1],
                xyzr[c*4+2],xyzr[c*4+3],
                this->addedTiDispl[c*4+0],
                this->addedTiDispl[c*4+1],
                this->addedTiDispl[c*4+2],
                this->addedTiDispl[c*4+3]);
    }*/ // DEBUG


    // compute both density map and floating point color texture map
    CUDAQuickSurf *cqs = (CUDAQuickSurf *) this->cudaqsurf;
    int rc = cqs->calc_map(dc->GetCellCnt(),
                &xyzr[0],
                &this->addedTiDispl[0],
                true, this->origin,
                this->numvoxels,
                2.5f, // Max radius
                1.0f, // radius scaling
                gridspacing, isoval, gausslim);

    if (rc == 0) {
        free(xyzr);
        return true;
    } else {
        free(xyzr);
        return false;
    }
}


/*
 * protein_cuda::DataWriter::GetNearestDistTi
 */
float protein_cuda::DataWriter::GetNearestDistTi(protein_calls::CrystalStructureDataCall *dc,
        int idx) {

    float *ti = new float[dc->GetCellCnt()*3];
    float *dist = new float[dc->GetCellCnt()];

    // Store ti atom positions
#pragma omp parallel for
    for (int c = 0; c < static_cast<int>(dc->GetCellCnt()); c++) {
        int idxTiAtom = dc->GetCells()[15*c+14];
        ti[c*3+0] = this->frameData0[idxTiAtom*7+0];
        ti[c*3+1] = this->frameData0[idxTiAtom*7+1];
        ti[c*3+2] = this->frameData0[idxTiAtom*7+2];
    }

    // Calculate distance to all other ti atoms
#pragma omp parallel for
    for (int c = 0; c < static_cast<int>(dc->GetCellCnt()); c++) {
        dist[c] = sqrt((ti[3*idx] - ti[3*c])*(ti[3*idx] - ti[3*c]) +
                       (ti[3*idx+1] - ti[3*c+1])*(ti[3*idx+1] - ti[3*c+1]) +
                       (ti[3*idx+2] - ti[3*c+2])*(ti[3*idx+2] - ti[3*c+2]));
    }

    float min = 1000.0;
    // Get minimum dist
    for (int c = 0; c < static_cast<int>(dc->GetCellCnt()); c++) {
        if((dist[c] < min)&&(idx != c)) {
            min = dist[c];
        }
    }

    delete[] ti;
    delete[] dist;

    return min;
}


/*
 * protein_cuda::DataWriter::create
 */
bool protein_cuda::DataWriter::create(void) {
    using namespace vislib::sys;
    // Create OpenGL interoperable CUDA device.
    //cudaGLSetGLDevice(cudaUtilGetMaxGflopsDeviceId());
    //printf("cudaGLSetGLDevice: %s\n", cudaGetErrorString(cudaGetLastError()));
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO, "Thrust Version: %d.%d.%d\n",
            THRUST_MAJOR_VERSION,
            THRUST_MINOR_VERSION,
            THRUST_SUBMINOR_VERSION);

    // Create quicksurf object
    if(!this->cudaqsurf ) {
        this->cudaqsurf = new CUDAQuickSurf();
    }
    return true;
}


/*
 * protein_cuda::DataWriter::release
 */
void protein_cuda::DataWriter::release(void) {
    if(this->frameData0 != NULL) delete[] this->frameData0;
    if(this->frameData1 != NULL) delete[] this->frameData1;
    if(this->frameDataDispl != NULL) delete[] this->frameDataDispl;
    if(this->addedPos != NULL) delete[] this->addedPos;
    if(this->cudaqsurf) {
        CUDAQuickSurf *cqs = (CUDAQuickSurf *) this->cudaqsurf;
        delete cqs;
    }
}


/*
 * protein_cuda::DataWriter::writeFrame2VTKLegacy
 */
bool protein_cuda::DataWriter::writeFrame2VTKLegacy(unsigned int frameIdx,
        float gridspacing,
        vislib::TString fileName) {

    using namespace vislib::sys;

    std::ofstream outfile;


    outfile.open(fileName.PeekBuffer(), std::ios::out | std::ios::binary);
    if(!outfile.good()) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
                "%s: Unable to open file %s\n",
                this->ClassName(),
                fileName.PeekBuffer());
        return false;
    }

    /// Write header data ///

    outfile << "# vtk DataFile Version 3.0" << std::endl;
    outfile << "Test" << std::endl; // TODO
    //outfile << "BINARY" << std::endl;
    outfile << "ASCII" << std::endl;
    outfile << "DATASET STRUCTURED_POINTS" << std::endl;
    outfile << "DIMENSIONS " << this->numvoxels[0] << " " << this->numvoxels[1] << " "  << this->numvoxels[2] << std::endl;
    outfile << "ORIGIN " << this->origin[0] << " " << this->origin[1] << " "  << this->origin[2] << std::endl;
    outfile << "SPACING " << gridspacing << " " << gridspacing << " " << gridspacing << std::endl;    // TODO Don't hardcode this
    outfile << "POINT_DATA " << this->numvoxels[0]*this->numvoxels[1]*this->numvoxels[2] << std::endl;

    CUDAQuickSurf *cqs = (CUDAQuickSurf *) this->cudaqsurf;
    float *testArr = new float[this->numvoxels[0]*this->numvoxels[1]*this->numvoxels[2]*3];
     checkCudaErrors(cudaMemcpy(
            testArr,
            cqs->getColorMap(),
            this->numvoxels[0]*this->numvoxels[1]*this->numvoxels[2]*3*sizeof(float),
            cudaMemcpyDeviceToHost));

    /// Write data ///

    outfile << "VECTORS displacement float" << std::endl;
    printf("Number of voxels (%i %i %i)\n", this->numvoxels[0], this->numvoxels[1], this->numvoxels[2]);
    /*for(int cnt = 0; cnt < this->numvoxels[0]*this->numvoxels[1]*this->numvoxels[2]; cnt++) {
        printf("TEST\n");
        printf("%f %f %f", cqs->getColorMap()[cnt*3], cqs->getColorMap()[cnt*3+1], cqs->getColorMap()[cnt*3+2]);

    }*/

    /*char *testArrBuff = (char *) testArr;
    // NOTE Paraview expects big endian byte ordering
    for(int cnt = 0; cnt < this->numvoxels[0]*this->numvoxels[1]*this->numvoxels[2]*3; cnt++) {
        outfile.write(&testArrBuff[cnt*4+3], 1);
        outfile.write(&testArrBuff[cnt*4+2], 1);
        outfile.write(&testArrBuff[cnt*4+1], 1);
        outfile.write(&testArrBuff[cnt*4+0], 1);
    }*/

    for(int cnt = 0; cnt < this->numvoxels[0]*this->numvoxels[1]*this->numvoxels[2]; cnt++) {
        outfile << testArr[3*cnt+0] << " " << testArr[3*cnt+1] << " " << testArr[3*cnt+2] << std::endl;
    }

    outfile.close();
    delete[] testArr;
    return true;
}


/*
 * protein_cuda::DataWriter::PutAvgCellLength
 */
bool protein_cuda::DataWriter::PutAvgCellLengthAlt(unsigned int idxStart, unsigned int idxEnd,
		protein_calls::CrystalStructureDataCall *dc) {
    using namespace vislib;
    using namespace vislib::math;
    std::ofstream outputStr;
    std::ofstream outputStr1;

	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetExtent)) {
        return false;
    }

    unsigned int idxEndChecked;
    if( dc->FrameCount() > idxEnd && idxEnd > idxStart )
        idxEndChecked = idxEnd;
    else
        idxEndChecked = dc->FrameCount();

    // filename
    unsigned int offset = 500 - dc->FrameCount() + 1;
    char buffer[30];
    sprintf( buffer, "CellVol_Avg%u.txt", offset);

    outputStr.open( buffer, std::ios::out);
    outputStr.precision(4);
    //outputStr1.open("MinVol_Avg25.txt", std::ios::out);
    //outputStr1.precision(4);

    for(unsigned int cnt = idxStart; cnt < idxEndChecked; cnt++) {

        float maxDist = -1.0f;
        float maxVol = -1.0f;
        float minVol = 1000.0f;
        float avgVol = 0.0f;
        unsigned int avgVolCellCnt = 0;
        float avgDist = 0.0f;
        unsigned int avgDistCellCnt = 0;

        printf("FRAME %u\n", cnt);
        // Get data from data source
        dc->SetFrameID(cnt, true);
        dc->SetCalltime(static_cast<float>(cnt));
		if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetData)) {
            this->jobDone = true;
            return false;
        }
        //memcpy(this->frameData0, dc->GetFrameData(), dc->GetAtomCnt()*7*sizeof(float));
        dc->Unlock();

        /*for(unsigned int c = 0; c < dc->GetCellCnt(); c++) {
            // Print cell info
            Vector<float, 3> A(&dc->GetAtomPos()[dc->GetCells()[15*c+0]*3]);
            Vector<float, 3> B(&dc->GetAtomPos()[dc->GetCells()[15*c+1]*3]);
            Vector<float, 3> C(&dc->GetAtomPos()[dc->GetCells()[15*c+2]*3]);
            Vector<float, 3> D(&dc->GetAtomPos()[dc->GetCells()[15*c+3]*3]);
            Vector<float, 3> E(&dc->GetAtomPos()[dc->GetCells()[15*c+4]*3]);
            Vector<float, 3> F(&dc->GetAtomPos()[dc->GetCells()[15*c+5]*3]);
            Vector<float, 3> G(&dc->GetAtomPos()[dc->GetCells()[15*c+6]*3]);
            Vector<float, 3> H(&dc->GetAtomPos()[dc->GetCells()[15*c+7]*3]);

            Vector<float, 3> O_0(&dc->GetAtomPos()[dc->GetCells()[15*c+8]*3]);
            Vector<float, 3> O_1(&dc->GetAtomPos()[dc->GetCells()[15*c+9]*3]);
            Vector<float, 3> O_2(&dc->GetAtomPos()[dc->GetCells()[15*c+10]*3]);
            Vector<float, 3> O_3(&dc->GetAtomPos()[dc->GetCells()[15*c+11]*3]);
            Vector<float, 3> O_4(&dc->GetAtomPos()[dc->GetCells()[15*c+12]*3]);
            Vector<float, 3> O_5(&dc->GetAtomPos()[dc->GetCells()[15*c+13]*3]);

            Vector<float, 3> Ti(&dc->GetAtomPos()[dc->GetCells()[15*c+14]*3]);

            outputStr << (O_1-Ti).Length() << ", " <<  (O_0-Ti).Length() << ", " << (O_2-Ti).Length() << ", ";
        }

        outputStr << std::endl;

        for(unsigned int c = 0; c < dc->GetCellCnt(); c++) {
            // Print cell info
            Vector<float, 3> A(&dc->GetAtomPos()[dc->GetCells()[15*c+0]*3]);
            Vector<float, 3> B(&dc->GetAtomPos()[dc->GetCells()[15*c+1]*3]);
            Vector<float, 3> C(&dc->GetAtomPos()[dc->GetCells()[15*c+2]*3]);
            Vector<float, 3> D(&dc->GetAtomPos()[dc->GetCells()[15*c+3]*3]);
            Vector<float, 3> E(&dc->GetAtomPos()[dc->GetCells()[15*c+4]*3]);
            Vector<float, 3> F(&dc->GetAtomPos()[dc->GetCells()[15*c+5]*3]);
            Vector<float, 3> G(&dc->GetAtomPos()[dc->GetCells()[15*c+6]*3]);
            Vector<float, 3> H(&dc->GetAtomPos()[dc->GetCells()[15*c+7]*3]);

            Vector<float, 3> O_0(&dc->GetAtomPos()[dc->GetCells()[15*c+8]*3]);
            Vector<float, 3> O_1(&dc->GetAtomPos()[dc->GetCells()[15*c+9]*3]);
            Vector<float, 3> O_2(&dc->GetAtomPos()[dc->GetCells()[15*c+10]*3]);
            Vector<float, 3> O_3(&dc->GetAtomPos()[dc->GetCells()[15*c+11]*3]);
            Vector<float, 3> O_4(&dc->GetAtomPos()[dc->GetCells()[15*c+12]*3]);
            Vector<float, 3> O_5(&dc->GetAtomPos()[dc->GetCells()[15*c+13]*3]);

            Vector<float, 3> Ti(&dc->GetAtomPos()[dc->GetCells()[15*c+14]*3]);

            outputStr << (O_4-Ti).Length() << ", " <<  (O_5-Ti).Length() << ", " << (O_3-Ti).Length() << ", ";
        }

        outputStr << std::endl;*/


        for(unsigned int c = 0; c < dc->GetCellCnt(); c++) {

            /*printf("FRAME %u, CELL %u, (", cnt, c);
            for( unsigned int aIdx = 0; aIdx < 15; aIdx++ ) {
                printf("%i ", dc->GetCells()[15*c+aIdx]);
            }
            printf(")");*/
            // check if cell is valid
            bool cellValid = true;
            for( unsigned int aIdx = 0; aIdx < 15; aIdx++ ) {
                if( dc->GetCells()[15*c+aIdx] < 0 ) {
                    cellValid = false;
                    //break;
                }
            }
            if( !cellValid ) {
                //printf("  NOT VALID\n");
                continue;
            }
            //printf("\n");

            // Print cell info
            Vector<float, 3> A(&dc->GetAtomPos()[dc->GetCells()[15*c+0]*3]);
            Vector<float, 3> B(&dc->GetAtomPos()[dc->GetCells()[15*c+1]*3]);
            Vector<float, 3> C(&dc->GetAtomPos()[dc->GetCells()[15*c+2]*3]);
            Vector<float, 3> D(&dc->GetAtomPos()[dc->GetCells()[15*c+3]*3]);
            Vector<float, 3> E(&dc->GetAtomPos()[dc->GetCells()[15*c+4]*3]);
            Vector<float, 3> F(&dc->GetAtomPos()[dc->GetCells()[15*c+5]*3]);
            Vector<float, 3> G(&dc->GetAtomPos()[dc->GetCells()[15*c+6]*3]);
            Vector<float, 3> H(&dc->GetAtomPos()[dc->GetCells()[15*c+7]*3]);

            Vector<float, 3> O_0(&dc->GetAtomPos()[dc->GetCells()[15*c+8]*3]);
            Vector<float, 3> O_1(&dc->GetAtomPos()[dc->GetCells()[15*c+9]*3]);
            Vector<float, 3> O_2(&dc->GetAtomPos()[dc->GetCells()[15*c+10]*3]);
            Vector<float, 3> O_3(&dc->GetAtomPos()[dc->GetCells()[15*c+11]*3]);
            Vector<float, 3> O_4(&dc->GetAtomPos()[dc->GetCells()[15*c+12]*3]);
            Vector<float, 3> O_5(&dc->GetAtomPos()[dc->GetCells()[15*c+13]*3]);

            Vector<float, 3> Ti(&dc->GetAtomPos()[dc->GetCells()[15*c+14]*3]);

            /*printf("FRAME %u, CELL %u, (", cnt, c);
            for( unsigned int aIdx = 0; aIdx < 15; aIdx++ ) {
                printf(" (%.4f %.4f %.4f) ",
                        dc->GetAtomPos()[dc->GetCells()[15*c+aIdx]*3+0],
                        dc->GetAtomPos()[dc->GetCells()[15*c+aIdx]*3+1],
                        dc->GetAtomPos()[dc->GetCells()[15*c+aIdx]*3+2]);
            }
            printf(" )\n");*/

            float vol = this->CalcCellVolume(A, B, C, D, E, F, G, H);
            if((vol > 100.0f)||(vol < 28.0f)) continue;

            //if(vol > maxVol) maxVol = vol;
            //if(vol < minVol) minVol = vol;

            /*float distX = fabs((O_1-Ti).Length() - (O_4-Ti).Length());
            float distY = fabs((O_0-Ti).Length() - (O_5-Ti).Length());
            float distZ = fabs((O_2-Ti).Length() - (O_3-Ti).Length());
            if( distX > 3.5f || distY > 3.5f || distZ > 3.5f ) {
                printf("CELL %u dist x %f, dist y %f, dist z %f\n", c, distX, distY,distZ);
                continue;

            }

            float dist = distX > distY ? distX : distY;
            dist = distZ > dist ? distZ : dist;

            avgDist += dist;
            avgDistCellCnt++;

            //if((maxDist < dist)&&(dist < 3.0f)) maxDist = dist;
            //outputStr << dist << ", ";*/

            // compute average volunme (sum up and count all valid cell volumes)
            avgVol += vol;
            avgVolCellCnt++;
        }
        //outputStr << static_cast<float>(cnt+25) << " " << maxDist;
        //outputStr << static_cast<float>(cnt+12) << " " << minVol;
        if( avgVolCellCnt > 0 ) {
            outputStr << static_cast<float>(cnt + offset / 2)  << " " << (avgVol / float(avgVolCellCnt));
            outputStr << std::endl;
        }
        if( avgDistCellCnt > 0 ) {
            outputStr << static_cast<float>(cnt + offset / 2)  << " " << (avgDist / float(avgDistCellCnt));
            outputStr << std::endl;
        }

        printf("FRAME %u, nCells %u\n", cnt, avgVolCellCnt);
        //outputStr1 << static_cast<float>(cnt+17) << " " << maxVol;
        //outputStr1 << std::endl;
    }

    outputStr.close();
    //outputStr1.close();

    return true;
}


/*
 * protein_cuda::DataWriter::PutAvgCellLength
 */
bool protein_cuda::DataWriter::PutAvgCellLength(unsigned int idxStart, unsigned int offs,
		protein_calls::CrystalStructureDataCall *dc) {
    using namespace vislib;
    using namespace vislib::math;

    unsigned int cellIdx = 29950; // zw. Mitte u. Rand
    //unsigned int cellIdx = 2450; // Mitte

    float invOffs = 1.0f/static_cast<float>(offs);
    float avgPos[15*3];
    unsigned int atmIdx[15]; // Atom indices of the cell


	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetExtent)) {
        return false;
    }

    // Get atom indices of the cell
#pragma omp parallel for
    for (int at = 0; at < 15; at++) {
        atmIdx[at] = dc->GetCells()[cellIdx*15+at];
    }

    for(unsigned int cnt = idxStart; cnt+offs < dc->FrameCount(); cnt++) {

        // Init with zero
#pragma omp parallel for
        for (int at = 0; at < 15; at++) {
            avgPos[at*3+0] = 0.0f;
            avgPos[at*3+1] = 0.0f;
            avgPos[at*3+2] = 0.0f;
        }


        // Calc avg position
        for(unsigned int fr = cnt; fr < cnt+offs; fr++) {
            // Get data from data source
            dc->SetFrameID(fr, true);                         // Set 'force' flag
            dc->SetCalltime(static_cast<float>(fr));
			if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetData)) {
                return false;
            }

#pragma omp parallel for
            for (int at = 0; at < 15; at++) {
                /*avgPos[at*3+0] += (dc->GetFrameData()[atmIdx[at]*7+0]*invOffs);
                avgPos[at*3+1] += (dc->GetFrameData()[atmIdx[at]*7+1]*invOffs);
                avgPos[at*3+2] += (dc->GetFrameData()[atmIdx[at]*7+2]*invOffs);*/
            }
            dc->Unlock();
        }

        // Print cell info

        Vector<float, 3> A(&avgPos[0*3]);
        Vector<float, 3> B(&avgPos[1*3]);
        Vector<float, 3> C(&avgPos[2*3]);
        Vector<float, 3> D(&avgPos[3*3]);
        Vector<float, 3> E(&avgPos[4*3]);
        Vector<float, 3> F(&avgPos[5*3]);
        Vector<float, 3> G(&avgPos[6*3]);
        Vector<float, 3> H(&avgPos[7*3]);

        Vector<float, 3> O_0(&avgPos[8*3]);
        Vector<float, 3> O_1(&avgPos[9*3]);
        Vector<float, 3> O_2(&avgPos[10*3]);
        Vector<float, 3> O_3(&avgPos[11*3]);
        Vector<float, 3> O_4(&avgPos[12*3]);
        Vector<float, 3> O_5(&avgPos[13*3]);

        Vector<float, 3> Ti(&avgPos[14*3]);

        //printf("%f (%f %f %f)\n", static_cast<float>(cnt), A.X(), A.Y(), A.Z());   // Position of first Ba atom

        float vol = this->CalcCellVolume(A, B, C, D, E, F, G, H);
        printf("CELL %i |  %5.1f %.4f ", cellIdx, static_cast<float>(cnt), vol);                           // Cell volume

        printf("x: (%.4f %.4f %.4f %.4f) %.4f (%.4f %.4f) ", (E-A).Length(), (G-C).Length(),
                (H-D).Length(), (F-B).Length(),
                ((E-A).Length()+(G-C).Length()+(H-D).Length()+(F-B).Length())*0.25,
                (O_1-Ti).Length(), (O_4-Ti).Length()
                ); // Cell Length x axis

        printf("y: (%.4f %.4f %.4f %.4f) %.4f (%.4f %.4f) ", (C-A).Length(), (G-E).Length(),
                (H-F).Length(), (D-B).Length(),
                ((C-A).Length()+(G-E).Length()+(H-F).Length()+(D-B).Length())*0.25,
                (O_0-Ti).Length(), (O_5-Ti).Length()); // Cell Length y axis

        printf("z: (%.4f %.4f %.4f %.4f) %.4f (%.4f %.4f)\n", (B-A).Length(), (F-E).Length(),
                (H-G).Length(), (D-C).Length(),
                ((B-A).Length()+(F-E).Length()+(H-G).Length()+(D-C).Length())*0.25,
                (O_2-Ti).Length(), (O_3-Ti).Length()); // Cell Length z axis


    }

    return true;
}


/*
 * protein_cuda::DataWriter::CalcCellVolume
 */
float protein_cuda::DataWriter::CalcVolTetrahedron(
        vislib::math::Vector<float, 3> A,
        vislib::math::Vector<float, 3> B,
        vislib::math::Vector<float, 3> C,
        vislib::math::Vector<float, 3> D) {

    using namespace vislib;
    using namespace vislib::math;

    Vector <float, 3> vecA = B-A;
    Vector <float, 3> vecB = C-A;
    Vector <float, 3> vecC = D-A;

    float res = fabs(vecA.Cross(vecB).Dot(vecC));
    res /= 6.0f;
    return res;
}

/*
 * protein_cuda::DataWriter::CalcCellVolume
 */
float protein_cuda::DataWriter::CalcCellVolume(
        vislib::math::Vector<float, 3> A,
        vislib::math::Vector<float, 3> B,
        vislib::math::Vector<float, 3> C,
        vislib::math::Vector<float, 3> D,
        vislib::math::Vector<float, 3> E,
        vislib::math::Vector<float, 3> F,
        vislib::math::Vector<float, 3> G,
        vislib::math::Vector<float, 3> H) {

    float res = 0.0f;

    res += this->CalcVolTetrahedron(A, B, C, E);
    res += this->CalcVolTetrahedron(D, C, B, H);
    res += this->CalcVolTetrahedron(G, C, H, E);
    res += this->CalcVolTetrahedron(F, B, H, E);
    res += this->CalcVolTetrahedron(H, C, B, E);

    return res;
}


/*
 * protein_cuda::DataWriter::WriteFrameFileBinAvg
 */
bool protein_cuda::DataWriter::WriteFrameFileBinAvg(protein_calls::CrystalStructureDataCall *dc) {

    const unsigned int NATOMS = 625000;

    // Get data from data source
	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetExtent)) {
        this->jobDone = true;
        return false;
    }

    std::ofstream outStr;
    std::string outFile = "scivis/atomPos_avg/bto_625000at_avg200.bin";

    outStr.open(outFile.c_str(),
            std::fstream::binary | std::fstream::app);
    if(!outStr.is_open()) {
        std::cout << "ERR could not open file '" << outFile << "'" << std::endl;
        return false;
    }

    unsigned int avgOffset = 200;
    //unsigned int avgOffset = 25;

    float invOffs = 1.0f/static_cast<float>(avgOffset);
    float *buff = new float[NATOMS*3];

    // Init with zero
#pragma omp parallel for
    for (int at = 0; at < static_cast<int>(dc->GetAtomCnt()); at++) {
        buff[3*at+0] = 0.0f;
        buff[3*at+1] = 0.0f;
        buff[3*at+2] = 0.0f;
    }

    printf("Writing frame %i ...\n", 0);
    // Compute average of first time window
    for(unsigned int fr = 0; fr < avgOffset; fr++) {

        // Get data from data source
        dc->SetFrameID(fr, true);                         // Set 'force' flag
        dc->SetCalltime(static_cast<float>(fr));
		if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetData)) {
            return false;
        }

#pragma omp parallel for
        for (int at = 0; at < static_cast<int>(dc->GetAtomCnt()); at++) {
            buff[3*at+0] += (dc->GetAtomPos()[3*at+0]*invOffs);
            buff[3*at+1] += (dc->GetAtomPos()[3*at+1]*invOffs);
            buff[3*at+2] += (dc->GetAtomPos()[3*at+2]*invOffs);
        }
        dc->Unlock();
    }
    outStr.write((char *)(buff), NATOMS*3*sizeof(float));

    for(unsigned int w = 1; w+avgOffset-1 < dc->FrameCount(); w++) {

        printf("Writing frame %u ...\n", w);

        // Subtract last frame
        dc->SetFrameID(w-1, true);                         // Set 'force' flag
        dc->SetCalltime(static_cast<float>(w-1));
		if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetData)) {
            return false;
        }
#pragma omp parallel for
        for (int at = 0; at < static_cast<int>(dc->GetAtomCnt()); at++) {
            buff[3*at+0] -= (dc->GetAtomPos()[3*at+0]*invOffs);
            buff[3*at+1] -= (dc->GetAtomPos()[3*at+1]*invOffs);
            buff[3*at+2] -= (dc->GetAtomPos()[3*at+2]*invOffs);
        }
        dc->Unlock();

        // Add new frame
        dc->SetFrameID(w+avgOffset-1, true);                         // Set 'force' flag
        dc->SetCalltime(static_cast<float>(w+avgOffset-1));
		if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetData)) {
            return false;
        }
#pragma omp parallel for
        for (int at = 0; at < static_cast<int>(dc->GetAtomCnt()); at++) {
            buff[3*at+0] += (dc->GetAtomPos()[3*at+0]*invOffs);
            buff[3*at+1] += (dc->GetAtomPos()[3*at+1]*invOffs);
            buff[3*at+2] += (dc->GetAtomPos()[3*at+2]*invOffs);
        }
        dc->Unlock();

        outStr.write((char *)(buff), NATOMS*3*sizeof(float)); // Write to file
    }

    delete[] buff;
    outStr.close();

    return true;
}


/*
 * protein_cuda::DataWriter::PutCubeSize
 */
bool protein_cuda::DataWriter::PutCubeSize(unsigned int frIdx0, unsigned int frIdx1,
		protein_calls::CrystalStructureDataCall *dc) {

    using namespace vislib::math;

	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetExtent)) {
        this->jobDone = true;
        return false;
    }

    for(unsigned int fr = frIdx0; fr <= frIdx1; fr++) {
        // Get data of the frame
        dc->SetFrameID(fr, true);                         // Set 'force' flag
        dc->SetCalltime(static_cast<float>(fr));
		if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetData)) {
            return false;
        }
        // Find Min and Max coords
        Vector<float, 3> maxCoord(0.0f, 0.0f, 0.0f), minCoord(0.0f, 0.0f, 0.0f);
        for(unsigned int at = 0; at < dc->GetAtomCnt(); at++) {
            if(dc->GetAtomPos()[3*at+0] > maxCoord.X()) {
                maxCoord.SetX(dc->GetAtomPos()[3*at+0]);
            }
            if(dc->GetAtomPos()[3*at+1] > maxCoord.Y()) {
                maxCoord.SetY(dc->GetAtomPos()[3*at+1]);
            }
            if(dc->GetAtomPos()[3*at+2] > maxCoord.Z()) {
                maxCoord.SetZ(dc->GetAtomPos()[3*at+2]);
            }
            if(dc->GetAtomPos()[3*at+0] < minCoord.X()) {
                minCoord.SetX(dc->GetAtomPos()[3*at+0]);
            }
            if(dc->GetAtomPos()[3*at+1] < minCoord.Y()) {
                minCoord.SetY(dc->GetAtomPos()[3*at+1]);
            }
            if(dc->GetAtomPos()[3*at+2] < minCoord.Z()) {
                minCoord.SetZ(dc->GetAtomPos()[3*at+2]);
            }
        }
        dc->Unlock();

        // Calculate volume
        float vol = (maxCoord.X() - minCoord.X())*(maxCoord.Y() - minCoord.Y())
                *(maxCoord.Z() - minCoord.Z());

        //printf("%f %f\n", static_cast<float>(fr+12), vol); // Print cube volume
        //printf("%f %f\n", static_cast<float>(fr+12), (maxCoord.X() - minCoord.X())); // Print x axis
        //printf("%f %f\n", static_cast<float>(fr+12), (maxCoord.Y() - minCoord.Y())); // Print y axis
        printf("%f %f\n", static_cast<float>(fr+12), (maxCoord.Z() - minCoord.Z())); // Print z axis
    }

    return true;
}


/*
 * protein_cuda::DataWriter::WriteTiDispl
 */
bool protein_cuda::DataWriter::ReadTiDispl(
		protein_calls::CrystalStructureDataCall *dc) {

    std::fstream inStr;
    std::string inFile = "avg25TiDispl.bin";

	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetExtent)) {
        this->jobDone = true;
        return false;
    }

    inStr.open(inFile.c_str(), std::fstream::binary | std::fstream::in);
    if(!inStr.is_open()) {
        std::cout << "ERR could not open file '" << inFile << "'" << std::endl;
        return false;
    }

    float *buff = new float[dc->GetCellCnt()*6];
    unsigned int frame = 0;
    while(inStr.good()) {
        inStr.read((char *)buff, dc->GetCellCnt()*24);
        for(int cnt = 0; cnt < 125000; cnt++) {
            printf("FRAME %u pos %f %f %f vec %f %f %f\n", frame, buff[cnt*6+0],
                    buff[cnt*6+1],buff[cnt*6+2], buff[cnt*6+3],
                    buff[cnt*6+4], buff[cnt*6+5]);
        }
        frame++;
    }

    delete[] buff;
    inStr.close();

    return true;
}


/*
 * protein_cuda::DataWriter::ReadDipole
 */
bool protein_cuda::DataWriter::ReadTiODipole(
		protein_calls::CrystalStructureDataCall *dc) {

    std::fstream inStr;
    std::string inFile = "TiODipole_avg25.bin";

	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetExtent)) {
        this->jobDone = true;
        return false;
    }

    inStr.open(inFile.c_str(), std::fstream::binary | std::fstream::in);
    if(!inStr.is_open()) {
        std::cout << "ERR could not open file '" << inFile << "'" << std::endl;
        return false;
    }

    float *buff = new float[dc->GetCellCnt()*6];
    unsigned int frame = 0;
    while(inStr.good()) {
        inStr.read((char *)buff, dc->GetCellCnt()*24);
        for(int cnt = 0; cnt <= 124999; cnt++) {
            printf("FRAME %u pos %f %f %f vec %f %f %f\n", frame, buff[cnt*6+0],
                    buff[cnt*6+1],buff[cnt*6+2], buff[cnt*6+3],
                    buff[cnt*6+4], buff[cnt*6+5]);
        }
        frame++;
    }

    delete[] buff;
    inStr.close();

    return true;
}


/*
 * protein_cuda::DataWriter::WriteTiDispl
 */
bool protein_cuda::DataWriter::WriteTiDispl(
		protein_calls::CrystalStructureDataCall *dc) {


    using namespace vislib::math;


	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetExtent)) {
        this->jobDone = true;
        return false;
    }

    std::ofstream outStr;
    std::string outFile = "avg25TiDispl.bin";
    bool ready;
    unsigned int tmpVal;

    outStr.open(outFile.c_str(),
            std::fstream::binary);
    if(!outStr.is_open()) {
        std::cout << "ERR could not open file '" << outFile << "'" << std::endl;
        return false;
    }

    unsigned int *idxSorted = new unsigned int[dc->GetCellCnt()];
    float *posKey = new float[dc->GetCellCnt()];

    // Init index
#pragma omp parallel for
    for(int idx = 0; idx < static_cast<int>(dc->GetCellCnt()); idx++) {
        idxSorted[idx] = idx;
    }

    // Get data of the first frame
    dc->SetFrameID(0, true);  // Set 'force' flag
    dc->SetCalltime(0.0f);
	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetData)) {
        return false;
    }

    // Get x coord
    printf("Sorting by x coord ...\n");
#pragma omp parallel for
    for(int idx = 0; idx < static_cast<int>(dc->GetCellCnt()); idx++) {
        posKey[idx] = dc->GetAtomPos()[dc->GetCells()[idx*15+14]*3+0];
    }

    // Sort by x position
    do{
        ready = true;
        for(int i = 0; i < 124999; i++) {
            if(posKey[idxSorted[i]] > posKey[idxSorted[i+1]]) {
                tmpVal = idxSorted[i];
                idxSorted[i] = idxSorted[i+1];
                idxSorted[i+1] = tmpVal;
                ready = false;
            }
        }
    } while(!ready);

    // Get y coord
#pragma omp parallel for
    for(int idx = 0; idx < static_cast<int>(dc->GetCellCnt()); idx++) {
        posKey[idx] = dc->GetAtomPos()[dc->GetCells()[idx*15+14]*3+1];
    }

    // Sort by y position
    printf("Sorting slabs by y coord ...\n");
    for(int s = 0; s < 50; s++) {
        do{
            ready = true;
            for(int i = s*2500; i < s*2500+2499; i++) {
                if(posKey[idxSorted[i]] > posKey[idxSorted[i+1]]) {
                    tmpVal = idxSorted[i];
                    idxSorted[i] = idxSorted[i+1];
                    idxSorted[i+1] = tmpVal;
                    ready = false;
                }
            }
        } while(!ready);
    }

    // Get z coord
#pragma omp parallel for
    for(int idx = 0; idx < static_cast<int>(dc->GetCellCnt()); idx++) {
        posKey[idx] = dc->GetAtomPos()[dc->GetCells()[idx*15+14]*3+2];
    }

    // Sort by z position
    printf("Sorting scanlines by z coord ...\n");
    for(int s = 0; s < 2500; s++) {
        do{
            ready = true;
            for(int i = s*50; i < s*50+49; i++) {
                if(posKey[idxSorted[i]] > posKey[idxSorted[i+1]]) {
                    tmpVal = idxSorted[i];
                    idxSorted[i] = idxSorted[i+1];
                    idxSorted[i+1] = tmpVal;
                    ready = false;
                }
            }
        } while(!ready);
    }

    // Debug print all positions
    /*for(int idx = 0; idx < static_cast<int>(dc->GetCellCnt()); idx++) {
        if(idx%2500 == 0) printf("======================================= mod2500 == 0\n");
        if(idx%50 == 0)   printf("======================================= mod50 == 0\n");
        printf("%u: (%f %f %f)\n", idx,
                dc->GetAtomPos()[3*dc->GetCells()[15*idxSorted[idx]+14]+0],
                dc->GetAtomPos()[3*dc->GetCells()[15*idxSorted[idx]+14]+1],
                dc->GetAtomPos()[3*dc->GetCells()[15*idxSorted[idx]+14]+2]);
    }*/

    float *buff = new float[6*dc->GetCellCnt()];

    for(unsigned int fr = 0; fr <= 474; fr++) {
        // Get data of the first frame
        dc->SetFrameID(fr, true);  // Set 'force' flag
        dc->SetCalltime(static_cast<float>(fr));
		if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetData)) {
            break;
        }

		for (int cnt = 0; cnt < (int)dc->GetCellCnt(); cnt++) {
            buff[cnt*6+0] = dc->GetAtomPos()[3*dc->GetCells()[idxSorted[cnt]]+0];
            buff[cnt*6+1] = dc->GetAtomPos()[3*dc->GetCells()[idxSorted[cnt]]+1];
            buff[cnt*6+2] = dc->GetAtomPos()[3*dc->GetCells()[idxSorted[cnt]]+2];
            buff[cnt*6+3] = dc->GetDipole()[3*dc->GetCells()[idxSorted[cnt]]+0];
            buff[cnt*6+4] = dc->GetDipole()[3*dc->GetCells()[idxSorted[cnt]]+1];
            buff[cnt*6+5] = dc->GetDipole()[3*dc->GetCells()[idxSorted[cnt]]+2];
        }

        dc->Unlock();
        printf("Writing frame %u ...\n", fr);
        outStr.write((char *)buff, 24*dc->GetCellCnt());
    }
    outStr.close();

    delete[] idxSorted;
    delete[] posKey;
    delete[] buff;

    return true;
}


/*
 * protein_cuda::DataWriter::WriteTiODipole
 */
bool protein_cuda::DataWriter::WriteTiODipole(
		protein_calls::CrystalStructureDataCall *dc) {


    using namespace vislib::math;


	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetExtent)) {
        this->jobDone = true;
        return false;
    }

    std::ofstream outStr;
    std::string outFile = "TiODipole_avg25.bin";
    bool ready;
    unsigned int tmpVal;

    outStr.open(outFile.c_str(),
            std::fstream::binary);
    if(!outStr.is_open()) {
        std::cout << "ERR could not open file '" << outFile << "'" << std::endl;
        return false;
    }

    unsigned int *idxSorted = new unsigned int[dc->GetCellCnt()];
    float *posKey = new float[dc->GetCellCnt()];

    // Init index
#pragma omp parallel for
    for(int idx = 0; idx < static_cast<int>(dc->GetCellCnt()); idx++) {
        idxSorted[idx] = idx;
    }

    // Get data of the first frame
    dc->SetFrameID(0, true);  // Set 'force' flag
    dc->SetCalltime(0.0f);
	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetData)) {
        return false;
    }

    // Get x coord
    printf("Sorting by x coord ...\n");
#pragma omp parallel for
    for(int idx = 0; idx < static_cast<int>(dc->GetCellCnt()); idx++) {
        posKey[idx] = dc->GetAtomPos()[dc->GetCells()[idx*15+14]*3+0];
    }

    // Sort by x position
    do{
        ready = true;
        for(int i = 0; i < 124999; i++) {
            if(posKey[idxSorted[i]] > posKey[idxSorted[i+1]]) {
                tmpVal = idxSorted[i];
                idxSorted[i] = idxSorted[i+1];
                idxSorted[i+1] = tmpVal;
                ready = false;
            }
        }
    } while(!ready);

    // Get y coord
#pragma omp parallel for
    for(int idx = 0; idx < static_cast<int>(dc->GetCellCnt()); idx++) {
        posKey[idx] = dc->GetAtomPos()[dc->GetCells()[idx*15+14]*3+1];
    }

    // Sort by y position
    printf("Sorting slabs by y coord ...\n");
    for(int s = 0; s < 50; s++) {
        do{
            ready = true;
            for(int i = s*2500; i < s*2500+2499; i++) {
                if(posKey[idxSorted[i]] > posKey[idxSorted[i+1]]) {
                    tmpVal = idxSorted[i];
                    idxSorted[i] = idxSorted[i+1];
                    idxSorted[i+1] = tmpVal;
                    ready = false;
                }
            }
        } while(!ready);
    }

    // Get z coord
#pragma omp parallel for
    for(int idx = 0; idx < static_cast<int>(dc->GetCellCnt()); idx++) {
        posKey[idx] = dc->GetAtomPos()[dc->GetCells()[idx*15+14]*3+2];
    }

    // Sort by z position
    printf("Sorting scanlines by z coord ...\n");
    for(int s = 0; s < 2500; s++) {
        do{
            ready = true;
            for(int i = s*50; i < s*50+49; i++) {
                if(posKey[idxSorted[i]] > posKey[idxSorted[i+1]]) {
                    tmpVal = idxSorted[i];
                    idxSorted[i] = idxSorted[i+1];
                    idxSorted[i+1] = tmpVal;
                    ready = false;
                }
            }
        } while(!ready);
    }

    // Debug print all positions
    /*for(int idx = 0; idx < static_cast<int>(dc->GetCellCnt()); idx++) {
        if(idx%2500 == 0) printf("======================================= mod2500 == 0\n");
        if(idx%50 == 0)   printf("======================================= mod50 == 0\n");
        printf("%u: (%f %f %f)\n", idx,
                dc->GetAtomPos()[3*dc->GetCells()[15*idxSorted[idx]+14]+0],
                dc->GetAtomPos()[3*dc->GetCells()[15*idxSorted[idx]+14]+1],
                dc->GetAtomPos()[3*dc->GetCells()[15*idxSorted[idx]+14]+2]);
    }*/

    float *buff = new float[6*dc->GetCellCnt()];
    float *dipole = new float[3*dc->GetCellCnt()];

    for(unsigned int fr = 0; fr <= 475; fr++) {
        // Get data of the first frame
        dc->SetFrameID(fr, true);  // Set 'force' flag
        dc->SetCalltime(static_cast<float>(fr));
		if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetData)) {
            break;
        }

        // Calc Ti/O-Dipole


        // Loop through all cells
#pragma omp parallel for
        for (int c = 0; c < static_cast<int>(dc->GetCellCnt()); c++) {

            int idxTiAtom = dc->GetCells()[15*c+14];

            // Check whether the cell is valid, if not go to next cell
            bool isValid = true;
            for (int cnt = 0; cnt < 15; cnt++) {
                if (dc->GetCells()[c*15+cnt] == -1) isValid = false;
            }
            if (!isValid) {
                dipole[3*c+0] = 0.0f;
                dipole[3*c+1] = 0.0f;
                dipole[3*c+2] = 0.0f;
                continue;
            }

            vislib::math::Vector<float, 3> tmpVec1, tmpVec2, diffVec;

            // Calculate spacial center of anions (= oxygen atoms)
            float anionCenter[] = {0.0, 0.0, 0.0};
            for(int oxy = 8; oxy < 14; oxy++) {
                anionCenter[0] += dc->GetAtomPos()[3 * dc->GetCells()[15*c+oxy] + 0];
                anionCenter[1] += dc->GetAtomPos()[3 * dc->GetCells()[15*c+oxy] + 1];
                anionCenter[2] += dc->GetAtomPos()[3 * dc->GetCells()[15*c+oxy] + 2];
            }
            anionCenter[0] /= 6.0f;
            anionCenter[1] /= 6.0f;
            anionCenter[2] /= 6.0f;

            tmpVec1.Set(anionCenter[0],anionCenter[1],anionCenter[2]);

            // Calculate spacial center of cations (= titanium and barium atoms)
            float cationCenter[] = {0.0f, 0.0f, 0.0f};
            /*for(int at = 0; at < 8; at++) {
                cationCenter[0] += atomPos[4 * dc->GetCells()[15*c+at] + 0];
                cationCenter[1] += atomPos[4 * dc->GetCells()[15*c+at] + 1];
                cationCenter[2] += atomPos[4 * dc->GetCells()[15*c+at] + 2];
            }*/
            cationCenter[0] += dc->GetAtomPos()[3 * idxTiAtom + 0];
            cationCenter[1] += dc->GetAtomPos()[3 * idxTiAtom + 1];
            cationCenter[2] += dc->GetAtomPos()[3 * idxTiAtom + 2];

            tmpVec2.Set(cationCenter[0], cationCenter[1], cationCenter[2]);

            diffVec = tmpVec2 - tmpVec1;

            dipole[3*c+0] = diffVec.X();
            dipole[3*c+1] = diffVec.Y();
            dipole[3*c+2] = diffVec.Z();

        }

#pragma omp parallel for
		for (int cnt = 0; cnt < (int)dc->GetCellCnt(); cnt++) {
            buff[cnt*6+0] = dc->GetAtomPos()[3*dc->GetCells()[15*idxSorted[cnt]+14]+0];
            buff[cnt*6+1] = dc->GetAtomPos()[3*dc->GetCells()[15*idxSorted[cnt]+14]+1];
            buff[cnt*6+2] = dc->GetAtomPos()[3*dc->GetCells()[15*idxSorted[cnt]+14]+2];
            buff[cnt*6+3] = dipole[3*idxSorted[cnt]+0];
            buff[cnt*6+4] = dipole[3*idxSorted[cnt]+1];;
            buff[cnt*6+5] = dipole[3*idxSorted[cnt]+2];;
        }

        dc->Unlock();
        printf("Writing frame %u ...\n", fr);
        outStr.write((char *)buff, 24*dc->GetCellCnt());
    }
    outStr.close();

    delete[] idxSorted;
    delete[] posKey;
    delete[] buff;
    delete[] dipole;

    return true;
}


/*
 * protein_cuda::DataWriter::sortByKey
 */
void protein_cuda::DataWriter::sortByKey(unsigned int *idx, unsigned int n, float *pos) {

}

void protein_cuda::DataWriter::PutVelocity() {

    printf("Put velocity\n");

    std::fstream inStr;
    std::string inFile = "/proj/SFB716/D4/Datensaetze/SciVisContest2012/bto_625000at_500fr_Velocity.bin";

    inStr.open(inFile.c_str(), std::fstream::binary | std::fstream::in);
    if(!inStr.is_open()) {
        std::cout << "ERR could not open file '" << inFile << "'" << std::endl;
        return;
    }

    std::fstream atomTypesStr;
    std::string atomTypesFile = "/proj/SFB716/D4/Datensaetze/SciVisContest2012/bto_625000at.bin";
    atomTypesStr.open(atomTypesFile.c_str(), std::fstream::binary | std::fstream::in);
    if(!atomTypesStr.is_open()) {
        std::cout << "ERR could not open file '" << atomTypesFile << "'" << std::endl;
        return;
    }

    // Get atom info
    int *atomInfo = new int[7*625000];
    atomTypesStr.read((char *)atomInfo , 625000*28);
    atomTypesStr.close();

    // Loop through all frames
    float *buff = new float[625000*3];
    unsigned int frame = 0;
    while(inStr.good()) {
        inStr.read((char *)buff, 625000*12);

        float avgMag = 0.0;
        unsigned int avgCnt = 0;

        // Print velocity and atom types of all atoms
        for(int cnt = 0; cnt < 625000; cnt++) {

            /*printf("FRAME %u ", frame);
            if(atomInfo[7*cnt] == 0)
                printf("Ba : %i ", atomInfo[7*cnt]);
            else if(atomInfo[7*cnt] == 1)
                printf("O : %i ", atomInfo[7*cnt]);
            else if(atomInfo[7*cnt] == 2)
                printf("Ti : %i ", atomInfo[7*cnt]);
            else
                printf("INVALID ATOMTYPE: %i ", atomInfo[7*cnt]);

            printf("vel %f %f %f\n",
                    buff[cnt*3+0],
                    buff[cnt*3+1],
                    buff[cnt*3+2]);*/

            if(atomInfo[7*cnt] == 2) {
                avgMag += sqrt(buff[cnt*3+0]*buff[cnt*3+0] +
                               buff[cnt*3+1]*buff[cnt*3+1] +
                               buff[cnt*3+2]*buff[cnt*3+2]);
                avgCnt++;
            }
        }
        avgMag /= static_cast<float>(avgCnt);
        printf("%f %f\n", static_cast<float>(frame), avgMag);
        frame++;
    }

    delete[] buff;
    delete[] atomInfo;

    inStr.close();
    atomTypesStr.close();
}


void protein_cuda::DataWriter::PutDisplacement(protein_calls::CrystalStructureDataCall *dc) {
    using namespace vislib::math;

    unsigned int frameWin = 0;

	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetExtent)) {
        this->jobDone = true;
    }

    float max=0.0f, min=10000.0f, mag;

    for(unsigned int fr = 0; fr <= dc->FrameCount()-1; fr++) {
        float avgMag = 0.0f;
        unsigned int avgCnt = 0;
        // Get data of the first frame
        dc->SetFrameID(fr, true);  // Set 'force' flag
        dc->SetCalltime(static_cast<float>(fr));
		if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetData)) {
            return;
        }

        for(unsigned int cnt = 0; cnt < dc->GetDipoleCnt(); cnt++) {
            /*//if(dc->GetAtomType()[cnt] == CrystalStructureDataCall::O) {
                avgMag += sqrt(dc->GetDipole()[3*cnt+0]*dc->GetDipole()[3*cnt+0] +
                               dc->GetDipole()[3*cnt+1]*dc->GetDipole()[3*cnt+1] +
                               dc->GetDipole()[3*cnt+2]*dc->GetDipole()[3*cnt+2]);
                avgCnt++;
            //}*/
			if (dc->GetAtomType()[cnt] == protein_calls::CrystalStructureDataCall::TI) {
                mag = sqrt(dc->GetDipole()[3*cnt+0]*dc->GetDipole()[3*cnt+0] +
                               dc->GetDipole()[3*cnt+1]*dc->GetDipole()[3*cnt+1] +
                               dc->GetDipole()[3*cnt+2]*dc->GetDipole()[3*cnt+2]);
                if(mag > max) max = mag;
                if(mag < min) min = mag;
            }
        }

        dc->Unlock();
        //avgMag /= static_cast<float>(avgCnt);
        //printf("%f %f\n", static_cast<float>(fr + (frameWin-1)/2)*20.0f, avgMag);
        //printf("%f %f\n", 20.0f*static_cast<float>(fr), avgMag); // No averaging
        //printf("%f %f\n", 20.0f*(static_cast<float>(fr)+25.0f), avgMag); // Avg 50
        printf("max %f, min %f\n", max, min);
    }
}


bool protein_cuda::DataWriter::GetMaxCoords(protein_calls::CrystalStructureDataCall *dc) {

    using namespace vislib::sys;

    if(dc == NULL) {
        this->jobDone = true;
        return false;
    }

    // Get extend
	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetExtent)) {
        this->jobDone = true;
        return false;
    }

    float maxCoordX = 0.0f;
    float maxCoordY = 0.0f;
    float maxCoordZ = 0.0f;

    for(unsigned int fr = 0; fr < dc->FrameCount(); fr++) {

        // Get data from data source
        dc->SetFrameID(fr, true);
        dc->SetCalltime(static_cast<float>(fr));
		if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetData)) {
            this->jobDone = true;
            return false;
        }

        for(unsigned int vec = 0; vec < dc->GetDipoleCnt(); vec++) {
            if(fabs(dc->GetDipole()[vec*3+0]) > maxCoordX) maxCoordX = fabs(dc->GetDipole()[vec*3+0]);
            if(fabs(dc->GetDipole()[vec*3+1]) > maxCoordX) maxCoordY = fabs(dc->GetDipole()[vec*3+1]);
            if(fabs(dc->GetDipole()[vec*3+2]) > maxCoordZ) maxCoordZ = fabs(dc->GetDipole()[vec*3+2]);
        }

        printf("FRAME %u (maxX %f, maxY %f, maxZ %f)\n", fr, maxCoordX, maxCoordY, maxCoordZ);
    }

    this->jobDone = true;
    return true;
}


bool protein_cuda::DataWriter::PutCubeVol(protein_calls::CrystalStructureDataCall *dc) {

    using namespace vislib::sys;

    if(dc == NULL) {
        this->jobDone = true;
        return false;
    }

    // Get extend
	if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetExtent)) {
        this->jobDone = true;
        return false;
    }


    for(unsigned int fr = 0; fr < dc->FrameCount(); fr++) {

        
        float maxCoordX = 0.0f;
        float maxCoordY = 0.0f;
        float maxCoordZ = 0.0f;
        float minCoordX = 0.0f;
        float minCoordY = 0.0f;
        float minCoordZ = 0.0f;

        // Get data from data source
        dc->SetFrameID(fr, true);
        dc->SetCalltime(static_cast<float>(fr));
		if (!(*dc)(protein_calls::CrystalStructureDataCall::CallForGetData)) {
            this->jobDone = true;
            return false;
        }

        for(unsigned int vec = 0; vec < dc->GetAtomCnt(); vec++) {
            if(dc->GetAtomPos()[vec*3+0] > maxCoordX) maxCoordX = dc->GetAtomPos()[vec*3+0];
            if(dc->GetAtomPos()[vec*3+1] > maxCoordY) maxCoordY = dc->GetAtomPos()[vec*3+1];
            if(dc->GetAtomPos()[vec*3+2] > maxCoordZ) maxCoordZ = dc->GetAtomPos()[vec*3+2];
            if(dc->GetAtomPos()[vec*3+0] < minCoordX) minCoordX = dc->GetAtomPos()[vec*3+0];
            if(dc->GetAtomPos()[vec*3+1] < minCoordY) minCoordY = dc->GetAtomPos()[vec*3+1];
            if(dc->GetAtomPos()[vec*3+2] < minCoordZ) minCoordZ = dc->GetAtomPos()[vec*3+2];
        }

        float vol = fabs(maxCoordX-minCoordX)*fabs(maxCoordY-minCoordY)*fabs(maxCoordZ-minCoordZ);
        printf("%f %f\n", static_cast<float>(fr), vol);
    }

    this->jobDone = true;
    return true;
}
