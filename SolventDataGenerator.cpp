/*
 * SolventDataGenerator.cpp
 *
 * Copyright (C) 2011 by University of Stuttgart (VISUS).
 * All rights reserved.
 */


#include "stdafx.h"
#include "SolventDataGenerator.h"
#include "param/FilePathParam.h"
#include "param/IntParam.h"
#include "param/BoolParam.h"
#include "param/StringParam.h"
#include "param/FloatParam.h"
#include "vislib/ArrayAllocator.h"
#include "vislib/Log.h"
#include "vislib/mathfunctions.h"
#include "vislib/MemmappedFile.h"
#include "vislib/SmartPtr.h"
#include "vislib/types.h"
#include "vislib/sysfunctions.h"
#include "vislib/StringConverter.h"
#include "vislib/StringTokeniser.h"
#include "vislib/ASCIIFileBuffer.h"
#include "vislib/ShallowPoint.h"
#include <ctime>
#include <iostream>
#include <fstream>

using namespace megamol;
using namespace megamol::core;
using namespace megamol::protein;


/**
 * Simple nearest-neighbour-search implementation which uses a regular grid to speed up search queries.
 */
template<class T/*, unigned int Dim> als template parameter?!*/>
class GridNeighbourFinder {
	typedef vislib::math::Point<T,3> Point;

public:
	GridNeighbourFinder(const T *pointData, unsigned int pointCount, vislib::math::Cuboid<T> boundingBox,
		vislib::math::Dimension<unsigned int,3> gridRes) :
		elementPositions(pointData), elementCount(pointCount), elementBBox(boundingBox)
	{
		Dimension<T, 3> bBoxDimension = elementBBox.GetSize();
		elementOrigin = elementBBox.Origin();
		for(int i = 0; i < 3; i++)
			gridResolution[i] = gridRes[i];

		initElementGrid();
	}

	GridNeighbourFinder(const T *pointData, unsigned int pointCount, vislib::math::Cuboid<T> boundingBox, T searchDistance) :
		elementPositions(pointData), elementCount(pointCount), elementBBox(boundingBox)
	{
		vislib::math::Dimension<T, 3> bBoxDimension = elementBBox.GetSize();
		elementOrigin = elementBBox.GetOrigin();
		for(int i = 0; i < 3; i++)
			gridResolution[i] =  floor(bBoxDimension[i] / (2/*4*/*searchDistance) + 1.0);

		initElementGrid();
	}

	~GridNeighbourFinder() {
		delete [] elementGrid;
	}

	//template<typename T>
	void FindNeighboursInRange(const T *point, T distance, vislib::Array<unsigned int>& resIdx) const {
		//Point relPos = sub(point, elementOrigin);
		T relPos[3] = {point[0] - elementOrigin[0],
			point[1] - elementOrigin[1],
			point[2] - elementOrigin[2]};

		// calculate range in the grid ...
		int min[3], max[3];
		for(int i = 0; i < 3; i++) {
			min[i] = floor((relPos[i]-distance)*gridResolutionFactors[i]);
			if (min[i] < 0)
				min[i] = 0;
			max[i] = ceil((relPos[i]+distance)*gridResolutionFactors[i]);
			if (max[i] >= gridResolution[i])
				max[i] = gridResolution[i]-1;
		}

		// loop over all cells inside the sphere (point, distance)
/*		for(float x = relPos[0]-distance; x <= relPos[0]+distance+cellSize[0]; x += cellSize[0]) {
			for(float y = relPos[1]-distance; y <= relPos[1]+distance+cellSize[1]; y += cellSize[1]) {
				for(float z = relPos[2]-distance; z <= relPos[2]+distance+cellSize[2]; z += cellSize[2]) {
					unsigned int indexX = x * gridResolutionFactors[0]; // floor()
					unsigned int indexY = y * gridResolutionFactors[1];
					unsigned int indexZ = z * gridResolutionFactors[2];
					if (indexX > 0 && indexX < ... && ... && ... )
				*/
		for(int indexX = min[0]; indexX <= max[0]; indexX++) {
			for(int indexY = min[1]; indexY <= max[1]; indexY++) {
				for(int indexZ = min[2]; indexZ <= max[2]; indexZ++) {
					//if ( (Point(x,y,z)-relPos).Length() < distance ) continue;
						findNeighboursInCell(elementGrid[cellIndex(indexX, indexY, indexZ)], point, distance, resIdx);
				}
			}
		}
	}


private:
	/** fill the internal grid structure */
	void initElementGrid() {
		vislib::math::Dimension<T, 3> bBoxDimension = elementBBox.GetSize();
		for(int i = 0 ; i < 3; i++) {
			gridResolutionFactors[i] = (T)gridResolution[i] / bBoxDimension[i];
			cellSize[i] = (T)bBoxDimension[i] / gridResolution[i]; //(T)1.0) / gridResolutionFactors[i];
		}
		gridSize = gridResolution[0]*gridResolution[1]*gridResolution[2];

		// initialize element grid
		elementGrid = new vislib::Array<const T *>[gridSize];

		// sort the element positions into the grid ...
		for(int i = 0; i < elementCount; i+=3) {
			ASSERT( elementBBox.Contains(vislib::math::ShallowPoint<T,3>(const_cast<T*>(&elementPositions[i]))) );
			insertPointIntoGrid(&elementPositions[i]);
		}
	}

	VISLIB_FORCEINLINE void insertPointIntoGrid(const T *point) {
		//Point relPos = sub(point, elementOrigin);
		unsigned int indexX = /*relPos.X()*/(point[0] - elementOrigin[0]) * gridResolutionFactors[0]; // floor()?
		unsigned int indexY = /*relPos.Y()*/(point[1] - elementOrigin[1]) * gridResolutionFactors[1];
		unsigned int indexZ = /*relPos.Z()*/(point[2] - elementOrigin[2]) * gridResolutionFactors[2];
		ASSERT(indexX < gridResolution[0] && indexY < gridResolution[1] && indexZ < gridResolution[2]);
		vislib::Array<const T *>& cell = elementGrid[cellIndex(indexX, indexY, indexZ)];
		cell.Add(point);
	}

	VISLIB_FORCEINLINE void findNeighboursInCell(const vislib::Array<const T *>& cell, const T* point, T distance, vislib::Array<unsigned int>& resIdx) const {
		for(int i = 0; i < cell.Count(); i++)
			if ( dist(cell[i],point) <= distance )
				resIdx.Add((cell[i]-elementPositions)/3); // store atom index
	}

	inline unsigned int cellIndex(unsigned int x, unsigned int y, unsigned int z) const {
		return x + (y + z*gridResolution[1]) * gridResolution[0];
	}

	inline static Point sub(const T* a, const Point& b) { return Point(a[0]-b[0],a[1]-b[1],a[2]-b[2]); };

	inline static T dist(const T *a, const T *b) {
		T x = a[0]-b[0]; T y = a[1]-b[1]; T z = a[2]-b[2];
		return sqrt(x*x + y*y + z*z);
	}

private:
	/** pointer to points/positions stored in triples (xyzxyz...) */
	const T *elementPositions;
	/** number of points of 'elementPositions' */
	unsigned int elementCount;
	/** array of position-pointers for each cell of the regular element grid */
	vislib::Array<const T *> *elementGrid;
	/** bounding box of all positions/points */
	vislib::math::Cuboid<T> elementBBox;
	/** origin of 'elementBBox' */
	Point elementOrigin;
	/** number of cells in each dimension */
	unsigned int gridResolution[3];
	/** factors to calculate cell index from a given point (inverse of 'cellSize') */
	T gridResolutionFactors[3];
	/** extends of each a grid cell */
	T cellSize[3];
	/** short for gridResolution[0]*gridResolution[1]*gridResolution[2] */
	unsigned int gridSize;
};


megamol::protein::SolventDataGenerator::SolventDataGenerator() :
		dataOutSlot( "dataout", "The slot providing the generated solvent data"),
		molDataInputCallerSlot( "getInputData", "molecular data source (usually the PDB loader)"),
		hBondDataFile( "hBondDataFile", "file to store hydrogen bond data"),
		hBondDistance("hBondDistance", "distance for hydrogen bonds (angstroem?)"),
		showMiddlePositions("showMiddlePositions", "show the middle of all atom positions over time")
{
	this->molDataInputCallerSlot.SetCompatibleCall<MolecularDataCallDescription>();
	this->MakeSlotAvailable( &this->molDataInputCallerSlot);

	this->dataOutSlot.SetCallback( MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(MolecularDataCall::CallForGetData), &SolventDataGenerator::getData);
	this->dataOutSlot.SetCallback( MolecularDataCall::ClassName(), MolecularDataCall::FunctionName(MolecularDataCall::CallForGetExtent), &SolventDataGenerator::getExtent);
	this->MakeSlotAvailable( &this->dataOutSlot);

	// distance for hydrogen bonds
	this->hBondDistance.SetParameter(new param::FloatParam(1.9f, 0.0f));
	this->MakeSlotAvailable( &this->hBondDistance);

	this->hBondDataFile.SetParameter(new param::StringParam("hbond.dat"));
	this->MakeSlotAvailable( &this->hBondDataFile);

	this->showMiddlePositions.SetParameter(new param::BoolParam(false));
	this->MakeSlotAvailable( &this->showMiddlePositions);

	for(int i = 0; i < HYDROGEN_BOND_IN_CORE; i++)
		curHBondFrame[i] = -1;
}

megamol::protein::SolventDataGenerator::~SolventDataGenerator() {
}

bool megamol::protein::SolventDataGenerator::create(void) {
	return true;
}

/**
 * -> preprocessing step
 *
 *- Aufenthaltswahrscheinlichkeit/-dauer (einzelne Moleküle oder Molekültypen über komplette Trajektorie berechnen & als Farbe auf statische Moleküloberfläche mappen) *
 * -> das geht so net! das läuft auf ne volumen-akkumulation hinaus ...
 */
void megamol::protein::SolventDataGenerator::calcSpatialProbabilities(MolecularDataCall *src, MolecularDataCall *dst) {
	int nFrames = src->FrameCount();
	int nAtoms = src->AtomCount();

	//middleAtomPos.AssertCapacity(nAtoms*3);
	middleAtomPos.SetCount(nAtoms*3);

	float *middlePosPtr = &middleAtomPos.First();
	const float *atomPositions = src->AtomPositions();
	memset(middlePosPtr, 0, sizeof(float)*3*nAtoms);

//#pragma omp parallel for private( ??? )
	for(int i = 0; i < nFrames; i++) {
		src->SetFrameID(i,true);
		if( !(*src)(MolecularDataCall::CallForGetData))
			continue; // return false;

		#pragma omp parallel for
		for(int aIdx = 0; aIdx < nAtoms; aIdx+=3) {
			middlePosPtr[aIdx] += atomPositions[aIdx];
			middlePosPtr[aIdx+1] += atomPositions[aIdx+1];
			middlePosPtr[aIdx+2] += atomPositions[aIdx+2];
		}
	}

	float normalize = 1.0f / nFrames;

	#pragma omp parallel for
	for(int aIdx = 0; aIdx < nAtoms*3; aIdx++)
		middlePosPtr[aIdx] *= normalize;

	//molDest->SetAtomPositions(middleAtomPos.PeekElements());
}

/*
 *
 *
void megamol::protein::SolventDataGenerator::findDonors(MolecularDataCall *data) {
	int nResidues = data->ResidueCount();

	for( int i = 0; i < nResidues; i++ ) {
		MolecularDataCall::Residue *residue = data->Residues()[i];
	}
}
*/


class HbondIO {
private:
	vislib::sys::File *readHandle, *writeHandle;

	/*vislib::sys::File::FileSize*/
	unsigned int dataStartOffset, atomCount, frameCount, frameSizeInBytes;

public:
	HbondIO(unsigned int atomCount, unsigned int frameCount, const vislib::StringA& fname, bool read)
		: dataStartOffset(0), atomCount(atomCount), frameCount(frameCount) {
		if (read) {
			readHandle = new vislib::sys::MemmappedFile();
			if (!readHandle->Open(fname, vislib::sys::File::READ_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::OPEN_ONLY)) {
				delete readHandle;
				readHandle = 0;
			}
			writeHandle = 0;
		} else {
			writeHandle = new vislib::sys::MemmappedFile();
			if (!writeHandle->Open(fname, vislib::sys::File::WRITE_ONLY, vislib::sys::File::SHARE_READ, vislib::sys::File::CREATE_OVERWRITE)) {
				delete writeHandle;
				writeHandle = 0;
			}
			readHandle = 0;
		}
	}
	~HbondIO() {
		delete readHandle;
		delete writeHandle;
	}

	bool writeFrame(int *frame, unsigned int frameId) {
		if (!writeHandle)
			return false;

		// write header at first
		if ( dataStartOffset == 0) {
			dataStartOffset = 4*sizeof(dataStartOffset); // store 4 ineteger entries as header for now ...
			frameSizeInBytes = atomCount * sizeof(int); // 1 signed integer per atom specifying the hbond connection to another atom
			writeHandle->Write(&dataStartOffset, sizeof(dataStartOffset));
			writeHandle->Write(&atomCount, sizeof(atomCount));
			writeHandle->Write(&frameCount, sizeof(frameCount));
			writeHandle->Write(&frameSizeInBytes, sizeof(frameSizeInBytes));
			ASSERT(writeHandle->Tell()==dataStartOffset);
			if (writeHandle->Tell()!=dataStartOffset)
				return false;
		}

		ASSERT(frameId < frameCount);
		writeHandle->Seek(dataStartOffset + frameId*frameSizeInBytes);
		if (writeHandle->Write(frame, frameSizeInBytes) != frameSizeInBytes)
			return false;
		return true;
	}

	bool readFrame(int *frame, unsigned int frameId) {
		if (!readHandle)
			return false;

		// read header at first
		if ( dataStartOffset == 0) {
			readHandle->Read(&dataStartOffset, sizeof(dataStartOffset));
			readHandle->Read(&atomCount, sizeof(atomCount));
			readHandle->Read(&frameCount, sizeof(frameCount));
			readHandle->Read(&frameSizeInBytes, sizeof(frameSizeInBytes));
			ASSERT(frameSizeInBytes==atomCount * sizeof(int)); // this is some sort of file type/integrity check ...
			if (frameSizeInBytes!=atomCount * sizeof(int))
				return false;
		}

		ASSERT(frameId < frameCount);
		readHandle->Seek(dataStartOffset + frameId*frameSizeInBytes);
		if (readHandle->Read(frame, frameSizeInBytes) != frameSizeInBytes)
			return false;
		return true;
	}
};


void megamol::protein::SolventDataGenerator::calcHydroBondsForCurFrame(MolecularDataCall *data, int *atomHydroBondsIndicesPtr) {
	float hbondDist = hBondDistance.Param<param::FloatParam>()->Value();

	GridNeighbourFinder<float> NNS(data->AtomPositions(), data->AtomCount(), data->AccessBoundingBoxes().ObjectSpaceBBox(), hbondDist);

	const float *atomPositions = data->AtomPositions();
	const MolecularDataCall::AtomType *atomTypes = data->AtomTypes();
	const unsigned int *atomTypeIndices = data->AtomTypeIndices();
	const int *atomResidueIndices = data->AtomResidueIndices();
	int nResidues = data->ResidueCount();

    // set all entries to "not connected"
	memset(atomHydroBondsIndicesPtr, -1, sizeof(int)*data->AtomCount());

	if (reverseConnection.Count() < data->AtomCount())
		reverseConnection.SetCount(data->AtomCount());
	int *reverseConnectionPtr = &reverseConnection[0];
	memset(reverseConnectionPtr, -1, sizeof(int)*data->AtomCount());

	// looping over residues may not be a good idea?! (index-traversal?) loop over all possible acceptors ...
	for( int rIdx = 0; rIdx < nResidues; rIdx++ ) {
		const MolecularDataCall::Residue *residue = data->Residues()[rIdx];

		// find possible acceptor atoms in the current residuum (for now just O, N, C(?)

		// vorerst nur Sauerstoff und Stickstoff als Akzeptor/Donator (N, O)
		int lastAtomIdx = residue->FirstAtomIndex()+residue->AtomCount();
		for(int aIdx = residue->FirstAtomIndex(); aIdx < lastAtomIdx; aIdx++) {
			// is this atom already connected?
			if (reverseConnectionPtr[aIdx] != -1)
				continue;

			const MolecularDataCall::AtomType& t = atomTypes[atomTypeIndices[aIdx]];
			const vislib::StringA& name = t.Name();
			char element = name[0];

/*
JW: ich fürchte für eine allgemeine Deffinition der Wasserstoffbrücken muß man über die Bindungsenergien gehen und diese berechnen.
Für meine Simulationen und alle Bio-Geschichten reicht die Annahme, dass Sauerstoff, Stickstoff und Fluor (was fast nie vorkommt)
Wasserstoffbrücken bilden und dabei als Donor und Aktzeptor dienen könne. Dabei ist der Wasserstoff am Donor gebunden und bildet die Brücke zum Akzeptor.
*/
			if (element=='N' || element=='O' /*|| element=='F' || element=='C'??*/) {
				neighbourIndices.Clear(); // clear, keep capacity ...
				NNS.FindNeighboursInRange(&atomPositions[aIdx*3], hbondDist, neighbourIndices);
				for(int nIdx = 0; nIdx<neighbourIndices.Count(); nIdx++) {
					int neighbIndex = neighbourIndices[nIdx];
					// atom from the current residue?
					if (atomResidueIndices[neighbIndex]==rIdx)
						continue;
					// check if a H-atom is in range and add a h-bond...
					if (atomTypes[atomTypeIndices[neighbIndex]].Name()[0]=='H') {
						atomHydroBondsIndicesPtr[aIdx] = neighbIndex;
						// avoid double checks - only one hydrogen bond per atom?!
						reverseConnectionPtr[neighbIndex] = aIdx;
					}
				}
			}
		}
	}
}


bool megamol::protein::SolventDataGenerator::getHBonds(MolecularDataCall *dataTarget, MolecularDataCall *dataSource) {
	int reqFrame = dataTarget->FrameID();
	int cacheIndex = reqFrame % HYDROGEN_BOND_IN_CORE;
	float hbondDist = hBondDistance.Param<param::FloatParam>()->Value();

	if (curHBondFrame[cacheIndex] == reqFrame) {
		// recalc hbonds if 'hBondDistance' has changed?!
		//if( this->hBondDistance.IsDirty() )
		dataTarget->SetAtomHydrogenBondIndices(atomHydroBondsIndices[cacheIndex].PeekElements());
		dataTarget->SetAtomHydrogenBondDistance(hbondDist);
		return true;
	}

	vislib::Array<int>& atomHydroBonds = this->atomHydroBondsIndices[cacheIndex];
	if (atomHydroBonds.Count() < dataSource->AtomCount())
		atomHydroBonds.SetCount(dataSource->AtomCount());

#if 0
	const vislib::TString fileName = hBondDataFile.Param<param::StringParam>()->Value();
	if (fileName.IsEmpty())
		return false;

	if (vislib::sys::File::Exists(fileName)) {
		HbondIO input(data->AtomCount(), data->FrameCount(), fileName, true);
		if (input.readFrame(&atomHydroBonds[0], reqFrame)) {
			curHBondFrame[cacheIndex] = reqFrame;
			data->SetAtomHydrogenBondIndices(atomHydroBonds.PeekElements());
			dataTarget->SetAtomHydrogenBondDistance(hbondDist);
			return true;
		}
		return false;
	}

	HbondIO *output = new HbondIO(data->AtomCount(), data->FrameCount(), fileName, false);

	int *tmpArray = new int[data->AtomCount()];

	// calculate hydrogen bounds for all frames and store them in a file ...
	for(int frameId = 0; frameId < data->FrameCount(); frameId++) {
		int *atomHydroBondsIndicesPtr = (frameId == reqFrame ? &atomHydroBonds[0] : tmpArray);

#if 0
		// workaround to avoid recursion ...
		{
			int tmp = curHBondFrame[frameId % HYDROGEN_BOND_IN_CORE];
			curHBondFrame[frameId % HYDROGEN_BOND_IN_CORE] = frameId;
			data->SetFrameID(frameId);
			if( !(*dataTarget)(MolecularDataCall::CallForGetData))
				return false;
			curHBondFrame[frameId % HYDROGEN_BOND_IN_CORE] = tmp;
			calcHydroBondsForCurFrame(dataTarget, atomHydroBondsIndicesPtr);
		}
#else
		dataSource->SetFrameId(frameId);
		if( !(*dataSource)(MolecularDataCall::CallForGetData))
			return false;
		calcHydroBondsForCurFrame(dataSource, atomHydroBondsIndicesPtr);
#endif

		output->writeFrame(atomHydroBondsIndicesPtr, frameId);
	}
	delete output;
	delete tmpArray;

#else
	dataSource->SetFrameID(reqFrame);
	if( !(*dataSource)(MolecularDataCall::CallForGetData))
		return false;
	calcHydroBondsForCurFrame(dataSource, &atomHydroBonds[0]);
#endif

	curHBondFrame[cacheIndex] = reqFrame;
	dataTarget->SetFrameID(reqFrame);
	dataTarget->SetAtomHydrogenBondDistance(hbondDist);
	dataTarget->SetAtomHydrogenBondIndices(atomHydroBonds.PeekElements());

	return true;
}

bool megamol::protein::SolventDataGenerator::getExtent(core::Call& call) {
	MolecularDataCall *molDest = dynamic_cast<MolecularDataCall*>( &call); // dataOutSlot ??
	MolecularDataCall *molSource = this->molDataInputCallerSlot.CallAs<MolecularDataCall>();

	if (!molDest || !molSource)
		return false;

	if( !(*molSource)(MolecularDataCall::CallForGetExtent))
		return false;

	// forward data ...
	molDest->AccessBoundingBoxes().Clear();
	molDest->AccessBoundingBoxes().SetObjectSpaceBBox( molSource->AccessBoundingBoxes().ObjectSpaceBBox() );
	molDest->AccessBoundingBoxes().SetObjectSpaceClipBox( molSource->AccessBoundingBoxes().ObjectSpaceClipBox() );
	molDest->SetFrameCount( molSource->FrameCount() );
	molDest->SetDataHash( molSource->DataHash() );
	return true;
}

bool megamol::protein::SolventDataGenerator::getData(core::Call& call) {
	MolecularDataCall *molDest = dynamic_cast<MolecularDataCall*>( &call); // dataOutSlot ??
	MolecularDataCall *molSource = this->molDataInputCallerSlot.CallAs<MolecularDataCall>();

	if (!molDest || !molSource)
		return false;

	molSource->SetFrameID( molDest->FrameID() ); // forward frame request
	if( !(*molSource)(MolecularDataCall::CallForGetData))
		return false;

	*molDest = *molSource;

	if (showMiddlePositions.Param<param::BoolParam>()->Value()) {
		if (!middleAtomPos.Count())
			calcSpatialProbabilities(molSource, molDest);
		molDest->SetAtomPositions(middleAtomPos.PeekElements());
	}

	// test: only compute hydrogen bounds once at startup ... (this is not suficcient for trajectories)
	getHBonds(molDest, molSource);

	return true;
}

void megamol::protein::SolventDataGenerator::release(void) {
}
