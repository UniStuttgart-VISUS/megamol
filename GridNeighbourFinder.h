/*
 * GridNeighbourFinder.cpp
 *
 * Copyright (C) 2011 by University of Stuttgart (VISUS).
 * All rights reserved.
 */


#include "stdafx.h"
#include "vislib/ArrayAllocator.h"
#include "vislib/Log.h"
#include "vislib/mathfunctions.h"
#include "vislib/SmartPtr.h"
#include "vislib/types.h"
#include "vislib/ShallowPoint.h"
#include <ctime>

using namespace megamol;

/**
 * Simple nearest-neighbour-search implementation which uses a regular grid to speed up search queries.
 */
namespace megamol {
namespace protein {
	template<class T/*, unigned int Dim> als template parameter?!*/>
	class GridNeighbourFinder {
		typedef vislib::math::Point<T,3> Point;

	public:
		GridNeighbourFinder(const T *pointData, unsigned int pointCount, vislib::math::Cuboid<T> boundingBox,
			vislib::math::Dimension<unsigned int,3> gridRes) :
			elementPositions(pointData), elementCount(pointCount), elementBBox(boundingBox)
		{
			Dimension<T, 3> bBoxDimension = elementBBox.GetSize();
			elementOrigin = elementBBox.GetOrigin();
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
				gridResolution[i] =  floor((bBoxDimension[i] / searchDistance) + 1.0);

			initElementGrid();
		}

		~GridNeighbourFinder() {
			delete [] elementGrid;
		}

		/**
		 * Set new point data to the neighbourhood search grid.
		 * @params TODO
		 */
		void SetNewPointData( cont T *pointData, unsigned int pointCount, vislib::math::Cuboid<T> boundingBox ) :
				elementPositions(pointData), elementCount(pointCount) {
			// if the bounding 
			if( elementBBox.Contains( boundingBox) ) {
				for( int cnt = 0; cnt < gridSize; ++cnt ) {
					elementGrid[cnt].Clear();
				}
			}
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
			int cnt;
			for( cnt = 0; cnt < gridSize; ++cnt) {
				elementGrid[cnt].Resize( 100);
				elementGrid[cnt].SetCapacityIncrement( 100);
			}

			// sort the element positions into the grid ...
			for(int i = 0; i < elementCount; i++) {
				ASSERT( elementBBox.Contains(vislib::math::ShallowPoint<T,3>(const_cast<T*>(&elementPositions[i*3]))) );
				insertPointIntoGrid(&elementPositions[i*3]);
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
}; //namespace megamol
}; //namespace protein
