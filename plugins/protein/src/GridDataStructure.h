/*
 * GridDataStructure.cpp
 *
 * Copyright (C) 2012 by University of Stuttgart (VISUS).
 * All rights reserved.
 */


#ifndef MEGAMOLPROTEIN_GRIDDATASTRUCTURE_H_INCLUDED
#define MEGAMOLPROTEIN_GRIDDATASTRUCTURE_H_INCLUDED
/*
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif // (defined(_MSC_VER) && (_MSC_VER > 1000))
*/

#include "stdafx.h"
#include "vislib/ArrayAllocator.h"
#include "vislib/sys/Log.h"
#include "vislib/math/mathfunctions.h"
#include "vislib/SmartPtr.h"
#include "vislib/types.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/math/Cuboid.h"
#include <ctime>

/**
 * Simple nearest-neighbour-search implementation which uses a regular grid to speed up search queries.
 */
namespace megamol {
namespace protein {
	template<class T>
	class GridDataStructure {
		typedef vislib::math::Point<T,3> Point;

	public:
		GridDataStructure() : elementGrid(0), elementCount(0), elementPositions(0), gridSize(0) {}

		~GridDataStructure() {
			delete [] elementGrid;
		}

        unsigned int getGridSize() const { return gridSize; };

        const unsigned int* getGridResolution() const { return &gridResolution[0]; };

		/**
		 * Set new point data to the neighbourhood search grid.
		 * @params TODO
		 */
		void setPointData( const T *pointData, unsigned int pointCount, vislib::math::Cuboid<T> boundingBox, T cellLength) {
			this->elementPositions = pointData;
			this->elementCount = pointCount;
            
            // clear element grid
			for (unsigned int i = 0; i < gridSize; i++ ) {
				elementGrid[i].Clear();
            }
			// does the new BBox fit into the old one?
			if( !elementGrid || !(this->elementBBox == boundingBox )  ) {
				// resize bounding-box and grid structure
				vislib::math::Dimension<T, 3> dim = boundingBox.GetSize();
                
                // compute the number of cells in each direction
                this->gridResolution[0] = (unsigned int)( dim[0] / cellLength + T(1.0));
                this->gridResolution[1] = (unsigned int)( dim[1] / cellLength + T(1.0));
                this->gridResolution[2] = (unsigned int)( dim[2] / cellLength + T(1.0));
				this->gridSize = this->gridResolution[0] * this->gridResolution[1] * this->gridResolution[2];
				this->elementBBox = boundingBox;

				// initialize element grid
				if (elementGrid)
					delete [] elementGrid;
				this->elementGrid = new vislib::Array<unsigned int>[gridSize];
				for (unsigned int i = 0; i < gridSize; i++) {
					elementGrid[i].Resize( 100);
					elementGrid[i].SetCapacityIncrement( 100);
				}
			}

			this->bBoxOrigin = this->elementBBox.GetOrigin();

			// fill the internal grid structure
			vislib::math::Dimension<T, 3> bBoxDimension = this->elementBBox.GetSize();
			for(int i = 0 ; i < 3; i++) {
				this->gridResolutionFactors[i] = (T)this->gridResolution[i] / bBoxDimension[i];
				this->cellSize[i] = (T)bBoxDimension[i] / this->gridResolution[i]; //(T)1.0) / gridResolutionFactors[i];
			}
			// sort the element positions into the grid ...
			for(unsigned int i = 0; i < this->elementCount; i++) {
				insertPointIndexIntoGrid(&this->elementPositions[i*3], i);
			}
		}
        
		void getPointsInCell( unsigned int indexX, unsigned int indexY, unsigned int indexZ, vislib::Array<unsigned int>& resIdx) const {
			ASSERT(indexX < gridResolution[0] && indexY < gridResolution[1] && indexZ < gridResolution[2]);
            getNeighboursInCell( elementGrid[cellIndex(indexX, indexY, indexZ)], resIdx);
		}

		const vislib::Array<unsigned int>& accessPointsInCell( unsigned int indexX, unsigned int indexY, unsigned int indexZ) const {
			ASSERT(indexX < gridResolution[0] && indexY < gridResolution[1] && indexZ < gridResolution[2]);
            return elementGrid[cellIndex(indexX, indexY, indexZ)];
		}

		VISLIB_FORCEINLINE void insertPointIndexIntoGrid(const T *point, unsigned int idx) {
			unsigned int indexX = (unsigned int)( (point[0] - bBoxOrigin[0]) * gridResolutionFactors[0]);
			unsigned int indexY = (unsigned int)( (point[1] - bBoxOrigin[1]) * gridResolutionFactors[1]);
			unsigned int indexZ = (unsigned int)( (point[2] - bBoxOrigin[2]) * gridResolutionFactors[2]);
			ASSERT(indexX < gridResolution[0] && indexY < gridResolution[1] && indexZ < gridResolution[2]);
			elementGrid[cellIndex(indexX, indexY, indexZ)].Add( idx);
        }

		VISLIB_FORCEINLINE void getNeighboursInCell(const vislib::Array<const T *>& cell, vislib::Array<unsigned int>& resIdx) const {
			for(int i = 0; i < cell.Count(); i++) {
				resIdx.Add( cell[i]); // store atom index
            }
		}

		inline unsigned int cellIndex(unsigned int x, unsigned int y, unsigned int z) const {
			return x + (y + z*gridResolution[1]) * gridResolution[0];
		}
        
		/** pointer to points/positions stored in triples (xyzxyz...) */
		const T *elementPositions;
		/** number of points of 'elementPositions' */
		unsigned int elementCount;
		/** array of position indices for each cell of the regular element grid */
		vislib::Array<unsigned int> *elementGrid;
		/** bounding box of all positions/points */
		vislib::math::Cuboid<T> elementBBox;
		/** origin of 'elementBBox' */
		Point bBoxOrigin;
		/** number of cells in each dimension */
		unsigned int gridResolution[3];
		/** factors to calculate cell index from a given point (inverse of 'cellSize') */
		T gridResolutionFactors[3];
		/** extends of each grid cell */
		T cellSize[3];
		/** short for gridResolution[0]*gridResolution[1]*gridResolution[2] */
		unsigned int gridSize;
	};
} //namespace megamol
} //namespace protein

#endif // MEGAMOLPROTEIN_GRIDDATASTRUCTURE_H_INCLUDED
