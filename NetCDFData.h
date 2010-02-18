/*
 * NetCDFData.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VISUS). Alle Rechte vorbehalten.
 */

/* netcdfdata is only compiled if path is defined in script */
#if (defined(WITH_NETCDF) && (WITH_NETCDF))

#ifndef MEGAMOL_NETCDFDATA_H_INCLUDED
#define MEGAMOL_NETCDFDATA_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "netcdf.h"
#include "CallProteinData.h"
#include "view/AnimDataModule.h"
#include "AnimDataTimer.h"
#include "Module.h"
#include "CalleeSlot.h"
#include "param/ParamSlot.h"
#include "vislib/Vector.h"
#include "vislib/Array.h"
#include <vector>
#include <map>
#include <iostream>
#include "Stride.h"


namespace megamol {
namespace protein {

	/**
	 * Data source class for amber netcdf datafiles
	 */
    class NetCDFData : public megamol::core::view::AnimDataModule {
		public:

			/** CTOR */
			NetCDFData ( void );

			/** DTOR */
			virtual ~NetCDFData ( void );

			/**
				* Answer the name of this module.
				*
				* @return The name of this module.
				*/
			static const char *ClassName ( void )
			{
				return "NetCDFData";
			}

			/**
				* Answer a human readable description of this module.
				*
				* @return A human readable description of this module.
				*/
			static const char *Description ( void )
			{
				return "Offers NetCDF data.";
			}

			/**
				* Answers whether this module is available on the current system.
				*
				* @return 'true' if the module is available, 'false' otherwise.
				*/
			static bool IsAvailable ( void )
			{
				return true;
			}

		protected:

			/**
				* Implementation of 'Create'.
				*
				* @return true if the initialisation was successful, false otherwise.
				*/
			virtual bool create ( void );

			/**
				* Implementation of 'Release'.
				*/
			virtual void release ( void ) {};

		private:

			/**
				*
				* NetCDFData callback.
				*
				* @param call The calling call.
				*
				* @return 'true' on success, 'false' otherwise.
				*/
		bool NetCDFDataCallback( megamol::core::Call& call);

			/** Nested class of frame data */
        class Frame : public megamol::core::view::AnimDataModule::Frame {
			public:

				/** Ctor. */
				Frame ( const NetCDFData * );

				/** Dtor. */
				virtual ~Frame ( void );

				/**
					* Clears the internal data buffers
					*/
				void Clear ( void );

				/**
					* Answers the size of the loaded data in bytes.
					*
					* @return The size of the loaded data in bytes.
					*/
				//SIZE_T SizeOf(void) const;

				/**
					* reads a designated frame from the netCDF file
					*
					* @param frameID the frame to read
					*/
				bool readFrameFromNetCDF ( unsigned int frameID );

				/**
					* Replaces the data of this object with the interpolated data
					* based on the two frames 'a', 'b', and the interpolation
					* parameter 'alpha' [0, 1].
					*
					* @param alpha The interpolation parameter.
					* @param a The first interpolation value, used if 'alpha' is zero.
					* @param b The second interpolation value, used if 'alpha' is one.
					*
					* @return The frame to be used after the interpolation.
					*/
				const Frame * MakeInterpolationFrame ( float alpha, const Frame &a,
																	const Frame &b );

				/**
					* get pointer to the current atom coordinates
					*/
				const float* GetCoords() const;

				/**
					* get current disulfid bonds
					*/
				std::vector<protein::CallProteinData::IndexPair> GetDisulfidBonds ( void );

			private:

				const NetCDFData *parent;

				/**
					* atom coordinates:
					* This variable shall contain the Cartesian coordinates of the
					* specified particle for the specified frame.
					*/
				float *coords;

				/**
					* time ellapsed since 1st frame:
					* When coordinates on the frame dimension have a temporal sequence
					* (e.g. they form a molecular dynamics trajectory), creators shall
					* define this dimension and write a float for each frame coordinate
					* representing the simulated time value in picoseconds associated with
					* the frame. Time zero is arbitrary, but typically will correspond to
					* the start of the simulation. When the file stores a collection of
					* conformations having no temporal sequence, creators shall omit this
					* variable.
					*/
				float* time;

				/**
					* When the coordinates variable is included and the data in the
					* coordinates variable come from a simulation with periodic
					* boundaries, creators shall include this variable. This variable
					* shall represent the lengths (a,b,c) of the unit cell for each
					* frame. The edge with length a lies along the x axis; the edge with
					* length b lies in the x-y plane. The origin (point of invariance
					* under scaling) of the unit cell is defined as (0,0,0). If the
					* simulation has one or two dimensional periodicity, then the
					* length(s) corresponding to spatial dimensions in which there is no
					* periodicity shall be set to zero.
					*/
				double* cell_lengths;

				/**
					* Creators shall include this variable if and only if they include
					* the cell_lengths variable. This variable shall represent the angles
					* ( $\alpha,\beta,\gamma$) defining the unit cell for each frame.
					* $\alpha$ defines the angle between the b and c vectors, $\beta$
					* defines the angle between the a and c vectors and $\gamma$ defines
					* the angle between the a and b vectors. Angles that are undefined
					* due to less than three dimensional periodicity shall be set to
					* zero.
					*/
				double* cell_angles;

				/**
					* When the velocities variable is present, it shall represent the
					* cartesian components of the velocity for the specified particle and
					* frame. It is recognized that due to the nature of commonly used
					* integrators in molecular dynamics, it may not be possible for the
					* creator to write a set of velocities corresponding to exactly the
					* same point in time as defined by the time variable and represented
					* in the coordinates variable. In such cases, the creator shall write
					* a set of velocities from the nearest point in time to that
					* represented by the specified frame.
					*/
				float* velocities;

				/** all disulfid bonds from dataset */
				std::vector<protein::CallProteinData::IndexPair> disulfidBondsVec;

		};

		/**********************************************************************
			* 'netcdf'-functions
			**********************************************************************/

		/**
			* Tries to load the file 'm_filename' into memory
			*
			* @ param rturn 'true' if file(s) could be loaded, 'false' otherwise
			*/
		bool tryLoadFile ( void );

		/**
			* Read the atom/amino acid table and build the connectivity table
			*
			* @return 'true' if header could be read, 'false' otherwise
			*/
		bool makeConnections ( void );

		/**
			* Reads the topology file
			*
			* @param topfile The topology file name
			*
			* @return 'true' if top file could be read, 'false' otherwise
			*/
		bool readTopfile ( const char* &topfile );

		/**
			* Reads the header from the netCDF file and checks validity
			*
			* @return 'true' if header of file could be read, 'false' otherwise
			*/
		bool readHeaderFromNetCDF ( void );

		/**
			* Set AminoAcids in each chain
			*/
		void initChains ( void );

		/**
			* Initialize solvent entries
			*/
		void initSolvent ( void );

		/**
			* Gets a netcdf attribute string. Must not be called if the netcdf file is
			* not open.
			*
			* @param varid The variable id or NC_GLOBAL
			* @param name The attribute name
			* @param outStr The string variable receiving the value.
			*
			* @return 'true' on success, 'false' on failure.
			*/
		bool netCDFGetAttrString ( int varid, const char *name, vislib::StringA& outStr );

		/**
			* Get the number of an element according to the periodic table of the elements
			*
			* @param
			*
			* @return
			*/
		//unsigned int getElementNumber(vislib::StringA name);

		/**
			* Reads a designated frame from the netCDF file
			*
			* @param frameID the frame to read
			*
			* @return
			*/
		//bool readFrameFromNetCDF(unsigned int frameID);

		/**
			* Builds up the frame index table.
			*/
		//void buildFrameTable(void);

		/**
			* Creates a frame to be used in the frame cache. This method will be
			* called from within 'initFrameCache'.
			*
			* @return The newly created frame object.
			*/
		virtual AnimDataModule::Frame* constructFrame ( void ) const;

		/**
			* Loads one frame of the data set into the given 'frame' object. This
			* method may be invoked from another thread. You must take
			* precausions in case you need synchronised access to shared
			* ressources.
			*
			* @param frame The frame to be loaded.
			* @param idx The index of the frame to be loaded.
			*/
		virtual void loadFrame ( AnimDataModule::Frame *frame, unsigned int idx );

		/**********************************************************************
			* variables
			**********************************************************************/

		/** Callee slot */
		megamol::core::CalleeSlot m_generalDataSlot;
		megamol::core::CalleeSlot m_RMSFrameDataSlot1;
		megamol::core::CalleeSlot m_RMSFrameDataSlot2;

		/** parameter slot for filename */
		megamol::core::param::ParamSlot m_filenameParam;

		/* frame objects used to store RMS data for each callee
		* (to prevent multiple loading of same frame) */
		Frame m_generalDataFrame;
		Frame m_RMSFrame1;
		Frame m_RMSFrame2;

		/** state variables for parameters */
		bool m_animation;

		/** the id of the netCDF file */
		int m_netcdfFileID;
		/** bool if the netCDF file is open */
		bool m_netcdfOpen;
		/** the id of the coordinates variable file */
		int m_coordsVarID;

		/** bool if the netCDF file header is read and varified */
		//bool m_netcdfHeaderVarified = false;

		/** IDs to validate open netCDF file */
		int m_frameDimID;
		int m_unlimID;
		int m_spatialDimID;
		int m_atomDimID;
		int m_cellspatialDimID;
		int m_cellangularDimID;
		int m_labelDimID;

		/** dimensions read from netCDF header */
		size_t m_framesize;
		size_t m_spatialsize;
		size_t m_atomsize;
		size_t m_cell_spatial_size;
		size_t m_cell_angular_size;
		size_t m_label_size;

		//int m_firstSolventAtomID;
		//int m_firstSolventMoleculeID;
		int m_frameID;

		/** total number of frames in dataset */
		int m_numFrames;

		/** total number of atoms in dataset */
		unsigned int m_numAtoms;

		/** total number of distinct atom types in dataset */
		unsigned int m_numAtomTypes;

		/** total number of residues in dataset */
		int m_numres;

		/** total number of distinct amino acid types in dataset */
		unsigned int m_numAminoAcidNames;

		/** NSP   : the total number of atoms in each molecule */
		//int *m_nsp;
		int *m_numAtomsPerMol;
		/** IPTRES : final residue that is considered part of the solute */
		//int m_iptres;
		int m_finSoluteRes;
		/** NSPM   : total number of molecules */
		//int m_nspm;
		int m_numMols;
		/** NSPSOL : the first solvent "molecule" */
		//int m_nspsol;
		int m_firstSolventMol;
		/* from LoadAmber */

		/** total number of protein atoms in dataset */
		unsigned int m_numProteinAtoms;
		/** Vector of chains in dataset */
		std::vector<protein::CallProteinData::Chain> m_proteinChains;

		/** index vector of sulfide atoms to calculate disulfide bonds*/
		std::vector<unsigned int> m_sulfides;

		/** all disulfid bonds from dataset */
		//std::vector<protein::CallProteinData::IndexPair> m_disulfidBondsVec;

		/** total number of solvent atoms in dataset */
		unsigned int m_numSolventAtoms;
		/** total number of solvent molecule types in dataset (from residue label) */
		unsigned int m_numSolventMolTypes;
		/** total number of solvent molecules per type in dataset */
		std::vector<unsigned int> m_numSolventMolsPerTypes;
		/** solvent molecule data */
		std::vector<protein::CallProteinData::SolventMoleculeData> m_solventData;

		/** index of first solvent "residue" (from residue label) */
		unsigned int m_firstSolRes;

		/** temporary struct for solvent data storage */
		struct solvent {
			unsigned int aminoAcidNameIdx;
			unsigned int numMolsPerType;
			unsigned int numAtomsPerType;
		};

		/** type vector of solvent data */
		std::vector<solvent> m_solventTypeVec;

		/** The frame object used to evaluate and store interpolated data */
		Frame m_interpolFrame;

		/** bool if loadFrame succeeded */
		//bool m_frameLoaded;

		/** The timer to control the animation */
		AnimDataTimer m_animTimer;

		/** Pointer to the frame used last time. */
		/*const*/ Frame *m_lastFrame;

		/** The time of the frame used last time. */
		float m_lastTime;

		// if 'true' a bond will not be added to the connectivity table if it's longer than 3 Angstrom
		//bool m_checkBondLength;

		/** atom coordinates */
		//float *m_coords;

		/** table of atom types */
		std::vector<protein::CallProteinData::AtomType> m_atomTypeTable;
		/** table of amino acid types */
		std::vector<vislib::StringA> m_aminoAcidTypeTable;
		/** all atoms in dataset */
		std::vector<protein::CallProteinData::AtomData> m_atomTable;
		/** only protein atoms */
		std::vector<protein::CallProteinData::AtomData> m_proteinAtomTable;
		/** only solvent atoms */
		std::vector<protein::CallProteinData::AtomData> m_solventAtomTable;
		/** all amino acids from dataset */
		std::vector<protein::CallProteinData::AminoAcid> m_aminoAcidTable;

		/** periodic box, angle between the XY and YZ planes - currently not supported in MegaMol */
		double m_box_beta;

		/** temporary storage of bounding box dimensions */
		float m_boundingBox[3];

		/** scale factor (for what ?) */
		float m_scale;

		/** disulfide bond table */
		//std::vector<Disulfide> m_disulfideBondTable;

		/** Stride secondary structure */
		Stride *m_stride;
		/** Stride sec struct computed */
		bool m_secondaryStructureComputed;

		/** the Id of the current frame of the NetCDF file */
		unsigned int currentFrameId;

		/** the maximum charge */
		float maxCharge;
		/** the minimum charge */
		float minCharge;

		/** the disulfide bonds */
		std::vector<protein::CallProteinData::IndexPair> disulfidBondsVec;

        /** the filename of the current netcdf file */
        vislib::StringA filename;
	};

} /* end namespace protein */
} /* end namespace megamol */

#endif /* MEGAMOL_NETCDFDATA_H_INCLUDED */

#endif /* (defined(WITH_NETCDF) && (WITH_NETCDF)) */
