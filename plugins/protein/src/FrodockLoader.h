/*
 * FrodockLoader.h
 *
 * Copyright (C) 2010 by University of Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_FRODOCKLOADER_H_INCLUDED
#define MMPROTEINPLUGIN_FRODOCKLOADER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

// define the name size in number of characters
#define NAMESIZE 200

#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "vislib/Array.h"
#include "vislib/math/Vector.h"
#include "vislib/math/Matrix.h"
#include "vislib/math/Cuboid.h"
#include "protein_calls/MolecularDataCall.h"
#include "Stride.h"
#include "vislib/net/Socket.h"
#include "vislib/RawStorage.h"

namespace megamol {
namespace protein {

    /**
     * Data source for PDB files
     */

    class FrodockLoader : public megamol::core::Module
    {
    public:

        struct Potential_FI{
	        char name[NAMESIZE]; 		//filename of the receptor potential
	        float weight;				//Multiplication factor for the correlation obtained with the potential
	        int type;					//Potential type
        };

        enum Convention {
            Rosseta, 
            ICM, 
            EEF1,
            Sybil
        };

        struct FrodockInput{
	        //PDB FILES. MANDATORY
	        char receptor[NAMESIZE]; 	//Filename with the receptor PDB structure
	        char ligand[NAMESIZE];      //Filename with the ligand PDB structure

	        //MAIN POTENTIALS. OPTIONALS
	        // Van der Waals potential
	        char vdw[NAMESIZE]; 		//Filename with the receptor Van der Waals potential
	        float vdw_weight; 			//Multiplication factor for the correlation obtained with the vdw potential (default 1.0)
	        //Electrostatic potentials
	        char ele[NAMESIZE];	        //Filename with the receptor Electrostatic potential
	        float ele_weight;			//Multiplication factor for the correlation obtained with the electrostatic potential (default 0.3)
	        //Desolvation potential
	        char desol_rec[NAMESIZE];	//Filename with the receptor desolvation potential
	        char desol_lig[NAMESIZE];   //Filename with the ligand desolvation potential
	        char asa_rec[NAMESIZE];	    //Filename with the receptor asa projection
	        char asa_lig[NAMESIZE];	    //Filename with the ligand asa projection
	        float desol_weight;			//Multiplication factor for the correlation obtained with the desolvation potential (default 0.5)

	        //EXTRA POTENTIALS. OPTIONALS
	        //Up to 50 potentials can be added
	        int num_pot; 				//Number of extra potentials
	        Potential_FI *potentials;   //entries for each potential (each one stores the filename of the receptor potential and Multiplication factor)

	        //SEARCH PARAMETERS
	        int bw;						//Bandwitdh in spherical harmonic representation. Define rotational stepsize (default: 32. Rotational stepsize ~11º)
	        float lmax;					//External Mask reduction ratio (default: 0.25).
	        float lmin;			      	//Internal Mask reduction ratio (default: 0.26).
	        float th;	      			//Electrostatic map threshold (default: 10.0)
	        float lw;		      		//Width between spherical layers in amstrongs (default: 1.0).
	        float st;			      	//Translational search stepsize in amstrongs (default: 2.0).
	        int np;				      	//Number of solutions stored per traslational position. (default: 4)
	        float rd;				   	//Minimal rotational distance allowed between close solutions in degrees (default: 12.0)
	        int nt;	      				//Number of solutions stored in the search (default: unlimited -1).
	        float td;	      			//Maximal translational distance to consider close solutions in grid units. (default: 0)

	        bool use_around;			//Limit the translational search to a region (default: false)
	        float around_point[3];		//Coordinates of the central point of the region

	        char points [NAMESIZE];		//File with points to explore

	        Convention conv; 			//Forcefiel convention
        }; // Frodock input structure

        /** Ctor */
        FrodockLoader(void);

        /** Dtor */
        virtual ~FrodockLoader(void);

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void)  {
            return "FrodockLoader";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Offers Frodock data.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        /**
         * Call callback to get the data
         *
         * @param c The calling call
         *
         * @return True on success
         */
        bool getData( core::Call& call);

        /**
         * Call callback to get the extent of the data
         *
         * @param c The calling call
         *
         * @return True on success
         */
        bool getExtent( core::Call& call);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

    private:

        /**
         * Loads a PDB file.
         *
         * @param filename The path to the file to load.
         */
        void loadFile( const vislib::TString& filename);

        /**
         * Applies a solution (computes new atom positions).
         *
         * @param solIdx The solution index.
         * @return 'true' if the solution could be applied, 'false' otherwise.
         */
		bool applySolution(const megamol::protein_calls::MolecularDataCall *ligand, unsigned int solIdx);

        // -------------------- variables --------------------

        /** The file name slot */
        core::param::ParamSlot filenameSlot;
        /** The data callee slot */
        core::CalleeSlot dataOutSlot;

        /** molecular data caller slot (receptor molecule) */
        megamol::core::CallerSlot receptorDataCallerSlot;
        /** molecular data caller slot (ligand) */
        megamol::core::CallerSlot ligandDataCallerSlot;

        /** The STRIDE usage flag slot */
        core::param::ParamSlot strideFlagSlot;

        /** The file server name slot */
        core::param::ParamSlot fileServerNameSlot;

        /** The host address */
        core::param::ParamSlot hostAddressSlot;
        /** The port */
        core::param::ParamSlot portSlot;

        /** The bounding box */
        vislib::math::Cuboid<float> bbox;

        /** The data hash */
        SIZE_T datahash;

        /** The frodock input file */
        FrodockInput frodockInput;
        
        /** The socket for Frodock connection */
        vislib::net::Socket socket;

        /** The return value of frodock */
        int frodockResult;
        /** The ligand center */
        float ligandCenter[3];
        /** The number of solutions */
        int numSolutions;
        /** The solutions */
        vislib::RawStorage solutions;

        /** transformed atom positions */
        vislib::Array<float> atomPos;
        /** currently applied solution */
        int currentSolution;
    };


} /* end namespace protein */
} /* end namespace megamol */

#endif // MMPROTEINPLUGIN_FRODOCKLOADER_H_INCLUDED
