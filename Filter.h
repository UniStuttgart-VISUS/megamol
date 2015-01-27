/*
 * Filter.h
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MMPROTEINPLUGIN_FILTER_H_INCLUDED
#define MMPROTEINPLUGIN_FILTER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */


#include "CalleeSlot.h"
#include "CallerSlot.h"
#include "param/ParamSlot.h"
#include "MolecularDataCall.h"
#include "view/Renderer3DModule.h"

#if (defined(WITH_CUDA) && (WITH_CUDA))

//#include <cudpp/cudpp.h>
#include "filter_cuda.cuh"

#endif // (defined(WITH_CUDA) && (WITH_CUDA))


namespace megamol {
namespace protein {


    /*
     * Filter class
     */
    class Filter : public megamol::core::Module {

    public:


        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) {
            return "Filter";
        }


        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) {
            return "Offers data filtering.";
        }


        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) {
            return true;
        }


        /** Ctor. */
        Filter(void);


        /** Dtor. */
        virtual ~Filter(void);

    protected:


        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);


        /**
         * Implementation of 'release'.
         */
        virtual void release(void);
        
        
        /**
         * Call callback to get the data
         *
         * @param c The calling call
         *
         * @return True on success
         */
        bool getData(core::Call& call);


        /**
         * Call callback to get the extent of the data
         *
         * @param c The calling call
         *
         * @return True on success
         */
        bool getExtent(core::Call& call);


    private:
    
    
        enum {TOPDOWN, BOTTOMUP};
        
        enum filterMode {NONE, SOLVENT, SOLVENTALT};
        
        
        /**
         * Helper class to unlock frame data.
         */
        class Unlocker : public MolecularDataCall::Unlocker {
        public:

            /**
             * Ctor.
             *
             * @param mol The molecular data call whos 'Unlock'-method is to be 
             *            called.
             */
            Unlocker(MolecularDataCall& mol) : MolecularDataCall::Unlocker(),
                mol(&mol){
                // intentionally empty
            }

            /** Dtor. */
            virtual ~Unlocker(void) {
                this->Unlock();
            }

            /** Unlocks the data */
            virtual void Unlock(void) {
                this->mol->Unlock();            
            }
            
        private:
        
            MolecularDataCall *mol;

        };        
        
        /**
         * Update all parameters.
         *
         * @param mol   Pointer to the data call.
         */
        void updateParams(MolecularDataCall *mol);
        
        
        /**
         * Initialize visibility information for all atoms.
         * 
         * @param visibility The visibility flag
         */
        void initAtomVisibility(int visibility);
        
        
        /**
         * Flag all solvent atoms of a given data source.
         * 
         * @param mol The molecular data call
         */
        void flagSolventAtoms(const MolecularDataCall *mol);
        
        
        /**
         * Gets positions of non-solvent atoms.
         * 
         * @param atomPos The current atom positions
         */
        void getProtAtoms(float *atomPos);
        
        
        /**
         * Set hierarchical visibility information.
         */
        void setHierarchicalVisibility(const MolecularDataCall *mol);
        
        
        /**
         * Filters solvent atoms according to the number of non-solvent atoms
         * within their neighbourhood cells defined by a given range.
         *
         * @param atomPos The current atom positions
         */
        void filterSolventAtoms(float *atomPos);
        
        
        /**
         * Filters solvent atoms according to the number of non-solvent atoms
         * within their neighbourhood cells defined by a given range.
         *
         * @param atomPos The current atom positions
         */
        void filterSolventAtomsAlt(float *atomPos);

        
        /** Caller/callee slot */
        megamol::core::CallerSlot molDataCallerSlot;
        megamol::core::CalleeSlot dataOutSlot;
        
        /** Parameter slots */
        megamol::core::param::ParamSlot hierarchyParam;
        megamol::core::param::ParamSlot solvRadiusParam;
        megamol::core::param::ParamSlot interpolParam;
        megamol::core::param::ParamSlot gridSizeParam;
        megamol::core::param::ParamSlot filterParam;
        
        /** The calltime of the last call */
        float calltimeOld;
        
        /** Number of atoms */
        unsigned int atmCnt;
        unsigned int solvAtmCnt;
        unsigned int protAtmCnt;
        
        /** Array with atom visibility information */
        int *atomVisibility; // note: 1 = visible, 0 = invisible
        
        /** Flags all solvent atoms */
        bool *isSolventAtom;
        
        /** Contains positions of non-solvent atoms */
        float *atmPosProt;
        
        
#if (defined(WITH_CUDA) && (WITH_CUDA))
        
        /** CUDA **/ 
        
        //CUDPPHandle sortHandle;
        FilterParams params;
        
        float *atomPosD;
        float *atomPosProtD;
        float *atomPosProtSortedD;
        
        unsigned int *gridAtomHashD;
        unsigned int *gridAtomIndexD;
        
        bool *isSolventAtomD;
        
        unsigned int *cellStartD;
        unsigned int *cellEndD;

        int *atomVisibilityD;
        
        int *neighbourCellPosD;

#endif // (defined(WITH_CUDA) && (WITH_CUDA))


    };


} /* end namespace protein */
} /* end namespace megamol */

#endif // MMPROTEINPLUGIN_FILTER_H_INCLUDED
