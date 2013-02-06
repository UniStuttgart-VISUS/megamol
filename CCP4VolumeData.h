/*
 * CCP4VolumeData.h
 *
 * Author: Michael Krone
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef MEGAMOL_PROTEIN_CCP4VOLUMEDATA_H_INCLUDED
#define MEGAMOL_PROTEIN_CCP4VOLUMEDATA_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "CallVolumeData.h"
#include "Module.h"
#include "CalleeSlot.h"
#include "param/ParamSlot.h"
#include "vislib/IllegalParamException.h"
#include "vislib/Vector.h"
#include "vislib/Array.h"
#include "vislib/String.h"
#include "vislib/memutils.h"
#include <vector>
#include <map>
#include "Stride.h"

namespace megamol {
namespace protein {

    /**
     * Data source for CCP4 files
     *
     * For a detailed description of the CCP4 file format, refer to:
     * http://www.ccp4.ac.uk/html/maplib.html#description
     *
     * TODO / Known Bugs and Issues:
     *  - Currently, only files in little endian encoding can be read
     *  - The space group number is assumed to be always 0
     *  - Symmetry information is not handled
     *  - Skew transformation is not supported
     *  - Reorder map data, if corresponding axis are not in order: (x, y, z)
     *  - 
     */

    class CCP4VolumeData : public megamol::core::Module
    {
    public:
        
        /**
         * Container class for CCP4 header information
         */
        class CCP4Header{
        public:
            // volume dimension in voxels (cols x rows x sections)
            int volDim[3];
            // data type
            // TODO: currently, only 2 (REAL) is supported
            int mode;
            // Number of first col/row/sec in map
            int start[3];
            // number of intervals along all three axes
            int intervals[3];
            // cell dimensions in Angstrom
            float cellDim[3];
            // cell angles in degrees
            // TODO: currently, only 90 is supported
            float cellAngles[3];
            // Corresponding axis for cols/rows/secs (1 = x, 2 = y, 3 = z)
            int correspondigAxis[3];
            // The minimum density
            float minDensity;
            // The maximum density
            float maxDensity;
            // The mean (or average) density
            float meanDensity;
            // Space group number
            int ispg;
            // Number of bytes used for storing symmetry operators
            int nsymbyte;
            // Flag for skew transformation, =0 none, =1 fill
            int skewTransformFlag;
            // Skew matrix S (in order S11, S12, S13, S21 etc)
            float skewMatrix[9];
            // Skew translation t
            float skewTranslation[3];
            // reserved for future use
            float future[11];
            // If this contains "ADP ", the next three values contain a float 
            // offset to add to the first positions of the map.
            char ADP[4];
            // Float offset for first position of map (i.e. origin offset)
            float offset[3];
            // Must contain "MAP "
            char MAP[4];
            // Machine stamp indicating the machine type which wrote the file
            int machine;
            // RMS deviation of map from mean density
            float rmsDeviation;
            // number of labels being used
            int numLabel;
            // 10 x 80 character text labels
            char label[10*80];
        };

        /** Ctor */
        CCP4VolumeData(void);

        /** Dtor */
        virtual ~CCP4VolumeData(void);

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static const char *ClassName(void) 
        {
            return "CCP4VolumeData";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static const char *Description(void) 
        {
            return "Offers volumetric data.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static bool IsAvailable(void) 
        {
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
         * Implementation of 'Release'.
         */
        virtual void release(void) {};

    private:

        /**
         * ProtData callback.
         *
         * @param call The calling call.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        bool VolumeDataCallback( megamol::core::Call& call);

        /**********************************************************************
         * 'volume'-functions
         **********************************************************************/

        /** Clear and reset all data structures */
        void ClearData();

        /**
         * Tries to load the file 'm_filename' into memory
         * 
         * @ param rturn 'true' if file(s) could be loaded, 'false' otherwise 
         */
        bool tryLoadFile(void);

        /**********************************************************************
         * variables
         **********************************************************************/

        /** Callee slot */
        megamol::core::CalleeSlot volumeDataCalleeSlot;

        /** The filename parameter */
        megamol::core::param::ParamSlot filename;

        ////////////////////////////////////////////////////////////
        // variables for storing all data needed by the interface //
        ////////////////////////////////////////////////////////////
        CCP4Header header;
        float *map;
        char *symmetry;
    };


} /* end namespace protein */
} /* end namespace megamol */

#endif //MEGAMOL_PROTEIN_CCP4VOLUMEDATA_H_INCLUDED
