#ifndef MEGAMOL_DATATOOLS_FLOATTABLETOPARTICLES_H_INCLUDED
#define MEGAMOL_DATATOOLS_FLOATTABLETOPARTICLES_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_datatools/floattable/CallFloatTableData.h"
#include <map>

namespace megamol {
namespace stdplugin {
namespace datatools {

    /**
     * This module converts from a generic float table to the MultiParticleDataCall.
     */
    class FloatTableToParticles : public megamol::core::Module {

    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static inline const char *ClassName(void)  {
            return "FloatTableToParticles";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static inline const char *Description(void) {
            return "Converts generic float tables to Particles.";
        }

        /**
         * Answers whether this module is available on the current system.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        static inline bool IsAvailable(void) {
            return true;
        }

        /**
         * Initialises a new instance.
         */
        FloatTableToParticles(void);

        /**
         * Finalises an instance.
         */
        virtual ~FloatTableToParticles(void);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        bool getMultiParticleData(core::Call& call);

        bool getMultiparticleExtent(core::Call& call);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

    private:

		bool assertData(floattable::CallFloatTableData *ft);

		bool anythingDirty();

		void resetAllDirty();

		std::string cleanUpColumnHeader(const std::string& header) const;
		std::string cleanUpColumnHeader(const vislib::TString& header) const;

		bool pushColumnIndex(std::vector<size_t>& cols, const vislib::TString& colName);

        /** Minimum coordinates of the bounding box. */
        float bboxMin[3];

        /** Maximum coordinates of the bounding box. */
        float bboxMax[3];

		float iMin, iMax;
        
        /** The slot for retrieving the data as multi particle data. */
        core::CalleeSlot slotCallMultiPart;

        /** The data callee slot. */
        core::CallerSlot slotCallFloatTable;

        /** The name of the float column holding the red colour channel. */
        core::param::ParamSlot slotColumnB;

        /** The name of the float column holding the green colour channel. */
        core::param::ParamSlot slotColumnG;

        /** The name of the float column holding the blue colour channel. */
        core::param::ParamSlot slotColumnR;
        	
        /** The name of the float column holding the intensity channel. */
        core::param::ParamSlot slotColumnI;
        
		/**
		* The constant color of spheres if the data set does not provide
		* one.
		*/
		core::param::ParamSlot slotGlobalColor;

		/** The color mode: explicit rgb, intensity or constant */
        core::param::ParamSlot slotColorMode;

        /** The name of the float column holding the particle radius. */
        core::param::ParamSlot slotColumnRadius;

		/**
		* The constant radius of spheres if the data set does not provide
		* one.
		*/
		core::param::ParamSlot slotGlobalRadius;

		/** The color mode: explicit rgb, intensity or constant */
		core::param::ParamSlot slotRadiusMode;

		/** The name of the float column holding the x-coordinate. */
        core::param::ParamSlot slotColumnX;

        /** The name of the float column holding the y-coordinate. */
        core::param::ParamSlot slotColumnY;

        /** The name of the float column holding the z-coordinate. */
        core::param::ParamSlot slotColumnZ;

		std::vector<float> everything;
		SIZE_T inputHash;
		SIZE_T myHash;
		std::map<std::string, size_t> columnIndex;
		size_t stride;

    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOL_DATATOOLS_FLOATTABLETOPARTICLES_H_INCLUDED */