#ifndef MEGAMOL_DATATOOLS_FLOATTABLEOBSERVERPLANE_H_INCLUDED
#define MEGAMOL_DATATOOLS_FLOATTABLEOBSERVERPLANE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/param/ParamSlot.h"
#include "mmstd_datatools/floattable/CallFloatTableData.h"
#include "mmcore/view/CallClipPlane.h"
#include <map>

namespace megamol {
namespace stdplugin {
namespace datatools {
namespace floattable {

    /**
     * This module converts from a generic float table to the MultiParticleDataCall.
     */
    class FloatTableObserverPlane : public megamol::core::Module {

    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static inline const char *ClassName(void)  {
            return "FloatTableObserverPlane";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static inline const char *Description(void) {
            return "A plane that observes relevant items in local (xy) coordinates over dicrete time steps and stacks them along the z axis.";
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
        FloatTableObserverPlane(void);

        /**
         * Finalises an instance.
         */
        virtual ~FloatTableObserverPlane(void);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        bool getObservedData(core::Call& call);

        bool getHash(core::Call& call);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

    private:

        bool assertData(floattable::CallFloatTableData *ft, megamol::core::view::CallClipPlane *cp, floattable::CallFloatTableData& out);

		bool anythingDirty();

		void resetAllDirty();

		std::string cleanUpColumnHeader(const std::string& header) const;
		std::string cleanUpColumnHeader(const vislib::TString& header) const;

        int getColumnIndex(const vislib::TString& colName);
		bool pushColumnIndex(std::vector<size_t>& cols, const vislib::TString& colName);

        /** Minimum coordinates of the bounding box. */
        float bboxMin[3];

        /** Maximum coordinates of the bounding box. */
        float bboxMax[3];

		float iMin, iMax;
        
        /** The slot for retrieving the stacked observed planes. */
        core::CalleeSlot slotCallObservedTable;

        /** The data callee slot. */
        core::CallerSlot slotCallFloatTable;

        /** The clip plane slot defining the observation plane. */
        core::CallerSlot slotCallClipPlane;

		/** The name of the float column holding the x-coordinate. */
        core::param::ParamSlot slotColumnX;

        /** The name of the float column holding the y-coordinate. */
        core::param::ParamSlot slotColumnY;

        /** The name of the float column holding the z-coordinate. */
        core::param::ParamSlot slotColumnZ;

        /** The name of the float column holding the particle radius. */
        core::param::ParamSlot slotColumnRadius;

        /**
        * The constant radius of spheres if the data set does not provide
        * one.
        */
        core::param::ParamSlot slotGlobalRadius;

        core::param::ParamSlot slotStartTime;
        core::param::ParamSlot slotEndTime;
        
        // makes no sense without float time in FTC
        //core::param::ParamSlot slotTimeIncrement;
        core::param::ParamSlot slotSliceOffset;

        /** The color mode: explicit rgb, intensity or constant */
        core::param::ParamSlot slotRadiusMode;

        /** how particles are chosen for the result planes */
        core::param::ParamSlot slotObservationStrategy;

		std::vector<float> everything;
		SIZE_T inputHash;
		SIZE_T myHash;
		std::map<std::string, size_t> columnIndex;
		size_t stride;
        int frameID;

    };

} /* end namespace floattable */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOL_DATATOOLS_FLOATTABLEOBSERVERPLANE_H_INCLUDED */