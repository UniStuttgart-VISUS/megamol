#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"
#include "geometry_calls/LinesDataCall.h"
#include "mmstd_datatools/table/TableDataCall.h"
#include <map>

namespace megamol {
namespace stdplugin {
namespace datatools {
        /**
         * This module converts from a generic table to the LineDataCall.
         */
        class TableToLines : public core::Module {

        public:

            /**
             * Answer the name of this module.
             *
             * @return The name of this module.
             */
            static inline const char *ClassName(void) {
                return "TableToLines";
            }

            /**
             * Answer a human readable description of this module.
             *
             * @return A human readable description of this module.
             */
            static inline const char *Description(void) {
                return "Converts generic tables to Lines.";
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
            TableToLines(void);

            /**
             * Finalises an instance.
             */
            virtual ~TableToLines(void);

        protected:

            /**
             * Implementation of 'Create'.
             *
             * @return 'true' on success, 'false' otherwise.
             */
            virtual bool create(void);

            bool getLineData(core::Call& call);

            bool getLineDataExtent(core::Call& call);

            /**
             * Implementation of 'Release'.
             */
            virtual void release(void);

        private:

            bool assertData(table::TableDataCall *ft);

            bool anythingDirty();

            void resetAllDirty();

            void colorTransferGray(std::vector<float> &grayArray, float const* transferTable, unsigned int tableSize, std::vector<float> &rgbaArray, unsigned int target_length);

            std::string cleanUpColumnHeader(const std::string& header) const;
            std::string cleanUpColumnHeader(const vislib::TString& header) const;

            bool pushColumnIndex(std::vector<size_t>& cols, const vislib::TString& colName);

            /** Minimum coordinates of the bounding box. */
            float bboxMin[3];

            /** Maximum coordinates of the bounding box. */
            float bboxMax[3];

            float iMin, iMax;

            /** The slot for retrieving the data as line data. */
            core::CalleeSlot slotDeployData;

            /** The data callee slot. */
            core::CallerSlot slotCallTable;

            /** The color transfer function slot. */
            core::CallerSlot slotTF;

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

            /** The name of the float column holding the x-coordinate. */
            core::param::ParamSlot slotColumnX;

            /** The name of the float column holding the y-coordinate. */
            core::param::ParamSlot slotColumnY;

            /** The name of the float column holding the z-coordinate. */
            core::param::ParamSlot slotColumnZ;

            /** The name of the float column holding the line index. */
            core::param::ParamSlot slotColumnIndex;

            /** The connection type of the data set. */
            core::param::ParamSlot slotConnectionType;

            std::vector<float> everything;
            SIZE_T inputHash;
            SIZE_T myHash;
            std::map<std::string, size_t> columnIndex;
            size_t stride;

            std::vector<float> allVerts;
            std::vector<float> allColor;
            std::vector<int> lineIndices;
            std::vector<std::vector<float>> lineVerts;
            std::vector<std::vector<float>> lineColor;

            std::vector<geocalls::LinesDataCall::Lines> lines;

        };
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */
