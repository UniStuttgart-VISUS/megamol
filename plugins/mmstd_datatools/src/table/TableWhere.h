/*
 * TableWhere.h
 *
 * Copyright (C) 2019 Visualisierungsinstitut der Universität Stuttgart
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmcore/param/ParamSlot.h"

#include "mmstd_datatools/table/TableDataCall.h"


namespace megamol {
namespace stdplugin {
namespace datatools {
namespace table {

    /**
     * This module selects rows from a table based on a filter.
     */
    class TableWhere : public core::Module {

    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static inline const char *ClassName(void) {
            return "TableWhere";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static inline const char *Description(void) {
            return "Selects rows from a data table";
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
        TableWhere(void);

        /**
         * Finalises an instance.
         */
        virtual ~TableWhere(void);

    protected:

        virtual bool create(void);

        virtual void release(void);

    private:

        typedef megamol::stdplugin::datatools::table::TableDataCall::ColumnInfo
            ColumnInfo;

        bool getData(core::Call& call);

        bool getHash(core::Call& call);

        inline std::size_t getHash(void) {
            auto retval = this->inputHash;
            retval ^= this->localHash + 0x9e3779b9 + (retval << 6)
                + (retval >> 2);
            return retval;
        }

        std::vector<ColumnInfo> columns;
        unsigned int frameID;
        std::size_t inputHash;
        std::size_t localHash;
        core::param::ParamSlot paramColumn;
        core::param::ParamSlot paramEpsilon;
        core::param::ParamSlot paramOperator;
        core::param::ParamSlot paramReference;
        core::CallerSlot slotInput;
        core::CalleeSlot slotOutput;
        std::vector<float> values;
    };

} /* end namespace table */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */
