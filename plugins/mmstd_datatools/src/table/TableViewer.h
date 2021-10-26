#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"

#include "mmstd_datatools/table/TableDataCall.h"

namespace megamol {
namespace stdplugin {
namespace datatools {

    /**
     * This module converts from a generic table to the MultiParticleDataCall.
     */
    class TableViewer : public megamol::core::Module {

    public:

        /**
         * Answer the name of this module.
         *
         * @return The name of this module.
         */
        static inline const char *ClassName(void)  {
            return "TableViewer";
        }

        /**
         * Answer a human readable description of this module.
         *
         * @return A human readable description of this module.
         */
        static inline const char *Description(void) {
            return "Direct inspection of table values, data is passed through.";
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
        TableViewer(void);

        /**
         * Finalises an instance.
         */
        virtual ~TableViewer(void);

    protected:

        /**
         * Implementation of 'Create'.
         *
         * @return 'true' on success, 'false' otherwise.
         */
        virtual bool create(void);

        bool getTableData(core::Call& call);

        bool getTableHash(core::Call& call);

        /**
         * Implementation of 'Release'.
         */
        virtual void release(void);

    private:

        void drawTable(table::TableDataCall *c);

        /** The slot for retrieving the data as multi particle data. */
        core::CalleeSlot slotTableOut;

        /** The data callee slot. */
        core::CallerSlot slotTableIn;
    };

} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */
