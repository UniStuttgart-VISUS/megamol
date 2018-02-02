/*
 * FloatTableJoin.h
 *
 * Copyright (C) 2016-2016 by VISUS (University of Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_DATATOOLS_FLOATTABLE_FLOATTABLEJOIN_H_INCLUDED
#define MEGAMOL_DATATOOLS_FLOATTABLE_FLOATTABLEJOIN_H_INCLUDED

#include "mmcore/Module.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"

#include "mmcore/param/ParamSlot.h"

#include "mmstd_datatools/floattable/CallFloatTableData.h"

namespace megamol {
namespace stdplugin {
namespace datatools {
namespace floattable {

/**
 * This module joins two float tables by copying the values together into one matrix
 */
class FloatTableJoin : public core::Module {
public:
    static std::string ModuleName;

    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char *ClassName(void) {
        return ModuleName.c_str();
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char *Description(void) {
        return "Joins two float tables (union of columns)";
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
    FloatTableJoin(void);

    /**
     * Finalises an instance.
     */
    virtual ~FloatTableJoin(void);

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
    virtual void release(void);

private:
    /** data callback */
    bool processData(core::Call &c);

    /** extent callback */
    bool getExtent(core::Call &c);

    /** concatenates two float tables */
    void concatenate(float * const out, const float * const first, const float * const second,
        const size_t rowsCount, const size_t columnCount, const size_t firstColumnCount,
        const size_t secondColumnCount);

    /** input slot of first float table */
    core::CallerSlot firstFloatTableInSlot;

    /** output slot of second float table */
    core::CallerSlot secondFloatTableInSlot;

    /** data output */
    core::CalleeSlot dataOutSlot;

    /** frameID */
    int frameID;

    /** datahash */
    size_t firstDataHash;
    size_t secondDataHash;

    /** number of rows of the float table */
    size_t rows_count;

    /** number of columns of the float table */
    size_t column_count;

    /** vector storing the meta information of each column */
    std::vector<CallFloatTableData::ColumnInfo> column_info;

    /** vector storing the data values of the float table */
    std::vector<float> data;
}; /* end class FloatTableJoin */

} /* end namespace floattable */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif // end ifndef MEGAMOL_DATATOOLS_FLOATTABLE_FLOATTABLEJOIN_H_INCLUDED