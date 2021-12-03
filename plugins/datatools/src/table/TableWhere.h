/*
 * TableWhere.h
 *
 * Copyright (C) 2019 Visualisierungsinstitut der Universität Stuttgart
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "TableProcessorBase.h"


namespace megamol {
namespace datatools {
namespace table {

/**
 * This module selects rows from a table based on a filter.
 */
class TableWhere : public TableProcessorBase {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName(void) {
        return "TableWhere";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description(void) {
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

    virtual bool prepareData(TableDataCall& src, const unsigned int frameID) override;

    virtual void release(void);

private:
    core::param::ParamSlot paramColumn;
    core::param::ParamSlot paramEpsilon;
    core::param::ParamSlot paramOperator;
    core::param::ParamSlot paramReference;
    core::param::ParamSlot paramUpdateRange;
};

} /* end namespace table */
} /* end namespace datatools */
} /* end namespace megamol */
