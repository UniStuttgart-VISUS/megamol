/*
 * TableSort.h
 *
 * Copyright (C) 2019 Visualisierungsinstitut der Universität Stuttgart
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "TableProcessorBase.h"


namespace megamol::datatools::table {

/**
 * This module sorts tabular data according to the specified column.
 */
class TableSort : public TableProcessorBase {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() {
        return "TableSort";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() {
        return "Selects rows from a data table";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable() {
        return true;
    }

    /**
     * Initialises a new instance.
     */
    TableSort();

    /**
     * Finalises an instance.
     */
    ~TableSort() override;

protected:
    bool create() override;

    bool prepareData(TableDataCall& src, const unsigned int frameID) override;

    void release() override;

private:
    core::param::ParamSlot paramColumn;
    core::param::ParamSlot paramIsDescending;
    core::param::ParamSlot paramIsStable;
};

} // namespace megamol::datatools::table
