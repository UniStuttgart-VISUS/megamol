/*
 * profiler/Connection.h
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_PROFILER_CONNECTION_H_INCLUDED
#define MEGAMOLCORE_PROFILER_CONNECTION_H_INCLUDED
#pragma once

#include "mmcore/Call.h"
#include "vislib/Array.h"
#include "vislib/Pair.h"
#include <memory>


namespace megamol {
namespace core {
namespace profiler {

/**
 * Connection of a call to the profiling manager
 */
class Connection {
public:
    /** smart pointer type */
    typedef std::shared_ptr<Connection> ptr_type;

    /** Ctor */
    Connection(void);

    /** Dtor */
    ~Connection(void);

    /** Begins the performance measurement of the single call invoke */
    void begin_measure(void);

    /** Ends the performance measurement of the single call invoke */
    void end_measure(void);

    /**
     * Sets the connected call
     *
     * @param c The connected call
     */
    inline void set_call(const Call* c) {
        this->call = c;
    }

    /**
     * Answer the connected call
     *
     * @return The connected call
     */
    inline const Call* get_call(void) const {
        return this->call;
    }

    /**
     * Sets the function id
     *
     * @param id The new function id
     */
    inline void set_function_id(unsigned int id) {
        this->func = id;
    }

    /**
     * Gets the function id
     *
     * @return The function id
     */
    inline unsigned int get_function_id(void) const {
        return this->func;
    }

    /**
     * Computes the mean call duration time (in seconds)
     *
     * @return The mean call duration time (in seconds)
     */
    double get_mean(void) const;

private:
    /** The log size */
    static const SIZE_T log_size;

    /** The connected call */
    const Call* call;

    /** The function number */
    unsigned int func;

    /** The measured performance values */
    vislib::Pair<double, double>* values;

    /** The count of valid values */
    SIZE_T values_cnt;

    /** The position of the current value */
    SIZE_T values_pos;
};

} /* end namespace profiler */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_PROFILER_CONNECTION_H_INCLUDED */
