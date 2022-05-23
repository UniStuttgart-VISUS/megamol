/*
 * profiler/Connection.cpp
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */
#include "mmcore/profiler/Connection.h"
#include "mmcore/profiler/Manager.h"
#include "vislib/memutils.h"

using namespace megamol;
using namespace megamol::core;


/*
 * profiler::Connection::log_size
 */
const SIZE_T profiler::Connection::log_size = 1000;


/*
 * profiler::Connection::Connection
 */
profiler::Connection::Connection(void) : call(nullptr), func(0), values(nullptr), values_cnt(0), values_pos(-1) {
    this->values = new vislib::Pair<double, double>[log_size];
}


/*
 * profiler::Connection::~Connection
 */
profiler::Connection::~Connection(void) {
    this->call = nullptr; // do not delete
    ARY_SAFE_DELETE(this->values);
}


/*
 * profiler::Connection::begin_measure
 */
void profiler::Connection::begin_measure(void) {
    this->values_pos = (this->values_pos + 1) % log_size;
    this->values[this->values_pos].SetFirst(Manager::Instance().Now());
    this->values[this->values_pos].SetSecond(-1.0);
}


/*
 * profiler::Connection::end_measure
 */
void profiler::Connection::end_measure(void) {
    this->values[this->values_pos].SetSecond(Manager::Instance().Now() - this->values[this->values_pos].First());
    if (this->values_cnt <= this->values_pos)
        this->values_cnt = this->values_pos + 1;
}


/*
 * profiler::Connection::get_mean
 */
double profiler::Connection::get_mean(void) const {
    double m = 0.0;
    for (unsigned int i = 0; i < this->values_cnt; i++) {
        m += this->values[i].Second();
    }
    m /= static_cast<double>(this->values_cnt);
    return m;
}
