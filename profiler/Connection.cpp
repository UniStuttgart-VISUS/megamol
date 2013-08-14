/*
 * profiler/Connection.cpp
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "profiler/Connection.h"

using namespace megamol;
using namespace megamol::core;


/*
 * profiler::Connection::Connection
 */
profiler::Connection::Connection(void) : call(nullptr), func(0) {

    // TODO: Implement

}


/*
 * profiler::Connection::~Connection
 */
profiler::Connection::~Connection(void) {
    this->call = nullptr; // do not delete

    // TODO: Implement

}


/*
 * profiler::Connection::begin_measure
 */
void profiler::Connection::begin_measure(void) {

    // TODO: Implement

}


/*
 * profiler::Connection::end_measure
 */
void profiler::Connection::end_measure(void) {

    // TODO: Implement

}
