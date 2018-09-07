/**
* timelog.cpp
*
* Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
* Alle Rechte vorbehalten.
*/


#include "mmcore/utility/timelog.h"


using namespace megamol;
using namespace megamol::core;
using namespace megamol::core::utility;


timelog::timelog(void) {

    this->set_time(timelog::TIME::START);
    this->set_time(timelog::TIME::WRITE);
    this->set_time(timelog::TIME::REQUEST);

    this->close_file();
}


timelog::~timelog(void) {

    this->close_file();
}


bool timelog::clear(vislib::StringA fn) {

    this->filename_ = fn;

    this->set_time(timelog::TIME::START);
    this->set_time(timelog::TIME::WRITE);
    this->set_time(timelog::TIME::REQUEST);

    this->close_file();
    if (this->logfile_.good()) {
        this->logfile_.open(this->filename_.PeekBuffer(), std::ios::binary);
        this->close_file();
    }

    return(this->logfile_.good());
}


double timelog::delta_time(timelog::TIME t) const {

    time_point ct;
    switch (t) {
        case (TIME::START):   ct = this->start_time_;   break;
        case (TIME::WRITE):   ct = this->write_time_;   break;
        case (TIME::REQUEST): ct = this->request_time_; break;
        default: break;
    }
    std::chrono::duration<double> diff = (std::chrono::system_clock::now() - ct);

    return diff.count();
}


bool timelog::write_time_line(vislib::StringA line) {
    bool retval = false;
    if (this->open_file()) {
        this->logfile_ << std::fixed << std::setprecision(6) << std::setw(12);
        this->logfile_ << this->delta_time(timelog::TIME::START) << " seconds: " << line.PeekBuffer() << "\n";
        this->close_file();
        retval = true;
    }

    return retval;
}


bool timelog::write_line(vislib::StringA line) {
    bool retval = false;
    if (this->open_file()) {
        this->logfile_ << std::fixed << std::setprecision(6) << std::setw(12);
        this->logfile_ << "                      " << line.PeekBuffer() << "\n";
        this->close_file();
        retval = true;
    }

    return retval;
}


bool timelog::open_file(void) {

    this->logfile_.clear();
    if (this->logfile_.good()) {
        if (!this->logfile_.is_open()) {
            this->logfile_.open(this->filename_.PeekBuffer(), std::ios::app | std::ios::binary);
        }
    }

    return(this->logfile_.good());
}


bool timelog::close_file(void) {

    if (this->logfile_.is_open()) {
        try { this->logfile_.flush(); }
        catch (...) {}
        try { this->logfile_.close(); }
        catch (...) {}
    }
    this->logfile_.clear();

    return(this->logfile_.good());
}