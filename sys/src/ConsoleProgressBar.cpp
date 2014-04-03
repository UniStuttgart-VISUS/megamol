/*
 * ConsoleProgressBar.cpp
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "vislib/ConsoleProgressBar.h"
#include "vislib/Console.h"
#include "vislib/mathfunctions.h"
#include "vislib/sysfunctions.h"
#include "the/string.h"
#include "the/text/string_builder.h"


#define PRINT_MILLISECONDS
#ifdef _WIN32
#define PBAR_FILLED_CHAR '\xFE' // cool ANSI-Character for Windows
#else /* _WIN32 */
#define PBAR_FILLED_CHAR '>'    // normal ASCII-Character for Non-Windows.
#endif /* _WIN32 */
#define PBAR_EMPTY_CHAR  ' '
#define PBAR_LEND_CHAR  '['
#define PBAR_REND_CHAR  ']'


/*
 * vislib::sys::ConsoleProgressBar::ConsoleProgressBar
 */
vislib::sys::ConsoleProgressBar::ConsoleProgressBar(void) : running(false), 
        maxValue(100), lastPers(-1.0f), startTime(0) {
}


/*
 * vislib::sys::ConsoleProgressBar::~ConsoleProgressBar
 */
vislib::sys::ConsoleProgressBar::~ConsoleProgressBar(void) {
    if (this->running) {
        this->Stop();
    }
}


/*
 * vislib::sys::ConsoleProgressBar::Set
 */
void vislib::sys::ConsoleProgressBar::Set(
        vislib::sys::ConsoleProgressBar::Size value) {
    if (this->running) {
        float pers = float(value) / float(this->maxValue);
        unsigned int now = vislib::sys::GetTicksOfDay();
        if (pers < 0.0f) {
            pers = 0.0f;
        }
        if (pers > 1.0f) {
            pers = 1.0f;
        }
        // update at least once per second
        // update at most four times per second
        // update whenever a tenth percent changed
        if (((int(pers * 
#ifdef PRINT_MILLISECONDS
            1000.0f
#else /* PRINT_MILLISECONDS */
            100.0f
#endif /* PRINT_MILLISECONDS */
            ) != int(this->lastPers * 
#ifdef PRINT_MILLISECONDS
            1000.0f
#else /* PRINT_MILLISECONDS */
            100.0f
#endif /* PRINT_MILLISECONDS */
            )) 
                && ((now - this->lastPersTime) > 250)) 
                || ((now - this->lastPersTime) > 1000)) {
            this->lastPers = pers;
            this->lastPersTime = now;
            this->update();
        }
    }
}


/*
 * vislib::sys::ConsoleProgressBar::Start
 */
void vislib::sys::ConsoleProgressBar::Start(const char *title, 
        vislib::sys::ConsoleProgressBar::Size maxValue) {
    if (this->running) {
        this->Stop();
    }
    this->title = title;
    this->title.append(": ");
    this->startTime = this->lastPersTime = vislib::sys::GetTicksOfDay();
    this->maxValue = maxValue;
    this->lastPers = 0.0f;
    this->running = true;
    this->update();
}


/*
 * vislib::sys::ConsoleProgressBar::Stop
 */
void vislib::sys::ConsoleProgressBar::Stop(void) {
    if (this->running) {
        this->lastPers = 1.0f;
        this->lastPersTime = vislib::sys::GetTicksOfDay();
        this->running = false;
        this->update();
        this->title.clear();
    }
}


/*
 * vislib::sys::ConsoleProgressBar::update
 */
void vislib::sys::ConsoleProgressBar::update(void) {
    static the::astring left;
    static the::astring right;
    static the::astring line;
    static the::astring tmp;
    unsigned int width = vislib::sys::Console::GetWidth() - 1;

    left = this->title;

    if (this->running) {
        the::text::astring_builder::format_to(right, "%3i%%", int(this->lastPers * 100.0f));
        if ((this->lastPersTime > this->startTime) 
                && (this->lastPers > 0.0f)) {
            unsigned int endTime = this->startTime 
                + int(float(this->lastPersTime - this->startTime) 
                / this->lastPers);
            unsigned int now = vislib::sys::GetTicksOfDay();
            // time elapsed
            printDuration(tmp, now - this->startTime);
            right.append(" (Times: ");
            right += tmp;
            // estimated time remaining
            printDuration(tmp, endTime - now);
            right.append("; ");
            right += tmp;
            // estimated overall time
            printDuration(tmp, endTime - this->startTime);
            right.append("; ");
            right += tmp;
            right.append(")");
        }

        int spc = int(width) - int(left.size() + right.size());

        if (spc < 10) {
            // console too tiny to print a progress bar.
            if (spc < 0) {
                int s = int(width) - int(left.size());
                if (s > 0) {
                    left += the::astring(' ', s);
                }
                s = int(width) - int(right.size());
                if (s > 0) {
                    right += the::astring(' ', s);
                }
                line = left + "\n" + right + "\n";
            } else {
                line = left + right + "\r";
            }
        } else {
            // print a progress bar.
            spc -= 2;
            unsigned int filled 
                = static_cast<unsigned int>(float(spc) * this->lastPers);
            line = left + PBAR_LEND_CHAR
                + the::astring(PBAR_FILLED_CHAR, filled)
                + the::astring(PBAR_EMPTY_CHAR, spc - filled)
                + PBAR_REND_CHAR + right + "\r";
        }

    } else {
        this->printDuration(tmp, this->lastPersTime - this->startTime);
        line = left + "Finished in " + tmp;
        int spc = int(width) - int(line.size());
        if (spc > 0) {
            line += the::astring(' ', spc);
        }
        line += "\n";
    }

    // the following two lines are the ONLY once realy printing output
    fprintf(stdout, "%s", line.c_str());
    fflush(stdout);
}


/*
 * vislib::sys::ConsoleProgressBar::printDuration
 */
void vislib::sys::ConsoleProgressBar::printDuration(the::astring& outStr, 
        unsigned int duration) {
#ifdef PRINT_MILLISECONDS
    unsigned int milli = duration % 1000;
#endif /* PRINT_MILLISECONDS */
    unsigned int sec = duration / 1000;
    unsigned int min = sec / 60;
    unsigned int hour = min / 60;
    min %= 60;
    sec %= 60;
    if (hour > 0) { 
        the::text::astring_builder::format_to(outStr, "%2u:%.2u:%.2u"
#ifdef PRINT_MILLISECONDS
            ".%.3u"
#endif /* PRINT_MILLISECONDS */
            , hour, min, sec
#ifdef PRINT_MILLISECONDS
            , milli
#endif /* PRINT_MILLISECONDS */
            );
    } else {
        the::text::astring_builder::format_to(outStr, "%.2u:%.2u"
#ifdef PRINT_MILLISECONDS
            ".%.3u"
#endif /* PRINT_MILLISECONDS */
            , min, sec
#ifdef PRINT_MILLISECONDS
            , milli
#endif /* PRINT_MILLISECONDS */
            );
    }
}
