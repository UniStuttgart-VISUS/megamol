/*
 * ConsoleProgressBar.h
 *
 * Copyright (C) 2006 - 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/String.h"


namespace vislib::sys {


/**
 * This class is a progress bar which is output to the text console on
 * 'stdout'.
 *
 * A typical Example:
 *
 *      ConsoleProgressBar pb;
 *
 *      pb.Start("Performing Operation", count);
 *      for (int i = 0; i < count; i++) {
 *
 *          doStuff(i);
 *
 *          pb.Set(i);
 *      }
 *      pb.Stop();
 *
 * Remark:
 * Since the console progress bar is written to 'stdout' you should not
 * output text on this stream or any similar stream shown in the same
 * console or else the resulting output might by mixed up.
 *
 * To ensure that this progress bar output does not have a to critical
 * impact on the applications performance, there will only be an output
 * at most four times a second.
 */
class ConsoleProgressBar {
public:
    /** typedef for the progress bar values */
    typedef unsigned int Size;

    /** ctor. */
    ConsoleProgressBar();

    /** dtor */
    ~ConsoleProgressBar();

    /**
     * Answer whether the progress bar is running.
     *
     * @return 'true' if the progress bar is running.
     */
    inline bool IsRunning() const {
        return this->running;
    }

    /**
     * Sets the value of the progress bar.
     *
     * @param value The new value of the progress bar. If the progress bar
     *              is not running, this call has no effect. The 'value' is
     *              clamped between zero and 'maxValue'.
     */
    void Set(Size value);

    /**
     * Starts the progress bar.
     *
     * @param title A very short title string for the progress. Should be
     *              less then 20 characters.
     * @param maxValue The maximum value to be reached at 100%
     */
    void Start(const char* title, Size maxValue);

    /**
     * Starts the progress bar.
     *
     * @param title A very short title string for the progress. Should be
     *              less then 20 characters.
     * @param maxValue The maximum value to be reached at 100%
     */
    inline void Start(const StringA& title, Size maxValue) {
        this->Start(title.PeekBuffer(), maxValue);
    }

    /**
     * Stops the progress bar.
     */
    void Stop();

private:
    /** updates the view of the progress bar */
    void update();

    /**
     * prints a duration.
     *
     * @param outStr The vislib string receiving the output.
     * @param duration The duration in milliseconds.
     */
    void printDuration(vislib::StringA& outStr, unsigned int duration);

    /** flag indicating if the progress bar is running */
    bool running;

    /** the max count */
    Size maxValue;

    /** the last percentage shown */
    float lastPers;

    /** the start time */
    unsigned int startTime;

    /** the time of last pers */
    unsigned int lastPersTime;

    /** the title string */
    vislib::StringA title;
};

} // namespace vislib::sys

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
