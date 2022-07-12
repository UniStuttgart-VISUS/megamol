/**
 * MegaMol
 * Copyright (c) 2011, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/Module.h"
#include "mmcore/param/ParamSlot.h"

namespace megamol::core::view {

/**
 * Class implementing time control
 */
class TimeControl {
public:
    /** Ctor. */
    TimeControl();

    /** Dtor. */
    virtual ~TimeControl();

    /**
     * Gets the idx'th slot
     *
     * @param idx Zero-based index
     *
     * @return The requested slot or NULL if idx is too large
     */
    AbstractSlot* GetSlot(unsigned int idx) {
        switch (idx) {
        case 0:
            return &this->animPlaySlot;
        case 1:
            return &this->animSpeedSlot;
        case 2:
            return &this->animTimeSlot;
        case 3:
            return &this->toggleAnimPlaySlot;
        case 4:
            return &this->animSpeedUpSlot;
        case 5:
            return &this->animSpeedDownSlot;
        case 6:
            return &this->masterSlot;
        case 7:
            return &this->slaveSlot;
        }
        return NULL;
    }

    /**
     * Gets the current time
     *
     * @param instTime The instance time code
     *
     * @return The current time
     */
    float Time(double instTime) const;

    /**
     * Sets the time extend of the source data
     *
     * @param frames The number of data frames or the number of the frame currently available at an in-situ source
     * @param isInSitu Flag whether or not the time control is to handle an in-situ data source
     */
    void SetTimeExtend(unsigned int frames, bool isInSitu);

private:
    /**
     * Toggles the animation button
     *
     * @param p The button parameter
     *
     * @return true;
     */
    bool onAnimToggleButton(param::ParamSlot& p);

    /**
     * Changes the animation speed
     *
     * @param p A button parameter
     *
     * @return true;
     */
    bool onAnimSpeedStep(param::ParamSlot& p);

    /**
     * Incoming call from a slave to this master
     *
     * @param c The incoming call
     *
     * @return Some return value
     */
    bool masterCallback(Call& c);

    /**
     * Gets the connected master
     *
     * @return The connected master or NULL
     */
    TimeControl* getMaster() const;

    /** Bool parameter to play/stop the animation */
    mutable param::ParamSlot animPlaySlot;

    /** Float parameter of animation speed in time frames per second */
    mutable param::ParamSlot animSpeedSlot;

    /** The slot holding the current time */
    mutable param::ParamSlot animTimeSlot;

    /** toggles the animation */
    param::ParamSlot toggleAnimPlaySlot;

    /** Speeds up the animation */
    param::ParamSlot animSpeedUpSlot;

    /** Slows down the animation */
    param::ParamSlot animSpeedDownSlot;

    /** Slot used if this time control is slave */
    mutable CallerSlot slaveSlot;

    /** Slot used if this time control is master */
    CalleeSlot masterSlot;

    /** The number of data frames or the number of the frame currently available at an in-situ source */
    unsigned int frames;

    /** Flag whether or not the time control is to handle an in-situ data source */
    bool isInSitu;

    /** Instance time offset */
    mutable double instOffset;
};

} // namespace megamol::core::view
