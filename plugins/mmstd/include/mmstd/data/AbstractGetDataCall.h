/**
 * MegaMol
 * Copyright (c) 2009, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "mmcore/BoundingBoxes.h"
#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"

namespace megamol::core {

/**
 * Abstract base class for calls for data
 */
class AbstractGetDataCall : public Call {
public:
    /**
     * Nested class interface for data unlockers. If data is returned with
     * an unlocker set. The caller must call 'Unlock' on the unlocker as
     * soon as the data is no longer required.
     */
    class Unlocker {
    public:
        /** ctor. */
        Unlocker() {
            // intentionally empty
        }

        /** dtor. */
        virtual ~Unlocker() {
            // intentionally empty
        }

        /** Unlocks the data */
        virtual void Unlock() = 0;
    };


    /** Ctor. */
    AbstractGetDataCall();

    /** Dtor. */
    ~AbstractGetDataCall() override;

    /**
     * Answer the unique hash number of the returned data, or zero if such
     * a number can not be provided.
     *
     * @return The unique hash number of the returned data
     */
    inline SIZE_T DataHash() const {
        return this->datahash;
    }

    /**
     * Answer the unlocker
     *
     * @return The unlocker
     */
    inline Unlocker* GetUnlocker() const {
        return this->unlocker;
    }

    /**
     * Sets the unique hash number for the returned data, or zero if such
     * a number can not be provided.
     *
     * @param hash The unique hash number
     */
    inline void SetDataHash(SIZE_T hash) {
        this->datahash = hash;
    }

    /**
     * Sets the data unlocker and optionally unlocks the old data if
     * present. The memory of the unlocker object 'unlocker' must be
     * allocated with defaut 'new'. The object will be owned by this call
     * object after this method returns. This means the caller must not
     * change the 'unlocker' object anymore, especially he must not delete
     * the object.
     *
     * @param unlocker The new unlocker object to use.
     * @param unlockOld If 'true' 'Unlock' is called before the new
     *                  unlocker is set.
     */
    inline void SetUnlocker(Unlocker* unlocker, bool unlockOld = true) {
        if (unlockOld)
            this->Unlock();
        this->unlocker = unlocker;
    }

    /**
     * Unlocks the data stored
     * This must be called after the data is no longer used to avoid
     * deadlocks in the out-of-core streaming mechanism.
     */
    inline void Unlock() {
        if (this->unlocker != NULL) {
            this->unlocker->Unlock();
            SAFE_DELETE(this->unlocker);
        }
    }

    /**
     * Assignment operator.
     * Makes a deep copy of all members. While for data these are only
     * pointers, the pointer to the unlocker object is also copied.
     *
     * @param rhs The right hand side operand
     *
     * @return A reference to this
     */
    AbstractGetDataCall& operator=(const AbstractGetDataCall& rhs);

protected:
    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount() {
        return 1;
    }

    /**
     * Answer the name of the function used for this call.
     *
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case 0:
            return "GetData";
        }
        return NULL;
    }

private:
    /**
     * A unique hash number of the returned data, or zero if such a number
     * can not be provided
     */
    SIZE_T datahash;

    /** the data unlocker */
    Unlocker* unlocker;
};

} // namespace megamol::core
