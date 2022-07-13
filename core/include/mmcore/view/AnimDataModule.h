/*
 * AnimDataModule.h
 *
 * Copyright (C) 2008 - 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_ANIMDATAMODULE_H_INCLUDED
#define MEGAMOL_ANIMDATAMODULE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <atomic>

#include "mmcore/Module.h"
#include "vislib/sys/CriticalSection.h"
#include "vislib/sys/Thread.h"


namespace megamol {
namespace core {
namespace view {

/**
 * Abstract base class for data sources for time-dependent data sets.
 */
class AnimDataModule : public Module {
public:
    /** Dtor. */
    virtual ~AnimDataModule(void);

    /**
     * Answer the number of time frames of the dataset.
     *
     * @return The number of time frames of the dataset.
     */
    inline unsigned int FrameCount(void) const {
        return this->frameCnt;
    }

protected:
    /**
     * Base class for holding all variable data of one time frame of the
     * dataset.
     */
    class Frame {
    public:
        friend class ::megamol::core::view::AnimDataModule;

        /** possible values for the state of the frame */
        enum State { STATE_INVALID, STATE_LOADING, STATE_AVAILABLE, STATE_INUSE };

        /**
         * Ctor.
         *
         * @param owner The owning AnimDataModule
         */
        Frame(AnimDataModule& owner) : frame(0), owner(owner), state(STATE_INVALID) {
            // intentionally empty
        }

        /** Dtor. */
        virtual ~Frame() {
            // intentionally empty
        }

        /**
         * Answer the number of the loaded frame.
         *
         * @return The number of the loaded frame.
         */
        inline unsigned int FrameNumber(void) const {
            return this->frame;
        }

        /**
         * Answers whether this frame is locked or not.
         *
         * @return 'true' if this frame is locked, 'false' otherwise.
         */
        inline bool IsLocked(void) const {
            return (this->state == STATE_INUSE);
        }

        /**
         * Unlocks the frame.
         * This should be called as soon as the frame is no longer used.
         * After you unlocked the frame, it's content is no longer
         * guaranteed to be stable.
         */
        inline void Unlock(void) {
            if (this->state == STATE_INUSE) {
                owner.unlock(this);
            }
        };

    protected:
        /** the number of the loaded frame. */
        unsigned int frame;

    private:
        /** The owning AnimDataModule */
        AnimDataModule& owner;

        /** the state of this frame */
        State state;
    };

    /**
     * hidden ctor. Derived classes should use a ctor with similar syntax!
     */
    AnimDataModule(void);

    /**
     * Answer the size of the cache
     *
     * @return The size of the cache
     */
    inline unsigned int CacheSize(void) const {
        return this->cacheSize;
    }

    /**
     * Creates a frame to be used in the frame cache. This method will be
     * called from within 'initFrameCache'.
     *
     * @return The newly created frame object.
     */
    virtual Frame* constructFrame(void) const = 0;

    /**
     * Initialises the frame cache to the given size. 'setFrameCount'
     * should be called before.
     *
     * @param cacheSize The number of frames to be held in cache.
     */
    void initFrameCache(unsigned int cacheSize);

    /**
     * Loads one frame of the data set into the given 'frame' object. This
     * method may be invoked from another thread. You must take
     * precausions in case you need synchronised access to shared
     * ressources.
     *
     * @param frame The frame to be loaded.
     * @param idx The index of the frame to be loaded.
     */
    virtual void loadFrame(Frame* frame, unsigned int idx) = 0;

    /**
     * Requests the frame from the frame cache, which is the best for the
     * requested frame index. Must not be called before the frame cache
     * has been initialised. The returned frame will be marked with state
     * 'STATE_INUSE'. You must call 'Unlock' on the Frame as soon as you
     * do not longer need this data. Not calling 'Unlock' will result in a
     * deadlock of the streaming mechanism loading the data.
     *
     * @param idx The index of the frame to be returned.
     * @param forceIdx If set to true, the frame is only returned for
     *                 exactly the requested idx, and not the closest
     *                 match.
     *
     * @return The frame most suitable to the request.
     */
    Frame* requestLockedFrame(unsigned int idx);
    Frame* requestLockedFrame(unsigned int idx, bool forceIdx);

    /**
     * Resets the whole module to the same state as directly after the
     * 'ctor' returned. You must call 'setFrameCount' and 'initFrameCache'
     * again before you can use the module.
     */
    void resetFrameCache(void);

    /**
     * Sets the number of time frames of the dataset. Must not be called
     * after the frame cache has been initialised!
     *
     * @param cnt The number of time frames of the dataset. Must not be
     *            zero.
     */
    void setFrameCount(unsigned int cnt);

    /** frame is a friend to be able to call 'unlock' */
    friend class ::megamol::core::view::AnimDataModule::Frame;

private:
    /**
     * The loader thread function.
     *
     * @param userData Pointer to this object.
     *
     * @return The return value.
     */
    static DWORD loaderFunction(void* userData);

    /**
     * Unlocks the given frame
     */
    void unlock(Frame* frame);

#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */

    /** The number of time frames of the dataset */
    unsigned int frameCnt;

    /** The loading thread */
    vislib::sys::Thread loader;

    /** The frame cache */
    Frame** frameCache;

    /** the number of frames to be held in cache. */
    unsigned int cacheSize;

    /**
     * The critical section to synchornise the state changes of the
     * cached frames.
     */
    vislib::sys::CriticalSection stateLock;

    /** The frame number requested the last time 'requestLockedFrame' was called */
    unsigned int lastRequested;

    /** TODO: The Mueller shalt document his stuff */
    std::atomic_bool isRunning;
#ifdef _WIN32
#pragma warning(default : 4251)
#endif /* _WIN32 */
};


} // namespace view
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOL_ANIMDATAMODULE_H_INCLUDED */
