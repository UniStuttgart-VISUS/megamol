/*
 * AnimDataModule.cpp
 *
 * Copyright (C) 2008 - 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "mmcore/view/AnimDataModule.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/assert.h"
#include "vislib/sys/Thread.h"
#include <chrono>

using namespace megamol::core;

#define MM_ADM_COUNT_LOCKED_FRAMES


/*
 * view::AnimDataModule::AnimDataModule
 */
view::AnimDataModule::AnimDataModule(void)
        : Module()
        , frameCnt(0)
        , loader(loaderFunction)
        , frameCache(NULL)
        , cacheSize(0)
        , stateLock()
        , lastRequested(0) {
    this->isRunning.store(false);
}


/*
 * view::AnimDataModule::~AnimDataModule
 */
view::AnimDataModule::~AnimDataModule(void) {
    this->Release();

    Frame** frames = this->frameCache;
    //    this->frameCache = NULL;
    this->isRunning.store(false);
    if (this->loader.IsRunning()) {
        this->loader.Join();
    }
    this->frameCache = NULL;
    if (frames != NULL) {
        for (unsigned int i = 0; i < this->cacheSize; i++) {
            delete frames[i];
        }
        delete[] frames;
    }
}


/*
 * view::AnimDataModule::initframeCache
 */
void view::AnimDataModule::initFrameCache(unsigned int cacheSize) {
    ASSERT(this->loader.IsRunning() == false);
    ASSERT(cacheSize > 0);
    ASSERT(this->frameCnt > 0);

    if (cacheSize > this->frameCnt) {
        cacheSize = this->frameCnt; // because we don't need more
    }

    if (this->frameCache != NULL) {
        for (unsigned int i = 0; i < this->cacheSize; i++) {
            delete this->frameCache[i];
        }
        delete[] this->frameCache;
    }

    this->cacheSize = cacheSize;
    this->frameCache = new Frame*[this->cacheSize];
    bool frameConstructionError = false;
    for (unsigned int i = 0; i < this->cacheSize; i++) {
        this->frameCache[i] = this->constructFrame();
        ASSERT(&this->frameCache[i]->owner == this);
        if (this->frameCache[i] != NULL) {
            this->frameCache[i]->state = Frame::STATE_INVALID;
        } else {
            frameConstructionError = true;
        }
    }

    if (!frameConstructionError) {
        this->frameCache[0]->state = Frame::STATE_LOADING;
        this->loadFrame(this->frameCache[0], 0); // load first frame directly.
        this->frameCache[0]->state = Frame::STATE_AVAILABLE;
        this->lastRequested = 0;

        this->isRunning.store(true);
        this->loader.Start(this);
        // XXX Is there a race condition that requires higher sleep time (value originally was 250)?
        // XXX Reduced for faster module creation, because called in ctor.
        vislib::sys::Thread::Sleep(10);
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "Unable to create frame data cache ('constructFrame' returned 'NULL').");
    }
}


/*
 * view::AnimDataModule::requestFrame
 */
view::AnimDataModule::Frame* view::AnimDataModule::requestLockedFrame(unsigned int idx) {
    Frame* retval = NULL;
    int dist, minDist = this->frameCnt;
    static bool deadlockwarning = true;

    this->stateLock.Lock();
    this->lastRequested = idx; // TODO: choose better caching strategy!!!
    for (unsigned int i = 0; i < this->cacheSize; i++) {
        if ((this->frameCache[i]->state == Frame::STATE_AVAILABLE) ||
            (this->frameCache[i]->state == Frame::STATE_INUSE)) {
            // note: do not wrap distance around!
            dist = labs(this->frameCache[i]->frame - idx);
            if (dist == 0) {
                retval = this->frameCache[i];
                break;
            } else if (dist < minDist) {
                retval = this->frameCache[i];
                minDist = dist;
            }
        }
    }
    if (retval != NULL) {
        retval->state = Frame::STATE_INUSE;
    }
    this->stateLock.Unlock();

    if (deadlockwarning
#if !(defined(DEBUG) || defined(_DEBUG))
        && (this->cacheSize < this->frameCnt)
    // streaming is required to handle this data set
#endif /* !(defined(DEBUG) || defined(_DEBUG)) */
    ) {
        unsigned int clcf = 0;
        for (unsigned int i = 0; i < this->cacheSize; i++) {
            if (this->frameCache[i]->state == Frame::STATE_INUSE) {
                clcf++;
            }
        }

        //printf("======== %u frames locked\n", clcf);

        if ((clcf == this->cacheSize) && (this->cacheSize > 2)) {
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(
                megamol::core::utility::log::Log::LEVEL_ERROR, "Possible data frame cache deadlock detected!");
            deadlockwarning = false;
        }
    }

    return retval;
}


/*
 * view::AnimDataModule::requestLockedFrame
 */
view::AnimDataModule::Frame* view::AnimDataModule::requestLockedFrame(unsigned int idx, bool forceIdx) {
    Frame* f = this->requestLockedFrame(idx);
    if ((f->FrameNumber() == idx) || (!forceIdx))
        return f;
    // wrong frame number and frame is forced

    // clamp idx
    if (idx >= this->frameCnt) {
        idx = this->frameCnt - 1;
        f->Unlock();
        f = this->requestLockedFrame(idx);
    }

    // wait for the new frame
    while (idx != f->FrameNumber()) {
        f->Unlock();

        // HAZARD: This will wait for all eternity if the requested frame is never loaded

        vislib::sys::Thread::Sleep(100); // time for the loader thread to load
        f = this->requestLockedFrame(idx);
    }

    return f;
}


/*
 * view::AnimDataModule::resetFrameCache
 */
void view::AnimDataModule::resetFrameCache(void) {
    Frame** frames = this->frameCache;
    //    this->frameCache = NULL;
    this->isRunning.store(false);
    if (this->loader.IsRunning()) {
        this->loader.Join();
    }
    this->frameCache = NULL;
    if (frames != NULL) {
        for (unsigned int i = 0; i < this->cacheSize; i++) {
            delete frames[i];
        }
        delete[] frames;
    }
    this->frameCnt = 0;
    this->cacheSize = 0;
    this->lastRequested = 0;
}


/*
 * view::AnimDataModule::setFrameCount
 */
void view::AnimDataModule::setFrameCount(unsigned int cnt) {
    ASSERT(this->loader.IsRunning() == false);
    ASSERT(cnt > 0);
    this->frameCnt = cnt;
}


/*
 * view::AnimDataModule::loaderFunction
 */
DWORD view::AnimDataModule::loaderFunction(void* userData) {
    AnimDataModule* This = static_cast<AnimDataModule*>(userData);
    ASSERT(This != NULL);
    unsigned int index, i, j, req;
#ifdef _LOADING_REPORTING
    unsigned int l;
#endif /* _LOADING_REPORTING */
    Frame* frame;
    vislib::StringA fullName(This->FullName());

    std::chrono::high_resolution_clock::duration accumDuration = std::chrono::seconds(0);
    unsigned int accumCount = 0;
    std::chrono::system_clock::time_point lastReportTime = std::chrono::system_clock::now();
    const std::chrono::system_clock::duration lastReportDistance = std::chrono::seconds(3);

    while (This->isRunning.load()) {
        // sleep to enforce thread changes
        vislib::sys::Thread::Sleep(1);
        if (!This->isRunning.load())
            break;

        // idea:
        //  1. search for the most important frame to be loaded.
        //  2. search for the best cached frame to be overwritten.
        //  3. load the frame

        // 1.
        // Note: we do not need to lock here, because we won't change the frame
        // state now, and the different states that can be set outside this
        // thread are aquivalent for us.
        index = req = This->lastRequested;
        for (j = 0; j < This->cacheSize; j++) {
            for (i = 0; i < This->cacheSize; i++) {
                if (!This->isRunning.load())
                    break;
                if (((This->frameCache[i]->state == Frame::STATE_AVAILABLE) ||
                        (This->frameCache[i]->state == Frame::STATE_INUSE)) &&
                    (This->frameCache[i]->frame == index)) {
                    break;
                }
            }
            if (!This->isRunning.load())
                break;
            if (i >= This->cacheSize) {
                break;
            }
            index = (index + 1) % This->frameCnt;
        }
        if (!This->isRunning.load())
            break;
        if (j >= This->cacheSize) {
            if (j >= This->frameCnt) {
                ASSERT(This->frameCnt == This->cacheSize);
                megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_INFO,
                    "All frames of the dataset loaded into cache. Terminating loading Thread.");
                break;
            }
            continue;
        }

        // 2.
        // Note: We now need to lock, because we must synchronise against
        // frames changing from 'STATE_AVAILABLE' to 'STATE_INUSE'.
        This->stateLock.Lock();
        // core idea: search for the frame with the largest distance to the requested frame
        frame = NULL; // the frame to be overwritten
        j = 0;        // the distance to the found frame to be overwritten
        for (i = 0; i < This->cacheSize; i++) {
            if (This->frameCache[i]->state == Frame::STATE_INVALID) {
                frame = This->frameCache[i];
                // j = UINT_MAX; // not required, since we instantly leave the loop
#ifdef _LOADING_REPORTING
                l = i;
#endif /* _LOADING_REPORTING */
                break;
            } else if (This->frameCache[i]->state == Frame::STATE_AVAILABLE) {
                // distance to the frame[i];
                long ld = static_cast<long>(This->frameCache[i]->frame) - static_cast<long>(req);
                if (ld < 0) {
                    if (ld < (static_cast<long>(This->frameCnt)) / 10) {
                        ld += static_cast<long>(This->frameCnt);
                        if (ld < 0)
                            ld = 0; // should never happen
                    } else {
                        ld = -10 * ld;
                    }
                }

                if (j < static_cast<unsigned int>(ld)) {
                    frame = This->frameCache[i];
                    j = static_cast<unsigned int>(ld);
#ifdef _LOADING_REPORTING
                    l = i;
#endif /* _LOADING_REPORTING */
                }
            }
            if (!This->isRunning.load())
                break;
        }

        // 3.
        if (frame != NULL) {
            frame->state = Frame::STATE_LOADING;
        }
        // if frame is NULL no suitable cache buffer found for loading. This is
        // mostly the case if the cache is too small or if the data source
        // locks too much frames.
        This->stateLock.Unlock();

        if ((frame != NULL) && This->isRunning.load()) {
#ifdef _LOADING_REPORTING
            printf("Loading frame %i into cache %i\n", index, l);
#endif /* _LOADING_REPORTING */

            std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

            This->loadFrame(frame, index);

            std::chrono::high_resolution_clock::duration duration = std::chrono::high_resolution_clock::now() - start;
            accumDuration += duration;
            accumCount++;

            std::chrono::system_clock::time_point reportTime = std::chrono::system_clock::now();
            if ((reportTime - lastReportTime) > lastReportDistance) {
                lastReportTime = reportTime;
                if (accumCount > 0) {
                    megamol::core::utility::log::Log::DefaultLog.WriteInfo(100, "[%s] Loading speed: %f ms/f (%u)",
                        fullName.PeekBuffer(),
                        1000.0 * std::chrono::duration_cast<std::chrono::duration<double>>(accumDuration).count() /
                            static_cast<double>(accumCount),
                        static_cast<unsigned int>(accumCount));
                }
            }

            // we no not need to lock here, because this transition from
            // 'STATE_LOADING' to 'STATE_AVAILABLE' is safe for the using
            // thread.
            frame->state = Frame::STATE_AVAILABLE;
        }
    }

    if (accumCount > 0) {
        megamol::core::utility::log::Log::DefaultLog.WriteInfo(100, "[%s] Loading speed: %f ms/f (%u)",
            fullName.PeekBuffer(),
            1000.0 * std::chrono::duration_cast<std::chrono::duration<double>>(accumDuration).count() /
                static_cast<double>(accumCount),
            static_cast<unsigned int>(accumCount));
    }

    megamol::core::utility::log::Log::DefaultLog.WriteInfo("The loader thread is exiting.");
    return 0;
}


/*
 * view::AnimDataModule::unlock
 */
void view::AnimDataModule::unlock(view::AnimDataModule::Frame* frame) {
    ASSERT(&frame->owner == this);
    ASSERT(frame->state == Frame::STATE_INUSE);
    this->stateLock.Lock();
    frame->state = Frame::STATE_AVAILABLE;
    this->stateLock.Unlock();
}
