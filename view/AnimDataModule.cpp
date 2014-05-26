/*
 * AnimDataModule.cpp
 *
 * Copyright (C) 2008 - 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "AnimDataModule.h"
#include "vislib/assert.h"
#include "vislib/Log.h"

using namespace megamol::core;

#define MM_ADM_COUNT_LOCKED_FRAMES


/*
 * view::AnimDataModule::AnimDataModule
 */
view::AnimDataModule::AnimDataModule(void) : frameCnt(0),
        loader(loaderFunction), frameCache(NULL), cacheSize(0),
        stateLock(), lastRequested(0) {
    this->isRunning.store(false);
}


/*
 * view::AnimDataModule::~AnimDataModule
 */
view::AnimDataModule::~AnimDataModule(void) {
    this->Release();

    Frame ** frames = this->frameCache;
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
    } else {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to create frame data cache ('constructFrame' returned 'NULL').");
    }
}


/*
 * view::AnimDataModule::requestFrame
 */
view::AnimDataModule::Frame * view::AnimDataModule::requestLockedFrame(unsigned int idx) {
    Frame *retval = NULL;
    int dist, minDist = this->frameCnt;
    static bool deadlockwarning = true;

    this->stateLock.Lock();
    this->lastRequested = idx; // TODO: choose better caching strategy!!!
    for (unsigned int i = 0; i < this->cacheSize; i++) {
        if ((this->frameCache[i]->state == Frame::STATE_AVAILABLE)
                || (this->frameCache[i]->state == Frame::STATE_INUSE)) {
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
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Possible data frame cache deadlock detected!");
            deadlockwarning = false;
        }
    }

    return retval;
}


/*
 * view::AnimDataModule::resetFrameCache
 */
void view::AnimDataModule::resetFrameCache(void) {
    Frame ** frames = this->frameCache;
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
DWORD view::AnimDataModule::loaderFunction(void *userData) {
    AnimDataModule *This = static_cast<AnimDataModule*>(userData);
    ASSERT(This != NULL);
    unsigned int index, i, j, k, req;
#ifdef _LOADING_REPORTING
    unsigned int l;
#endif /* _LOADING_REPORTING */
    Frame *frame;

    while (This->isRunning.load()) {
        // sleep to enforce thread changes
        vislib::sys::Thread::Sleep(1);
        if (!This->isRunning.load()) break;

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
                if (!This->isRunning.load()) break;
                if (((This->frameCache[i]->state == Frame::STATE_AVAILABLE)
                        || (This->frameCache[i]->state == Frame::STATE_INUSE)) 
                        && (This->frameCache[i]->frame == index)) {
                    break;
                }
            }
            if (!This->isRunning.load()) break;
            if (i >= This->cacheSize) {
                break;
            }
            index = (index + 1) % This->frameCnt;
        }
        if (!This->isRunning.load()) break;
        if (j >= This->cacheSize) {
            if (j >= This->frameCnt) {
                ASSERT(This->frameCnt == This->cacheSize);
                vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_INFO,
                    "All frames of the dataset loaded into cache. Terminating loading Thread.");
                break;
            }
            continue;
        }

        // 2.
        // Note: We now need to lock, because we must synchronise against 
        // frames changing from 'STATE_AVAILABLE' to 'STATE_INUSE'.
        This->stateLock.Lock();
        frame = NULL;
        j = (req + This->cacheSize) % This->frameCnt;
        for (i = 0; i < This->cacheSize; i++) {
            if (This->frameCache[i]->state == Frame::STATE_INVALID) {
                frame = This->frameCache[i];
#ifdef _LOADING_REPORTING
                l = i;
#endif /* _LOADING_REPORTING */
                break;
            } else if (This->frameCache[i]->state == Frame::STATE_AVAILABLE) {
                k = This->frameCache[i]->frame;
                if (j < req) {
                    if ((k > j) && (k < req)) {
                        frame = This->frameCache[i];
#ifdef _LOADING_REPORTING
                        l = i;
#endif /* _LOADING_REPORTING */
                    }
                } else {
                    if ((k < req) || (k > j)) {
                        frame = This->frameCache[i];
#ifdef _LOADING_REPORTING
                        l = i;
#endif /* _LOADING_REPORTING */
                    }
                }
            }
        }

        // 3.
        if (frame != NULL) {
            frame->state = Frame::STATE_LOADING;
        }
        // if frame is NULL no suitable cache buffer found for loading. This is
        // mostly the case if the cache is too small or if the data source 
        // locks too much frames.
        This->stateLock.Unlock();

        if (frame != NULL) {
#ifdef _LOADING_REPORTING
            printf("Loading frame %i into cache %i\n", index, l);
#endif /* _LOADING_REPORTING */
            This->loadFrame(frame, index);
            // we no not need to lock here, because this transition from 
            // 'STATE_LOADING' to 'STATE_AVAILABLE' is safe for the using 
            // thread.
            frame->state = Frame::STATE_AVAILABLE;
        }
    }

    return 0;
}


/*
 * view::AnimDataModule::unlock
 */
void view::AnimDataModule::unlock(view::AnimDataModule::Frame *frame) {
    ASSERT(&frame->owner == this);
    ASSERT(frame->state == Frame::STATE_INUSE);
    this->stateLock.Lock();
    frame->state = Frame::STATE_AVAILABLE;
    this->stateLock.Unlock();
}
