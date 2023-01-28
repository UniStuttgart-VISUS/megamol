/*
 * VolumetricDataSource.h
 *
 * Copyright (C) 2014 by Visualisierungsinstitut der Universität Stuttgart.
 * Alle rechte vorbehalten.
 */

#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include "datRaw.h"

#include "geometry_calls/VolumetricDataCall.h"

#include "mmcore/param/ParamSlot.h"

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/Module.h"

#include "vislib/PtrArray.h"
#include "vislib/RawStorage.h"
#include "vislib/sys/Event.h"
#include "vislib/sys/Thread.h"

namespace megamol::volume {
/**
 * Reads volumetric data from a dat/raw data source.
 */
class VolumetricDataSource : public core::Module {

public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() {
        return "VolumetricDataSource";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() {
        return "Data source for dat/raw-encoded volumetric data.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable() {
        return true;
    }

    /**
     * Initialises a new instance.
     */
    VolumetricDataSource();

    /**
     * Finalises an instance.
     */
    ~VolumetricDataSource() override;

protected:
    /** Superclass typedef. */
    typedef core::Module Base;

    /**
     * Computes the size of a single frame.
     *
     * The method assumes that the frame is read as it is, ie without
     * converting the voxels to a user-defined scalar format.
     *
     * @return The size of a single frame stored in the raw file in bytes.
     */
    size_t calcFrameSize() const;

    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Gets the requested output data format if any. Otherwise, the
     * format of the input data is returned.
     */
    DatRawDataFormat getOutputDataFormat() const;

    /**
     * Handles a change of 'paramFileName'.
     *
     * @param slot The updated ParamSlot.
     *
     * @return true, unconditionally.
     */
    bool onFileNameChanged(core::param::ParamSlot& slot);

    /**
     * Gets the data from the source.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool onGetData(core::Call& call);

    /**
     * Gets the data extents.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool onGetExtents(core::Call& call);

    /**
     * Handles a change of 'paramLoadAsync'.
     *
     * @param slot The updated ParamSlot.
     *
     * @return true, unconditionally.
     */
    bool onLoadAsyncChanged(core::param::ParamSlot& slot);

    /**
     * Handles a change of 'paramMemorySaturation'.
     *
     * @param slot The updated ParamSlot.
     *
     * @return true if the local memory was changed/reallocated,
     *         false otherwise.
     */
    bool onMemorySaturationChanged(core::param::ParamSlot& slot);

    /**
     * Gets the meta data.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool onGetMetadata(core::Call& call);

    /**
     * Starts the asynchronous loading thread.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool onStartAsync(core::Call& call);

    /**
     * Cancels all pending asynchronous load operations and stops
     * the asynchronous loading thread.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool onStopAsync(core::Call& call);

    /**
     * Tries to retrieve a frame that has been preloaded asynchronously.
     *
     * @param caller The calling call.
     *
     * @return 'true' on success, 'false' on failure.
     */
    bool onTryGetData(core::Call& call);

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /** Resume the asynchronous loading thread if it was suspended. */
    bool resumeAsyncLoad();

    /**
     * Start the asynchronous loading thread.
     */
    bool startAsyncLoad();

    /**
     * Stop the asynchronous loading thread and optionally wait for it to
     * exit.
     */
    void stopAsyncLoad(const bool isWait);

    /**
     * Instruct the asynchronous loading thread to suspend processing of
     * buffers. The thread will, however, not be stopped and can be resumed
     * again.
     *
     * @param isWait If true, the calling thread will wait until the loader
     *               thread confirms that it is suspended.
     *
     * @return true if the loader thread was actually suspended (ie was
     *         running before), false if it was already not running.
     */
    bool suspendAsyncLoad(const bool isWait);

private:
    /** Enapsulates all information about a buffer for a single frame. */
    typedef struct BufferSlot_t {
        vislib::RawStorage Buffer;
        unsigned int FrameID;
        std::atomic_int status;
    } BufferSlot;

    /**
     * Class for unlocking a BufferSlot after it has been rendered.
     */
    class BufferSlotUnlocker : public core::AbstractGetDataCall::Unlocker {

    public:
        /**
         * Initialises a new instance.
         *
         * @param buffer THe buffer that should be unlocked.
         */
        inline BufferSlotUnlocker(BufferSlot* buffer) : Base(), buffers(2) {
            this->AddBuffer(buffer);
        }

        /** Dtor. */
        ~BufferSlotUnlocker() override;

        inline void AddBuffer(BufferSlot* buffer) {
            ASSERT(buffer != nullptr);
            this->buffers.Add(buffer);
        }

        /** Unlocks the data */
        void Unlock() override;

    private:
        /** Base class. */
        typedef core::AbstractGetDataCall::Unlocker Base;

        /** The buffer slot to unlock */
        vislib::Array<BufferSlot*> buffers;
    };

    /** Indicates that a buffer is about to be deleted by the UI thread. */
    static const int BUFFER_STATUS_DELETING;

    /** Indicates that a buffer is used by the loader thread. */
    static const int BUFFER_STATUS_LOADING;

    /** Indicates that the buffer should be loaded by the loader thread. */
    static const int BUFFER_STATUS_PENDING;

    /** Indicates that the buffer is ready for use. */
    static const int BUFFER_STATUS_READY;

    /** Indiciates that the buffer contains no valid data. */
    static const int BUFFER_STATUS_UNUSED;

    /** Indicates that the renderer is using the data in the buffer. */
    static const int BUFFER_STATUS_USED;

    /**
     * Indicates that the loader thread is running, but not performing
     * work.
     */
    static const int LOADER_STATUS_PAUSED;

    /**
     * Indicates that the UI thread requested the loader to pause, but the
     * loader did not yet suspend its work. The loader will try to go to
     * 'LOADER_STATUS_PAUSED' as soon as possible.
     */
    static const int LOADER_STATUS_PAUSING;

    /** Indicates that the loader thread is performing work. */
    static const int LOADER_STATUS_RUNNING;

    /** Indicates that the loader thread stopped. */
    static const int LOADER_STATUS_STOPPED;

    /** Indicates that the UI thread requested the loader thread to exit. */
    static const int LOADER_STATUS_STOPPING;

    /** Executes the given loading call asynchronously. */
    static DWORD loadAsync(void* userData);

    /**
     * Add an unlocker to 'call' that will eventually unlock 'buffer'.
     */
    static void setUnlocker(geocalls::VolumetricDataCall& call, BufferSlot* buffer);

    /**
     * Tries to set 'dst' to 'value' using an interlocked CAS operation
     * while the expected value is one of 'expected'. The method implements
     * a user mode spin lock that tries to set the value until success
     * execpt fpr 'canFail' is set true.
     */
    static bool spinExchange(
        std::atomic_int& dst, const int value, const int* expected, const size_t cntExpected, const bool canFail);

    /**
     * Try to ensure that the available buffers for frames match the
     * requested memory saturation.
     *
     * This method is NOT thread-safe!
     *
     * @param cntFrames The number of frames to prepare. If zero, the value
     *                  is computes from the current mode and allowed memory
     *                  saturation.
     * @param doNotFree If true, prevents buffers being freed by the method.
     *
     * @return The number of buffers allocated.
     */
    size_t assertBuffersUnsafe(size_t cntFrames = 0, bool doNotFree = false);

    /**
     * Search 'buffers' for a slot with the specified frame ID without
     * locking 'lockBuffers'. If no such buffer exists, return a negative
     * index.
     *
     * This method is NOT thread-safe!
     */
    int bufferForFrameIDUnsafe(const unsigned int frameID) const;

    /** The buffers that volume data can be loaded to. */
    vislib::PtrArray<BufferSlot> buffers;

    /** Hash for the data set. */
    unsigned int dataHash;

    /** Wakes the loader thread after there was nothing to do. */
    vislib::sys::Event evtStartLoading;

    /** The content of the dat file. */
    DatRawFileInfo* fileInfo;

    /** The status of the asynchronous loading thread. */
    std::atomic_int loaderStatus;

    /** The thread handling asynchronous loading requests. */
    vislib::sys::Thread loaderThread;

    /**
     * The metadata of the current dat file. These must be updated every
     * time the 'fileInfo' changes, ie if a new file is loaded.
     */
    geocalls::VolumetricDataCall::Metadata metadata;

    /**
     * Determines how long the loader thread sleeps after it loaded a
     * frame. If zero, the loader will not sleep.
     */
    core::param::ParamSlot paramAsyncSleep;

    /**
     * Determines how long the loader thread waits until it wakes itself
     * for checking whether something has to be done. If zero, the loader
     * will not wake itself.
     */
    core::param::ParamSlot paramAsyncWake;

    /**
     * The number of buffers that should be allocated for (pre-) loading
     * frames.
     */
    core::param::ParamSlot paramBuffers;

    /** The path to the dat file. */
    core::param::ParamSlot paramFileName;

    /**
     * The number of bytes the data set should be converted to during
     * loading.
     */
    core::param::ParamSlot paramOutputDataSize;

    /**
     * The data type that the data set should be converted to during
     * loading.
     */
    core::param::ParamSlot paramOutputDataType;

    /** Enables or disables asynchronous loading. */
    core::param::ParamSlot paramLoadAsync;

    /** The slot that requests the data. */
    core::CalleeSlot slotGetData;

    std::vector<double> mins, maxes;

    template<class T>
    void calcMinMax(void const* vol_ptr, std::vector<double>& mins, std::vector<double>& maxes,
        DatRawFileInfo const& fileinfo, geocalls::VolumetricDataCall::Metadata const& metadata) {
        mins.resize(metadata.Components);
        maxes.resize(metadata.Components);
        for (auto c = 0; c < metadata.Components; ++c) {
            double min = std::numeric_limits<double>::max();
            double max = std::numeric_limits<double>::lowest();
            size_t totalLength = 1;
            for (int i = 0; i < fileinfo.dimensions; ++i) {
                totalLength *= fileinfo.resolution[i];
            }
            auto* vol = reinterpret_cast<const T*>(vol_ptr);
            for (size_t x = c; x < totalLength; x += metadata.Components) {
                if (vol[x] < min)
                    min = vol[x];
                if (vol[x] > max)
                    max = vol[x];
            }
            mins[c] = min;
            maxes[c] = max;
        }
    }
};

} // namespace megamol::volume
