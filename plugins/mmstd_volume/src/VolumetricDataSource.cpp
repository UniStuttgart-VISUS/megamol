/*
 * VolumetricDataSource.cpp
 *
 * Copyright (C) 2014 by Visualisierungsinstitut der Universit�t Stuttgart.
 * Alle rechte vorbehalten.
 */

#include "stdafx.h"
#include "VolumetricDataSource.h"

#include "mmcore/misc/VolumetricDataCall.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"

#include "vislib/sys/Log.h"


#define STATIC_ARRAY_COUNT(ary) (sizeof(ary) / sizeof(*(ary)))

#ifndef FALSE
#    define FALSE (0)
#endif /* !FALSE */


// static int CompareBufferSlots(
//        const megamol::datraw::VolumetricDataSource::BufferSlot *& lhs,
//        const megamol::datraw::VolumetricDataSource::BufferSlot *& rhs) {
//}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::VolumetricDataSource
 */
megamol::stdplugin::volume::VolumetricDataSource::VolumetricDataSource(void)
    : Base()
    , dataHash(-234895)
    , fileInfo(nullptr)
    , loaderThread(VolumetricDataSource::loadAsync)
    , paramAsyncSleep("AsyncSleep", "The time in milliseconds that the loader sleeps between two frames.")
    , paramAsyncWake("AsyncWake", "The time in milliseconds after that the loader wakes itself.")
    , paramBuffers("Buffers", "The number of buffers for loading frames asynchronously.")
    , paramFileName("FileName", "The path to the dat file to be loaded.")
    , paramOutputDataSize("OutputDataSize", "Forces the scalar type to the specified size.")
    , paramOutputDataType("OutputDataType", "Enforces the type of a scalar during loading.")
    , paramLoadAsync("LoadAsync", "Start asynchronous loading of frames.")
    , slotGetData("GetData", "Slot for requesting data from the source.") {
    using core::misc::VolumetricDataCall;
    core::param::EnumParam* enumParam = nullptr;

    this->loaderStatus.store(LOADER_STATUS_STOPPED);

    this->paramAsyncSleep.SetParameter(new core::param::IntParam(0, 0));
    this->MakeSlotAvailable(&this->paramAsyncSleep);

    this->paramAsyncWake.SetParameter(new core::param::IntParam(0, 0));
    this->MakeSlotAvailable(&this->paramAsyncWake);

    this->paramBuffers.SetParameter(new core::param::IntParam(2, 2));
    this->MakeSlotAvailable(&this->paramBuffers);

    this->paramFileName.SetParameter(new core::param::FilePathParam(_T("")));
    this->paramFileName.SetUpdateCallback(&VolumetricDataSource::onFileNameChanged);
    this->MakeSlotAvailable(&this->paramFileName);

    enumParam = new core::param::EnumParam(-1);
    enumParam->SetTypePair(-1, _T("Auto"));
    enumParam->SetTypePair(1, _T("1 Byte/Scalar"));
    enumParam->SetTypePair(2, _T("2 Bytes/Scalar"));
    enumParam->SetTypePair(4, _T("4 Bytes/Scalar"));
    enumParam->SetTypePair(8, _T("8 Bytes/Scalar"));
    this->paramOutputDataSize.SetParameter(enumParam);
    this->MakeSlotAvailable(&this->paramOutputDataSize);

    enumParam = new core::param::EnumParam(-1);
    enumParam->SetTypePair(-1, _T("Auto"));
    enumParam->SetTypePair(VolumetricDataCall::ScalarType::SIGNED_INTEGER, _T("int"));
    enumParam->SetTypePair(VolumetricDataCall::ScalarType::UNSIGNED_INTEGER, _T("uint"));
    enumParam->SetTypePair(VolumetricDataCall::ScalarType::FLOATING_POINT, _T("float"));
    this->paramOutputDataType.SetParameter(enumParam);
    this->MakeSlotAvailable(&this->paramOutputDataType);

    this->paramLoadAsync.SetParameter(new core::param::BoolParam(FALSE));
    // this->paramLoadAsync.SetUpdateCallback(
    //    &VolumetricDataSource::onLoadAsyncChanged);
    this->MakeSlotAvailable(&this->paramLoadAsync);

    this->slotGetData.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_DATA), &VolumetricDataSource::onGetData);
    this->slotGetData.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_EXTENTS), &VolumetricDataSource::onGetExtents);
    this->slotGetData.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_GET_METADATA), &VolumetricDataSource::onGetMetadata);
    this->slotGetData.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_START_ASYNC), &VolumetricDataSource::onStartAsync);
    this->slotGetData.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_STOP_ASYNC), &VolumetricDataSource::onStopAsync);
    this->slotGetData.SetCallback(VolumetricDataCall::ClassName(),
        VolumetricDataCall::FunctionName(VolumetricDataCall::IDX_TRY_GET_DATA), &VolumetricDataSource::onTryGetData);
    this->MakeSlotAvailable(&this->slotGetData);
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::~VolumetricDataSource
 */
megamol::stdplugin::volume::VolumetricDataSource::~VolumetricDataSource(void) {
    this->Release();
    ASSERT(this->fileInfo == nullptr);
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::calcFrameSize
 */
size_t megamol::stdplugin::volume::VolumetricDataSource::calcFrameSize(void) const {
    ASSERT(this->fileInfo != nullptr);
    return ::datRaw_getBufferSize(this->fileInfo, this->getOutputDataFormat());
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::create
 */
bool megamol::stdplugin::volume::VolumetricDataSource::create(void) {
    if (!this->paramFileName.Param<core::param::FilePathParam>()->Value().IsEmpty()) {
        this->onFileNameChanged(this->paramFileName);
    }
    return true;
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::getOutputDataFormat
 */
DatRawDataFormat megamol::stdplugin::volume::VolumetricDataSource::getOutputDataFormat(void) const {
    using core::misc::VolumetricDataCall;
    using vislib::sys::Log;

    auto sizeParam = this->paramOutputDataSize.Param<core::param::EnumParam>();
    auto typeParam = this->paramOutputDataType.Param<core::param::EnumParam>();

    VolumetricDataCall::ScalarType scalarType = (typeParam->Value() >= 0)
                                                    ? static_cast<VolumetricDataCall::ScalarType>(typeParam->Value())
                                                    : this->metadata.ScalarType;
    size_t scalarLength = (sizeParam->Value() >= 0) ? sizeParam->Value() : this->metadata.ScalarLength;

    switch (scalarType) {
    case VolumetricDataCall::ScalarType::SIGNED_INTEGER:
        switch (scalarLength) {
        case 1:
            return DatRawDataFormat::DR_FORMAT_CHAR;
        case 2:
            return DatRawDataFormat::DR_FORMAT_SHORT;
        case 4:
            return DatRawDataFormat::DR_FORMAT_INT;
        case 8:
            return DatRawDataFormat::DR_FORMAT_LONG;
        default:
            Log::DefaultLog.WriteError(_T("Unsupported scalar ")
                                       _T("length %u in combination with type %d."),
                scalarLength, scalarType);
            return DatRawDataFormat::DR_FORMAT_NONE;
        }

    case VolumetricDataCall::ScalarType::UNSIGNED_INTEGER:
        switch (scalarLength) {
        case 1:
            return DatRawDataFormat::DR_FORMAT_UCHAR;
        case 2:
            return DatRawDataFormat::DR_FORMAT_USHORT;
        case 4:
            return DatRawDataFormat::DR_FORMAT_UINT;
        case 8:
            return DatRawDataFormat::DR_FORMAT_ULONG;
        default:
            Log::DefaultLog.WriteError(_T("Unsupported scalar ")
                                       _T("length %u in combination with type %d."),
                scalarLength, scalarType);
            return DatRawDataFormat::DR_FORMAT_NONE;
        }

    case VolumetricDataCall::ScalarType::FLOATING_POINT:
        switch (scalarLength) {
        case 2:
            return DatRawDataFormat::DR_FORMAT_HALF;
        case 4:
            return DatRawDataFormat::DR_FORMAT_FLOAT;
        case 8:
            return DatRawDataFormat::DR_FORMAT_DOUBLE;
        default:
            Log::DefaultLog.WriteError(_T("Unsupported scalar ")
                                       _T("length %u in combination with type %d."),
                scalarLength, scalarType);
            return DatRawDataFormat::DR_FORMAT_NONE;
        }

    case VolumetricDataCall::ScalarType::BITS:
        return DatRawDataFormat::DR_FORMAT_RAW;

    case VolumetricDataCall::ScalarType::UNKNOWN:
    default:
        return DatRawDataFormat::DR_FORMAT_NONE;
    }
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::onFileNameChanged
 */
bool megamol::stdplugin::volume::VolumetricDataSource::onFileNameChanged(core::param::ParamSlot& slot) {
    using core::misc::VolumetricDataCall;
    using vislib::sys::Log;

    /* Allocate header or prepare it for re-use. */
    if (this->fileInfo == nullptr) {
        this->fileInfo = new DatRawFileInfo();
    } else {
        ::datRaw_freeInfo(this->fileInfo);
    }

    bool isAsync = this->paramLoadAsync.Param<core::param::BoolParam>()->Value();
    if (isAsync) {
        // Cancel loading the previous data set before settings a new one.
        Log::DefaultLog.WriteInfo(_T("Halting asynchronous loading thread ")
                                  _T("in preparation for changing the data set."));
        this->stopAsyncLoad(true);
        Log::DefaultLog.WriteInfo(_T("Halting asynchronous loading thread ")
                                  _T("in preparation for changing the data set."));
    }

    /* Read the header. */
    vislib::StringA fileName(this->paramFileName.Param<core::param::FilePathParam>()->Value());
    if (::datRaw_readHeader(fileName.PeekBuffer(), this->fileInfo, nullptr) != FALSE) {
        Log::DefaultLog.WriteInfo(_T("Successfully loaded dat file %hs."), fileName.PeekBuffer());

        static_assert(STATIC_ARRAY_COUNT(this->metadata.SliceDists) == STATIC_ARRAY_COUNT(this->metadata.Resolution),
            "Dimension of "
            "slice distance and resolution arrays match.");
        static_assert(STATIC_ARRAY_COUNT(this->metadata.SliceDists) == STATIC_ARRAY_COUNT(this->metadata.IsUniform),
            "Dimension of "
            "slice distance and uniformity flag arrays match.");
        int dimensions = vislib::math::Min(
            static_cast<int>(STATIC_ARRAY_COUNT(this->metadata.SliceDists)), this->fileInfo->dimensions);

        // Erase *all* old slice distances from heap.
        ::ZeroMemory(this->metadata.SliceDists, sizeof(this->metadata.SliceDists));

        switch (this->fileInfo->gridType) {
        case DR_GRID_CARTESIAN:
            this->metadata.GridType = VolumetricDataCall::GridType::CARTESIAN;
            Log::DefaultLog.WriteInfo(_T("The grid is cartesian."));
            for (int i = 0; i < dimensions; ++i) {
                this->metadata.SliceDists[i] = this->fileInfo->sliceDist + i;
                this->metadata.IsUniform[i] = true;
                Log::DefaultLog.WriteInfo(_T("The grid is uniform in ")
                                          _T("dimension %d and has a slice distance of %f."),
                    i, this->metadata.SliceDists[i][0]);
            }
            break;

        case DR_GRID_RECTILINEAR:
            this->metadata.GridType = VolumetricDataCall::GridType::RECTILINEAR;
            Log::DefaultLog.WriteInfo(_T("The grid is rectilinear."));
            for (int i = 0, j = 0; i < dimensions; ++i) {
                this->metadata.SliceDists[i] = this->fileInfo->sliceDist + j;

                /* Determine uniformity of the dimension. */
                this->metadata.IsUniform[i] = true;
                float reference = this->fileInfo->sliceDist[j];
                for (; j < this->fileInfo->resolution[i]; ++j) {
                    if (this->fileInfo->sliceDist[j] != reference) {
                        this->metadata.IsUniform[i] = false;
                        // Do *not* break! We need 'j' to determine the
                        // start of the next dimension.
                    }
                }

                Log::DefaultLog.WriteInfo(_T("The grid is%s uniform ")
                                          _T("in dimension %d."),
                    this->metadata.IsUniform[i] ? _T("") : _T(" not"), i);
            }
            break;

        case DR_GRID_TETRAHEDRAL:
            this->metadata.GridType = VolumetricDataCall::GridType::TETRAHEDRAL;
            Log::DefaultLog.WriteWarn(1, _T("The grid is tetrahedral. ")
                                         _T("Tetrahedral grids are currently not supported."));
            break;

        default:
            Log::DefaultLog.WriteError(1,
                _T("The data set uses the ")
                _T("unexpected grid type %d."),
                this->fileInfo->gridType);
        case DR_GRID_NONE:
            this->metadata.GridType = VolumetricDataCall::GridType::NONE;
            break;
        }

        ::ZeroMemory(&this->metadata.Resolution, sizeof(this->metadata.Resolution));
        for (int i = 0; i < dimensions; ++i) {
            ASSERT(this->fileInfo->resolution[i] >= 0);
            this->metadata.Resolution[i] = this->fileInfo->resolution[i];
            Log::DefaultLog.WriteInfo(_T("Resolution in dimension %d ")
                                      _T("is %u."),
                i, this->metadata.Resolution[i]);
        }
        ::ZeroMemory(&this->metadata.Origin, sizeof(this->metadata.Origin));
        for (int i = 0; i < dimensions; ++i) {
            this->metadata.Origin[i] = this->fileInfo->origin[i];
            Log::DefaultLog.WriteInfo(_T("Origin in dimension %d ")
                                      _T("is %f."),
                i, this->metadata.Origin[i]);
        }

        this->metadata.Components = this->fileInfo->numComponents;
        Log::DefaultLog.WriteInfo(_T("Each voxel comprises %u scalars."), this->metadata.Components);

        switch (this->fileInfo->dataFormat) {
        case DR_FORMAT_CHAR:
        case DR_FORMAT_SHORT:
        case DR_FORMAT_INT:
        case DR_FORMAT_LONG:
            this->metadata.ScalarType = VolumetricDataCall::ScalarType::SIGNED_INTEGER;
            Log::DefaultLog.WriteInfo(_T("Scalars are signed ")
                                      _T("integers."));
            break;

        case DR_FORMAT_UCHAR:
        case DR_FORMAT_USHORT:
        case DR_FORMAT_UINT:
        case DR_FORMAT_ULONG:
            this->metadata.ScalarType = VolumetricDataCall::ScalarType::UNSIGNED_INTEGER;
            Log::DefaultLog.WriteInfo(_T("Scalars are unsigned ")
                                      _T("integers."));
            break;

        case DR_FORMAT_HALF:
        case DR_FORMAT_FLOAT:
        case DR_FORMAT_DOUBLE:
            this->metadata.ScalarType = VolumetricDataCall::ScalarType::FLOATING_POINT;
            Log::DefaultLog.WriteInfo(_T("Scalars are floating ")
                                      _T("point numbers."));
            break;

        case DR_FORMAT_RAW:
            this->metadata.ScalarType = VolumetricDataCall::ScalarType::BITS;
            Log::DefaultLog.WriteInfo(_T("Scalars are raw bits."));
            break;

        case DR_FORMAT_NONE:
        default:
            this->metadata.ScalarType = VolumetricDataCall::ScalarType::UNKNOWN;
            Log::DefaultLog.WriteWarn(_T("The type of the scalar ")
                                      _T("values is unkown."));
            break;
        }

        this->metadata.ScalarLength = ::datRaw_getFormatSize(this->fileInfo->dataFormat);
        Log::DefaultLog.WriteInfo(_T("Scalar values comprise %u bytes."), this->metadata.ScalarLength);

        this->metadata.NumberOfFrames = this->fileInfo->timeSteps;
        Log::DefaultLog.WriteInfo(_T("The data set comprises %u frames."), this->metadata.NumberOfFrames);

        /* Compute extents. */
        ::ZeroMemory(this->metadata.Extents, sizeof(this->metadata.Extents));
        for (int d = 0; (d < STATIC_ARRAY_COUNT(this->metadata.Extents)) && (d < this->fileInfo->dimensions); ++d) {
            switch (this->fileInfo->gridType) {
            case DatRawGridType::DR_GRID_CARTESIAN:
                this->metadata.Extents[d] =
                    this->fileInfo->sliceDist[d] * static_cast<float>(this->fileInfo->resolution[d] - 1);
                break;

            case DatRawGridType::DR_GRID_RECTILINEAR:
                if (this->metadata.IsUniform[d]) {
                    this->metadata.Extents[d] =
                        this->fileInfo->sliceDist[d] * static_cast<float>(this->fileInfo->resolution[d] - 1);
                } else {
                    int r = this->fileInfo->resolution[d];
                    this->metadata.Extents[d] = this->fileInfo->sliceDist[r - 1] * static_cast<float>(r - 1);
                }

            case DatRawGridType::DR_GRID_NONE:
            case DatRawGridType::DR_GRID_TETRAHEDRAL:
            default:
                throw vislib::IllegalStateException(_T("The available ")
                                                    _T("grid type is not supported."),
                    __FILE__, __LINE__);
            }
        }

    } else {
        Log::DefaultLog.WriteError(1,
            _T("Failed to read and parse dat file ")
            _T("%hs."),
            fileName.PeekBuffer());
        SAFE_DELETE(this->fileInfo);
    }

    /* Signal data having changed (this is always the case). */
    ++this->dataHash;

    /* Restart loader if asynchronous loading was selected. */
    if (isAsync) {
        Log::DefaultLog.WriteInfo(_T("Resuming asynchronous loading thread ")
                                  _T("after changing the data set."));
        this->startAsyncLoad();
    }

    return true;
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::onGetData
 */
bool megamol::stdplugin::volume::VolumetricDataSource::onGetData(core::Call& call) {
    using core::misc::VolumetricDataCall;
    using core::param::BoolParam;
    using core::param::IntParam;
    using vislib::sys::Log;

    int expected = 0;
    bool retval = false;

    VolumetricDataCall& c = dynamic_cast<VolumetricDataCall&>(call);

    if (c.DataHash() != this->dataHash) {
        try {
            /* Evaluate parameter changes. */
            bool isAsync = this->paramLoadAsync.Param<BoolParam>()->Value();
            size_t cntBuffers = this->paramBuffers.Param<IntParam>()->Value();

            if (cntBuffers != this->buffers.Count()) {
                bool isResume = this->suspendAsyncLoad(true);
                this->assertBuffersUnsafe(cntBuffers);
                if (isResume) {
                    this->resumeAsyncLoad();
                }
            }

            if (isAsync && this->loaderStatus != LOADER_STATUS_RUNNING) {
                this->startAsyncLoad();
            }

            /* Do the actual work. */
            c.SetDataHash(this->dataHash);

            /* Sanity check. */
            if (this->fileInfo == nullptr) {
                throw vislib::IllegalStateException(_T("A valid dat file must be ")
                                                    _T("loaded before the data can be read."),
                    __FILE__, __LINE__);
            }

            /*
             * Search whether we have the requested frame buffered. The synchronous
             * and the asynchronous case will handle this differently.
             */
            auto bufferIdx = this->bufferForFrameIDUnsafe(c.FrameID());

            if (isAsync) {
                /* Loader is running asynchronously, try to get data from buffer. */

                if (c.GetData() != nullptr) {
                    throw vislib::IllegalStateException(_T("A user-defined ")
                                                        _T("destination buffer cannot be used if asynchronous ")
                                                        _T("loading has been enabled."),
                        __FILE__, __LINE__);
                }

                // The loader cannot change the number of buffers. Therefore, we do
                // not lock the buffers here.
                // Note: Only the UI thread shall reallocate buffers, so we to not
                // lock here and allow the loader thread to continue.
                ASSERT(this->buffers.Count() > 0);

                /* Get the buffer we use for loading the requested frame. */
                auto buffer = (bufferIdx >= 0) ? this->buffers[bufferIdx] : nullptr;

                /*
                 * Cancel all non-running requests and search buffers we can
                 * potentially reuse to load the requested frame and the ones after
                 * that (both, for the current frame as well as for the follow-up
                 * frames).
                 */
                vislib::Array<BufferSlot*> oldBuffers(this->buffers.Count());
                vislib::Array<BufferSlot*> unusedBuffers(this->buffers.Count());
                for (size_t i = 0; i < this->buffers.Count(); ++i) {
                    auto b = this->buffers[i % this->buffers.Count()];
                    if (b != buffer) {
                        if (b->status.load() == BUFFER_STATUS_UNUSED) {
                            unusedBuffers.Add(b);
                        }
                        if (b->status.load() == BUFFER_STATUS_READY) {
                            oldBuffers.Add(b);
                        }
                        expected = BUFFER_STATUS_PENDING;
                        if (b->status.compare_exchange_strong(expected, BUFFER_STATUS_UNUSED)) {
                            unusedBuffers.Add(b);
                        }
                    } /* end if (b != buffer) */
                }     /* end for (size_t i = 0; i < this->buffers.Count(); ++i) */

                if (buffer == nullptr) {
                    /*
                     * If we do not have the frame, request it using a free or
                     * reusable buffer.
                     */
                    if (!unusedBuffers.IsEmpty()) {
                        buffer = unusedBuffers[0];
                        unusedBuffers.RemoveFirst();
                    } else if (!oldBuffers.IsEmpty()) {
                        buffer = oldBuffers[0];
                        oldBuffers.RemoveFirst();
                    }
                }

                if (buffer != nullptr) {
                    /*
                     * Request loading only if the buffer is not yet ready or
                     * already being queued.
                     */
                    if (buffer->FrameID != c.FrameID()) {
                        buffer->FrameID = c.FrameID();
                        buffer->status.store(BUFFER_STATUS_UNUSED);
                    }
                    expected = BUFFER_STATUS_UNUSED;
                    buffer->status.compare_exchange_strong(expected, BUFFER_STATUS_PENDING);

                    /* Ensure that loading the requested frame starts now. */
                    this->evtStartLoading.Set();

                    /* Request follow-up frames. */
                    for (size_t i = 1; (i < this->metadata.NumberOfFrames) && (i < this->buffers.Count() - 1); ++i) {
                        unsigned int frameID = (c.FrameID() + i) % (unsigned int)this->metadata.NumberOfFrames;
                        if (this->bufferForFrameIDUnsafe(frameID) < 0) {
                            BufferSlot* bs = nullptr;
                            if (!unusedBuffers.IsEmpty()) {
                                bs = unusedBuffers.First();
                                unusedBuffers.RemoveFirst();
                            } else if (!oldBuffers.IsEmpty()) {
                                bs = oldBuffers.First();
                                oldBuffers.RemoveFirst();
                            }
                            if (bs != nullptr) {
#if defined(DEBUG) || defined(_DEBUG)
                                Log::DefaultLog.WriteInfo(_T("Queueing frame %u ")
                                                          _T("to be loaded for future use."),
                                    frameID);
#endif /* defined(DEBUG) || defined(_DEBUG) */
                                bs->FrameID = frameID;
                                bs->status.store(BUFFER_STATUS_PENDING);
                                ASSERT(this->bufferForFrameIDUnsafe(frameID) != -1);
                            } /* end if (bs != nullptr) */
                        }     /* end if (this->bufferForFrameIDUnsafe(frameID) == -1) */
                    }

                    /* Ensure that the loader thread is awake. */
                    this->evtStartLoading.Set();

                    /* Wait for the requested data if not yet loaded. */
#if 0
                expected = BUFFER_STATUS_READY;
                while (!buffer->status.compare_exchange_strong(expected,
                        BUFFER_STATUS_USED)) {
                    expected = BUFFER_STATUS_READY;
                    //vislib::sys::Thread::Reschedule();
                    this->evtStartLoading.Set();
                }
#else
                    int expecteds[] = {BUFFER_STATUS_READY, BUFFER_STATUS_USED};
                    this->spinExchange(
                        buffer->status, BUFFER_STATUS_USED, expecteds, STATIC_ARRAY_COUNT(expecteds), false);
#endif
                    ASSERT(buffer->status == BUFFER_STATUS_USED);

                    /* Move the stuff to the call and set the unlocker. */
                    c.SetData(buffer->Buffer.At(0), 1);
                    VolumetricDataSource::setUnlocker(c, buffer);

                    retval = true;

                } else {
                    /* We do not have buffers left, which is a fatal error. */
                    Log::DefaultLog.WriteError(1,
                        _T("There are insufficient ")
                        _T("buffers for loading the frame with ID %u."),
                        c.FrameID());
                    retval = false;
                }

            } else {
                /* Loader is running synchronously, complete the request. */
                size_t cnt = vislib::math::Max<size_t>(1, c.GetAvailableFrames());
                size_t frameSize = this->calcFrameSize();
                size_t loadStart = 0;
                vislib::Array<void*> dst(cnt);

                if (c.GetData() == nullptr) {
                    /*
                     * If the call does not request a direct copy to a target buffer
                     * that the caller provides, allocate local memory and return
                     * the data from there.
                     *
                     * In this case, we might already have preloaded the requested
                     * frame into an existing buffer. If so, reuse the existing
                     * buffer and preload the following frames if requested.
                     *
                     * Note that preloading to a data source-local buffer does not
                     * enable the caller to immediately get the data.
                     */
                    dst.SetCount(vislib::math::Min(this->assertBuffersUnsafe(cnt, true), cnt));

                    if ((bufferIdx >= 0) && (this->buffers[bufferIdx]->status == BUFFER_STATUS_READY)) {
                        loadStart = 1;
                        dst[0] = this->buffers[bufferIdx]->Buffer.At(0);
                        retval = true;
                    } else {
                        loadStart = 0;
                        bufferIdx = 0;
                    }

                    for (size_t i = loadStart; i < dst.Count(); ++i) {
                        size_t idx = i % dst.Count();
                        this->buffers[idx]->FrameID = c.FrameID() + (unsigned int)i;
                        this->buffers[idx]->Buffer.AssertSize(frameSize);
                        this->buffers[idx]->status.store(BUFFER_STATUS_READY);
                        dst[i] = this->buffers[idx]->Buffer.At(0);
                    }

                    ASSERT(this->buffers[bufferIdx]->Buffer.At(0) == dst[0]);
                    this->buffers[bufferIdx]->status.store(BUFFER_STATUS_USED);
                    c.SetData(dst[0], 1);
                    VolumetricDataSource::setUnlocker(c, this->buffers[bufferIdx]);

                } else {
                    /*
                     * The caller requests us to copy directly to the memory that is
                     * provided in the call. We assume this memory to be contiguous
                     * and large enough to hold all requested frames.
                     */
                    dst.SetCount(cnt);
                    for (size_t i = 0; i < dst.Count(); ++i) {
                        dst[i] = static_cast<BYTE*>(c.GetData()) + i * frameSize;
                    }
                    loadStart = 0;

                    // No local buffer is required here. The data pointer is already
                    // set by the caller.
                } /* end if (c.GetData() == nullptr) */

                retval = true;
                for (size_t i = loadStart; i < cnt; ++i) {
                    auto buffer = dst[i];
                    auto frameID = c.FrameID() + i;
                    auto format = this->getOutputDataFormat();
#if (defined(DEBUG) || defined(_DEBUG))
                    Log::DefaultLog.WriteInfo(_T("Loading frame %u in format ")
                                              _T("%hs to 0x%p"),
                        frameID, ::datRaw_getDataFormatName(format), dst.PeekElements());
#endif /* (defined(DEBUG) || defined(_DEBUG)) */
                    retval = (::datRaw_loadStep(this->fileInfo, static_cast<int>(frameID), &buffer, format) != 0);
                    ::datRaw_close(this->fileInfo);
                    if (!retval) {
                        Log::DefaultLog.WriteError(_T("Loading frame %u failed."), frameID);
                        break;
                    }
                } /* end for (size_t i = 0; i < dst.Count(); ++i) */
            }     /* end if (this->paramLoadAsync.Param<BoolParam>()->Value()) */

        } catch (vislib::Exception& e) {
            Log::DefaultLog.WriteError(1, e.GetMsg());
            retval = false;
        } catch (...) {
            Log::DefaultLog.WriteError(1, _T("Unexpected exception in callback ")
                                          _T("onGetData (please check the call)."));
            retval = false;
        }

        if (retval) {
            const VolumetricDataCall& vdc = dynamic_cast<VolumetricDataCall&>(call);
            switch (this->getOutputDataFormat()) {
            case DR_FORMAT_UCHAR:
                this->calcMinMax<uint8_t>(vdc.GetData(), this->mins, this->maxes, *fileInfo, metadata);
                break;
            case DR_FORMAT_FLOAT:
                this->calcMinMax<float>(vdc.GetData(), this->mins, this->maxes, *fileInfo, metadata);
                break;
            case DR_FORMAT_DOUBLE:
                this->calcMinMax<double>(vdc.GetData(), this->mins, this->maxes, *fileInfo, metadata);
                break;
            case DR_FORMAT_USHORT:
                this->calcMinMax<uint16_t>(vdc.GetData(), this->mins, this->maxes, *fileInfo, metadata);
                break;
            case DR_FORMAT_SHORT:
                this->calcMinMax<int16_t>(vdc.GetData(), this->mins, this->maxes, *fileInfo, metadata);
                break;
            case DR_FORMAT_RAW:
                vislib::sys::Log::DefaultLog.WriteWarn("Cannot determine min/max of BITS volume. Setting to [0,1].");
                this->mins.resize(this->metadata.Components, 0.0);
                this->maxes.resize(this->metadata.Components, 1.0);
                break;
            default:
                vislib::sys::Log::DefaultLog.WriteWarn("Cannot determine min/max of unknown volume. Setting to [0,1].");
                this->mins.resize(this->metadata.Components, 0.0);
                this->maxes.resize(this->metadata.Components, 1.0);
                break;
            }
            this->metadata.MinValues = this->mins.data();
            this->metadata.MaxValues = this->maxes.data();
        }
    } else {
        retval = true;
    }

    return retval;
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::onGetExtents
 */
bool megamol::stdplugin::volume::VolumetricDataSource::onGetExtents(core::Call& call) {
    using core::misc::VolumetricDataCall;
    using vislib::sys::Log;

    try {
        VolumetricDataCall& c = dynamic_cast<VolumetricDataCall&>(call);

        /* Sanity check. */
        if (this->fileInfo == nullptr) {
            throw vislib::IllegalStateException(_T("A valid dat file must be ")
                                                _T("loaded before the extents can be retrieved."),
                __FILE__, __LINE__);
        }
        /*
         * If the file info is available, metadata are also available. We can
         * use them to avoid doing the same thing twice.
         */

        /* Complete request. */
        c.SetExtent((unsigned int)this->metadata.NumberOfFrames, this->metadata.Origin[0], this->metadata.Origin[1],
            this->metadata.Origin[2], this->metadata.Extents[0] + this->metadata.Origin[0],
            this->metadata.Extents[1] + this->metadata.Origin[1], this->metadata.Extents[2] + this->metadata.Origin[2]);

        // Log::DefaultLog.WriteInfo(100, _T("Volume bounding box is ")
        //    _T("(%f, %f, %f) - (%f, %f, %f)."), 0.0f, 0.0f, 0.0f,
        //    this->metadata.Extents[0], this->metadata.Extents[1],
        //    this->metadata.Extents[2]);
        return true;

    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteError(1, e.GetMsg());
        return false;
    } catch (...) {
        Log::DefaultLog.WriteError(1, _T("Unexpected exception in callback ")
                                      _T("onGetExtents (please check the call)."));
        return false;
    }
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::onGetMetadata
 */
bool megamol::stdplugin::volume::VolumetricDataSource::onGetMetadata(core::Call& call) {
    using core::misc::VolumetricDataCall;
    using vislib::sys::Log;

    try {
        VolumetricDataCall& c = dynamic_cast<VolumetricDataCall&>(call);

        /* Sanity check. */
        if (this->fileInfo == nullptr) {
            throw vislib::IllegalStateException(_T("A valid dat file must be ")
                                                _T("loaded before the meta data can be retrieved."),
                __FILE__, __LINE__);
        }
        /*
         * If the file info was set, we can assume that the metadata are
         * up-to-date, because we update them immediately after parsing the
         * dat file in the handler for file name changes.
         */

        c.SetMetadata(&this->metadata);
        return true;
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteError(1, e.GetMsg());
        return false;
    } catch (...) {
        Log::DefaultLog.WriteError(1, _T("Unexpected exception in callback ")
                                      _T("onGetMetadata (please check the call)."));
        return false;
    }
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::onTryGetData
 */
bool megamol::stdplugin::volume::VolumetricDataSource::onTryGetData(core::Call& call) {
    using core::misc::VolumetricDataCall;
    using vislib::sys::Log;

    int expected = 0;
    bool retval = true;

    try {
        VolumetricDataCall& c = dynamic_cast<VolumetricDataCall&>(call);
        c.SetDataHash(this->dataHash);

        /* Sanity check. */
        if (this->fileInfo == nullptr) {
            throw vislib::IllegalStateException(_T("A valid dat file must be ")
                                                _T("loaded before the data can be read."),
                __FILE__, __LINE__);
        }
        if (c.GetAvailableFrames() != 1) {
            Log::DefaultLog.WriteWarn(_T("When invoking TryGetData, exactly ")
                                      _T("one frame can be requested from the data source."));
        }

        /* Try to get the data. */
        auto bufferIdx = this->bufferForFrameIDUnsafe(c.FrameID());
        void* data = nullptr;
        size_t cnt = 0;
        if (bufferIdx >= 0) {
            auto buffer = this->buffers[bufferIdx];
            int expected = BUFFER_STATUS_READY;
            if (buffer->status.compare_exchange_strong(expected, BUFFER_STATUS_USED)) {
                data = buffer->Buffer.At(0);
                cnt = 1;
            }
        }

        /* Update the call. */
        if (data != nullptr) {
            if (c.GetData() == nullptr) {
                /*
                 * No user-provided buffer, pass the pointer and give
                 * ownership to the call by registering the unlocker.
                 */
                c.SetData(data, cnt);
                VolumetricDataSource::setUnlocker(c, this->buffers[bufferIdx]);

            } else if (data != nullptr) {
                /*
                 * Copy data to user-defined buffer and release local
                 * ownership (unlock immediately).
                 */
                ASSERT(bufferIdx >= 0);
                ::memcpy(c.GetData(), data, this->calcFrameSize());
                c.SetData(c.GetData(), 1);
                this->buffers[bufferIdx]->status.store(BUFFER_STATUS_READY);
            }

        } else {
            /* Have no data, which is OK. No need to unlock anything here. */
            c.SetData(nullptr, 0);
        }

    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteError(1, e.GetMsg());
        retval = false;
    } catch (...) {
        Log::DefaultLog.WriteError(1, _T("Unexpected exception in callback ")
                                      _T("onTryGetData (please check the call)."));
        retval = false;
    }

    return retval;
}

/*
 * megamol::stdplugin::volume::VolumetricDataSource::BUFFER_STATUS_DELETING
 */
const int megamol::stdplugin::volume::VolumetricDataSource::BUFFER_STATUS_DELETING = 5;


/*
 * megamol::stdplugin::volume::VolumetricDataSource::BUFFER_STATUS_LOADING
 */
const int megamol::stdplugin::volume::VolumetricDataSource::BUFFER_STATUS_LOADING = 2;


/*
 * megamol::stdplugin::volume::VolumetricDataSource::BUFFER_STATUS_PENDING
 */
const int megamol::stdplugin::volume::VolumetricDataSource::BUFFER_STATUS_PENDING = 1;


/*
 * megamol::stdplugin::volume::VolumetricDataSource::BUFFER_STATUS_READY
 */
const int megamol::stdplugin::volume::VolumetricDataSource::BUFFER_STATUS_READY = 3;


/*
 * megamol::stdplugin::volume::VolumetricDataSource::BUFFER_STATUS_UNUSED
 */
const int megamol::stdplugin::volume::VolumetricDataSource::BUFFER_STATUS_UNUSED = 0;


/*
 * megamol::stdplugin::volume::VolumetricDataSource::BUFFER_STATUS_USED
 */
const int megamol::stdplugin::volume::VolumetricDataSource::BUFFER_STATUS_USED = 4;


/*
 * megamol::stdplugin::volume::VolumetricDataSource::LOADER_STATUS_PAUSED
 */
const int megamol::stdplugin::volume::VolumetricDataSource::LOADER_STATUS_PAUSED = 2;


/*
 * megamol::stdplugin::volume::VolumetricDataSource::LOADER_STATUS_PAUSING
 */
const int megamol::stdplugin::volume::VolumetricDataSource::LOADER_STATUS_PAUSING = 1;


/*
 * megamol::stdplugin::volume::VolumetricDataSource::LOADER_STATUS_RUNNING
 */
const int megamol::stdplugin::volume::VolumetricDataSource::LOADER_STATUS_RUNNING = 0;


/*
 * megamol::stdplugin::volume::VolumetricDataSource::LOADER_STATUS_STOPPED
 */
const int megamol::stdplugin::volume::VolumetricDataSource::LOADER_STATUS_STOPPED = 4;


/*
 * megamol::stdplugin::volume::VolumetricDataSource::LOADER_STATUS_STOPPING
 */
const int megamol::stdplugin::volume::VolumetricDataSource::LOADER_STATUS_STOPPING = 3;


/*
 * megamol::stdplugin::volume::VolumetricDataSource::onLoadAsyncChanged
 */
bool megamol::stdplugin::volume::VolumetricDataSource::onLoadAsyncChanged(core::param::ParamSlot& slot) {
    // VLAUTOSTACKTRACE;
    // using vislib::sys::Log;

    // bool isEnabled = this->paramLoadAsync.Param<core::param::BoolParam>(
    //    )->Value();

    // try {
    //    Log::DefaultLog.WriteInfo(_T("Asynchronous loading changed to %d ")
    //        _T("on user request."), isEnabled);
    //    this->continueLoading.store(isEnabled);
    //    if (isEnabled && !this->loaderThread.IsRunning()) {
    //        // Start loader if not running.
    //        this->assertBuffers();
    //        this->loaderThread.Start(this);
    //    }
    //    this->evtStartLoading.Set();
    //} catch (vislib::Exception e) {
    //    Log::DefaultLog.WriteError(1, e.GetMsg());
    //} catch (...) {
    //    Log::DefaultLog.WriteError(1, _T("Unexpected exception in callback ")
    //        _T("onLoadAsyncChanged."));
    //}

    return true;
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::onMemorySaturationChanged
 */
bool megamol::stdplugin::volume::VolumetricDataSource::onMemorySaturationChanged(core::param::ParamSlot& slot) {
    // TODO: Free memory if allowed staturation was shrinked.
    return false;
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::onStartAsync
 */
bool megamol::stdplugin::volume::VolumetricDataSource::onStartAsync(core::Call& call) {
    this->paramLoadAsync.Param<core::param::BoolParam>()->SetValue(true);
    return true;
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::onStopAsync
 */
bool megamol::stdplugin::volume::VolumetricDataSource::onStopAsync(core::Call& call) {
    this->paramLoadAsync.Param<core::param::BoolParam>()->SetValue(false);
    return true;
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::release
 */
void megamol::stdplugin::volume::VolumetricDataSource::release(void) {
    using vislib::sys::Log;

    try {
        if (this->loaderThread.IsRunning()) {
            this->stopAsyncLoad(true);
        }
        ASSERT(!this->loaderThread.IsRunning());
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteError(1, e.GetMsg());
    } catch (...) {
        Log::DefaultLog.WriteError(1, _T("Unexpected exception while ")
                                      _T("stopping volume loader thread during release of data source."));
    }

    if (this->fileInfo != nullptr) {
        Log::DefaultLog.WriteInfo(10, _T("Releasing dat file..."));
        ::datRaw_close(this->fileInfo);
        ::datRaw_freeInfo(this->fileInfo);
        SAFE_DELETE(this->fileInfo);
    }
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::resumeAsyncLoad
 */
bool megamol::stdplugin::volume::VolumetricDataSource::resumeAsyncLoad(void) {
    using vislib::sys::Log;
    Log::DefaultLog.WriteInfo(_T("Resuming volume loader thread..."));
    int expected[] = {LOADER_STATUS_PAUSING, LOADER_STATUS_PAUSED};
    this->spinExchange(this->loaderStatus, LOADER_STATUS_RUNNING, expected, STATIC_ARRAY_COUNT(expected), true);
    this->evtStartLoading.Set();
    return (this->loaderStatus.load() == LOADER_STATUS_RUNNING);
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::startAsyncLoad
 */
bool megamol::stdplugin::volume::VolumetricDataSource::startAsyncLoad(void) {
    using vislib::sys::Log;
    try {
        Log::DefaultLog.WriteInfo(_T("Starting volume loader thread..."));
        this->loaderStatus.store(LOADER_STATUS_RUNNING);
        this->evtStartLoading.Set();
        if (!this->loaderThread.IsRunning()) {
            this->loaderThread.Start(this);
        }
        return true;
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteError(1, e.GetMsg());
        return false;
    }
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::stopAsyncLoad
 */
void megamol::stdplugin::volume::VolumetricDataSource::stopAsyncLoad(const bool isWait) {
    using vislib::sys::Log;
    try {
        Log::DefaultLog.WriteInfo(_T("Stopping volume loader thread..."));
        this->loaderStatus.store(LOADER_STATUS_STOPPING);
        this->evtStartLoading.Set();
        if (this->loaderThread.IsRunning() && isWait) {
            this->loaderThread.Join();
            ASSERT(this->loaderStatus.load() == LOADER_STATUS_STOPPED);
        }
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteError(1, e.GetMsg());
    }
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::suspendAsyncLoad
 */
bool megamol::stdplugin::volume::VolumetricDataSource::suspendAsyncLoad(const bool isWait) {
    using vislib::sys::Log;
    using vislib::sys::Thread;

    int expected = LOADER_STATUS_RUNNING;
    bool retval = false;

    Log::DefaultLog.WriteInfo(_T("Suspending volume loader thread..."));
    if (this->loaderStatus.compare_exchange_strong(expected, LOADER_STATUS_PAUSING)) {
        ASSERT(this->loaderThread.IsRunning());
        this->evtStartLoading.Set();
        if (isWait) {
            while (this->loaderStatus.load() != LOADER_STATUS_PAUSED) {
                Thread::Reschedule();
            }
        }
        retval = true;
    }
    ASSERT(this->loaderStatus != LOADER_STATUS_RUNNING);

    return retval;
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::BufferSlotUnlocker::~BufferSlotUnlocker
 */
megamol::stdplugin::volume::VolumetricDataSource::BufferSlotUnlocker::~BufferSlotUnlocker(void) { this->Unlock(); }


/*
 * megamol::stdplugin::volume::VolumetricDataSource::BufferSlotUnlocker::Unlock
 */
void megamol::stdplugin::volume::VolumetricDataSource::BufferSlotUnlocker::Unlock(void) {
    for (size_t i = 0; i < this->buffers.Count(); ++i) {
        ASSERT(this->buffers[i]->status == BUFFER_STATUS_USED);
        this->buffers[i]->status.store(BUFFER_STATUS_READY);
    }
    this->buffers.Clear();
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::loadAsync
 */
DWORD megamol::stdplugin::volume::VolumetricDataSource::loadAsync(void* userData) {
    using core::misc::VolumetricDataCall;
    using core::param::IntParam;
    using vislib::sys::Log;

    int expected = 0;
    DWORD retval = 0;
    VolumetricDataSource* that = static_cast<VolumetricDataSource*>(userData);
    ASSERT(that != nullptr);
    auto format = that->getOutputDataFormat();

    try {
        Log::DefaultLog.WriteInfo(_T("The asynchronous loading thread of ")
                                  _T("VolumetricDataSource is starting..."));
        expected = LOADER_STATUS_STOPPING;
        while (!that->loaderStatus.compare_exchange_strong(expected, LOADER_STATUS_STOPPED)) {

            /* Check all buffers whether there is something to do. */
            for (size_t i = 0; (i < that->buffers.Count()) && (that->loaderStatus.load() == LOADER_STATUS_RUNNING);
                 ++i) {
                expected = BUFFER_STATUS_PENDING;
                if (that->buffers[i]->status.compare_exchange_strong(expected, BUFFER_STATUS_LOADING)) {
                    // We assume that the loader thread is not running while the
                    // data set or the number of buffers are changed. If this is
                    // the case, the following code might crash!
                    that->buffers[i]->Buffer.AssertSize(that->calcFrameSize());
                    auto dst = that->buffers[i]->Buffer.At(0);

#if (defined(DEBUG) || defined(_DEBUG))
                    Log::DefaultLog.WriteInfo(_T("Loading frame %u in format ")
                                              _T("%hs to 0x%p (async)."),
                        that->buffers[i]->FrameID, ::datRaw_getDataFormatName(format), dst);
#endif /* (defined(DEBUG) || defined(_DEBUG)) */
                    if (::datRaw_loadStep(that->fileInfo, static_cast<int>(that->buffers[i]->FrameID), &dst, format)) {
                        that->buffers[i]->status.store(BUFFER_STATUS_READY);
                    } else {
                        Log::DefaultLog.WriteError(_T("Loading frame %u ")
                                                   _T("failed."),
                            that->buffers[i]->FrameID);
                        that->buffers[i]->status.store(BUFFER_STATUS_UNUSED);
                    }
                    ::datRaw_close(that->fileInfo);

                    /* Sleep if requested by user. */
                    auto asyncSleep = that->paramAsyncSleep.Param<IntParam>()->Value();
                    if (asyncSleep > 0) {
                        vislib::sys::Thread::Sleep(asyncSleep);
                    }
                } /* end if (that->buffers[i]->status. ... */
            }     /* end for (size_t i = 0; (i < that->buffers.Count()) ... */

            /*
             * Confirm that we are not loading if the UI requested the loader
             * to suspend loading.
             */
            expected = LOADER_STATUS_PAUSING;
            that->loaderStatus.compare_exchange_strong(expected, LOADER_STATUS_PAUSED);

            /*
             * Wait until we have something to do. If requested, wake the thread
             * automatically to check for status updates after some time.
             */
            auto asyncWake = that->paramAsyncWake.Param<IntParam>()->Value();
            if (asyncWake > 0) {
                that->evtStartLoading.Wait(asyncWake);
            } else {
                that->evtStartLoading.Wait();
            }

            // Prepare for stopping.
            expected = LOADER_STATUS_STOPPING;
        }
        ASSERT(that->loaderStatus.load() == LOADER_STATUS_STOPPED);
        Log::DefaultLog.WriteInfo(_T("The asynchronous loading thread of ")
                                  _T("VolumetricDataSource is exiting..."));
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteError(1, e.GetMsg());
        retval = 1;
    } catch (...) {
        Log::DefaultLog.WriteError(1, _T("Unexpected exception in ")
                                      _T("asynchronous loader thread. Most likely, there is not enough")
                                      _T("memory available for storing the data set."));
        retval = 2;
    }

    return retval;
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::setUnlocker
 */
void megamol::stdplugin::volume::VolumetricDataSource::setUnlocker(
    core::misc::VolumetricDataCall& call, BufferSlot* buffer) {
    ASSERT(buffer != nullptr);
    auto ul = dynamic_cast<BufferSlotUnlocker*>(call.GetUnlocker());
    if (ul != nullptr) {
        ul->AddBuffer(buffer);
    } else {
        call.SetUnlocker(new BufferSlotUnlocker(buffer));
    }
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::spinExchange
 */
bool megamol::stdplugin::volume::VolumetricDataSource::spinExchange(
    std::atomic_int& dst, const int value, const int* expected, const size_t cntExpected, const bool canFail) {
    ASSERT(expected != nullptr);
    ASSERT(cntExpected >= 1);

    int e = *expected;
    bool isUnexpected;

    while (!dst.compare_exchange_strong(e, value)) {
        isUnexpected = true;
        for (size_t i = 0; i < cntExpected; ++i) {
            if (e == expected[i]) {
                isUnexpected = false;
                break;
            }
        }
        if (isUnexpected) {
            if (canFail) {
                return false;
            } else {
                e = *expected;
            }
        }
    }

    return true;
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::assertBuffersUnsafe
 */
size_t megamol::stdplugin::volume::VolumetricDataSource::assertBuffersUnsafe(size_t cntFrames, bool doNotFree) {
    using core::param::IntParam;
    using vislib::sys::Log;

    size_t frameSize = this->calcFrameSize();
    size_t retval = 0;

    if (cntFrames < 2) {
        cntFrames = this->paramBuffers.Param<IntParam>()->Value();
    }
    ASSERT(cntFrames >= 2);

    try {
        size_t cntBuffers = this->buffers.Count();

        if (cntBuffers < cntFrames) {
            for (size_t i = cntBuffers; i < cntFrames; ++i) {
                auto bufferSlot = new BufferSlot();
                bufferSlot->Buffer.AssertSize(frameSize);
                bufferSlot->status = BUFFER_STATUS_UNUSED;
                this->buffers.Add(bufferSlot);
            }

        } else if ((cntBuffers > cntFrames) && !doNotFree) {
            int deletable[] = {BUFFER_STATUS_READY, BUFFER_STATUS_UNUSED, BUFFER_STATUS_PENDING};
            for (size_t i = 0; (i < this->buffers.Count()) && (this->buffers.Count() > cntFrames); ++i) {
                if (VolumetricDataSource::spinExchange(this->buffers[i]->status, BUFFER_STATUS_DELETING, deletable,
                        STATIC_ARRAY_COUNT(deletable), true)) {
                    ASSERT(this->buffers[i]->status == BUFFER_STATUS_DELETING);
                    this->buffers.RemoveAt(i--);
                }
            }
        }

        retval = this->buffers.Count();
    } catch (vislib::Exception e) {
        Log::DefaultLog.WriteError(1, e.GetMsg());
    } catch (...) {
        Log::DefaultLog.WriteError(1, _T("Unexpected exception while ")
                                      _T("preparing data buffers. Not all of the requested memory ")
                                      _T("might be available."));
    }

    return retval;
}


/*
 * megamol::stdplugin::volume::VolumetricDataSource::bufferForFrameIDUnsafe
 */
int megamol::stdplugin::volume::VolumetricDataSource::bufferForFrameIDUnsafe(const unsigned int frameID) const {
    for (size_t i = 0; i < this->buffers.Count(); ++i) {
        if (this->buffers[i]->FrameID == frameID) {
            return static_cast<int>(i);
        }
    }
    /* Not found. */

    return -1;
}
