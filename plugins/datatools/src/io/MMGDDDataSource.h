/*
 * MMGDDDataSource.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_DATATOOLS_IO_MMGDDDATASOURCE_H_INCLUDED
#define MEGAMOL_DATATOOLS_IO_MMGDDDATASOURCE_H_INCLUDED
#pragma once

#include "datatools/GraphDataCall.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd/data/AnimDataModule.h"
#include "vislib/sys/File.h"
#include <cstdint>
#include <vector>


namespace megamol {
namespace datatools {
namespace io {


/**
 * Data source module for MegaMol GraphData Dump files.
 */
class MMGDDDataSource : public core::view::AnimDataModule {
public:
    static const char* ClassName(void) {
        return "MMGDDDataSource";
    }
    static const char* Description(void) {
        return "Data source module for MegaMol GraphData Dump files.";
    }
    static bool IsAvailable(void) {
        return true;
    }

    MMGDDDataSource(void);
    ~MMGDDDataSource(void) override;

protected:
    core::view::AnimDataModule::Frame* constructFrame(void) const override;
    bool create(void) override;
    void loadFrame(core::view::AnimDataModule::Frame* frame, unsigned int idx) override;
    void release(void) override;

    /** Nested class of frame data */
    class Frame : public core::view::AnimDataModule::Frame {
    public:
        Frame(core::view::AnimDataModule& owner)
                : core::view::AnimDataModule::Frame(owner)
                , isDirected(false)
                , edges() {
            // intentionally empty
        }
        ~Frame() override {
            // intentionally empty
        }

        void Clear() {
            edges.clear();
        }
        bool LoadFrame(vislib::sys::File* file, unsigned int idx, UINT64 size) {
            unsigned char flags;
            file->Read(&flags, 1);
            isDirected = (flags & 1);
            edges.resize(
                static_cast<std::vector<GraphDataCall::edge>::size_type>((size - 1) / sizeof(GraphDataCall::edge)));
            file->Read(edges.data(), edges.size() * sizeof(GraphDataCall::edge));
            frame = idx;
            return true;
        }
        void SetData(GraphDataCall& call) {
            call.Set(edges.data(), edges.size(), isDirected);
        }

    private:
        bool isDirected;
        std::vector<GraphDataCall::edge> edges;
    };

    /**
     * Helper class to unlock frame data when 'CallSimpleSphereData' is
     * used.
     */
    class Unlocker : public GraphDataCall::Unlocker {
    public:
        Unlocker(Frame& frame) : GraphDataCall::Unlocker(), frame(&frame) {
            // intentionally empty
        }
        ~Unlocker(void) override {
            this->Unlock();
            ASSERT(this->frame == nullptr);
        }
        void Unlock(void) override {
            if (this->frame != nullptr) {
                this->frame->Unlock();
                this->frame = nullptr;
            }
        }

    private:
        Frame* frame;
    };

    bool filenameChanged(core::param::ParamSlot& slot);

    bool getDataCallback(core::Call& caller);

    bool getExtentCallback(core::Call& caller);

    core::param::ParamSlot filename;
    core::CalleeSlot getData;

    vislib::sys::File* file;
    std::vector<uint64_t> frameIdx;
    size_t data_hash;
};

} /* end namespace io */
} /* end namespace datatools */
} /* end namespace megamol */

#endif /* MEGAMOL_DATATOOLS_IO_MMGDDDATASOURCE_H_INCLUDED */
