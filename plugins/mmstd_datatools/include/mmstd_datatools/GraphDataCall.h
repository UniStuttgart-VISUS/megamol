/*
 * GraphDataCall.h
 *
 * Copyright (C) 2016 by MegaMol Team
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/AbstractGetDataCall.h"
#include "mmcore/factories/CallAutoDescription.h"
#include <iterator>
#include <cstdint>

namespace megamol {
namespace stdplugin {
namespace datatools {

    /**
     * Call to transport graph data as edges between pairs (indices) of particles
     */
    class GraphDataCall : public core::AbstractGetDataCall {
    public:

        /** Particle index type */
        typedef uint32_t index_t;

        /** Edge type, tightly packed index pair */
        typedef struct _edge_t {
            index_t i1;
            index_t i2;
        } edge;

        /** Call function names */
        enum CallFunctionNames : int {
            GET_DATA = 0,
            GET_EXTENT = 1
        };

        /** factory info */
        static const char *ClassName(void) {
            return "GraphDataCall";
        }
        static const char *Description(void) {
            return "Call to get graph edge data";
        }
        static unsigned int FunctionCount(void) {
            return 2;
        }
        static const char * FunctionName(unsigned int idx) {
            switch (idx) {
            case GET_DATA: return "GetData";
            case GET_EXTENT: return "GetExtent";
            }
            return "";
        }

        /** ctor */
        GraphDataCall();
        /** dtor */
        virtual ~GraphDataCall();

        /**
         * Returns number of edges
         * Edge data will contain two index values (start and end point) for each edge)
         */
        inline size_t GetEdgeCount() const {
            return edge_count;
        }
        /**
         * Return edge data, i.e. continuous array of edges holding index pairs.
         */
        inline edge const * GetEdgeData() const {
            return edge_data;
        }
        /**
         * True if edges are directed.
         */
        inline bool IsDirected() const {
            return is_directed;
        }

        /**
         * Number of frames in time-dependent data
         */
        inline unsigned int FrameCount(void) const {
            return frameCnt;
        }
        /**
         * Current frame id (zero-based) in time-dependent data
         */
        inline unsigned int FrameID(void) const {
            return frameID;
        }

        /** Sets data pointers. No deep copy */
        inline void Set(edge const* data, size_t count, bool is_dir = false) {
            edge_data = data;
            edge_count = count;
            is_directed = is_dir;
        }
        /** Sets the number of frames in time-dependent data. Should never smaller than one */
        inline void SetFrameCount(unsigned int frameCnt) {
            this->frameCnt = frameCnt;
        }
        /** Sets the current frame id (should be smaller than frameCount */
        inline void SetFrameID(unsigned int frameID) {
            this->frameID = frameID;
        }

    private:

        size_t edge_count;
        edge const *edge_data;
        bool is_directed;
        unsigned int frameCnt;
        unsigned int frameID;

    };

    /** Description typedef */
    typedef core::factories::CallAutoDescription<GraphDataCall> GraphDataCallDescription;

}
}
}

namespace std {
    /** Utility functions allowing ranged-based for iterating over edges */
    inline const megamol::stdplugin::datatools::GraphDataCall::edge* begin(megamol::stdplugin::datatools::GraphDataCall& gdc) {
        return gdc.GetEdgeData();
    }
    inline const megamol::stdplugin::datatools::GraphDataCall::edge* end(megamol::stdplugin::datatools::GraphDataCall& gdc) {
        return gdc.GetEdgeData() + gdc.GetEdgeCount();
    }
}
