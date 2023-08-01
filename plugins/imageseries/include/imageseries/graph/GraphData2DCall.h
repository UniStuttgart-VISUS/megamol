/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include "GraphData2D.h"

#include "mmcore/factories/CallAutoDescription.h"

#include "mmstd/data/AbstractGetDataCall.h"

#include <glm/mat3x2.hpp>

#include <memory>

namespace megamol::ImageSeries {

class GraphData2DCall : public megamol::core::AbstractGetDataCall {

public:
    using CallDescription = typename megamol::core::factories::CallAutoDescription<GraphData2DCall>;

    /** Function index for obtaining the graph data. **/
    static const unsigned int CallGetData = 0;

    /**
     * Answer the name of this call.
     *
     * @return The name of this call.
     */
    static const char* ClassName() {
        return "GraphData2DCall";
    }

    /**
     * Answer a human readable description of this call.
     *
     * @return A human readable description of this call.
     */
    static const char* Description() {
        return "Call to transport a graph structure generated from a labeled image series";
    }

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
     * @param idx The index of the function to return the name of.
     *
     * @return The name of the requested function, or a nullpointer for out-of-bounds requests.
     */
    static const char* FunctionName(unsigned int idx) {
        switch (idx) {
        case CallGetData:
            return "GetData";
        }
        return nullptr;
    }

    GraphData2DCall() = default;
    ~GraphData2DCall() override = default;

    struct Output {
        std::shared_ptr<const graph::AsyncGraphData2D> graph;
    };

    const Output& GetOutput() const {
        return output;
    }

    void SetOutput(Output output) {
        this->output = std::move(output);
    }

private:
    Output output;
};

} // namespace megamol::ImageSeries
