#pragma once

#include <array>
#include <functional>
#include <string>
#include <tuple>

#include "mmcore/Call.h"

#include "mmcore/factories/CallAutoDescription.h"

namespace megamol::infovis {

/**
 * Call transporting a function pointer for pushing series info
 * into info container of the renderer
 */
class DiagramSeriesCall : public core::Call {
public:
    /*typedef struct _DiagramSeriesTuple {
        uint32_t id;
        uint32_t col;
        std::string name;
        float scaling;
    } DiagramSeriesTuple;*/

    typedef std::tuple<uint32_t, size_t, std::string, float, std::array<float, 3>> DiagramSeriesTuple;

    typedef std::function<void(const DiagramSeriesCall::DiagramSeriesTuple& tuple)> fpSeriesInsertionCB;

    static const unsigned int IdIdx;

    static const unsigned int ColIdx;

    static const unsigned int NameIdx;

    static const unsigned int ScalingIdx;

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName() {
        return "DiagramSeriesCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return "Call to get diagram series selection";
    }

    /** Index of the 'GetSeries' function */
    static const unsigned int CallForGetSeries;

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
            return "getSeries";
        }
        return "";
    }

    /** ctor */
    DiagramSeriesCall();

    /** dtor */
    ~DiagramSeriesCall() override;

    /** Copy operator of the funtion pointer for cascaded DiagramSeries modules */
    DiagramSeriesCall& operator=(const DiagramSeriesCall& rhs);

    fpSeriesInsertionCB GetSeriesInsertionCB() const;

    void SetSeriesInsertionCB(const fpSeriesInsertionCB& fpsicb);

private:
    /** Function pointer to push operation in renderer */
    fpSeriesInsertionCB ptmSeriesInsertionCB;
}; /* end class DiagramSeriesCall */

typedef core::factories::CallAutoDescription<DiagramSeriesCall> DiagramSeriesCallDescription;

} // namespace megamol::infovis
