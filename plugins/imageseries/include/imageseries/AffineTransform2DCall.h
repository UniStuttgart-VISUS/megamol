#ifndef INCLUDE_IMAGESERIES_AFFINETRANSFORM2DCALL_H_
#define INCLUDE_IMAGESERIES_AFFINETRANSFORM2DCALL_H_

#include "mmstd/data/AbstractGetDataCall.h"
#include "mmcore/factories/CallAutoDescription.h"

#include "glm/mat3x2.hpp"

#include <memory>

namespace megamol::ImageSeries {

class AffineTransform2DCall : public megamol::core::AbstractGetDataCall {

public:
    using CallDescription = typename megamol::core::factories::CallAutoDescription<AffineTransform2DCall>;

    /** Function index for obtaining the transformation matrix. **/
    static const unsigned int CallGetTransform = 0;

    /**
     * Answer the name of this call.
     *
     * @return The name of this call.
     */
    static const char* ClassName(void) {
        return "AffineTransform2DCall";
    }

    /**
     * Answer a human readable description of this call.
     *
     * @return A human readable description of this call.
     */
    static const char* Description(void) {
        return "Call to transport a 2D affine transformation matrix";
    }

    /**
     * Answer the number of functions used for this call.
     *
     * @return The number of functions used for this call.
     */
    static unsigned int FunctionCount(void) {
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
        case CallGetTransform:
            return "GetTransform";
        }
        return nullptr;
    }

    AffineTransform2DCall() = default;
    ~AffineTransform2DCall() override = default;

    struct Output {
        glm::mat3x2 matrix;
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

#endif
