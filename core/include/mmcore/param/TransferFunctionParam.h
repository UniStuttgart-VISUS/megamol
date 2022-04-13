/*
 * TransferFunctionParam.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <cmath>
#include <string>

#include "AbstractParam.h"
#include "mmcore/utility/JSONHelper.h"
#include "mmcore/utility/log/Log.h"

#include "vislib/String.h"


#define TFP_VAL_CNT (6)

namespace megamol {
namespace core {
namespace param {


/**
 * Class for parameter holding transfer function as JSON string.
 */
class TransferFunctionParam : public AbstractParam {
public:
    enum InterpolationMode { LINEAR = 0, GAUSS = 1 };

    typedef std::array<float, TFP_VAL_CNT> NodeData_t;
    typedef std::vector<NodeData_t> NodeVector_t;

    // ------------------------------------------------------------------------

    // Example JSON output format (empty string is also excepted for initialisation):
    //{
    //   "IgnoreProjectRange" : false,
    //    "Interpolation": "LINEAR",
    //    "Nodes" : [
    //        [
    //            0.0,     // = Red
    //            0.0,     // = Green
    //            0.0,     // = Blue
    //            0.0,     // = Alpha
    //            0.0,     // = Value (always in range [0,1], independent of actual given value range!)
    //            0.05     // = Sigma (only used for gauss interpolation)
    //        ],
    //        [
    //            1.0,
    //            1.0,
    //            1.0,
    //            1.0,
    //            1.0,
    //            0.05
    //        ]
    //    ],
    //    "TextureSize": 128,
    //    "ValueRange": [
    //        0.0,        // = minimum value of node range
    //        1.0         // = maximum value of node range
    //   ],
    //}

    /**
     * Set transfer function data from JSON string.
     *
     * @param in_tfs                                 The transfer function input encoded as string in JSON format.
     * @param out_nodes                              The transfer function node output.
     * @param out_interpolmode                       The transfer function interpolation mode output.
     * @param out_texsize                            The transfer function texture size output.
     *
     * @return True if JSON string was successfully converted into transfer function data, false otherwise.
     */
    static bool GetParsedTransferFunctionData(const std::string& in_tfs, NodeVector_t& out_nodes,
        InterpolationMode& out_interpolmode, unsigned int& out_texsize, std::array<float, 2>& out_range);

    /**
     * Get transfer function JSON string from data.
     *
     * @param out_tfs                       The transfer function output encoded as string in JSON format.
     * @param in_nodes                      The transfer function node input.
     * @param in_interpolmode               The transfer function interpolation mode input.
     * @param in_texsize                    The transfer function texture size input.
     * @param in_ignore_project_range_once  Flag to ignore range of tf parameter given in project file.
     *
     * @return True if transfer function data was successfully converted into JSON string, false otherwise.
     */
    static bool GetDumpedTransferFunction(std::string& out_tfs, const NodeVector_t& in_nodes,
        InterpolationMode in_interpolmode, unsigned int in_texsize, const std::array<float, 2>& in_range,
        bool in_ignore_project_range = true);

    /**
     * Check given transfer function data.
     *
     * @param nodes         The transfer function nodes input.
     * @param interpolmode  The transfer function interpolation mode input.
     * @param texsize       The transfer function texture size input.
     *
     * @return True if given data is valid, false otherwise.
     */
    static bool CheckTransferFunctionData(const NodeVector_t& nodes, InterpolationMode interpolmode,
        unsigned int texsize, const std::array<float, 2>& range);

    /**
     * Check given transfer function JSON string.
     *
     * @param tfs  The transfer function JSON string.
     *
     * @return True if given data is valid, false otherwise.
     */
    static bool CheckTransferFunctionString(const std::string& tfs);

    /**
     * Linear interpolation of transfer function data in range [0..texsize]
     *
     * @param out_texdata  The generated texture from the input transfer function.
     * @param in_texsize   The transfer function texture size input.
     * @param in_nodes    The transfer function node data input.
     */
    static std::vector<float> LinearInterpolation(unsigned int in_texsize, const NodeVector_t& in_nodes);

    /**
     * Gauss interpolation of transfer function data in range [0..texsize]
     *
     * @param out_texdata  The generated texture from the input transfer function.
     * @param in_texsize   The transfer function texture size input.
     * @param in_nodes    The transfer function node data input.
     */
    static std::vector<float> GaussInterpolation(unsigned int in_texsize, const NodeVector_t& in_nodes);

    /**
     * Return texture data for given transfer function JSON string.
     *
     * @param in_tfs          The transfer function JSON string as input.
     * @param out_tex_data    The returned texture data.
     * @param out_width       The returned texture width.
     * @param out_height      The returned texture height.
     *
     * @return True on success, false otherwise.
     */
    static bool GetTextureData(
        const std::string& in_tfs, std::vector<float>& out_tex_data, int& out_width, int& out_height);

    /**
     * Return flag 'IgnoreProjectRange'
     */
    static bool IgnoreProjectRange(const std::string& in_tfs);

    /** Calculates gauss function value.
     *
     * @param x  The current x position to calculate the gauss function value for.
     * @param a  The scaling of the gauss function (bell) in direction of the y axis.
     * @param b  The translation of the gauss function (bell) center in direction of the x axis.
     * @param c  The width (= sigma) of the gauss function (bell).
     *
     * @return The gauss funtion value.
     */
    static inline float gauss(float x, float a, float b, float c) {
        return (a * expf(-1.0f * (x - b) * (x - b) / (2.0f * c * c)));
    }

    // ------------------------------------------------------------------------

    /**
     * Ctor.
     *
     * @param initVal The initial value
     */
    explicit TransferFunctionParam(const std::string& initVal = "");
    explicit TransferFunctionParam(const char* initVal);
    explicit TransferFunctionParam(const vislib::StringA& initVal);

    /**
     * Dtor.
     */
    ~TransferFunctionParam(void) override = default;

    /**
     * Returns a machine-readable definition of the parameter.
     *
     * @param outDef A memory block to receive a machine-readable
     *               definition of the parameter.
     */
    std::string Definition() const override;

    /**
     * Tries to parse the given string as value for this parameter and
     * sets the new value if successful. This also triggers the update
     * mechanism of the slot this parameter is assigned to.
     *
     * @param v The new value for the parameter as string.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool ParseValue(std::string const& v) override;

    /**
     * Sets the value of the parameter and optionally sets the dirty flag
     * of the owning parameter slot.
     *
     * @param v the new value for the parameter
     * @param setDirty If 'true' the dirty flag of the owning parameter
     *                 slot is set and the update callback might be called.
     */
    void SetValue(const std::string& v, bool setDirty = true);

    /**
     * Returns the value of the parameter as string.
     *
     * @return The value of the parameter as string.
     */
    std::string ValueString(void) const override;

    /**
     * Gets the value of the parameter
     *
     * @return The value of the parameter
     */
    inline const std::string& Value(void) const {
        return this->val;
    }

    /**
     * Gets the value of the parameter
     *
     * @return The value of the parameter
     */
    inline operator const std::string&(void) const {
        return this->val;
    }

    /**
     * Gets the hash of the current value
     *
     * @return The hash of the parameter value
     */
    inline size_t ValueHash(void) const {
        return this->hash;
    }

private:
#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */

    /** The value of the parameter */
    std::string val;

    /** Hash of current parameter value. */
    size_t hash;

#ifdef _WIN32
#pragma warning(disable : 4251)
#endif /* _WIN32 */

}; /* end class TransferFunctionParam */

} /* end namespace param */
} /* end namespace core */
} /* end namespace megamol */
