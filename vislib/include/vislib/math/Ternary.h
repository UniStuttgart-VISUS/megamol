/*
 * Ternary.h
 *
 * Copyright (C) 2006 - 2009 by Visualisierungsinstitut Universitaet Stuttgart.
 * Alle Rechte vorbehalten.
 */

#pragma once
#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */

#include "vislib/String.h"
#include "vislib/forceinline.h"


namespace vislib::math {


/**
 * This class implements the three-state logical type used by ternary
 * logic as introduced by Lukasiewicz, Lewis and Sulski. It uses the value
 * constants 'True', 'False', and 'Unknown', whereas 'True' and 'False'
 * behave almost like their boolean counterparts. The value is stored as
 * an integer following the definition of a balanced ternary system:
 *  True    =  1
 *  Unknown =  0 (Differs from the definition of the boolean 'false')
 *  False   = -1
 * However, whenever casting from a numeric value, all positive numbers
 * will be converted to 'True', and all negative value will be converted
 * to 'False'.
 */
class Ternary {
public:
    /** The symbolic constant for 'True' with the value of 1 */
    static const Ternary TRI_TRUE;

    /** The symbolic constant for 'Unknown' with the value of 0 */
    static const Ternary TRI_UNKNOWN;

    /** The symbolic constant for 'False' with the value of -1 */
    static const Ternary TRI_FALSE;

    /**
     * Ctor. (TRI_UNKNOWN)
     */
    Ternary();

    /**
     * Ctor.
     *
     * @param src The source value to be copied.
     */
    Ternary(const Ternary& src);

    /**
     * Ctor.
     *
     * @param value The value to be copied. All positive numbers will be
     *              converted to 'True', all negative numbers will be
     *              converted to 'False, and zero will be converted to
     *              'Unknown'.
     */
    explicit Ternary(int value);

    /**
     * Ctor.
     *
     * @param value The value to be copied.
     */
    explicit Ternary(bool value);

    /** Dtor. */
    ~Ternary();

    /**
     * Answer whether the value is 'False'
     *
     * @return 'true' if the value is 'False'
     */
    inline bool IsFalse() const {
        return this->value == -1;
    }

    /**
     * Answer whether the value is 'True'
     *
     * @return 'true' if the value is 'True'
     */
    inline bool IsTrue() const {
        return this->value == 1;
    }

    /**
     * Answer whether the value is 'Unknown'
     *
     * @return 'true' if the value is 'Unknown'
     */
    inline bool IsUnknown() const {
        return this->value == 0;
    }

    /**
     * Parses the string 'str' to be a ternary value.
     *
     * @param str The string to be parsed as ternary value.
     *
     * @return 'true' on success, 'false' if the string could not be
     *         parsed. In the later case the value of 'this' was not
     *         changed.
     */
    bool Parse(const vislib::StringA& str);

    /**
     * Parses the string 'str' to be a ternary value.
     *
     * @param str The string to be parsed as ternary value.
     *
     * @return 'true' on success, 'false' if the string could not be
     *         parsed. In the later case the value of 'this' was not
     *         changed.
     */
    bool Parse(const vislib::StringW& str);

    /**
     * Answers a string representation of the value. The returned string
     * will be one of these strings: "true", "false", or "unknown".
     *
     * @return A string representation of the value.
     */
    vislib::StringA ToStringA() const;

    /**
     * Answers a string representation of the value. The returned string
     * will be one of these strings: "true", "false", or "unknown".
     *
     * @return A string representation of the value.
     */
    vislib::StringW ToStringW() const;

    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand.
     *
     * @return Reference to this
     */
    Ternary& operator=(const Ternary& rhs);

    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand.
     *
     * @return Reference to this
     */
    Ternary& operator=(int rhs);

    /**
     * Assignment operator
     *
     * @param rhs The right hand side operand.
     *
     * @return Reference to this
     */
    Ternary& operator=(bool rhs);

    /**
     * Test for equality.
     *
     * @param rhs The right hand side operand.
     *
     * @return 'true' if 'this' and 'rhs' are equal, 'false' otherwise.
     */
    bool operator==(const Ternary& rhs) const;

    /**
     * Test for equality.
     *
     * @param rhs The right hand side operand.
     *
     * @return 'true' if 'this' and 'rhs' are equal, 'false' otherwise.
     */
    bool operator==(int rhs) const;

    /**
     * Test for equality.
     *
     * @param rhs The right hand side operand.
     *
     * @return 'true' if 'this' and 'rhs' are equal, 'false' otherwise.
     */
    bool operator==(bool rhs) const;

    /**
     * Test for inequality.
     *
     * @param rhs The right hand side operand.
     *
     * @return 'false' if 'this' and 'rhs' are equal, 'true' otherwise.
     */
    inline bool operator!=(const Ternary& rhs) const {
        return !(*this == rhs);
    }

    /**
     * Test for inequality.
     *
     * @param rhs The right hand side operand.
     *
     * @return 'false' if 'this' and 'rhs' are equal, 'true' otherwise.
     */
    inline bool operator!=(int rhs) const {
        return !(*this == rhs);
    }

    /**
     * Test for inequality.
     *
     * @param rhs The right hand side operand.
     *
     * @return 'false' if 'this' and 'rhs' are equal, 'true' otherwise.
     */
    inline bool operator!=(bool rhs) const {
        return !(*this == rhs);
    }

    /**
     * Answers the order of 'this' and 'rhs'
     *
     * @param rhs The right hand side operand.
     *
     * @return 'true' if the value of 'this' is less then the value of
     *         'rhs', 'false' otherwise.
     */
    inline bool operator<(const Ternary& rhs) const {
        return this->value < rhs.value;
    }

    /**
     * Answers the order of 'this' and 'rhs'
     *
     * @param rhs The right hand side operand.
     *
     * @return 'true' if the value of 'this' is greater then the value of
     *         'rhs', 'false' otherwise.
     */
    inline bool operator>(const Ternary& rhs) const {
        return this->value > rhs.value;
    }

    /**
     * Answers the order of 'this' and 'rhs'
     *
     * @param rhs The right hand side operand.
     *
     * @return 'true' if the value of 'this' is less then or equal to the
     *         value of 'rhs', 'false' otherwise.
     */
    inline bool operator<=(const Ternary& rhs) const {
        return this->value <= rhs.value;
    }

    /**
     * Answers the order of 'this' and 'rhs'
     *
     * @param rhs The right hand side operand.
     *
     * @return 'true' if the value of 'this' is greater then or equal to
     *         the value of 'rhs', 'false' otherwise.
     */
    inline bool operator>=(const Ternary& rhs) const {
        return this->value >= rhs.value;
    }

    /**
     * Answers the strong negation of the ternary value. A strong negation
     * answers 'Unknown' for a 'Unknown' (different from weak negation).
     *
     * @return The strong negation of this value.
     */
    Ternary operator!() const;

    /**
     * Answers the strong negation of the ternary value. A strong negation
     * answers 'Unknown' for a 'Unknown' (different from weak negation).
     *
     * @return The strong negation of this value.
     */
    Ternary operator-() const;

    /**
     * Answers the weak negation of the ternary value. A weak negation
     * answers 'True' for a 'Unknown' (different from strong negation).
     *
     * @return The weak negation of this value.
     */
    Ternary operator~() const;

    /**
     * Performs a logical 'and' of 'this' and 'rhs' and returns the value
     * without changing the value of 'this'
     *
     * @param rhs The right hand side operand.
     *
     * @return The resulting value of the operation.
     */
    Ternary operator&(const Ternary& rhs) const;

    /**
     * Performs a logical 'and' of 'this' and 'rhs' and stores the result
     * in 'this'.
     *
     * @param rhs The right hand side operand.
     *
     * @return A reference to this.
     */
    Ternary& operator&=(const Ternary& rhs);

    /**
     * Performs a logical 'or' of 'this' and 'rhs' and returns the value
     * without changing the value of 'this'
     *
     * @param rhs The right hand side operand.
     *
     * @return The resulting value of the operation.
     */
    Ternary operator|(const Ternary& rhs) const;

    /**
     * Performs a logical 'or' of 'this' and 'rhs' and stores the result
     * in 'this'.
     *
     * @param rhs The right hand side operand.
     *
     * @return A reference to this.
     */
    Ternary& operator|=(const Ternary& rhs);

    /**
     * Cast to integer. This method will only return one of the three
     * values 1, 0, and -1.
     *
     * @return The integer representation [-1 .. 1] of the value.
     */
    operator int() const;

private:
    /**
     * Calculates the value for the input 'v'
     *
     * @param v The input value
     *
     * @return The value for the internal representation
     */
    VISLIB_FORCEINLINE int getValue(int v) const;

    /**
     * Calculates the value for the input 'v'
     *
     * @param v The input value
     *
     * @return The value for the internal representation
     */
    VISLIB_FORCEINLINE int getValue(bool v) const;

    /** The three-state value */
    int value;
};

} // namespace vislib::math

#if defined(_WIN32) && defined(_MANAGED)
#pragma managed(pop)
#endif /* defined(_WIN32) && defined(_MANAGED) */
