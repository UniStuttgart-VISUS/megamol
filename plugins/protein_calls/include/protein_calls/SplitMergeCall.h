/*
 * SplitMergeCall.h
 *
 * Author: Guido Reina
 * Copyright (C) 2012 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#ifndef MEGAMOL_PROTEIN_CALLS_SPLITMERGECALL_H_INCLUDED
#define MEGAMOL_PROTEIN_CALLS_SPLITMERGECALL_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/Array.h"
#include "vislib/Pair.h"
#include "vislib/PtrArray.h"
#include "vislib/String.h"
#include "vislib/forceinline.h"
#include "vislib/math/Vector.h"

namespace megamol {
namespace protein_calls {

/**
 * Base class for graph calls and data interfaces.
 *
 * Graphs based on coordinates can contain holes where the respective
 * getters return false for the abscissae. For categorical graphs this
 * seems useless as the abscissa is sparse anyway, but the interface
 * allows for that as well.
 */

class SplitMergeCall : public megamol::core::Call {
public:
    /**
     * The guide types that can be associated with a diagram.
     */
    enum GuideTypes { SPLITMERGE_GUIDE_VERTICAL = 0, SPLITMERGE_GUIDE_HORIZONTAL = 1 };

    class SplitMergeGuide {
    public:
        /** Ctor. */
        SplitMergeGuide(float position, GuideTypes type) : pos(position), type(type) {}

        /** Dtor. */
        ~SplitMergeGuide() {}

        /**
         * Return the position where the guide should be placed.
         *
         * @return the position where the guide is placed
         */
        VISLIB_FORCEINLINE float GetPosition() const {
            return pos;
        }

        /**
         * Return the type of this guide to choose a suitable
         * representation.
         *
         * @return the type of the guide
         */
        VISLIB_FORCEINLINE GuideTypes GetType() const {
            return type;
        }

        /**
         * Set the position where the guide should be placed.
         *
         * @param position the position
         */
        VISLIB_FORCEINLINE void SetPosition(float position) {
            this->pos = position;
        }

        /**
         * Set the type or visual representation of this guide.
         *
         * @param type the type
         */
        VISLIB_FORCEINLINE void SetType(GuideTypes type) {
            this->type = type;
        }

        /**
         * Equality operator for guides. Compares type and index.
         *
         * @param other the guide to compare this to
         *
         * @return whether this and other are equal
         */
        VISLIB_FORCEINLINE bool operator==(const SplitMergeGuide& other) const {
            return this->pos == other.pos && other.type == this->type;
        }

    private:
        float pos;
        GuideTypes type;
    };

    /**
     * Interface for ensuring that an arbitrary data source can be used
     * with a diagram. It transforms the input data into a tuple of multiple
     * abscissae (categoric or float) plus a single ordinate.
     * Asking a categorical abscissa for its float value and vice versa
     * results in undefined behavior.
     */
    class SplitMergeMappable {
    public:
        /**
         * Return the number of data points.
         *
         * @return the number of data points
         */
        virtual int GetDataCount() const = 0;

        /**
         * Return the float value of abscissa number abscissaIndex at
         * data point index. If abscissaIndex is categorical, the
         * result is undefined. If there is no value for index,
         * false is returned.
         *
         * @param index the index of the data point
         * @param abscissaIndex the index of the abscissa
         * @param[out] value returns the abscissa value
         *
         * @return whether there was a value
         */
        virtual bool GetAbscissaValue(const SIZE_T index, float* value) const = 0;

        /**
         * Return the ordinate at data point index. If there
         * is no value for index, returns 0.0f.
         *
         * @return the ordinate at index or 0.0f in non-existent
         */
        virtual float GetOrdinateValue(const SIZE_T index) const = 0;

        /**
         * Return the range [min,max] covered by abscissa at abscissaIndex
         *
         * @param abscissaIndex the abscissa to get the range for
         *
         * @return a pair of floats representing min and max
         */
        virtual vislib::Pair<float, float> GetAbscissaRange() const = 0;

        /**
         * Return the range [min,max] covered by the ordinate
         *
         * @return a pair of floats representing min and max
         */
        virtual vislib::Pair<float, float> GetOrdinateRange() const = 0;
    };

    /**
     * A series of data points that are to be represented by a diagram. The
     * data source proper is encapsulated in mappable, while the 'data space'
     * markers and parameters are set here.
     */
    class SplitMergeSeries {
    public:
        /** Ctor. */
        SplitMergeSeries(vislib::StringA name, SplitMergeMappable* mappable) : mappable(mappable), visible(true) {
            this->color = new vislib::math::Vector<float, 4>(0.5f, 0.5f, 0.5f, 1.0f);
            this->name = new vislib::StringA();
            *this->name = name;
        }

        /** Dtor. */
        ~SplitMergeSeries() {
            delete this->color;
            delete this->name;
        }

        /**
         * Answer the color of this series. This should be used to
         * represent it graphically.
         *
         * @return the four-component color of the series
         */
        VISLIB_FORCEINLINE const vislib::math::Vector<float, 4> GetColor() const {
            return *this->color;
        }

        /**
         * Answer the color of this series. This should be used to
         * represent it graphically.
         *
         * @return the three-component color of the series
         */
        VISLIB_FORCEINLINE const vislib::math::Vector<float, 3> GetColorRGB() const {
            return vislib::math::Vector<float, 3>(this->color->X(), this->color->Y(), this->color->Z());
        }

        /**
         * Get the mappable associated with the series, i.e. the raw
         * data source that is queried via the SplitMergeMappable interface.
         *
         * @return the mappable
         */
        VISLIB_FORCEINLINE SplitMergeMappable* GetMappable() {
            return this->mappable;
        }

        /**
         * Answer the name of the series. This also identifies a series.
         * See SplitMergeSeries::operator ==.
         *
         * @return the name of the series.
         */
        VISLIB_FORCEINLINE const vislib::StringA GetName() const {
            return *this->name;
        }

        /**
         * Answer whether this series is visible.
         *
         * @return the visibility of the series.
         */
        VISLIB_FORCEINLINE const bool GetVisible() const {
            return this->visible;
        }

        /**
         * Set the color the series should be represented with.
         *
         * @param r the red component of the color
         * @param g the green component of the color
         * @param b the blue component of the color
         */
        VISLIB_FORCEINLINE void SetColor(const float r, const float g, const float b) {
            this->color->Set(r, g, b, 1.0f);
        }

        /**
         * Set the color the series should be represented with.
         *
         * @param r the red component of the color
         * @param g the green component of the color
         * @param b the blue component of the color
         * @param a the alpha component of the color
         */
        VISLIB_FORCEINLINE void SetColor(const float r, const float g, const float b, const float a) {
            this->color->Set(r, g, b, a);
        }

        /**
         * Manipulate visibility of this SplitMergeSeries
         *
         * @param visible whether it is visible
         */
        VISLIB_FORCEINLINE void SetVisible(bool visible) {
            this->visible = visible;
        }

        /**
         * Set the mappable, i.e. the raw data source of the data.
         * The mappable is still owned by the caller!
         *
         * @param mappable the mappable
         */
        VISLIB_FORCEINLINE void SetMappable(SplitMergeMappable* mappable) {
            this->mappable = mappable;
        }

        /**
         * Set the name and identifier of the series. See also
         * SplitMergeSeries::operator ==.
         *
         * @param name the name to set.
         */
        VISLIB_FORCEINLINE void SetName(const vislib::StringA name) {
            *this->name = name;
        }

        /**
         * Equality operator for the series. Currently this only uses
         * the name(=identifier).
         *
         * @param other the series to compare to
         *
         * @return whether this and other have the same name.
         */
        VISLIB_FORCEINLINE bool operator==(const SplitMergeSeries& other) const {
            return this->name->Equals(*other.name);
        }

    private:
        /** the color of the series */
        vislib::math::Vector<float, 4>* color;

        /** the name of the series */
        vislib::StringA* name;

        /** the mappable associated with the series */
        SplitMergeMappable* mappable;

        /** whether this series is visible */
        bool visible;
    };

    /**
     * A transition which connects two data points in two series.
     */
    class SplitMergeTransition {
    public:
        /** Ctor. */
        SplitMergeTransition(unsigned int sourceSeries, unsigned int sourceSeriesIdx, float sourceWidth,
            unsigned int destinationSeries, unsigned int destinationSeriesIdx, float destinationWidth);

        /** Dtor. */
        ~SplitMergeTransition() {}

        /**
         * Get the index of the source series.
         *
         * @return the source series
         */
        VISLIB_FORCEINLINE const unsigned int SourceSeries() const {
            return this->srcSeries;
        }

        /**
         * Get the index of the destination series.
         *
         * @return the destination series
         */
        VISLIB_FORCEINLINE const unsigned int DestinationSeries() const {
            return this->dstSeries;
        }

        /**
         * Get the index of the data point in the source series.
         *
         * @return the source series
         */
        VISLIB_FORCEINLINE const unsigned int SourceSeriesDataIndex() const {
            return this->srcSeriesIdx;
        }

        /**
         * Get the index of the data point in the destination series.
         *
         * @return the destination series
         */
        VISLIB_FORCEINLINE const unsigned int DestinationSeriesDataIndex() const {
            return this->dstSeriesIdx;
        }

        /**
         * Get the transition width at the data point in the source series.
         *
         * @return the source width
         */
        VISLIB_FORCEINLINE const float SourceWidth() const {
            return this->srcWidth;
        }

        /**
         * Get the transition width at the data point in the destination series.
         *
         * @return the destination width
         */
        VISLIB_FORCEINLINE const float DestinationWidth() const {
            return this->dstWidth;
        }

        /**
         * Answer whether this transition is visible.
         *
         * @return the visibility of the transition.
         */
        VISLIB_FORCEINLINE const bool GetVisible() const {
            return this->visible;
        }

        /**
         * Manipulate visibility of this SplitMergeTransition
         *
         * @param visible whether it is visible
         */
        VISLIB_FORCEINLINE void SetVisible(bool visible) {
            this->visible = visible;
        }

        /**
         * Equality operator for the series. Currently this only uses
         * the series numbers and indices.
         *
         * @param other the series to compare to
         *
         * @return whether this and other have the same name.
         */
        VISLIB_FORCEINLINE bool operator==(const SplitMergeTransition& other) const {
            return (this->srcSeries == other.srcSeries && this->srcSeriesIdx == other.srcSeriesIdx &&
                    this->dstSeries == other.dstSeries && this->dstSeriesIdx == other.dstSeriesIdx);
        }

    private:
        /** the source series */
        unsigned int srcSeries;
        /** the index of the data point in the source series */
        unsigned int srcSeriesIdx;
        /** the width of the transition at the source */
        float srcWidth;
        /** the destination series */
        unsigned int dstSeries;
        /** the index of the data point in the destination series */
        unsigned int dstSeriesIdx;
        /** the width of the transition at the destination */
        float dstWidth;

        /** whether this series is visible */
        bool visible;
    };

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName(void) {
        return "SplitMergeCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description(void) {
        return "Call to get splitmerge data";
    }

    /** Index of the 'GetData' function */
    static const unsigned int CallForGetData;

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
     * @param idx The index of the function to return it's name.
     *
     * @return The name of the requested function.
     */
    static const char* FunctionName(unsigned int idx) {
        return "GetData";
    }

    /** Ctor. */
    SplitMergeCall(void);

    /** Dtor. */
    ~SplitMergeCall(void) override;

    /**
     * Add a diagram series to theData.
     *
     * @param series the series to add
     */
    VISLIB_FORCEINLINE void AddSeries(SplitMergeSeries* series) {
        this->theData->Append(series);
    }

    /**
     * Remove a diagram series from theData.
     *
     * @param series the series to remove.
     */
    VISLIB_FORCEINLINE void DeleteSeries(SplitMergeSeries* series) {
        this->theData->Remove(series);
    }

    /**
     * Return a diagram series stored at index.
     *
     * @param index the diagram series index inside theData.
     *
     * @return the diagram series at index.
     */
    VISLIB_FORCEINLINE SplitMergeSeries* GetSeries(SIZE_T index) const {
        return this->theData->operator[](index);
    }

    /**
     * Answer the number of diagram series in this call.
     *
     * @return the number of diagram series
     */
    VISLIB_FORCEINLINE const SIZE_T GetSeriesCount() const {
        return this->theData->Count();
    }

    /**
     * Add a transition to theTransitions.
     *
     * @param trans the transition to add
     */
    VISLIB_FORCEINLINE void SetTransitions(vislib::PtrArray<SplitMergeTransition>* trans) {
        this->theTransitions = trans;
    }

    /**
     * Return a transition stored at index.
     *
     * @param index the transition index inside theTransitions.
     *
     * @return the transition at index.
     */
    VISLIB_FORCEINLINE SplitMergeTransition* GetTransition(SIZE_T index) const {
        return (*this->theTransitions)[index];
    }

    /**
     * Answer the number of transitions in this call.
     *
     * @return the number of transitions
     */
    VISLIB_FORCEINLINE const SIZE_T GetTransitionCount() const {
        if (!this->theTransitions)
            return 0;
        return this->theTransitions->Count();
    }

    /**
     * Add a guide to guides.
     *
     * @param position where the guide should be placed
     * @param type the type of guide
     */
    VISLIB_FORCEINLINE void AddGuide(float position, GuideTypes type) {
        this->AddGuide(new SplitMergeGuide(position, type));
    }

    /**
     * Add a guide to guides.
     *
     * @param g the guide to add
     */
    VISLIB_FORCEINLINE void AddGuide(SplitMergeGuide* g) {
        this->guides->Append(g);
    }

    /**
     * Clear all guides stored in this call.
     */
    VISLIB_FORCEINLINE void ClearGuides() {
        this->guides->Clear();
    }

    /**
     * Retrieve a specific guide associated with the call.
     *
     * @param index the index the guide is stored at
     *
     * @return A pointer to the marker. The memory is still owned by
     *         the series.
     */
    VISLIB_FORCEINLINE SplitMergeGuide* GetGuide(SIZE_T index) const {
        return this->guides->operator[](index);
    }

    /**
     * Answer the number of guides stored in the call.
     *
     * @return the number of guides.
     */
    VISLIB_FORCEINLINE const SIZE_T GetGuideCount() const {
        return this->guides->Count();
    }

private:
    /** Data store for all contained DiagramSeries */
    vislib::Array<SplitMergeSeries*>* theData;

    /** Data store for all contained SplitMergeTransitions */
    vislib::PtrArray<SplitMergeTransition>* theTransitions;

    /**
     * The guides associated with the diagram. They belong to the
     * diagram and will be deallocated alongside it.
     */
    vislib::PtrArray<SplitMergeGuide>* guides;
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<SplitMergeCall> SplitMergeCallDescription;

} /* end namespace protein_calls */
} /* end namespace megamol */

#endif /* MEGAMOL_PROTEIN_CALLS_SPLITMERGECALL_H_INCLUDED */
