/*
 * DiagramCall.h
 *
 * Author: Guido Reina
 * Copyright (C) 2012 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */


#pragma once

#include "mmcore/Call.h"
#include "mmcore/factories/CallAutoDescription.h"
#include "vislib/Array.h"
#include "vislib/Pair.h"
#include "vislib/PtrArray.h"
#include "vislib/String.h"
#include "vislib/forceinline.h"
#include "vislib/math/Vector.h"


namespace megamol::protein_calls {

/**
 * Base class for graph calls and data interfaces.
 *
 * Graphs based on coordinates can contain holes where the respective
 * getters return false for the abscissae. For categorical graphs this
 * seems useless as the abscissa is sparse anyway, but the interface
 * allows for that as well.
 */

class DiagramCall : public megamol::core::Call {
public:
    /**
     * The guide types that can be associated with a diagram.
     */
    enum GuideTypes { DIAGRAM_GUIDE_VERTICAL = 0, DIAGRAM_GUIDE_HORIZONTAL = 1 };

    class DiagramGuide {
    public:
        /** Ctor. */
        DiagramGuide(float position, GuideTypes type) : pos(position), type(type) {}

        /** Dtor. */
        ~DiagramGuide() {}

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
        VISLIB_FORCEINLINE bool operator==(const DiagramGuide& other) const {
            return this->pos == other.pos && other.type == this->type;
        }

    private:
        float pos;
        GuideTypes type;
    };

    /**
     * The marker types that can be placed in a diagram. A visual
     * representation should be meaningful WRT these types.
     */
    enum MarkerTypes {
        DIAGRAM_MARKER_INVALID = 0,
        DIAGRAM_MARKER_DISAPPEAR = 1,
        DIAGRAM_MARKER_MERGE = 2,
        DIAGRAM_MARKER_SPLIT = 4,
        DIAGRAM_MARKER_BOOKMARK = 8,
        DIAGRAM_MARKER_EXIT = 16
    };

    /**
     * Represents some kind of Marker positioned identically to the data
     * at index. How the MarkerType is rendered graphically is up to the
     * implementing diagram. userData must NOT be an array. The userdata
     * ownership is passed to the marker and destroyed alongside it.
     */
    class DiagramMarker {
    public:
        /** Ctor. */
        DiagramMarker(SIZE_T index, MarkerTypes type, void* userData = NULL)
                : index(index)
                , type(type)
                , userData(userData) {
            tooltip = new vislib::StringA();
        }

        /** Dtor. */
        ~DiagramMarker() {
            delete userData;
            delete tooltip;
        }

        /**
         * Return the data index where the marker should be placed.
         *
         * @return the index at whose abscissa/ordinate the marker is
         *         placed
         */
        VISLIB_FORCEINLINE SIZE_T GetIndex() const {
            return index;
        }

        /**
         * Return the tooltip of this marker.
         *
         * @return the tooltip text
         */
        VISLIB_FORCEINLINE vislib::StringA GetTooltip() const {
            return *tooltip;
        }

        /**
         * Return the type of this marker to choose a suitable
         * representation.
         *
         * @return the type of the marker
         */
        VISLIB_FORCEINLINE MarkerTypes GetType() const {
            return type;
        }

        /**
         * Return the userdata associated with this marker.
         *
         * @return the userdata.
         */
        VISLIB_FORCEINLINE void* GetUserData() const {
            return userData;
        }

        /**
         * Set the data index where the marker should be placed.
         *
         * @param index the placement index
         */
        VISLIB_FORCEINLINE void SetIndex(SIZE_T index) {
            this->index = index;
        }

        /**
         * Set the tooltip of this marker.
         *
         * @param text the tooltip content.
         */
        VISLIB_FORCEINLINE void SetTooltip(vislib::StringA text) {
            *this->tooltip = text;
        }

        /**
         * Set the type or visual representation of this marker.
         *
         * @param type the type
         */
        VISLIB_FORCEINLINE void SetType(MarkerTypes type) {
            this->type = type;
        }

        /**
         * Associate arbitrary userdata with this marker. Ownership
         * is passed as well!
         *
         * @param userData the userdata
         */
        VISLIB_FORCEINLINE void SetUserData(void* userData) {
            this->userData = userData;
        }

        /**
         * Equality operator for markers. Compares type and index.
         *
         * @param other the marker to compare this to
         *
         * @return whether this and other are equal
         */
        VISLIB_FORCEINLINE bool operator==(const DiagramMarker& other) const {
            return this->index == other.index && other.type == this->type;
        }

    protected:
    private:
        /** the index of the marker */
        SIZE_T index;

        /** the type of the marker */
        MarkerTypes type;

        /** a short desciption for hovering over */
        vislib::StringA* tooltip;

        /** arbitrary data the user can append */
        void* userData;
    };

    /**
     * Interface for ensuring that an arbitrary data source can be used
     * with a diagram. It transforms the input data into a tuple of multiple
     * abscissae (categoric or float) plus a single ordinate.
     * Asking a categorical abscissa for its float value and vice versa
     * results in undefined behavior.
     */
    class DiagramMappable {
    public:
        /**
         * Return the number of abscissae. There might be some diagram
         * types that can stack several abscissae. Abscissae are ordered
         * from fastest-changing (idx 0) to slowest-changing
         * (idx GetAbscissaeCount)
         *
         * @return the number of abscissae
         */
        virtual int GetAbscissaeCount() const = 0;

        /**
         * Return the number of data points.
         *
         * @return the number of data points
         */
        virtual int GetDataCount() const = 0;

        /**
         * Answer whether the abscissa number abscissa is categorical.
         *
         * @param abscissa the index of the abscissa
         *
         * @return whether abscissa is categorical.
         */
        virtual bool IsCategoricalAbscissa(const SIZE_T abscissa) const = 0;

        /**
         * Return the string value of abscissa number abscissaIndex at
         * data point index. If abscissaIndex is not categorical, the
         * result is undefined. If there is no value for index,
         * false is returned.
         *
         * @param index the index of the data point
         * @param abscissaIndex the index of the abscissa
         * @param[out] category returns the category name
         *
         * @return whether there was a value
         */
        virtual bool GetAbscissaValue(
            const SIZE_T index, const SIZE_T abscissaIndex, vislib::StringA* category) const = 0;

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
        virtual bool GetAbscissaValue(const SIZE_T index, const SIZE_T abscissaIndex, float* value) const = 0;

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
        virtual vislib::Pair<float, float> GetAbscissaRange(const SIZE_T abscissaIndex) const = 0;

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
    class DiagramSeries {
    public:
        /** Ctor. */
        DiagramSeries(vislib::StringA name, DiagramMappable* mappable)
                : mappable(mappable)
        /*markers(), visible(true)*/ {
            this->color = new vislib::math::Vector<float, 4>(0.5f, 0.5f, 0.5f, 1.0f);
            this->name = new vislib::StringA(name);
            this->markers = new vislib::PtrArray<DiagramMarker>();
        }

        /** Dtor. */
        ~DiagramSeries() {
            delete this->color;
            delete this->markers;
        }

        /**
         * Add a marker to markers.
         *
         * @param index the index of the data point this marker
         *              is associated with
         * @param type  the type of marker that should be presented,
         *              see DiagramCall::MarkerTypes
         */
        VISLIB_FORCEINLINE void AddMarker(SIZE_T index, MarkerTypes type) {
            this->AddMarker(new DiagramMarker(index, type));
        }

        /**
         * Add a marker to markers. Note that this instance is
         * deallocated upon destruction of the DiagramSeries!
         *
         * @param m the marker to add and pass ownership of to the series
         */
        VISLIB_FORCEINLINE void AddMarker(DiagramMarker* m) {
            this->markers->Append(m);
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
         * Clear all markers stored in this series.
         */
        VISLIB_FORCEINLINE void ClearMarkers() {
            this->markers->Clear();
        }

        /**
         * Get the mappable associated with the series, i.e. the raw
         * data source that is queried via the DiagramMappable interface.
         *
         * @return the mappable
         */
        VISLIB_FORCEINLINE DiagramMappable* GetMappable() {
            return this->mappable;
        }

        /**
         * Retrieve a specific marker associated with the series.
         *
         * @param index the index the marker is stored at
         *
         * @return A pointer to the marker. The memory is still owned by
         *         the series.
         */
        VISLIB_FORCEINLINE DiagramMarker* GetMarker(SIZE_T index) const {
            return this->markers->operator[](index);
        }

        /**
         * Answer the number of markers stored in the series.
         *
         * @return the number of markers.
         */
        VISLIB_FORCEINLINE const SIZE_T GetMarkerCount() const {
            return this->markers->Count();
        }

        /**
         * Answer the name of the series. This also identifies a series.
         * See DiagramSeries::operator ==.
         *
         * @return the name of the series.
         */
        VISLIB_FORCEINLINE const vislib::StringA GetName() const {
            return *this->name;
        }

        ///**
        // * Answer whether this series is visible.
        // *
        // * @return the visibility of the series.
        // */
        //VISLIB_FORCEINLINE const bool GetVisible() const {
        //    return this->visible;
        //}

        /**
         * Set the color the series should be represented with.
         *
         * @param r the red component of the color
         * @param g the green component of the color
         * @param b the blue component of the color
         */
        VISLIB_FORCEINLINE void SetColor(const vislib::math::Vector<float, 4> col) {
            this->color->Set(col.X(), col.Y(), col.Z(), col.W());
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

        ///**
        // * Manipulate visibility of this DiagramSeries
        // *
        // * @param visible whether it is visible
        // */
        //VISLIB_FORCEINLINE void SetVisible(bool visible) {
        //    this->visible = visible;
        //}

        /**
         * Set the mappable, i.e. the raw data source of the data.
         * The mappable is still owned by the caller!
         *
         * @param mappable the mappable
         */
        VISLIB_FORCEINLINE void SetMappable(DiagramMappable* mappable) {
            this->mappable = mappable;
        }

        /**
         * Set the name and identifier of the series. See also
         * DiagramSeries::operator ==.
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
        VISLIB_FORCEINLINE bool operator==(const DiagramSeries& other) const {
            return this->name->Equals(*other.name);
        }

    private:
        /** the color of the series */
        vislib::math::Vector<float, 4>* color;

        /** the name of the series */
        vislib::StringA* name;

        /**
         * The markers associated with the series. They belong to the
         * series and will be deallocated alongside it.
         */
        vislib::PtrArray<DiagramMarker>* markers;

        /** the mappable associated with the series */
        DiagramMappable* mappable;

        /** whether this series is visible */
        //bool visible;
    };

    /**
     * Answer the name of the objects of this description.
     *
     * @return The name of the objects of this description.
     */
    static const char* ClassName() {
        return "DiagramCall";
    }

    /**
     * Gets a human readable description of the module.
     *
     * @return A human readable description of the module.
     */
    static const char* Description() {
        return "Call to get diagram data";
    }

    /** Index of the 'GetData' function */
    static const unsigned int CallForGetData;

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
        return "GetData";
    }

    /** Ctor. */
    DiagramCall();

    /** Dtor. */
    ~DiagramCall() override;

    /**
     * Add a guide to guides.
     *
     * @param position where the guide should be placed
     * @param type the type of guide
     */
    VISLIB_FORCEINLINE void AddGuide(float position, GuideTypes type) {
        this->AddGuide(new DiagramGuide(position, type));
    }

    /**
     * Add a guide to guides.
     *
     * @param g the guide to add
     */
    VISLIB_FORCEINLINE void AddGuide(DiagramGuide* g) {
        this->guides->Append(g);
    }

    /**
     * Add a diagram series to theData.
     *
     * @param series the series to add
     */
    VISLIB_FORCEINLINE void AddSeries(DiagramSeries* series) {
        this->theData->Append(series);
    }

    /**
     * Clear all guides stored in this call.
     */
    VISLIB_FORCEINLINE void ClearGuides() {
        this->guides->Clear();
    }

    /**
     * Remove a diagram series from theData.
     *
     * @param series the series to remove.
     */
    VISLIB_FORCEINLINE void DeleteSeries(DiagramSeries* series) {
        this->theData->Remove(series);
    }

    /**
     * Retrieve a specific guide associated with the call.
     *
     * @param index the index the guide is stored at
     *
     * @return A pointer to the marker. The memory is still owned by
     *         the series.
     */
    VISLIB_FORCEINLINE DiagramGuide* GetGuide(SIZE_T index) const {
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

    /**
     * Return a diagram series stored at index.
     *
     * @param index the diagram series index inside theData.
     *
     * @return the diagram series at index.
     */
    VISLIB_FORCEINLINE DiagramSeries* GetSeries(SIZE_T index) const {
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

private:
    /** Data store for all contained DiagramSeries */
    vislib::Array<DiagramSeries*>* theData;

    /**
     * The guides associated with the diagram. They belong to the
     * diagram and will be deallocated alongside it.
     */
    vislib::PtrArray<DiagramGuide>* guides;
};

/** Description class typedef */
typedef megamol::core::factories::CallAutoDescription<DiagramCall> DiagramCallDescription;

} // namespace megamol::protein_calls
