#include "DiagramSeries.h"

#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/ColourParser.h"
#include "vislib/StringConverter.h"

using namespace megamol::infovis;
using namespace megamol::datatools;


/*
 * DiagramSeries::DiagramSeries
 */
DiagramSeries::DiagramSeries(void)
        : core::Module()
        , seriesInSlot("seriesIn", "Input of series for passthrough")
        , seriesOutSlot("seriesOut", "Output of owned series")
        , ftInSlot("ftIn", "Input table to select series from")
        , columnSelectorParam("columnSelector", "Param to select specific column from the table")
        , scalingParam("scaling", "Param to set a scaling factor for selected column")
        , colorParam("color", "Selecto color for series")
        , myHash(0)
        , inputHash((std::numeric_limits<size_t>::max)()) {
    this->color = {1.0f, 1.0f, 1.0f};

    this->seriesInSlot.SetCompatibleCall<DiagramSeriesCallDescription>();
    this->MakeSlotAvailable(&this->seriesInSlot);

    this->seriesOutSlot.SetCallback(DiagramSeriesCall::ClassName(),
        DiagramSeriesCall::FunctionName(DiagramSeriesCall::CallForGetSeries), &DiagramSeries::seriesSelectionCB);
    this->MakeSlotAvailable(&this->seriesOutSlot);

    this->ftInSlot.SetCompatibleCall<table::TableDataCallDescription>();
    this->MakeSlotAvailable(&this->ftInSlot);

    core::param::FlexEnumParam* columnSelectorEP = new core::param::FlexEnumParam("undef");
    this->columnSelectorParam << columnSelectorEP;
    this->MakeSlotAvailable(&this->columnSelectorParam);

    this->scalingParam << new core::param::FloatParam(1.0f, (std::numeric_limits<float>::min)());
    this->MakeSlotAvailable(&this->scalingParam);

    this->colorParam << new core::param::StringParam(
        core::utility::ColourParser::ToString(this->color[0], this->color[1], this->color[2]).PeekBuffer());
    this->MakeSlotAvailable(&this->colorParam);
}


/*
 * DiagramSeries::~DiagramSeries
 */
DiagramSeries::~DiagramSeries(void) {
    this->Release();
}


/*
 * DiagramSeries::create
 */
bool DiagramSeries::create(void) {
    return true;
}


/*
 * DiagramSeries::release
 */
void DiagramSeries::release(void) {}


/*
 * DiagramSeries::seriesSelectionCB
 */
bool megamol::infovis::DiagramSeries::seriesSelectionCB(core::Call& c) {
    try {
        DiagramSeriesCall* outSeries = dynamic_cast<DiagramSeriesCall*>(&c);
        if (outSeries == NULL)
            return false;

        DiagramSeriesCall* inSeries = this->seriesInSlot.CallAs<DiagramSeriesCall>();

        table::TableDataCall* ft = this->ftInSlot.CallAs<table::TableDataCall>();
        if (ft == NULL)
            return false;
        if (!(*ft)(1))
            return false;
        if (!(*ft)(0))
            return false;

        if (!assertData(ft))
            return false;

        //(*(outSeries->GetSeriesInsertionCB()))(this->series);
        outSeries->GetSeriesInsertionCB()(this->series);

        if (inSeries != NULL) {
            *inSeries = *outSeries;

            if (!(*inSeries)(DiagramSeriesCall::CallForGetSeries)) {
                return false;
            }
        }
    } catch (...) {
        return false;
    }

    return true;
}


/*
 * DiagramSeries::assertData
 */
bool DiagramSeries::assertData(const table::TableDataCall* const ft) {
    if (this->inputHash == ft->DataHash() && !isAnythingDirty())
        return true;

    if (this->inputHash != ft->DataHash()) {
        this->columnSelectorParam.Param<core::param::FlexEnumParam>()->ClearValues();

        for (size_t i = 0; i < ft->GetColumnsCount(); i++) {
            auto name = ft->GetColumnsInfos()[i].Name();

            this->columnSelectorParam.Param<core::param::FlexEnumParam>()->AddValue(name.c_str());
        }
    }

    auto colname = this->columnSelectorParam.Param<core::param::FlexEnumParam>()->ValueString();

    uint32_t colIdx = 0;
    if (!this->getColumnIdx(colIdx, colname.c_str(), ft))
        return false;

    core::utility::ColourParser::FromString(this->colorParam.Param<core::param::StringParam>()->Value().c_str(),
        this->color[0], this->color[1], this->color[2]);

    this->series = std::make_tuple(0, colIdx, std::string(T2A(colname)),
        this->scalingParam.Param<core::param::FloatParam>()->Value(), this->color);

    /*this->series.col = colIdx;
    this->series.name = std::string(T2A(colname));
    this->series.scaling = this->scalingParam.Param<core::param::FloatParam>()->Value();*/

    this->resetDirtyFlags();
    this->myHash++;
    this->inputHash = ft->DataHash();

    return true;
}


/*
 * DiagramSeries::isAnythingDirty
 */
bool DiagramSeries::isAnythingDirty(void) const {
    return this->columnSelectorParam.IsDirty() || this->scalingParam.IsDirty() || this->colorParam.IsDirty();
}

/*
 * DiagramSeries::resetDirtyFlags
 */
void DiagramSeries::resetDirtyFlags(void) {
    this->columnSelectorParam.ResetDirty();
    this->scalingParam.ResetDirty();
    this->colorParam.ResetDirty();
}


/*
 * DiagramSeries::getColumnIdx
 */
bool DiagramSeries::getColumnIdx(
    uint32_t& colIdx, const vislib::TString& columnName, const table::TableDataCall* const ft) const {
    std::string name = std::string(T2A(columnName));

    for (size_t i = 0; i < ft->GetColumnsCount(); i++) {
        if (ft->GetColumnsInfos()[i].Name().compare(name) == 0) {
            colIdx = i;
            return true;
        }
    }

    return false;
}
