/*
 * CallFloatTableData.h
 *
 * Copyright (C) 2015-2016 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_DATATOOLS_CallFloatTableData_H_INCLUDED
#define MEGAMOL_DATATOOLS_CallFloatTableData_H_INCLUDED
#pragma once

#include "mmstd_datatools/mmstd_datatools.h"
#include "mmcore/AbstractGetDataCall.h"
#include "vislib/String.h"
#include "mmcore/factories/CallAutoDescription.h"
#include <string>
#include <type_traits>
#include "vislib/macro_utils.h"

namespace megamol {
namespace stdplugin {
namespace datatools {
namespace floattable {

    class MMSTD_DATATOOLS_API CallFloatTableData : public core::AbstractGetDataCall {
    public:

        static const char *ClassName(void) { return "CallFloatTableData"; }
        static const char *Description(void) { return "Data of a table of floats"; }
        static unsigned int FunctionCount(void) { return 2; }
        static const char * FunctionName(unsigned int idx) { 
            switch (idx) {
            case 0: return "GetData";
            case 1: return "GetHash";
            }
            return nullptr;
        }

        enum class ColumnType {
            CATEGORICAL,
            QUANTITATIVE
        };

        class MMSTD_DATATOOLS_API ColumnInfo {
        public:
            ColumnInfo();
            ColumnInfo(const ColumnInfo& src);
            ~ColumnInfo();
            ColumnInfo& operator=(const ColumnInfo& rhs);
            bool operator==(const ColumnInfo& rhs) const;

            inline const std::string& Name(void) const { return name; }
            inline ColumnType Type(void) const { return type; }
            inline float MinimumValue(void) const { return minVal; }
            inline float MaximumValue(void) const { return maxVal; }

            inline ColumnInfo& SetName(const std::string& n) {
                name = n;
                return *this;
            }
            inline ColumnInfo& SetType(ColumnType t) {
                type = t;
                return *this;
            }
            inline ColumnInfo& SetMinimumValue(float v) {
                minVal = v;
                return *this;
            }
            inline ColumnInfo& SetMaximumValue(float v) {
                maxVal = v;
                return *this;
            }

        private:
            VISLIB_MSVC_SUPPRESS_WARNING(4251)
            std::string name;
            ColumnType type;
            float minVal;
            float maxVal;
        };

        CallFloatTableData(void);
        virtual ~CallFloatTableData(void);

        inline size_t GetColumnsCount(void) const {
            return columns_count;
        }
        inline size_t GetRowsCount(void) const {
            return rows_count;
        }
        inline const ColumnInfo* GetColumnsInfos(void) const {
            return columns;
        }
        inline const float* GetData(void) const {
            return data;
        }
        inline const float* GetData(size_t row) const {
            assert(row >= 0);
            assert(row < rows_count);
            return data + row * columns_count;
        }
        inline float GetData(size_t col, size_t row) const {
            assert(col >= 0);
            assert(col < columns_count);
            assert(row >= 0);
            assert(row < rows_count);
            return data[col + row * columns_count];
        }

        inline void Set(size_t col_cnt, size_t row_cnt, const ColumnInfo* info, const float* d) {
            columns_count = col_cnt;
            rows_count = row_cnt;
            columns = info;
            data = d;
        }
        
        inline size_t GetFirstCategoricalColumnIndex() const {
            for (size_t i = 0; i < columns_count; ++i) {
                if (columns[i].Type() == ColumnType::CATEGORICAL) {
                    return i;
                }
            }
            
            return -1;
        }

    private:

        size_t columns_count;
        size_t rows_count;
        const ColumnInfo *columns;
        const float *data; // data is stored row major order, aka array of structs

    };

    typedef core::factories::CallAutoDescription<CallFloatTableData> CallFloatTableDataDescription;

} /* end namespace floattable */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif /* MEGAMOL_LSP_DATA_CallFloatTableData_H_INCLUDED */
