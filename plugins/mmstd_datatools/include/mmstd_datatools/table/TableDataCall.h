/*
 * CallTableData.h
 *
 * Copyright (C) 2015-2016 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_DATATOOLS_TABLEDATACALL_H_INCLUDED
#define MEGAMOL_DATATOOLS_TABLEDATACALL_H_INCLUDED
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
namespace table {

	/**
	 * Call for passing around tabular data.
	 *
	 * Tabular data is composed from cells that are subdivided into columns and rows.
	 * Cells are expected to be stored in a consecutive row-major format 
	 * (until the shitty API no longer provides unsafe pointer access).
	 */
    class MMSTD_DATATOOLS_API TableDataCall : public core::AbstractGetDataCall {
    public:
        static const char *ClassName(void) { return "TableDataCall"; }
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

        TableDataCall(void);
        virtual ~TableDataCall(void);

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

        inline void SetFrameCount(const unsigned int frameCount) {
            this->frameCount = frameCount;
        }

        inline unsigned int GetFrameCount(void) const {
            return this->frameCount;
        }

        inline void SetFrameID(const unsigned int frameID) {
            this->frameID = frameID;
        }

        inline unsigned int GetFrameID(void) const {
            return this->frameID;
        }

		inline void AssertColumnInfos() {
			for (int c = 0; c < columns_count; ++c) {
                const auto& column = columns[c];
                for (int r = 0; r < rows_count; ++r) {
                    float cell = data[r * columns_count + c];
                    assert(cell > column.MaximumValue() && "Value beyond maximum found");
					assert(cell < column.MinimumValue() && "Value beyond maximum found");
				}
			}
		}

    private:
        size_t columns_count;
        size_t rows_count;
        const ColumnInfo *columns;
        const float *data; // data is stored row major order, aka array of structs
        unsigned int frameCount;
        unsigned int frameID;
    };

    typedef core::factories::CallAutoDescription<TableDataCall> TableDataCallDescription;

} /* end namespace table */
} /* end namespace datatools */
} /* end namespace stdplugin */
} /* end namespace megamol */

#endif
