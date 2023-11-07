#include "ParquetWriter.h"

#ifdef MEGAMOL_USE_POWER

#include "mmcore/utility/log/Log.h"

#include <arrow/io/file.h>
#include <arrow/util/key_value_metadata.h>
#include <parquet/api/reader.h>
#include <parquet/api/writer.h>

#include "ColumnNames.h"

namespace megamol::power {
template<typename T>
void BatchWriter(parquet::RowGroupWriter* writer, std::vector<T> const& data, int c_idx, std::size_t min_field_size) {
    if constexpr (std::is_same_v<T, float>) {
        auto float_writer = static_cast<parquet::FloatWriter*>(writer->column(c_idx));
        float_writer->WriteBatch(min_field_size, nullptr, nullptr, data.data());
    } else if (std::is_same_v<T, int64_t>) {
        auto int_writer = static_cast<parquet::Int64Writer*>(writer->column(c_idx));
        int_writer->WriteBatch(min_field_size, nullptr, nullptr, data.data());
    } else {
        throw std::invalid_argument("Unsupported type");
    }
}

void WriteMetaData(std::unique_ptr<parquet::ParquetFileWriter>& file_writer, MetaData const* meta) {
    auto md = std::make_shared<parquet::KeyValueMetadata>();
    md->Append(std::string("project_file"), meta->project_file);
    for (auto const& [key, value] : meta->oszi_configs) {
        md->Append(key, value);
    }
    md->Append(std::string("runtime_libraries"), meta->runtime_libs);
    for (auto const& [key, value] : meta->hardware_software_info) {
        md->Append(key, value);
    }
    for (auto const& [key, value] : meta->analysis_recipes) {
        md->Append(key, value);
    }
    file_writer->AddKeyValueMetadata(md);
}

void ParquetWriter(std::filesystem::path const& file_path, value_map_t const& values_map, MetaData const* meta) {
    using namespace parquet;
    using namespace parquet::schema;

    try {
        std::size_t min_field_size = 0;
        bool first_time = true;

        // create scheme
        NodeVector fields;
        fields.reserve(values_map.size());
        for (auto const& [name, v_values] : values_map) {
            if (std::holds_alternative<std::vector<float>>(v_values)) {
                fields.push_back(PrimitiveNode::Make(name, Repetition::REQUIRED, Type::FLOAT, ConvertedType::NONE));
                if (first_time) {
                    min_field_size = std::get<std::vector<float>>(v_values).size();
                    first_time = false;
                } else {
                    min_field_size = std::min(min_field_size, std::get<std::vector<float>>(v_values).size());
                }
            } else if (std::holds_alternative<std::vector<int64_t>>(v_values)) {
                fields.push_back(PrimitiveNode::Make(name, Repetition::REQUIRED, Type::INT64, ConvertedType::NONE));
                if (first_time) {
                    min_field_size = std::get<std::vector<int64_t>>(v_values).size();
                    first_time = false;
                } else {
                    min_field_size = std::min(min_field_size, std::get<std::vector<int64_t>>(v_values).size());
                }
            } else {
                throw std::runtime_error("Unexpected type");
            }
        }
        auto schema = std::static_pointer_cast<GroupNode>(GroupNode::Make("schema", Repetition::REQUIRED, fields));

        // open file
        std::shared_ptr<::arrow::io::FileOutputStream> file;
        PARQUET_ASSIGN_OR_THROW(file, ::arrow::io::FileOutputStream::Open(file_path.string()));

        // configure
        WriterProperties::Builder builder;
        builder.compression(Compression::BROTLI);
        auto props = builder.build();

        // create instance
        auto file_writer = ParquetFileWriter::Open(file, schema, props);

        // write meta data
        if (meta) {
            WriteMetaData(file_writer, meta);
        }

        // write data
        auto rg_writer = file_writer->AppendBufferedRowGroup();

        int c_idx = 0;
        for (auto const& [name, v_values] : values_map) {
            if (std::holds_alternative<std::vector<float>>(v_values)) {
                auto const& values = std::get<std::vector<float>>(v_values);
                BatchWriter(rg_writer, values, c_idx++, min_field_size);
            } else if (std::holds_alternative<std::vector<int64_t>>(v_values)) {
                auto const& values = std::get<std::vector<int64_t>>(v_values);
                BatchWriter(rg_writer, values, c_idx++, min_field_size);
            } else {
                throw std::runtime_error("Unexpected type");
            }
        }

        // close
        rg_writer->Close();
        file_writer->Close();

#ifdef _DEBUG
        std::unique_ptr<parquet::ParquetFileReader> reader = parquet::ParquetFileReader::OpenFile(file_path.string());
        PrintSchema(reader->metadata()->schema()->schema_root().get(), std::cout);
#endif
    } catch (std::exception const& ex) {
        core::utility::log::Log::DefaultLog.WriteError("[ParquetWriter]: %s", ex.what());
    }
}

void ParquetWriter(
    std::filesystem::path const& file_path, std::vector<SampleBuffer> const& buffers, MetaData const* meta) {
    using namespace parquet;
    using namespace parquet::schema;

    if (buffers.empty())
        return;

    try {
        std::size_t min_field_size = buffers[0].ReadSamples().size();

        // create scheme
        NodeVector fields;
        fields.reserve(buffers.size() * 2);
        for (auto const& b : buffers) {
            fields.push_back(PrimitiveNode::Make(
                b.Name() + "_" + global_samples_name, Repetition::REQUIRED, Type::FLOAT, ConvertedType::NONE));
            fields.push_back(PrimitiveNode::Make(
                b.Name() + "_" + global_ts_name, Repetition::REQUIRED, Type::INT64, ConvertedType::NONE));
            min_field_size = std::min(min_field_size, b.ReadSamples().size());
        }
        auto schema = std::static_pointer_cast<GroupNode>(GroupNode::Make("schema", Repetition::REQUIRED, fields));

        // open file
        std::shared_ptr<::arrow::io::FileOutputStream> file;
        PARQUET_ASSIGN_OR_THROW(file, ::arrow::io::FileOutputStream::Open(file_path.string()));

        // configure
        WriterProperties::Builder builder;
        builder.compression(Compression::BROTLI);
        auto props = builder.build();

        // create instance
        auto file_writer = ParquetFileWriter::Open(file, schema, props);

        // write meta data
        if (meta) {
            WriteMetaData(file_writer, meta);
        }

        // write data
        auto rg_writer = file_writer->AppendBufferedRowGroup();

        int c_idx = 0;
        for (auto const& b : buffers) {
            auto const& s_values = b.ReadSamples();
            BatchWriter(rg_writer, s_values, c_idx++, min_field_size);
            auto const& t_values = b.ReadTimestamps();
            BatchWriter(rg_writer, t_values, c_idx++, min_field_size);
        }

        // close
        rg_writer->Close();
        file_writer->Close();

#ifdef _DEBUG
        std::unique_ptr<parquet::ParquetFileReader> reader = parquet::ParquetFileReader::OpenFile(file_path.string());
        PrintSchema(reader->metadata()->schema()->schema_root().get(), std::cout);
#endif
    } catch (std::exception const& ex) {
        core::utility::log::Log::DefaultLog.WriteError("[ParquetWriter]: %s", ex.what());
    }
}
} // namespace megamol::power

#endif
