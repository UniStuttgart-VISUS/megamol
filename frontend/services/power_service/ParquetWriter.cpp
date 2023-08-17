#include "ParquetWriter.h"

#ifdef MEGAMOL_USE_POWER

#include "mmcore/utility/log/Log.h"

#include <arrow/io/file.h>
#include <parquet/api/reader.h>
#include <parquet/api/writer.h>

namespace megamol::power {
void ParquetWriter(std::filesystem::path const& file_path,
    std::unordered_map<std::string, std::variant<std::vector<float>, std::vector<int64_t>>> const& values_map) {
    using namespace parquet;
    using namespace parquet::schema;

    try {
        // create scheme
        NodeVector fields;
        fields.reserve(values_map.size());
        for (auto const& [name, v_values] : values_map) {
            if (std::holds_alternative<std::vector<float>>(v_values)) {
                fields.push_back(PrimitiveNode::Make(name, Repetition::REQUIRED, Type::FLOAT, ConvertedType::NONE));
            } else if (std::holds_alternative<std::vector<int64_t>>(v_values)) {
                fields.push_back(PrimitiveNode::Make(name, Repetition::REQUIRED, Type::INT64, ConvertedType::NONE));
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

        // write data
        auto rg_writer = file_writer->AppendBufferedRowGroup();

        int c_idx = 0;
        for (auto const& [name, v_values] : values_map) {
            if (std::holds_alternative<std::vector<float>>(v_values)) {
                auto const& values = std::get<std::vector<float>>(v_values);
                auto float_writer = static_cast<FloatWriter*>(rg_writer->column(c_idx++));
                float_writer->WriteBatch(values.size(), nullptr, nullptr, values.data());
            } else if (std::holds_alternative<std::vector<int64_t>>(v_values)) {
                auto const& values = std::get<std::vector<int64_t>>(v_values);
                auto int_writer = static_cast<Int64Writer*>(rg_writer->column(c_idx++));
                int_writer->WriteBatch(values.size(), nullptr, nullptr, values.data());
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

void ParquetWriter(std::filesystem::path const& file_path, std::vector<SampleBuffer> const& buffers) {
    using namespace parquet;
    using namespace parquet::schema;

    if (buffers.empty())
        return;

    try {
        // create scheme
        NodeVector fields;
        fields.reserve(buffers.size() * 3);
        for (auto const& b : buffers) {
            fields.push_back(
                PrimitiveNode::Make(b.Name() + "_samples", Repetition::REQUIRED, Type::FLOAT, ConvertedType::NONE));
            fields.push_back(
                PrimitiveNode::Make(b.Name() + "_ts", Repetition::REQUIRED, Type::INT64, ConvertedType::NONE));
            fields.push_back(
                PrimitiveNode::Make(b.Name() + "_wt", Repetition::REQUIRED, Type::INT64, ConvertedType::NONE));
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

        // write data
        auto rg_writer = file_writer->AppendBufferedRowGroup();

        int c_idx = 0;
        for (auto const& b : buffers) {
            auto const& s_values = b.ReadSamples();
            auto float_writer = static_cast<FloatWriter*>(rg_writer->column(c_idx++));
            float_writer->WriteBatch(s_values.size(), nullptr, nullptr, s_values.data());
            auto const& t_values = b.ReadTimestamps();
            auto int_writer = static_cast<Int64Writer*>(rg_writer->column(c_idx++));
            int_writer->WriteBatch(t_values.size(), nullptr, nullptr, t_values.data());
            auto const& w_values = b.ReadWalltimes();
            auto int_writer2 = static_cast<Int64Writer*>(rg_writer->column(c_idx++));
            int_writer2->WriteBatch(w_values.size(), nullptr, nullptr, w_values.data());
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
