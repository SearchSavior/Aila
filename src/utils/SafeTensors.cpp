#include "SafeTensors.hpp"
#include <iostream>
#include <stdexcept>
#include <algorithm>

// ============================================================
// ModelWeights 实现
// ============================================================

Tensor& ModelWeights::get(const std::string& name) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Weight not found: " + name);
    }
    return it->second;
}

const Tensor& ModelWeights::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Weight not found: " + name);
    }
    return it->second;
}

bool ModelWeights::has(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

void ModelWeights::put(const std::string& name, Tensor tensor) {
    tensors_.emplace(name, std::move(tensor));
}

void ModelWeights::replace(const std::string& name, Tensor tensor) {
    tensors_.erase(name);
    tensors_.emplace(name, std::move(tensor));
}

std::vector<std::string> ModelWeights::names() const {
    std::vector<std::string> result;
    result.reserve(tensors_.size());
    for (const auto& pair : tensors_) {
        result.push_back(pair.first);
    }
    std::sort(result.begin(), result.end());
    return result;
}

// ============================================================
// DataType -> oneDNN data_type 转换
// ============================================================

dnnl::memory::data_type to_dnnl_dtype(DataType dt) {
    switch (dt) {
        case DT_F16:  return dnnl::memory::data_type::f16;
        case DT_BF16: return dnnl::memory::data_type::bf16;
        case DT_F32:  return dnnl::memory::data_type::f32;
        case DT_F64:  return dnnl::memory::data_type::f64;
        case DT_U8:   return dnnl::memory::data_type::u8;
        case DT_S8:   return dnnl::memory::data_type::s8;
        default:
            throw std::runtime_error("Unsupported data type");
    }
}

// ============================================================
// format_tag 辅助
// ============================================================

static dnnl::memory::format_tag get_format_tag(int ndims) {
    switch (ndims) {
        case 1: return dnnl::memory::format_tag::a;
        case 2: return dnnl::memory::format_tag::ab;
        case 3: return dnnl::memory::format_tag::abc;
        case 4: return dnnl::memory::format_tag::abcd;
        case 5: return dnnl::memory::format_tag::abcde;
        case 6: return dnnl::memory::format_tag::abcdef;
        default: return dnnl::memory::format_tag::a;
    }
}

// ============================================================
// 解析 safetensors JSON header
// ============================================================

std::vector<std::string> ParseHeader(const std::string& header,
                                     std::unordered_map<std::string, TensorMeta>& metadata) {
    std::vector<std::string> layer_names;
    try {
        simdjson::padded_string padded_json(header);
        simdjson::ondemand::parser parser;
        simdjson::ondemand::document doc = parser.iterate(padded_json);

        for (auto field : doc.get_object()) {
            std::string_view key = field.unescaped_key();
            if (key == "__metadata__") continue;

            simdjson::ondemand::object tensor_info = field.value().get_object();
            std::string_view dtype = tensor_info["dtype"];

            TensorMeta meta;
            meta.shape.ndims = 0;

            // dtype 转换
            if (dtype == "F16" || dtype == "float16") {
                meta.dtype = DT_F16;
            } else if (dtype == "BF16" || dtype == "bfloat16") {
                meta.dtype = DT_BF16;
            } else if (dtype == "F32" || dtype == "float32") {
                meta.dtype = DT_F32;
            } else if (dtype == "F64" || dtype == "float64") {
                meta.dtype = DT_F64;
            } else if (dtype == "U8" || dtype == "uint8") {
                meta.dtype = DT_U8;
            } else if (dtype == "S8" || dtype == "int8") {
                meta.dtype = DT_S8;
            } else {
                meta.dtype = DT_UNKNOWN;
                std::cerr << "[Warning] Unknown dtype: " << dtype << std::endl;
            }

            for (int64_t dim : tensor_info["shape"].get_array()) {
                meta.shape.push_back(dim);
            }

            auto offsets = tensor_info["data_offsets"].get_array();
            auto it = offsets.begin();
            meta.byte_offset_start = static_cast<size_t>(int64_t(*it));
            ++it;
            meta.byte_offset_end = static_cast<size_t>(int64_t(*it));

            std::string layer_name(key);
            metadata[layer_name] = meta;
            layer_names.push_back(layer_name);
        }
    } catch (const std::exception& e) {
        std::cerr << "[ParseHeader] Failed: " << e.what() << std::endl;
    }
    return layer_names;
}

// ============================================================
// 将单个 tensor 从 mmap 加载到 GPU, 返回 Tensor 对象
// ============================================================

static Tensor LoadTensorToGPU(const uint8_t* mmap_base_ptr,
                               const TensorMeta& meta,
                               Context& ctx) {
    size_t size_in_bytes = meta.byte_offset_end - meta.byte_offset_start;
    const void* host_src_ptr = mmap_base_ptr + meta.byte_offset_start;

    // GPU 分配
    void* device_ptr = ctx.alloc_device(size_in_bytes);

    // Host -> Device
    ctx.memcpy_h2d(device_ptr, host_src_ptr, size_in_bytes);

    // 构造 shape vector
    std::vector<int64_t> shape(meta.shape.dims, meta.shape.dims + meta.shape.ndims);
    dnnl::memory::data_type dnnl_dtype = to_dnnl_dtype(meta.dtype);

    // 创建 dnnl memory descriptor + memory 对象
    dnnl::memory::dims dnnl_dims(shape.begin(), shape.end());
    auto format = get_format_tag(meta.shape.ndims);
    dnnl::memory::desc md(dnnl_dims, dnnl_dtype, format);

    dnnl::memory gpu_mem = dnnl::sycl_interop::make_memory(
        md, ctx.engine(),
        dnnl::sycl_interop::memory_kind::usm,
        device_ptr
    );

    // 创建 Tensor (from_dnnl 不拥有内存，但我们需要它拥有)
    // 使用 view 包装，设置 owns_memory 通过 allocate 路径
    Tensor t = Tensor::view(ctx, device_ptr, shape, dnnl_dtype);
    // 注意: 这里的 tensor 不拥有内存，由 ModelWeights 整体管理
    // dnnl::memory 对象存储在内部以供 oneDNN primitive 使用

    return t;
}

// ============================================================
// 主加载函数
// ============================================================

ModelWeights LoadSafetensors(const std::string& path, Context& ctx) {
    ModelWeights weights;

    try {
        // 打开 mmap 文件
        weights.mmap_file_ = std::make_unique<MemoryMappedFile>(path);
        const uint8_t* raw_ptr = weights.mmap_file_->data();

        // 解析 header
        uint64_t header_size = *reinterpret_cast<const uint64_t*>(raw_ptr);
        std::string json_str(reinterpret_cast<const char*>(raw_ptr + 8), header_size);
        std::cout << "[SafeTensors] Header size: " << header_size << " bytes" << std::endl;

        const uint8_t* tensor_data_start = raw_ptr + 8 + header_size;

        std::unordered_map<std::string, TensorMeta> metadata;
        auto layer_names = ParseHeader(json_str, metadata);
        std::cout << "[SafeTensors] Found " << layer_names.size() << " tensors" << std::endl;

        // 加载所有 tensor 到 GPU
        size_t total_bytes = 0;
        int success = 0;
        int fail = 0;

        for (const auto& name : layer_names) {
            const auto& meta = metadata[name];
            try {
                Tensor t = LoadTensorToGPU(tensor_data_start, meta, ctx);
                size_t bytes = meta.byte_offset_end - meta.byte_offset_start;
                total_bytes += bytes;
                weights.put(name, std::move(t));
                success++;
            } catch (const std::exception& e) {
                std::cerr << "  [FAIL] " << name << ": " << e.what() << std::endl;
                fail++;
            }
        }

        std::cout << "[SafeTensors] Loaded " << success << " tensors ("
                  << total_bytes / (1024.0 * 1024.0) << " MB) to GPU"
                  << (fail > 0 ? ", " + std::to_string(fail) + " failed" : "")
                  << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[SafeTensors] Load failed: " << e.what() << std::endl;
        throw;
    }

    return weights;
}