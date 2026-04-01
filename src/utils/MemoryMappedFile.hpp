// MemoryMappedFile.hpp
#pragma once // 极其重要：防止头文件在同一个编译单元中被重复包含

#include <windows.h>
#include <string>
#include <cstdint>
#include <iostream>

class MemoryMappedFile {
private:
    HANDLE hFile;
    HANDLE hMapping;
    const void* mappedData;
    size_t fileSize;

    // 私有清理函数声明
    void cleanup() noexcept;

public:
    // 构造与析构声明
    explicit MemoryMappedFile(const std::string& filepath);
    ~MemoryMappedFile();

    // 禁用拷贝，允许移动 (这些可以直接在头文件里声明完)
    MemoryMappedFile(const MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;
    MemoryMappedFile(MemoryMappedFile&& other) noexcept;

    // 简短的 Getter 推荐直接在头文件实现 (这会被编译器自动内联，性能极高)
    const uint8_t* data() const { return static_cast<const uint8_t*>(mappedData); }
    size_t size() const { return fileSize; }
};