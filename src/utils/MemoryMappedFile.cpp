// MemoryMappedFile.cpp
#include <stdexcept>            // 具体的实现逻辑需要用到的库在这里引入
#include "MemoryMappedFile.hpp" // 必须包含自己的头文件

// 1. 实现构造函数 (注意这里的初始化列表)
MemoryMappedFile::MemoryMappedFile(const std::string& filepath) 
    : hFile(INVALID_HANDLE_VALUE), hMapping(NULL), mappedData(nullptr), fileSize(0) 
{
    hFile = CreateFileA(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, 
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("无法打开文件: " + filepath);
    }

    LARGE_INTEGER size;
    if (!GetFileSizeEx(hFile, &size)) {
        cleanup();
        throw std::runtime_error("无法获取文件大小");
    }
    fileSize = static_cast<size_t>(size.QuadPart);

    if (fileSize == 0) return;

    hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (hMapping == NULL) {
        cleanup();
        throw std::runtime_error("创建文件映射失败");
    }

    mappedData = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (mappedData == nullptr) {
        cleanup();
        throw std::runtime_error("映射文件视图失败");
    }
}

// 2. 实现析构函数
MemoryMappedFile::~MemoryMappedFile() {
    cleanup();
}

// 3. 实现移动构造函数
MemoryMappedFile::MemoryMappedFile(MemoryMappedFile&& other) noexcept 
    : hFile(other.hFile), hMapping(other.hMapping), mappedData(other.mappedData), fileSize(other.fileSize) 
{
    other.hFile = INVALID_HANDLE_VALUE;
    other.hMapping = NULL;
    other.mappedData = nullptr;
    other.fileSize = 0;
}

// 4. 实现私有清理方法
void MemoryMappedFile::cleanup() noexcept {
    if (mappedData != nullptr) {
        UnmapViewOfFile(mappedData);
        mappedData = nullptr;
    }
    if (hMapping != NULL) {
        CloseHandle(hMapping);
        hMapping = NULL;
    }
    if (hFile != INVALID_HANDLE_VALUE) {
        CloseHandle(hFile);
        hFile = INVALID_HANDLE_VALUE;
    }
}