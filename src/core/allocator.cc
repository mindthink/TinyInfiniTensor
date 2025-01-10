#include "core/allocator.h"
#include <utility>
#include <map>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // 首先检查是否有空闲块在末尾
        auto lastBlock = freeBlocks.rbegin();
        if (lastBlock != freeBlocks.rend() && lastBlock->first + lastBlock->second == peak) {
            // 找到了末尾的空闲块
            size_t addr = lastBlock->first;
            size_t blockSize = lastBlock->second;
            freeBlocks.erase(std::next(lastBlock).base());

            if (blockSize >= size) {
                // 末尾空闲块足够大，直接使用
                if (blockSize > size) {
                    // 如果有多余空间，保留为新的空闲块
                    freeBlocks[addr + size] = blockSize - size;
                }
                used += size;
                return addr;
            }
            // 末尾空闲块不够大，扩展它
            peak = addr + size;
            used += size;
            return addr;
        }

        // 如果末尾没有空闲块，尝试找到最合适的空闲块
        auto bestIt = freeBlocks.end();
        size_t bestSize = size_t(-1);

        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it) {
            if (it->second >= size && it->second < bestSize) {
                bestSize = it->second;
                bestIt = it;
            }
        }

        if (bestIt != freeBlocks.end()) {
            // 使用找到的最佳空闲块
            size_t addr = bestIt->first;
            size_t blockSize = bestIt->second;
            freeBlocks.erase(bestIt);

            if (blockSize > size) {
                freeBlocks[addr + size] = blockSize - size;
            }

            used += size;
            return addr;
        }

        // 没有合适的空闲块，从末尾分配新空间
        size_t addr = peak;
        peak += size;
        used += size;
        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);
        used -= size;

        // Add the block to free list
        auto it = freeBlocks.lower_bound(addr);
        
        // Try to merge with next block
        if (it != freeBlocks.end() && addr + size == it->first) {
            size += it->second;
            freeBlocks.erase(it);
        }
        
        // Try to merge with previous block
        if (it != freeBlocks.begin()) {
            auto prev = std::prev(it);
            if (prev->first + prev->second == addr) {
                prev->second += size;
                return;
            }
        }
        
        // No merging possible, just add the block
        freeBlocks[addr] = size;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
