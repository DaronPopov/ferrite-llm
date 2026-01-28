#pragma once
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <unordered_map>

namespace spc {

/**
 * Advanced TLSF O(1) Allocator - Host-Resident Metadata, GPU-Resident Data
 */
class TLSF {
public:
    static constexpr int FL_INDEX_MAX = 32;
    static constexpr int SL_INDEX_COUNT = 16;
    static constexpr int SL_INDEX_LOG2 = 4;
    static constexpr size_t MIN_BLOCK_SIZE = 64;
    
    // Safety Markers (for metadata)
    static constexpr uint32_t BLOCK_SIG   = 0x50434944; 

    struct Block {
        uint32_t signature;
        size_t size;
        bool is_free;
        void* gpu_ptr; // Pointer to actual GPU memory
        Block* prev_physical;
        Block* next_physical;
        Block* prev_free;
        Block* next_free;
    };

    TLSF(size_t total_size, void* pool_ptr) : total_size(total_size), pool_ptr(pool_ptr) {
        memset(blocks, 0, sizeof(blocks));
        fl_bitmap = 0;
        memset(sl_bitmaps, 0, sizeof(sl_bitmaps));

        // Initial big block (Host-resident metadata)
        Block* initial = new Block();
        initial->signature = BLOCK_SIG;
        initial->size = total_size;
        initial->is_free = true;
        initial->gpu_ptr = pool_ptr;
        initial->prev_physical = nullptr;
        initial->next_physical = nullptr;
        initial->prev_free = nullptr;
        initial->next_free = nullptr;

        insert_block(initial);
    }

    void* malloc(size_t size) {
        size_t total_needed = align_up(size, 8);
        if (total_needed < MIN_BLOCK_SIZE) total_needed = MIN_BLOCK_SIZE;

        int fl, sl;
        mapping(total_needed, fl, sl);
        
        Block* block = find_suitable_block(fl, sl);
        if (!block) return nullptr;

        remove_block(block);

        // Split block if possible
        if (block->size >= total_needed + MIN_BLOCK_SIZE) {
            Block* remaining = new Block();
            remaining->signature = BLOCK_SIG;
            remaining->size = block->size - total_needed;
            remaining->is_free = true;
            remaining->gpu_ptr = (char*)block->gpu_ptr + total_needed;
            
            remaining->next_physical = block->next_physical;
            if (remaining->next_physical) remaining->next_physical->prev_physical = remaining;
            
            remaining->prev_physical = block;
            block->next_physical = remaining;
            
            block->size = total_needed;
            insert_block(remaining);
        }

        block->is_free = false;
        allocation_map[block->gpu_ptr] = block;
        return block->gpu_ptr;
    }

    void free(void* gpu_ptr) {
        if (!gpu_ptr) return;
        if (allocation_map.find(gpu_ptr) == allocation_map.end()) {
            throw std::runtime_error("TLSF: Invalid free - pointer not managed by this allocator");
        }

        Block* block = allocation_map[gpu_ptr];
        block->is_free = true;
        allocation_map.erase(gpu_ptr);

        // Coalesce Next
        if (block->next_physical && block->next_physical->is_free) {
            Block* next = block->next_physical;
            remove_block(next);
            block->size += next->size;
            block->next_physical = next->next_physical;
            if (block->next_physical) block->next_physical->prev_physical = block;
            delete next;
        }

        // Coalesce Prev
        if (block->prev_physical && block->prev_physical->is_free) {
            Block* prev = block->prev_physical;
            remove_block(prev);
            prev->size += block->size;
            prev->next_physical = block->next_physical;
            if (prev->next_physical) prev->next_physical->prev_physical = prev;
            delete block;
            block = prev;
        }

        insert_block(block);
    }

private:
    void mapping(size_t size, int& fl, int& sl) {
        fl = 31 - __builtin_clz(size);
        sl = (size >> (fl - SL_INDEX_LOG2)) - SL_INDEX_COUNT;
    }

    void insert_block(Block* block) {
        int fl, sl;
        mapping(block->size, fl, sl);
        block->next_free = blocks[fl][sl];
        if (blocks[fl][sl]) blocks[fl][sl]->prev_free = block;
        blocks[fl][sl] = block;
        block->prev_free = nullptr;
        fl_bitmap |= (1U << fl);
        sl_bitmaps[fl] |= (1U << sl);
    }

    void remove_block(Block* block) {
        int fl, sl;
        mapping(block->size, fl, sl);
        if (block->prev_free) block->prev_free->next_free = block->next_free;
        if (block->next_free) block->next_free->prev_free = block->prev_free;
        if (blocks[fl][sl] == block) blocks[fl][sl] = block->next_free;
        if (!blocks[fl][sl]) {
            sl_bitmaps[fl] &= ~(1U << sl);
            if (!sl_bitmaps[fl]) fl_bitmap &= ~(1U << fl);
        }
    }

    Block* find_suitable_block(int& fl, int& sl) {
        uint32_t sl_map = sl_bitmaps[fl] & (~0U << sl);
        if (!sl_map) {
            uint32_t fl_map = fl_bitmap & (~0U << (fl + 1));
            if (!fl_map) return nullptr;
            fl = __builtin_ctz(fl_map);
            sl_map = sl_bitmaps[fl];
        }
        sl = __builtin_ctz(sl_map);
        return blocks[fl][sl];
    }

    size_t align_up(size_t n, size_t align) { return (n + align - 1) & ~(align - 1); }

    size_t total_size;
    void* pool_ptr;
    Block* blocks[FL_INDEX_MAX][SL_INDEX_COUNT];
    uint32_t fl_bitmap;
    uint32_t sl_bitmaps[FL_INDEX_MAX];
    std::unordered_map<void*, Block*> allocation_map;
};

} // namespace spc
