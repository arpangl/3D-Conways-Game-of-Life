#include "life_common.h"

#include <stdexcept>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace gol3d {

void step_avx512(const Grid3D& current, Grid3D& next, const LifeConfig3D& cfg) {
#if !defined(__AVX512BW__)
    // Either AVX-512 is unavailable or we lack byte/word support; fall back to the AVX2 path.
    step_avx(current, next, cfg);
#else
    if (current.size() != cfg.width * cfg.height * cfg.depth) {
        throw std::invalid_argument("current grid size mismatch (AVX-512)");
    }

    if (next.size() != current.size()) {
        next.resize(current.size());
    }

    // Use the AVX2 implementation to populate boundary cells and handle wrapping.
    step_avx(current, next, cfg);

    if (cfg.wrap) {
        return;
    }
    if (cfg.width < 34 || cfg.height < 3 || cfg.depth < 3) {
        return;
    }

    const auto width = cfg.width;
    const auto height = cfg.height;
    const auto depth = cfg.depth;
    const auto rowStride = width;
    const auto sliceStride = width * height;

    const auto* data = current.data();

    const __m512i four = _mm512_set1_epi16(4);
    const __m512i five = _mm512_set1_epi16(5);
    const __m512i zero = _mm512_setzero_si512();

    alignas(64) std::uint16_t tmp[32];

    const auto load_expand = [](const std::uint8_t* ptr) -> __m512i {
        const __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
        return _mm512_cvtepu8_epi16(bytes);
    };

    for (std::size_t z = 1; z + 1 < depth; ++z) {
        for (std::size_t y = 1; y + 1 < height; ++y) {
            const auto* rowCenter = data + index_3d(cfg, 0, y, z);
            const auto* rowAbove = rowCenter - rowStride;
            const auto* rowBelow = rowCenter + rowStride;

            const auto* sliceAboveRowCenter = rowCenter - sliceStride;
            const auto* sliceBelowRowCenter = rowCenter + sliceStride;

            for (std::size_t x = 1; x + 32 <= width - 1; x += 32) {
                auto acc = _mm512_setzero_si512();

                const auto accumulate_row = [&](const std::uint8_t* rowPtr) {
                    acc = _mm512_add_epi16(acc, load_expand(rowPtr + x - 1));
                    acc = _mm512_add_epi16(acc, load_expand(rowPtr + x));
                    acc = _mm512_add_epi16(acc, load_expand(rowPtr + x + 1));
                };

                const auto* sliceAboveRowAbove = sliceAboveRowCenter - rowStride;
                const auto* sliceAboveRowBelow = sliceAboveRowCenter + rowStride;
                const auto* sliceBelowRowAbove = sliceBelowRowCenter - rowStride;
                const auto* sliceBelowRowBelow = sliceBelowRowCenter + rowStride;

                // Previous slice (z - 1)
                accumulate_row(sliceAboveRowAbove);
                accumulate_row(sliceAboveRowCenter);
                accumulate_row(sliceAboveRowBelow);

                // Current slice (z)
                accumulate_row(rowAbove);
                accumulate_row(rowCenter);
                accumulate_row(rowBelow);

                // Next slice (z + 1)
                accumulate_row(sliceBelowRowAbove);
                accumulate_row(sliceBelowRowCenter);
                accumulate_row(sliceBelowRowBelow);

                const auto center = load_expand(rowCenter + x);
                acc = _mm512_sub_epi16(acc, center);

                const auto eq5 = _mm512_cmpeq_epi16_mask(acc, five);
                const auto eq4 = _mm512_cmpeq_epi16_mask(acc, four);

                const auto aliveMask = _mm512_cmpgt_epi16_mask(center, zero);
                const auto surviveMask = aliveMask & (eq4 | eq5);
                const auto birthMask = (~aliveMask) & eq5;
                const auto resultMask = surviveMask | birthMask;

                const auto result = _mm512_mask_set1_epi16(zero, resultMask, 1);
                _mm512_storeu_si512(reinterpret_cast<void*>(tmp), result);

                auto baseIndex = index_3d(cfg, x, y, z);
                for (int lane = 0; lane < 32; ++lane) {
                    next[baseIndex + static_cast<std::size_t>(lane)] = static_cast<std::uint8_t>(tmp[lane] != 0);
                }
            }
        }
    }
#endif
}

}  // namespace gol3d
