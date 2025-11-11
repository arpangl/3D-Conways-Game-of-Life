#include "life_common.h"

#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace gol3d {

void step_avx(const Grid3D& current, Grid3D& next, const LifeConfig3D& cfg) {
#if !defined(__AVX2__)
    // Fallback to scalar implementation when AVX2 is unavailable.
    step_single(current, next, cfg);
#else
    if (current.size() != cfg.width * cfg.height * cfg.depth) {
        throw std::invalid_argument("current grid size mismatch (AVX)");
    }

    if (next.size() != current.size()) {
        next.resize(current.size());
    }

    // Compute a scalar baseline so that boundary cells and wrap-around logic remain correct.
    step_single(current, next, cfg);

    if (cfg.wrap) {
        return;  // Wrapping makes aligned vector neighborhoods awkward; rely on scalar result.
    }
    if (cfg.width < 18 || cfg.height < 3 || cfg.depth < 3) {
        return;  // Not enough interior cells for the vectorized path.
    }

    const auto width = cfg.width;
    const auto height = cfg.height;
    const auto depth = cfg.depth;
    const auto rowStride = width;
    const auto sliceStride = width * height;

    const auto* data = current.data();

    const __m256i four = _mm256_set1_epi16(4);
    const __m256i five = _mm256_set1_epi16(5);

    alignas(32) std::uint16_t tmp[16];

    const auto load_expand = [](const std::uint8_t* ptr) -> __m256i {
        const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
        return _mm256_cvtepu8_epi16(bytes);
    };

    for (std::size_t z = 1; z + 1 < depth; ++z) {
        for (std::size_t y = 1; y + 1 < height; ++y) {
            const auto* rowCenter = data + index_3d(cfg, 0, y, z);
            const auto* rowAbove = rowCenter - rowStride;
            const auto* rowBelow = rowCenter + rowStride;

            const auto* sliceAboveRowCenter = rowCenter - sliceStride;
            const auto* sliceBelowRowCenter = rowCenter + sliceStride;

            for (std::size_t x = 1; x + 16 <= width - 1; x += 16) {
                __m256i acc = _mm256_setzero_si256();

                const auto accumulate_row = [&](const std::uint8_t* rowPtr) {
                    acc = _mm256_add_epi16(acc, load_expand(rowPtr + x - 1));
                    acc = _mm256_add_epi16(acc, load_expand(rowPtr + x));
                    acc = _mm256_add_epi16(acc, load_expand(rowPtr + x + 1));
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

                // Remove the contribution of the center cells (dz = dy = dx = 0).
                const __m256i center = load_expand(rowCenter + x);
                acc = _mm256_sub_epi16(acc, center);

                const __m256i eq5 = _mm256_cmpeq_epi16(acc, five);
                const __m256i eq4 = _mm256_cmpeq_epi16(acc, four);

                const __m256i surviveMask = _mm256_and_si256(center, _mm256_or_si256(eq4, eq5));
                const __m256i birthMask = _mm256_andnot_si256(center, eq5);
                const __m256i result16 = _mm256_or_si256(surviveMask, birthMask);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), result16);

                auto baseIndex = index_3d(cfg, x, y, z);
                for (int lane = 0; lane < 16; ++lane) {
                    next[baseIndex + static_cast<std::size_t>(lane)] = static_cast<std::uint8_t>(tmp[lane] != 0);
                }
            }
        }
    }
#endif
}

}  // namespace gol3d

