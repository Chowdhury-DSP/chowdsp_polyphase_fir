#include "../chowdsp_polyphase_fir.h"

#if defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>

namespace chowdsp::polyphase_fir::avx
{
void process_fir_interp (const Polyphase_FIR_State* state,
                         const float* ch_state,
                         float* y_data,
                         int n_samples_in,
                         float* scratch)
{
    static constexpr int v_size = 8;
    const auto n_taps_v = state->taps_per_filter_padded / v_size;
    const auto* coeffs_v = reinterpret_cast<const __m256*> (state->coeffs);
    const auto one_avx = _mm256_set1_ps (1.0f);

    for (int filter_idx = 0; filter_idx < state->factor; ++filter_idx)
    {
        const auto* filter_coeffs = coeffs_v + filter_idx * n_taps_v;
        for (int n = 0; n < n_samples_in; ++n)
        {
            auto accum = _mm256_setzero_ps();
            for (int k = 0; k < n_taps_v; ++k)
            {
                const auto z = _mm256_loadu_ps (ch_state + n + k * v_size);
                accum = _mm256_fmadd_ps (z, filter_coeffs[k], accum);
            }
            __m256 rr = _mm256_dp_ps (accum, one_avx, 0xff);
            __m256 tmp = _mm256_permute2f128_ps (rr, rr, 1);
            rr = _mm256_add_ps (rr, tmp);
            scratch[n] = _mm256_cvtss_f32 (rr);
        }

        for (int n = 0; n < n_samples_in; ++n)
            y_data[n * state->factor + filter_idx] = scratch[n];
    }
}

void process_fir_decim (const Polyphase_FIR_State* state,
                        const float* ch_state,
                        float* y_data,
                        int n_samples_out)
{
    static constexpr int v_size = 8;
    const auto n_taps_v = state->taps_per_filter_padded / v_size;
    const auto* coeffs_v = reinterpret_cast<const __m256*> (state->coeffs);
    const auto one_avx = _mm256_set1_ps (1.0f);

    int filter_idx = 0;
    const auto* filter_coeffs = coeffs_v + filter_idx * n_taps_v;
    const auto* filter_state = ch_state + filter_idx * state->state_per_filter_padded;
    for (int n = 0; n < n_samples_out; ++n)
    {
        auto accum = _mm256_setzero_ps();
        for (int k = 0; k < n_taps_v; ++k)
        {
            const auto z = _mm256_loadu_ps (filter_state + n + k * v_size);
            accum = _mm256_fmadd_ps (z, filter_coeffs[k], accum);
        }
        __m256 rr = _mm256_dp_ps (accum, one_avx, 0xff);
        __m256 tmp = _mm256_permute2f128_ps (rr, rr, 1);
        rr = _mm256_add_ps (rr, tmp);
        y_data[n] = _mm256_cvtss_f32 (rr);
    }

    for (filter_idx = 1; filter_idx < state->factor; ++filter_idx)
    {
        filter_coeffs = coeffs_v + filter_idx * n_taps_v;
        filter_state = ch_state + filter_idx * state->state_per_filter_padded;
        for (int n = 0; n < n_samples_out; ++n)
        {
            auto accum = _mm256_setzero_ps();
            for (int k = 0; k < n_taps_v; ++k)
            {
                const auto z = _mm256_loadu_ps (filter_state + n + k * v_size);
                accum = _mm256_fmadd_ps (z, filter_coeffs[k], accum);
            }
            __m256 rr = _mm256_dp_ps (accum, one_avx, 0xff);
            __m256 tmp = _mm256_permute2f128_ps (rr, rr, 1);
            rr = _mm256_add_ps (rr, tmp);
            y_data[n] += _mm256_cvtss_f32 (rr);
        }
    }
}
} // namespace chowdsp::polyphase_fir::avx
#endif
