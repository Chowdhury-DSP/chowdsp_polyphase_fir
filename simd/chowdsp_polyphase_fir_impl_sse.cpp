#include <immintrin.h>

namespace chowdsp::polyphase_fir::sse
{
static void process_fir_interp (const Polyphase_FIR_State* state,
                                const float* ch_state,
                                float* y_data,
                                int n_samples_in,
                                float* scratch)
{
    static constexpr int v_size = 4;
    const auto n_taps_v = state->taps_per_filter_padded / v_size;
    const auto* coeffs_v = reinterpret_cast<const __m128*> (state->coeffs);

    for (int filter_idx = 0; filter_idx < state->factor; ++filter_idx)
    {
        const auto* coeffs = state->coeffs + filter_idx * state->taps_per_filter_padded;
        const auto* filter_coeffs = coeffs_v + filter_idx * n_taps_v;
        for (int n = 0; n < n_samples_in; ++n)
        {
            auto accum = _mm_setzero_ps();
            for (int k = 0; k < n_taps_v; ++k)
            {
                const auto z = _mm_loadu_ps (ch_state + n + k * v_size);
                accum = _mm_add_ps (accum, _mm_mul_ps (z, filter_coeffs[k]));
            }

            auto rr = _mm_add_ps (_mm_shuffle_ps (accum, accum, 0x4e), accum);
            rr = _mm_add_ps (rr, _mm_shuffle_ps (rr, rr, 0xb1));
            scratch[n] = _mm_cvtss_f32 (rr);
        }

        for (int n = 0; n < n_samples_in; ++n)
            y_data[n * state->factor + filter_idx] = scratch[n];
    }
}
} // namespace chowdsp::polyphase_fir::sse
