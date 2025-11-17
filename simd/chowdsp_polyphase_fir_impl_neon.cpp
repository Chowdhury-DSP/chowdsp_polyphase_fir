#include <arm_neon.h>

namespace chowdsp::polyphase_fir::neon
{
static void process_fir_interp (const Polyphase_FIR_State* state,
                                const float* ch_state,
                                float* y_data,
                                int n_samples_in,
                                float* scratch)
{
    static constexpr int v_size = 4;
    const auto n_taps_v = state->taps_per_filter_padded / v_size;
    const auto* coeffs_v = reinterpret_cast<const float32x4_t*> (state->coeffs);

    for (int filter_idx = 0; filter_idx < state->factor; ++filter_idx)
    {
        const auto* filter_coeffs = coeffs_v + filter_idx * n_taps_v;
        for (int n = 0; n < n_samples_in; ++n)
        {
            float32x4_t accum_0 {};
            float32x4_t accum_1 {};
            int k = 0;
            for (; k < n_taps_v; k += 2)
            {
                const auto z0 = vld1q_f32 (ch_state + n + k * v_size);
                accum_0 = vfmaq_f32 (accum_0, z0, filter_coeffs[k]);
                const auto z1 = vld1q_f32 (ch_state + n + (k + 1) * v_size);
                accum_1 = vfmaq_f32 (accum_1, z1, filter_coeffs[k + 1]);
            }
            for (; k < n_taps_v; ++k)
            {
                const auto z = vld1q_f32 (ch_state + n + k * v_size);
                accum_0 = vfmaq_f32 (accum_0, z, filter_coeffs[k]);
            }

            const auto accum = vaddq_f32 (accum_0, accum_1);
            auto rr = vadd_f32 (vget_high_f32 (accum), vget_low_f32 (accum));
            scratch[n] = vget_lane_f32 (vpadd_f32 (rr, rr), 0);
        }

        for (int n = 0; n < n_samples_in; ++n)
            y_data[n * state->factor + filter_idx] = scratch[n];
    }
}

static void process_fir_decim (const Polyphase_FIR_State* state,
                               const float* channel_state,
                               float* y_data,
                               int n_samples_out,
                               float* scratch)
{
    static constexpr int v_size = 4;
    const auto n_taps_v = state->taps_per_filter_padded / v_size;
    const auto* coeffs_v = reinterpret_cast<const float32x4_t*> (state->coeffs);
    auto* scratch_v = reinterpret_cast<float32x4_t*> (scratch);

    int filter_idx = 0;
    const auto* filter_coeffs = coeffs_v + filter_idx * n_taps_v;
    const auto* filter_state = channel_state + filter_idx * state->state_per_filter_padded;
    for (int n = 0; n < n_samples_out; ++n)
    {
        float32x4_t accum_0 {};
        float32x4_t accum_1 {};
        int k = 0;
        for (; k < n_taps_v; k += 2)
        {
            const auto z0 = vld1q_f32 (filter_state + n + k * v_size);
            accum_0 = vfmaq_f32 (accum_0, z0, filter_coeffs[k]);
            const auto z1 = vld1q_f32 (filter_state + n + (k + 1) * v_size);
            accum_1 = vfmaq_f32 (accum_1, z1, filter_coeffs[k + 1]);
        }
        for (; k < n_taps_v; ++k)
        {
            const auto z = vld1q_f32 (filter_state + n + k * v_size);
            accum_0 = vfmaq_f32 (accum_0, z, filter_coeffs[k]);
        }

        scratch_v[n] = vaddq_f32 (accum_0, accum_1);
    }

    for (filter_idx = 1; filter_idx < state->factor; ++filter_idx)
    {
        filter_coeffs = coeffs_v + filter_idx * n_taps_v;
        filter_state = channel_state + filter_idx * state->state_per_filter_padded;
        for (int n = 0; n < n_samples_out; ++n)
        {
            float32x4_t accum_0 {};
            float32x4_t accum_1 {};
            int k = 0;
            for (; k < n_taps_v; k += 2)
            {
                const auto z0 = vld1q_f32 (filter_state + n + k * v_size);
                accum_0 = vfmaq_f32 (accum_0, z0, filter_coeffs[k]);
                const auto z1 = vld1q_f32 (filter_state + n + (k + 1) * v_size);
                accum_1 = vfmaq_f32 (accum_1, z1, filter_coeffs[k + 1]);
            }
            for (; k < n_taps_v; ++k)
            {
                const auto z = vld1q_f32 (filter_state + n + k * v_size);
                accum_0 = vfmaq_f32 (accum_0, z, filter_coeffs[k]);
            }

            scratch_v[n] = vaddq_f32 (scratch_v[n], vaddq_f32 (accum_0, accum_1));
            // scratch_v[n] += vaddq_f32 (accum_0, accum_1);
        }
    }

    for (int n = 0; n < n_samples_out; ++n)
    {
        auto rr = vadd_f32 (vget_high_f32 (scratch_v[n]), vget_low_f32 (scratch_v[n]));
        y_data[n] = vget_lane_f32 (vpadd_f32 (rr, rr), 0);
    }
}
} // namespace chowdsp::polyphase_fir::neon
