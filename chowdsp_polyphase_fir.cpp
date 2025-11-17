#include "chowdsp_polyphase_fir.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <tuple>

#if defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)
#include "simd/chowdsp_polyphase_fir_impl_sse.cpp"
#if CHOWDSP_POLYPHASE_FIR_COMPILER_SUPPORTS_AVX
namespace chowdsp::polyphase_fir::avx
{
void process_fir_interp (const Polyphase_FIR_State* state,
                         const float* ch_state,
                         float* y_data,
                         int n_samples_in,
                         float* scratch);
void process_fir_decim (const Polyphase_FIR_State* state,
                        const float* ch_state,
                        float* y_data,
                        int n_samples_out);
} // namespace chowdsp::polyphase_fir::avx
#endif
#elif defined(__ARM_NEON__) || defined(_M_ARM64)
#include "simd/chowdsp_polyphase_fir_impl_neon.cpp"
#endif

namespace chowdsp::polyphase_fir
{
static int min_int (int a, int b)
{
    return (b < a) ? b : a;
}

static int max_int (int a, int b)
{
    return (b > a) ? b : a;
}

template <typename T>
constexpr T ceiling_divide (T num, T den)
{
    return (num + den - 1) / den;
}

template <typename T>
constexpr T round_to_next_multiple (T value, T multiplier)
{
    return ceiling_divide (value, multiplier) * multiplier;
}

static int get_taps_per_filter_padded (int n_taps, int factor, int alignment)
{
    const auto taps_per_filter = ceiling_divide (n_taps, factor);
    return round_to_next_multiple (taps_per_filter,
                                   alignment / (int) sizeof (float));
}

static int get_state_per_filter_padded (int taps_per_filter_padded, int max_samples_in, int alignment)
{
    const auto state_required = taps_per_filter_padded + max_samples_in;
    return round_to_next_multiple (state_required,
                                   alignment / (int) sizeof (float));
}

static auto get_coeffs_state_bytes (int n_channels, int n_taps, int factor, int max_samples_in, int alignment)
{
    const auto taps_per_filter_padded = get_taps_per_filter_padded (n_taps, factor, alignment);
    const auto coeffs_bytes = taps_per_filter_padded * factor * sizeof (float);

    const auto interp_state_per_filter_padded = get_state_per_filter_padded (taps_per_filter_padded, max_samples_in, alignment);
    const auto interp_state_bytes = interp_state_per_filter_padded * n_channels * sizeof (float);

    const auto decim_state_per_filter_padded = get_state_per_filter_padded (taps_per_filter_padded, max_samples_in, alignment);
    const auto decim_state_bytes = decim_state_per_filter_padded * factor * n_channels * sizeof (float);

    return std::make_tuple (coeffs_bytes, interp_state_bytes, decim_state_bytes);
}

size_t persistent_bytes_required (int n_channels, int n_taps, int factor, int max_samples_in, int alignment)
{
    const auto state_object_bytes = round_to_next_multiple ((int) sizeof (Polyphase_FIR_State), alignment);
    const auto [coeffs_bytes, interp_state_bytes, decim_state_bytes] = get_coeffs_state_bytes (n_channels, n_taps, factor, max_samples_in, alignment);
    return state_object_bytes + coeffs_bytes + interp_state_bytes + decim_state_bytes;
}

Polyphase_FIR_State* init (int n_channels, int n_taps, int factor, int max_samples_in, void* persistent_data, int alignment)
{
    auto* data = (std::byte*) persistent_data;

    // "allocate" state object
    const auto state_object_bytes = round_to_next_multiple ((int) sizeof (Polyphase_FIR_State), alignment);
    auto* state = reinterpret_cast<Polyphase_FIR_State*> (data);
    data += state_object_bytes;

    // initialize state
    *state = {};
    state->n_channels = n_channels;
    state->taps_per_filter_padded = get_taps_per_filter_padded (n_taps, factor, alignment);
    state->state_per_filter_padded = get_state_per_filter_padded (state->taps_per_filter_padded, max_samples_in, alignment);
    state->factor = factor;

    const auto [coeffs_bytes, interp_state_bytes, decim_state_bytes] = get_coeffs_state_bytes (n_channels, n_taps, factor, max_samples_in, alignment);
    state->coeffs = reinterpret_cast<float*> (data);
    data += coeffs_bytes;
    state->interp_state = reinterpret_cast<float*> (data);
    data += interp_state_bytes;
    state->decim_state = reinterpret_cast<float*> (data);
    data += decim_state_bytes;

    reset (state);

    return state;
}

void load_coeffs (Polyphase_FIR_State* state, const float* coeffs, int n_taps)
{
    const auto coeffs_bytes = state->taps_per_filter_padded * state->factor * sizeof (float);
    std::memset (state->coeffs, 0, coeffs_bytes);

    for (int i = 0; i < state->factor; ++i)
    {
        auto* filter_coeffs = state->coeffs + state->taps_per_filter_padded * i;
        for (int j = 0; j < state->taps_per_filter_padded; ++j)
        {
            const auto src_idx = i + j * state->factor;
            const auto dest_idx = state->taps_per_filter_padded - j - 1;
            filter_coeffs[dest_idx] = src_idx >= n_taps ? 0.0f : coeffs[src_idx]; // reverse coefficients
        }
    }
}

void reset (Polyphase_FIR_State* state)
{
    const auto interp_state_bytes = state->state_per_filter_padded * state->n_channels * sizeof (float);
    std::memset (state->interp_state, 0, interp_state_bytes);

    const auto decim_state_bytes = state->state_per_filter_padded * state->factor * state->n_channels * sizeof (float);
    std::memset (state->decim_state, 0, decim_state_bytes);
}

size_t scratch_bytes_required (int n_taps, int factor, int max_samples_in, int alignment)
{
    const auto taps_per_filter_padded = get_taps_per_filter_padded (n_taps, factor, alignment);
    const auto buffer_bytes_padded = round_to_next_multiple (
        (taps_per_filter_padded + max_samples_in) * (int) sizeof (float),
        alignment);
    return buffer_bytes_padded;
}

void process_interpolate (Polyphase_FIR_State* state,
                          const float* const* in,
                          float* const* out,
                          int n_channels,
                          int n_samples_in,
                          void* scratch_data,
                          [[maybe_unused]] bool use_avx)
{
    auto* scratch_start = (float*) scratch_data;
    [[maybe_unused]] const auto n_samples_out = n_samples_in * state->factor;

    for (int ch = 0; ch < n_channels; ++ch)
    {
        auto* ch_state = state->interp_state + ch * state->state_per_filter_padded;

        { // copy x_data into ch_state
            auto* x_data = in[ch];
            std::memcpy (ch_state + state->taps_per_filter_padded - 1,
                         x_data,
                         n_samples_in * sizeof (float));
        }

        // apply filters
#if defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)
        if (use_avx)
            avx::process_fir_interp (state, ch_state, out[ch], n_samples_in, scratch_start);
        else
            sse::process_fir_interp (state, ch_state, out[ch], n_samples_in, scratch_start);
#else
        neon::process_fir_interp (state, ch_state, out[ch], n_samples_in, scratch_start);
#endif

        { // save channel state for next buffer
            auto* scratch = scratch_start;
            const auto samples_to_save = state->taps_per_filter_padded - 1;
            std::memcpy (scratch,
                         ch_state + n_samples_in,
                         samples_to_save * sizeof (float));
            std::memcpy (ch_state,
                         scratch,
                         samples_to_save * sizeof (float));
        }
    }
}

void process_decimate (struct Polyphase_FIR_State* state,
                       const float* const* in,
                       float* const* out,
                       int n_channels,
                       int n_samples_in,
                       void* scratch_data,
                       [[maybe_unused]] bool use_avx)
{
    auto* scratch_start = (float*) scratch_data;
    [[maybe_unused]] const auto n_samples_out = n_samples_in / state->factor;

    for (int ch = 0; ch < n_channels; ++ch)
    {
        auto* ch_state = state->interp_state + ch * (state->state_per_filter_padded * state->factor);

        { // copy x_data into ch_state
            auto* x_data = in[ch];
            int filter_idx = 0;
            auto* filter_state = ch_state + filter_idx * state->state_per_filter_padded;
            for (int n = 0; n < n_samples_out; ++n)
                filter_state[state->taps_per_filter_padded - 1 + n] = x_data[n * state->factor + filter_idx];

            for (filter_idx = 1; filter_idx < state->factor; ++filter_idx)
            {
                filter_state = ch_state + (state->factor - filter_idx) * state->state_per_filter_padded;
                for (int n = 0; n < n_samples_out; ++n)
                    filter_state[state->taps_per_filter_padded + n] = x_data[n * state->factor + filter_idx];
            }
        }

        // apply filters
#if defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)
        if (use_avx)
            avx::process_fir_decim (state, ch_state, out[ch], n_samples_out);
        else
            sse::process_fir_decim (state, ch_state, out[ch], n_samples_out);
#else
        neon::process_fir_decim (state, ch_state, out[ch], n_samples_out);
#endif

        { // save channel state for next buffer
            int filter_idx = 0;
            auto* filter_state = ch_state + filter_idx * state->state_per_filter_padded;
            auto* scratch = scratch_start;
            auto samples_to_save = state->taps_per_filter_padded - 1;
            std::memcpy (scratch,
                         filter_state + n_samples_out,
                         samples_to_save * sizeof (float));
            std::memcpy (filter_state,
                         scratch,
                         samples_to_save * sizeof (float));
            for (filter_idx = 1; filter_idx < state->factor; ++filter_idx)
            {
                filter_state = ch_state + filter_idx * state->state_per_filter_padded;
                samples_to_save = state->taps_per_filter_padded;
                std::memcpy (scratch,
                             filter_state + n_samples_out,
                             samples_to_save * sizeof (float));
                std::memcpy (filter_state,
                             scratch,
                             samples_to_save * sizeof (float));
            }
        }
    }
}
} // namespace chowdsp::polyphase_fir
