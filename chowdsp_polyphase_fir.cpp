#include "chowdsp_polyphase_fir.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <tuple>

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

    const auto state_per_filter_padded = get_state_per_filter_padded (taps_per_filter_padded, max_samples_in, alignment);
    const auto state_bytes = state_per_filter_padded * n_channels * sizeof (float);

    return std::make_tuple (coeffs_bytes, state_bytes);
}

size_t persistent_bytes_required (int n_channels, int n_taps, int factor, int max_samples_in, int alignment)
{
    const auto state_object_bytes = round_to_next_multiple ((int) sizeof (Polyphase_FIR_State), alignment);
    const auto [coeffs_bytes, state_bytes] = get_coeffs_state_bytes (n_channels, n_taps, factor, max_samples_in, alignment);
    return state_object_bytes + coeffs_bytes + state_bytes;
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
    state->state_per_channel_padded = get_state_per_filter_padded (state->taps_per_filter_padded, max_samples_in, alignment);
    state->factor = factor;

    const auto [coeffs_bytes, state_bytes] = get_coeffs_state_bytes (n_channels, n_taps, factor, max_samples_in, alignment);
    state->coeffs = reinterpret_cast<float*> (data);
    data += coeffs_bytes;
    state->state = reinterpret_cast<float*> (data);
    data += state_bytes;

    reset (state);

    return state;
}

void load_coeffs (Polyphase_FIR_State* state, const float* coeffs, int n_taps)
{
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
    const auto state_bytes = state->state_per_channel_padded * state->n_channels * sizeof (float);
    std::memset (state->state, 0, state_bytes);
}

size_t scratch_bytes_required (int n_taps, int factor, int max_samples_in, int alignment)
{
    const auto taps_per_filter_padded = get_taps_per_filter_padded (n_taps, factor, alignment);
    const auto buffer_bytes_padded = round_to_next_multiple (
        (taps_per_filter_padded - 1 + max_samples_in) * (int) sizeof (float),
        alignment);
    return buffer_bytes_padded;
}

void process (Polyphase_FIR_State* state,
              const float* const* in,
              float* const* out,
              int n_channels,
              int n_samples_in,
              void* scratch_data)
{
    auto* scratch_start = (float*) scratch_data;
    const auto num_samples_out = n_samples_in * state->factor;

    for (int ch = 0; ch < n_channels; ++ch)
    {
        auto* ch_state = state->state + ch * state->state_per_channel_padded;

        { // copy x_data into ch_state
            auto* x_data = in[ch];
            std::memcpy (ch_state + state->taps_per_filter_padded - 1,
                         x_data,
                         n_samples_in * sizeof (float));
        }

        // apply filters
        auto* y_data = out[ch];
        for (int filter_idx = 0; filter_idx < state->factor; ++filter_idx)
        {
            auto* scratch = scratch_start;
            const auto* coeffs = state->coeffs + filter_idx * state->taps_per_filter_padded;
            for (int n = 0; n < n_samples_in; ++n)
            {
                scratch[n] = 0.0f;
                for (int k = 0; k < state->taps_per_filter_padded; ++k)
                    scratch[n] += coeffs[k] * ch_state[n + k];
            }

            for (int n = 0; n < n_samples_in; ++n)
                y_data[n * state->factor + filter_idx] = scratch[n];
        }

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
} // namespace chowdsp::polyphase_fir
