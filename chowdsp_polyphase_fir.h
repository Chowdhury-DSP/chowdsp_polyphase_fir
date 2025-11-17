#pragma once

#ifdef __cplusplus
#include <cstddef>
extern "C"
{
namespace chowdsp::polyphase_fir
{
#else
#include <stddef.h>
#endif

/**
 * Object to hold the filter's persistent state.
 *
 * Users should not instantiate this object directly,
 * it will be provided by the `init()` method.
 */
struct Polyphase_FIR_State
{
    float* coeffs {};
    float* interp_state {};
    float* decim_state {};
    int n_channels {};
    int taps_per_filter_padded {};
    int state_per_filter_padded {};
    int factor {};
};

/** Returns the number of bytes needed to construct the filter state. */
size_t persistent_bytes_required (int n_channels, int n_taps, int factor, int max_samples_in, int alignment);

/*
 * Initializes the filter and returns a state object.
 *
 * Note that the number of taps must be greater than or equal to 16.
 *
 * The `max_samples_in` argument is relative to the "interpolation" mode of the filter.
 * For "decimation", the naximum input size if `max_samples_in * factor`.
 *
 * The returned pointer will be allocated into the provided block of persistent data.
 * This means that you should not free the pointer since it will be freed automatically
 * when you free the persistent data.
 */
struct Polyphase_FIR_State* init (int n_channels, int n_taps, int factor, int max_samples_in, void* persistent_data, int alignment);

/** Loads a set of filter coefficients into the filter */
void load_coeffs (struct Polyphase_FIR_State* state, const float* coeffs, int n_taps);

/** Resets the filter state */
void reset (struct Polyphase_FIR_State* state);

/** Returns the scratch memory required by the filter */
size_t scratch_bytes_required (int n_taps, int factor, int max_samples_in, int alignment);

/** Process data through the "interpolation" mode of the filter */
void process_interpolate (struct Polyphase_FIR_State* state,
                          const float* const* in,
                          float* const* out,
                          int n_channels,
                          int n_samples_in,
                          void* scratch_data,
                          bool use_avx);

/** Process data through the "decimation" mode of the filter */
void process_decimate (struct Polyphase_FIR_State* state,
                       const float* const* in,
                       float* const* out,
                       int n_channels,
                       int n_samples_in,
                       void* scratch_data,
                       bool use_avx);

#ifdef __cplusplus
} // namespace chowdsp::polyphase_fir
} // extern "C"
#endif
