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

size_t persistent_bytes_required (int n_channels, int n_taps, int factor, int max_samples_in, int alignment);

struct Polyphase_FIR_State* init (int n_channels, int n_taps, int factor, int max_samples_in, void* persistent_data, int alignment);

void load_coeffs (struct Polyphase_FIR_State* state, const float* coeffs, int n_taps);

void reset (struct Polyphase_FIR_State* state);

size_t scratch_bytes_required (int n_taps, int factor, int max_samples_in, int alignment);

void process_interpolate (struct Polyphase_FIR_State* state,
                          const float* const* in,
                          float* const* out,
                          int n_channels,
                          int n_samples_in,
                          void* scratch_data,
                          bool use_avx);

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
