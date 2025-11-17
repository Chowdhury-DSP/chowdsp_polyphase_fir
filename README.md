# chowdsp_polyphase_fir

[![Test](https://github.com/Chowdhury-DSP/chowdsp_polyphase_fir/actions/workflows/test.yml/badge.svg)](https://github.com/Chowdhury-DSP/chowdsp_polyphase_fir/actions/workflows/test.yml)

This repository contains a minimal C/C++ library for computing [polyphase FIR filters](https://www.dsprelated.com/blogimages/JosefHoffmann/decim_interp_polyphase.pdf).

## Usage

First, determine your memory requirements:
```cpp
const auto persistent_bytes = persistent_bytes_required (n_channels, n_taps, interpolation_factor, max_block_size, alignment);
const auto scratch_bytes = scratch_bytes_required (n_taps, interpolation_factor, max_block_size, alignment);
```

Next, you can create a "state" object:
```cpp
auto* state = init (n_channels,
                    n_taps,
                    interpolation_factor,
                    max_block_size,
                    allocate_bytes (persistent_bytes, alignment),
                    alignment);
```

And load in our filter coefficients:
```cpp
load_coeffs (state, coeffs, n_taps);
```

Now we're ready to do our processing! We can do polyphase interpolation:
```cpp
process_interpolate (state,
                     input_buffer,
                     output_buffer,
                     n_channels,
                     n_samples,
                     scratch_data,
                     use_avx);
```

Or decimation:
```cpp
process_decimate (state,
                  input_buffer,
                  output_buffer,
                  n_channels,
                  n_samples,
                  scratch_data,
                  use_avx);
```

## License

This code is licensed under the BSD 3-clause license. Enjoy!
