#include <chowdsp_filters/chowdsp_filters.h>
#include <chowdsp_polyphase_fir.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

static constexpr int n_taps = 25;
static constexpr float coeffs[n_taps] {
    0.000410322870809364f,
    -0.000000000000000001f,
    -0.002230285518524044f,
    0.000000000000000003f,
    0.007100857132041434f,
    -0.000000000000000006f,
    -0.017917030421899412f,
    0.000000000000000010f,
    0.040107417651266832f,
    -0.000000000000000015f,
    -0.090106922079616902f,
    0.000000000000000018f,
    0.312633321620530980f,
    0.500004637490783610f,
    0.312633321620530980f,
    0.000000000000000018f,
    -0.090106922079616916f,
    -0.000000000000000015f,
    0.040107417651266852f,
    0.000000000000000010f,
    -0.017917030421899436f,
    -0.000000000000000006f,
    0.007100857132041438f,
    0.000000000000000003f,
    -0.002230285518524042f,
};

template <int factor>
static void test_interp (int n_channels, int n_samples)
{
    chowdsp::Buffer<float> buffer_in { n_channels, n_samples };
    for (auto [ch, data] : chowdsp::buffer_iters::channels (buffer_in))
        for (auto [n, x] : chowdsp::enumerate (data))
            x = static_cast<float> (n + (size_t) ch);

    chowdsp::Buffer<float> ref_buffer_out { n_channels, n_samples * factor };
    {
        chowdsp::ArenaAllocator<> arena { 1 << 14 };
        chowdsp::FIRPolyphaseInterpolator<float, factor, n_taps> ref_filter;
        ref_filter.prepare (n_channels, n_samples, coeffs, arena);
        ref_filter.processBlock (buffer_in, ref_buffer_out);
    }

    chowdsp::Buffer<float> test_buffer_out { n_channels, n_samples * factor };
    {
        namespace pfir = chowdsp::polyphase_fir;
#if defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)
        static constexpr int alignment = 32;
#else
        static constexpr int alignment = 16;
#endif

        const auto block_size_1 = n_samples / 2;
        const auto block_size_2 = n_samples - block_size_1;
        const auto max_block_size = std::max (block_size_1, block_size_2);

        const auto persistent_bytes = pfir::persistent_bytes_required (n_channels, n_taps, factor, max_block_size, alignment);
        const auto scratch_bytes = pfir::scratch_bytes_required (n_taps, factor, max_block_size, alignment);
        chowdsp::ArenaAllocator<> arena { persistent_bytes + scratch_bytes + alignment };

        auto state = pfir::init (n_channels,
                                 n_taps,
                                 factor,
                                 max_block_size,
                                 arena.allocate_bytes (persistent_bytes, 32),
                                 alignment);
        pfir::load_coeffs (state, coeffs, n_taps);

        auto* scratch_data = arena.allocate_bytes (scratch_bytes, alignment);
        auto half_buffer_in = chowdsp::BufferView { buffer_in, 0, block_size_1 };
        auto half_buffer_out = chowdsp::BufferView { test_buffer_out, 0, block_size_1 * factor };
        pfir::process_interpolate (state,
                                   half_buffer_in.getArrayOfReadPointers(),
                                   half_buffer_out.getArrayOfWritePointers(),
                                   n_channels,
                                   block_size_1,
                                   scratch_data,
                                   true);

        half_buffer_in = chowdsp::BufferView { buffer_in, block_size_1, block_size_2 };
        half_buffer_out = chowdsp::BufferView { test_buffer_out, block_size_1 * factor, block_size_2 * factor };
        pfir::process_interpolate (state,
                                   half_buffer_in.getArrayOfReadPointers(),
                                   half_buffer_out.getArrayOfWritePointers(),
                                   n_channels,
                                   block_size_2,
                                   scratch_data,
                                   true);
    }

    for (const auto [ch, ref_data, test_data] : chowdsp::buffer_iters::zip_channels (std::as_const (ref_buffer_out),
                                                                                     std::as_const (test_buffer_out)))
    {
        for (const auto [ref, test] : chowdsp::zip (ref_data, test_data))
            REQUIRE (test == Catch::Approx { ref }.margin (1.0e-6));
    }
}

template <int factor>
static void test_decim (int n_channels, int n_samples)
{
    chowdsp::Buffer<float> buffer_in { n_channels, n_samples * factor };
    for (auto [ch, data] : chowdsp::buffer_iters::channels (buffer_in))
        for (auto [n, x] : chowdsp::enumerate (data))
            x = static_cast<float> (n + (size_t) ch);

    chowdsp::Buffer<float> ref_buffer_out { n_channels, n_samples };
    {
        chowdsp::ArenaAllocator<> arena { 1 << 14 };
        chowdsp::FIRPolyphaseDecimator<float, factor, n_taps> ref_filter;
        ref_filter.prepare (n_channels, n_samples * factor, coeffs, arena);
        ref_filter.processBlock (buffer_in, ref_buffer_out);
    }

    chowdsp::Buffer<float> test_buffer_out { n_channels, n_samples };
    {
        namespace pfir = chowdsp::polyphase_fir;
#if defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)
        static constexpr int alignment = 32;
#else
        static constexpr int alignment = 16;
#endif

        const auto block_size_1 = n_samples / 2;
        const auto block_size_2 = n_samples - block_size_1;
        const auto max_block_size = std::max (block_size_1, block_size_2);

        const auto persistent_bytes = pfir::persistent_bytes_required (n_channels, n_taps, factor, max_block_size, alignment);
        const auto scratch_bytes = pfir::scratch_bytes_required (n_taps, factor, max_block_size, alignment);
        chowdsp::ArenaAllocator<> arena { persistent_bytes + scratch_bytes + alignment };

        auto state = pfir::init (n_channels,
                                 n_taps,
                                 factor,
                                 max_block_size,
                                 arena.allocate_bytes (persistent_bytes, 32),
                                 alignment);
        pfir::load_coeffs (state, coeffs, n_taps);

        auto* scratch_data = arena.allocate_bytes (scratch_bytes, alignment);
        auto half_buffer_in = chowdsp::BufferView { buffer_in, 0, block_size_1 * factor };
        auto half_buffer_out = chowdsp::BufferView { test_buffer_out, 0, block_size_1 };
        pfir::process_decimate (state,
                                half_buffer_in.getArrayOfReadPointers(),
                                half_buffer_out.getArrayOfWritePointers(),
                                n_channels,
                                block_size_1 * factor,
                                scratch_data,
                                true);

        half_buffer_in = chowdsp::BufferView { buffer_in, block_size_1 * factor, block_size_2 * factor };
        half_buffer_out = chowdsp::BufferView { test_buffer_out, block_size_1, block_size_2 };
        pfir::process_decimate (state,
                                half_buffer_in.getArrayOfReadPointers(),
                                half_buffer_out.getArrayOfWritePointers(),
                                n_channels,
                                block_size_2 * factor,
                                scratch_data,
                                true);
    }

    for (const auto [ch, ref_data, test_data] : chowdsp::buffer_iters::zip_channels (std::as_const (ref_buffer_out),
                                                                                     std::as_const (test_buffer_out)))
    {
        for (const auto [ref, test] : chowdsp::zip (ref_data, test_data))
            REQUIRE (test == Catch::Approx { ref }.margin (1.0e-6));
    }
}

TEST_CASE ("Polyphase Interpolation")
{
    const int channels[] = { 1, 2 };
    const int samples[] = { 16, 127 };

    for (auto n_channels : channels)
    {
        for (auto n_samples : samples)
        {
            test_interp<1> (n_channels, n_samples);
            test_interp<2> (n_channels, n_samples);
            test_interp<3> (n_channels, n_samples);
        }
    }
}

TEST_CASE ("Polyphase Decimation")
{
    const int channels[] = { 1, 2 };
    const int samples[] = { 16, 127 };

    for (auto n_channels : channels)
    {
        for (auto n_samples : samples)
        {
            test_decim<1> (n_channels, n_samples);
            test_decim<2> (n_channels, n_samples);
            test_decim<3> (n_channels, n_samples);
        }
    }
}
