// Asserts the values returned by the host-side MMQ tuning accessors
// (get_mmq_x_max_host / get_mmq_y_host / mmq_get_nwarps_host) for a
// representative set of (cc, warp_size) inputs covering every arch
// bucket the dispatcher branches on.
//
// Run on the same build config on master and on a refactor branch.
// Identical PASS output across the two runs proves value parity.
//
// Manual build on HIP (no CMake integration required):
//   $LLAMA_DIR/build-hip is your existing HIP build of llama.cpp.
//   hipcc -std=c++17 -O2 \
//     -I ggml/src/ggml-cuda -I ggml/src -I ggml/include \
//     --offload-arch=gfx1151 \
//     -DGGML_USE_HIP -D__HIP_PLATFORM_AMD__ \
//     tests/test-mmq-config.cu -o build-hip/bin/test-mmq-config
//   ./build-hip/bin/test-mmq-config
//
// Assertions assume GGML_CUDA_FORCE_MMQ is not defined (default).

#include "mmq.cuh"

#include <cstdio>
#include <cstring>

struct test_case {
    int          cc;
    int          warp_size;
    int          expected_x_max;
    int          expected_y;
    int          expected_nwarps;
    const char * name;
};

#ifdef GGML_CUDA_FORCE_MMQ
static constexpr int volta_x_max = 128;
#else
static constexpr int volta_x_max = MMQ_DP4A_MAX_BATCH_SIZE;
#endif

static const test_case cases[] = {
    // AMD MFMA bucket (CDNA, wave64)
    { GGML_CUDA_CC_CDNA1,       64,  64, 128, 8, "AMD CDNA1 (gfx908)"       },
    { GGML_CUDA_CC_CDNA2,       64,  64, 128, 8, "AMD CDNA2 (gfx90a)"       },
    { GGML_CUDA_CC_CDNA3,       64,  64, 128, 8, "AMD CDNA3 (gfx942)"       },
    { GGML_CUDA_CC_CDNA4,       64,  64, 128, 8, "AMD CDNA4 (gfx950)"       },

    // AMD WMMA bucket (RDNA3+, wave32)
    { GGML_CUDA_CC_RDNA3,       32, 128, 128, 8, "AMD RDNA3 (gfx1100)"      },
    { GGML_CUDA_CC_RDNA3_5,     32, 128, 128, 8, "AMD RDNA3.5 (gfx1151)"    },
    { GGML_CUDA_CC_RDNA4,       32, 128, 128, 8, "AMD RDNA4 (gfx1200)"      },

    // AMD RDNA1 bucket
    { GGML_CUDA_CC_RDNA1,       32,  64,  64, 8, "AMD RDNA1 wave32"         },
    { GGML_CUDA_CC_RDNA1,       64,  64,  64, 4, "AMD RDNA1 wave64"         },

    // AMD "other" bucket: RDNA2 + GCN/Vega (preserves existing fallthrough)
    { GGML_CUDA_CC_RDNA2,       32,  64, 128, 8, "AMD RDNA2 (gfx1030)"      },
    { GGML_CUDA_CC_VEGA,        64,  64, 128, 4, "AMD Vega wave64"          },
    { GGML_CUDA_CC_GCN4,        64,  64, 128, 4, "AMD GCN4/Polaris wave64"  },

    // Nvidia Turing+ bucket
    { GGML_CUDA_CC_TURING,       32, 128, 128, 8, "Nvidia Turing sm_75"     },
    { GGML_CUDA_CC_AMPERE,       32, 128, 128, 8, "Nvidia Ampere sm_80"     },
    { GGML_CUDA_CC_ADA_LOVELACE, 32, 128, 128, 8, "Nvidia Ada sm_89"        },
    { GGML_CUDA_CC_BLACKWELL,    32, 128, 128, 8, "Nvidia Blackwell sm_120" },

    // Nvidia Volta-only bucket (no Turing in highest compiled)
    { GGML_CUDA_CC_VOLTA,        32, volta_x_max, 128, 8, "Nvidia Volta sm_70" },

    // Default bucket (pre-Volta Nvidia)
    { GGML_CUDA_CC_PASCAL,       32,  64,  64, 8, "Nvidia Pascal sm_60"     },
};

int main() {
    int passes = 0;
    int fails  = 0;

    for (const test_case & tc : cases) {
        const int x_max  = get_mmq_x_max_host(tc.cc);
        const int y      = get_mmq_y_host(tc.cc);
        const int nwarps = mmq_get_nwarps_host(tc.cc, tc.warp_size);

        const bool ok =
            x_max  == tc.expected_x_max &&
            y      == tc.expected_y     &&
            nwarps == tc.expected_nwarps;

        printf("[%s] %-30s cc=0x%07x warp=%d  =>  x_max=%-3d y=%-3d nwarps=%-2d  (expected %d/%d/%d)\n",
               ok ? "PASS" : "FAIL",
               tc.name,
               (unsigned) tc.cc, tc.warp_size,
               x_max, y, nwarps,
               tc.expected_x_max, tc.expected_y, tc.expected_nwarps);

        if (ok) {
            passes++;
        } else {
            fails++;
        }
    }

    printf("\n%d passed, %d failed (%zu total)\n",
           passes, fails, sizeof(cases) / sizeof(cases[0]));

    return fails > 0 ? 1 : 0;
}
