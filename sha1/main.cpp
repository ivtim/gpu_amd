#include <iostream>
#include <vector>
#include <thread>
#include <hip/hip_runtime.h>
#include <random>
#include <chrono>


#define SHA1_SIZE 20

struct sha1_block  {
    uint8_t block[64];
};

__device__ __forceinline__ uint32_t rotl(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

__global__
void sha1_kernel(const sha1_block* blocks, uint8_t* out_hashes, uint32_t count) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    const uint8_t* data = blocks[tid].block;

    uint32_t w[80];

    // load 16 words (big-endian)
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        int j = i * 4;
        w[i] =
            (uint32_t(data[j + 0]) << 24) |
            (uint32_t(data[j + 1]) << 16) |
            (uint32_t(data[j + 2]) << 8) |
            (uint32_t(data[j + 3]));
    }

    // extend
#pragma unroll
    for (int i = 16; i < 80; ++i) {
        w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    }

    uint32_t a = 0x67452301;
    uint32_t b = 0xEFCDAB89;
    uint32_t c = 0x98BADCFE;
    uint32_t d = 0x10325476;
    uint32_t e = 0xC3D2E1F0;

#pragma unroll
    for (int i = 0; i < 80; ++i) {
        uint32_t f, k;

        if (i < 20) {
            f = (b & c) | (~b & d);
            k = 0x5A827999;
        }
        else if (i < 40) {
            f = b ^ c ^ d;
            k = 0x6ED9EBA1;
        }
        else if (i < 60) {
            f = (b & c) | (b & d) | (c & d);
            k = 0x8F1BBCDC;
        }
        else {
            f = b ^ c ^ d;
            k = 0xCA62C1D6;
        }

        uint32_t temp = rotl(a, 5) + f + e + k + w[i];
        e = d;
        d = c;
        c = rotl(b, 30);
        b = a;
        a = temp;
    }

    uint32_t h0 = a + 0x67452301;
    uint32_t h1 = b + 0xEFCDAB89;
    uint32_t h2 = c + 0x98BADCFE;
    uint32_t h3 = d + 0x10325476;
    uint32_t h4 = e + 0xC3D2E1F0;

    uint8_t* out = out_hashes + tid * 20;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[i + 0] = (h0 >> (24 - i * 8)) & 0xFF;
        out[i + 4] = (h1 >> (24 - i * 8)) & 0xFF;
        out[i + 8] = (h2 >> (24 - i * 8)) & 0xFF;
        out[i + 12] = (h3 >> (24 - i * 8)) & 0xFF;
        out[i + 16] = (h4 >> (24 - i * 8)) & 0xFF;
    }
}

void prepare_sha1_block(const uint8_t* input, uint8_t input_size, sha1_block& block) {
    memset(block.block, 0, 64);

    memcpy(block.block, input, input_size);
    block.block[input_size] = 0x80;

    uint64_t bit_len = static_cast<uint64_t>(input_size) * 8;
    for (int i = 0; i < 8; ++i) {
        block.block[56 + i] =
            static_cast<uint8_t>((bit_len >> (56 - 8 * i)) & 0xFF);
    }
}

void prepare_sha1_blocks_parallel(std::vector<sha1_block>& blocks, const uint8_t* input, uint8_t input_size, unsigned int thread_count = 0)
{
    const size_t n = blocks.size();
    if (n == 0) return;

    if (thread_count == 0) {
        thread_count = std::thread::hardware_concurrency();
        if (thread_count == 0) thread_count = 1;
    }
    thread_count = static_cast<unsigned int>(std::min<size_t>(thread_count, n));

    const size_t base = n / thread_count;
    const size_t rem = n % thread_count;

    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    size_t offset = 0;
    for (unsigned int t = 0; t < thread_count; ++t) {
        const size_t chunk = base + (t < rem ? 1 : 0);
        const size_t start = offset;
        const size_t end = start + chunk;
        offset = end;

        threads.emplace_back([start, end, &blocks, input, input_size]() {
            for (size_t i = start; i < end; ++i) {
                prepare_sha1_block(input, input_size, blocks[i]);
            }
            });
    }

    for (auto& th : threads) {
        th.join();
    }
}

void prepare_sha1_blocks_parallel2(std::vector<sha1_block>& blocks, std::vector<uint8_t[16]>& input, uint8_t input_size)
{
    const size_t n = blocks.size();
    if (n == 0) return;

    unsigned int thread_count = std::thread::hardware_concurrency();

    const size_t base = n / thread_count;
    const size_t rem = n % thread_count;

    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    size_t offset = 0;
    for (unsigned int t = 0; t < thread_count; ++t) {
        const size_t chunk = base + (t < rem ? 1 : 0);
        const size_t start = offset;
        const size_t end = start + chunk;
        offset = end;

        threads.emplace_back([start, end, &blocks, &input, input_size]() {
            for (size_t i = start; i < end; ++i) {
                prepare_sha1_block(input[i], input_size, blocks[i]);
            }
            });
    }

    for (auto& th : threads) {
        th.join();
    }
}

static inline uint64_t rotl(uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

class splitmix64 {
public:
    explicit splitmix64(uint64_t seed) : state(seed) {}

    uint64_t next() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

private:
    uint64_t state;
};

class xoshiro256plus {
public:
    explicit xoshiro256plus(uint64_t seed) {
        splitmix64 sm(seed);
        for (int i = 0; i < 4; ++i)
            s[i] = sm.next();
    }

    uint64_t next() {
        const uint64_t result = s[0] + s[3];
        const uint64_t t = s[1] << 17;

        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];

        s[2] ^= t;
        s[3] = rotl(s[3], 45);

        return result;
    }

private:
    uint64_t s[4];
};

void generate_blocks_mt2(std::vector<uint8_t[16]>& blocks)
{
    std::size_t count = blocks.size();
    std::size_t thread_count = std::thread::hardware_concurrency();
    thread_count = std::min(thread_count, count);

    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    const std::size_t base_chunk = count / thread_count;
    const std::size_t remainder = count % thread_count;

    uint64_t global_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    std::size_t offset = 0;

    for (std::size_t t = 0; t < thread_count; ++t)
    {
        const std::size_t chunk = base_chunk + (t < remainder ? 1 : 0);
        const std::size_t start = offset;

        threads.emplace_back([start, chunk, &blocks, global_seed, t]()
            {
                splitmix64 seeder(global_seed + t);
                xoshiro256plus rng(seeder.next());

                for (std::size_t i = 0; i < chunk; ++i)
                {
                    uint64_t r1 = rng.next();
                    uint64_t r2 = rng.next();

                    std::memcpy(blocks[start + i], &r1, 8);
                    std::memcpy(blocks[start + i] + 8, &r2, 8);
                }
            });

        offset += chunk;
    }

    for (auto& th : threads)
        th.join();
}

int main()
{
    uint32_t count = 90000000;

    // init
    std::vector<uint8_t[16]> data_list(count);
    std::vector<sha1_block> h_blocks(count);
    std::vector<uint8_t[SHA1_SIZE]> h_hashes(count);

    // init gpu
    sha1_block* d_blocks;
    uint8_t* d_hashes;
    hipMalloc(&d_blocks, count * sizeof(sha1_block));
    hipMalloc(&d_hashes, count * SHA1_SIZE);

    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);

    for (size_t i = 0; i < 50; i++)
    {
        generate_blocks_mt2(data_list);
        prepare_sha1_blocks_parallel2(h_blocks, data_list, 16);

        hipMemcpy(d_blocks, h_blocks.data(), count * sizeof(sha1_block), hipMemcpyHostToDevice);

        hipLaunchKernelGGL(
            sha1_kernel,
            grid, block, 0, 0,
            d_blocks, d_hashes, count
        );

        hipMemcpy(h_hashes.data(), d_hashes, count * 20, hipMemcpyDeviceToHost);
    }

    hipFree(d_hashes);
    hipFree(d_blocks);
    return 0;
}
