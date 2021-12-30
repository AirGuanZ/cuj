#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuj.h>

#include "../../test/test/cuda/cuda.h"
#include "vec.h"

constexpr float EPS = 1e-4f;
constexpr float INF = 1e10f;

constexpr float PI = 3.1415926535f;

constexpr float CAMERA_POS[] = { 0.0f, 0.32f, 3.7f };
constexpr float LIGHT_POS[] = { -1.5f, 0.6f, 0.3f };
constexpr float LIGHT_NOR[] = { 1.0f, 0.0f, 0.0f };
constexpr float LIGHT_RADIUS = 2.0f;

constexpr float FOV = 0.23f;
constexpr float DIST_LIMIT = 100.0f;
constexpr int   MAX_DEPTH = 6;

constexpr int   WIDTH = 1280;
constexpr int   HEIGHT = 720;
constexpr float ASPECT = static_cast<float>(WIDTH) / HEIGHT;

void check_cuda_error(cudaError_t err)
{
    if(err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
}

i32 floor_div_rem(i32 a, i32 b)
{
    var r = a % b;
    $if(r > 0 & b < 0)
    {
        r = r + b;
    }
    $elif(r < 0 & b > 0)
    {
        r = r + b;
    };
    return r;
}

f32 intersect_light(const Vec3f &pos, const Vec3f &dir)
{
    var light_loc = make_vec3f(LIGHT_POS[0], LIGHT_POS[1], LIGHT_POS[2]);
    var dotv = -dot(dir, make_vec3f(LIGHT_NOR[0], LIGHT_NOR[1], LIGHT_NOR[2]));
    var dist = dot(light_loc - pos, dir);
    var dist_to_light = INF;
    $if(dotv > 0 & dist > 0)
    {
        var D = dist / dotv;
        var dist_to_center = length_square(light_loc - (pos + D * dir));
        $if(dist_to_center < LIGHT_RADIUS *LIGHT_RADIUS)
        {
            dist_to_light = D;
        };
    };
    return dist_to_light;
}

Vec3f out_dir(const Vec3f &n, ptr<cstd::LCG> rng)
{
    var u = make_vec3f(1.0f, 0.0f, 0.0f);
    $if(cstd::abs(n.y) < 1 - EPS)
    {
        u = normalize(cross(n, make_vec3f(0.0f, 1.0f, 0.0f)));
    };
    var v = cross(n, u);
    var phi = 2 * PI * rng->uniform_float();
    var ay = cstd::sqrt(rng->uniform_float());
    var ax = cstd::sqrt(1.0f - ay * ay);
    return ax * (cstd::cos(phi) * u + cstd::sin(phi) * v) + ay * n;
}

f32 make_nested(f32 f)
{
    f = f * 40;
    var i = i32(f);
    $if(f < 0)
    {
        $if(floor_div_rem(i, 2) == 1)
        {
            f = f - cstd::floor(f);
        }
        $else
        {
            f = cstd::floor(f) + 1.0f - f;
        };
    };
    f = (f - 0.2f) * (1.0f / 40);
    return f;
}

f32 sdf(Vec3f o)
{
    var wall = cstd::min(o.y + 0.1f, o.z + 0.4f);
    var sphere = length(o - make_vec3f(0.0f, 0.35f, 0.0f)) - 0.36f;

    var q = abs(o - make_vec3f(0.8f, 0.3f, 0.0f))
          - make_vec3f(0.3f, 0.3f, 0.3f);
    var box = length(make_vec3f(
                        cstd::max(f32(0), q.x),
                        cstd::max(f32(0), q.y),
                        cstd::max(f32(0), q.z)))
            + cstd::min(
                cstd::max(q.x, cstd::max(q.y, q.z)),
                f32(0));

    var O = o - make_vec3f(-0.8f, 0.3f, 0.0f);
    var d = make_vec2f(
                length(make_vec2f(O.x, O.z)) - 0.3f,
                cstd::abs(O.y) - 0.3f);
    var cylinder = cstd::min(cstd::max(d.x, d.y), f32(0))
                 + length(make_vec2f(
                            cstd::max(f32(0), d.x),
                            cstd::max(f32(0), d.y)));

    var geometry = make_nested(cstd::min(sphere, cstd::min(box, cylinder)));
    geometry = cstd::max(geometry, -(0.32f - (o.y * 0.6f + o.z * 0.8f)));
    return cstd::min(wall, geometry);
}

f32 ray_march(const Vec3f &p, const Vec3f &d)
{
    var j = 0;
    var dist = 0.0f;
    $while(true)
    {
        $if(j >= 100 | dist >= INF)
        {
            $break;
        };

        f32 s = sdf(p + dist * d);
        $if(s <= 1e-6f)
        {
            $break;
        };

        dist = dist + sdf(p + dist * d);
        j = j + 1;
    };
    return cstd::min(f32(INF), dist);
}

Vec3f sdf_normal(const Vec3f &p)
{
    constexpr float d = 1e-3f;
    var n = make_vec3f(0.0f, 0.0f, 0.0f);
    var sdf_center = sdf(p);
    $if(sdf_center < 1e-3f)
    {
        for(int i = 0; i < 3; ++i)
        {
            var inc = p;
            inc[i] = inc[i] + d;
            n[i] = 1 / d * (sdf(inc) - sdf_center);
        }
        n = normalize(n);
    };
    return n;
}

f32 next_hit(
    const Vec3f &pos,
    const Vec3f &d,
    ptr<Vec3f>   normal,
    ptr<Vec3f>   c)
{
    var closest = INF;
    *normal = make_vec3f(0.0f, 0.0f, 0.0f);
    *c = make_vec3f(0.0f, 0.0f, 0.0f);

    var ray_march_dist = ray_march(pos, d);
    $if(ray_march_dist < DIST_LIMIT)
    {
        closest = ray_march_dist;
        *normal = sdf_normal(pos + closest * d);

        var hit_pos = pos + closest * d;
        var t = i32((hit_pos.x + 10) * 1.1f + 0.5f) % 3;
        *c = make_vec3f(
            0.4f + 0.3f * cstd::select(t == 0, f32(1), f32(0)),
            0.4f + 0.2f * cstd::select(t == 1, f32(1), f32(0)),
            0.4f + 0.3f * cstd::select(t == 2, f32(1), f32(0)));
    };

    return closest;
}

void render_pixel(
    ptr<Vec3f> color_buffer, i32 x, i32 y, ptr<cstd::LCG> rng)
{
    var pos = make_vec3f(CAMERA_POS[0], CAMERA_POS[1], CAMERA_POS[2]);

    Vec3f d;
    d.x = 2 * FOV * (f32(x) + rng->uniform_float()) * (1.0f / HEIGHT) - FOV * ASPECT - 1e-5f;
    d.y = 2 * FOV * (f32(y) + rng->uniform_float()) * (1.0f / HEIGHT) - FOV - 1e-5f;
    d.z = -1.0f;
    d = normalize(d);

    var throughput = make_vec3f(1.0f, 1.0f, 1.0f);

    var depth = 0;
    var hit_light = false;

    $while(depth < MAX_DEPTH)
    {
        depth = depth + 1;

        Vec3f normal, c;
        var closest = next_hit(pos, d, normal.address(), c.address());
        var dist_to_light = intersect_light(pos, d);

        $if(dist_to_light < closest)
        {
            hit_light = true;
            depth = MAX_DEPTH;
        }
        $else
        {
            var hit_pos = pos + closest * d;
            $if(length_square(normal) > 0.0f)
            {
                d = out_dir(normal, rng);
                pos = hit_pos + 1e-3f * d;
                throughput = c * throughput;
            }
            $else
            {
                throughput = 0.8f * throughput;
                depth = MAX_DEPTH;
            };
        };
    };

    $if(hit_light)
    {
        var idx = (HEIGHT - 1 - y) * WIDTH + x;
        var pixel = color_buffer[idx].address();
        *pixel = *pixel + throughput;
    };
}

std::string generate_ptx()
{
    ScopedModule mod;

    const auto start_time = std::chrono::steady_clock::now();

    auto render = kernel(
        "render", [&](ptr<Vec3f> color_buffer, ptr<cstd::LCG> rngs)
    {
        var x = cstd::block_idx_x() * cstd::block_dim_x() + cstd::thread_idx_x();
        var y = cstd::block_idx_y() * cstd::block_dim_y() + cstd::thread_idx_y();
        $if(x < WIDTH & y < HEIGHT)
        {
            render_pixel(color_buffer, x, y, rngs + y * WIDTH + x);
        };
    });

    const auto end_record_time = std::chrono::steady_clock::now();

    PTXGenerator ptx_gen;
    ptx_gen.set_options(Options{
        .opt_level        = OptimizationLevel::O3,
        .fast_math        = true,
        .approx_math_func = true
    });
    ptx_gen.generate(mod);
    
    const auto end_compile_time = std::chrono::steady_clock::now();

    std::cout << "record time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_record_time - start_time).count()
              << "ms" << std::endl;
    
    std::cout << "compile time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_compile_time - end_record_time).count()
              << "ms" << std::endl;

    return ptx_gen.get_ptx();
}

void run()
{
    const auto ptx = generate_ptx();

    CUdevice cuDevice;
    CUcontext context;
    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&context, 0, cuDevice);
    CUJ_SCOPE_EXIT{ cuCtxDestroy(context); };

    CUDAModule cuda_module;
    cuda_module.load_ptx_from_memory(ptx.data(), ptx.size());

    float *device_color_buffer = nullptr;
    check_cuda_error(cudaMalloc(
        &device_color_buffer, sizeof(float) * 3 * WIDTH * HEIGHT));
    CUJ_SCOPE_EXIT{ cudaFree(device_color_buffer); };

    check_cuda_error(cudaMemset(
        device_color_buffer, 0, sizeof(float) * 3 * WIDTH * HEIGHT));

    uint32_t *device_rng_buffer = nullptr;
    check_cuda_error(cudaMalloc(
        &device_rng_buffer, sizeof(uint32_t) * WIDTH * HEIGHT));
    CUJ_SCOPE_EXIT{ cudaFree(device_rng_buffer); };

    {
        std::vector<uint32_t> rng_data(WIDTH * HEIGHT);
        uint32_t i = 1;
        for(uint32_t &s : rng_data)
            s = i++;
        check_cuda_error(cudaMemcpy(
            device_rng_buffer, rng_data.data(),
            sizeof(uint32_t) * rng_data.size(), cudaMemcpyHostToDevice));
    }

    constexpr int BLOCK_SIZE_X = 16;
    constexpr int BLOCK_SIZE_Y = 8;

    constexpr int BLOCK_COUNT_X = (WIDTH  + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    constexpr int BLOCK_COUNT_Y = (HEIGHT + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;


    cuda_module.launch(
        "render",
        { BLOCK_COUNT_X, BLOCK_COUNT_Y, 1 },
        { BLOCK_SIZE_X, BLOCK_SIZE_Y, 1 },
        device_color_buffer, device_rng_buffer, 0);
    cudaDeviceSynchronize();

        std::cout << "start rendering" << std::endl;
        const auto start_time = std::chrono::steady_clock::now();
    for(int i = 0; i < 4096; ++i)
    {
        cuda_module.launch(
            "render",
            { BLOCK_COUNT_X, BLOCK_COUNT_Y, 1 },
            { BLOCK_SIZE_X, BLOCK_SIZE_Y, 1 },
            device_color_buffer, device_rng_buffer, i);
    }

    cudaDeviceSynchronize();

    const auto end_time = std::chrono::steady_clock::now();
    std::cout << "render time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - start_time).count()
              << "ms" << std::endl;

    std::vector<float> color_buffer(WIDTH * HEIGHT * 3);
    check_cuda_error(cudaMemcpy(
        color_buffer.data(), device_color_buffer,
        sizeof(float) * color_buffer.size(),
        cudaMemcpyDeviceToHost));

    float mean = 0;
    for(float x : color_buffer)
        mean += x;
    mean /= color_buffer.size();
    for(float &x : color_buffer)
        x = x / mean * 0.24f;

    std::ofstream fout("output.ppm");
    if(!fout)
    {
        throw std::runtime_error(
            "failed to create output image: output.ppm");
    }
    
    fout << "P3\n" << WIDTH << " " << HEIGHT << std::endl << 255 << std::endl;
    for(int i = 0, j = 0; i < WIDTH * HEIGHT; ++i, j += 3)
    {
        const float rf = color_buffer[j];
        const float gf = color_buffer[j + 1];
        const float bf = color_buffer[j + 2];

        const int ri = std::min(255, static_cast<int>(std::pow(rf, 1 / 2.2f) * 255));
        const int gi = std::min(255, static_cast<int>(std::pow(gf, 1 / 2.2f) * 255));
        const int bi = std::min(255, static_cast<int>(std::pow(bf, 1 / 2.2f) * 255));

        fout << ri << " " << gi << " " << bi << " ";
    }

    fout.close();
    std::cout << "result written to output.ppm" << std::endl;
}

int main()
{
    try
    {
        run();
    }
    catch(const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }
}
