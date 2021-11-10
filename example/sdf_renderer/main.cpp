#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuj/cuj.h>

// reference: https://github.com/taichi-dev/taichi/blob/master/examples/rendering/sdf_renderer.py

using namespace cuj;
using namespace builtin::math;
using namespace builtin::cuda;

constexpr float EPS = 1e-4f;
constexpr float INF = 1e10f;

constexpr float PI = 3.1415926535f;

constexpr float CAMERA_POS[] = { 0.0f, 0.32f, 3.7f };
constexpr float LIGHT_POS[]  = { -1.5f, 0.6f, 0.3f };
constexpr float LIGHT_NOR[]  = { 1.0f, 0.0f, 0.0f };
constexpr float LIGHT_RADIUS = 2.0f;

constexpr float FOV        = 0.23f;
constexpr float DIST_LIMIT = 100.0f;
constexpr int   MAX_DEPTH  = 6;

constexpr int   WIDTH  = 1280;
constexpr int   HEIGHT = 720;
constexpr float ASPECT = static_cast<float>(WIDTH) / HEIGHT;

void check_cuda_error(cudaError_t err)
{
    if(err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
}

Vec3f abs(const Vec3f &v)
{
    return Vec3f(abs(v.x), abs(v.y), abs(v.z));
}

i32 floor_div_rem(i32 a, i32 b)
{
    i32 r = a % b;
    $if((r > 0 && b < 0) || (r < 0 && b > 0))
    {
        r = r + b;
    };
    return r;
}

std::string generate_ptx()
{
    ScopedContext ctx;

    auto intersect_light = to_callable<f32>(
        "intersect_light", [](const Vec3f &pos, const Vec3f &dir)
    {
        Vec3f light_loc(LIGHT_POS[0], LIGHT_POS[1], LIGHT_POS[2]);
        f32 dotv = -dot(dir, Vec3f(LIGHT_NOR[0], LIGHT_NOR[1], LIGHT_NOR[2]));
        f32 dist = dot(light_loc - pos, dir);
        f32 dist_to_light = INF;
        $if(dotv > 0 && dist > 0)
        {
            f32 D = dist / dotv;
            f32 dist_to_center = (light_loc - (pos + D * dir)).length_square();
            $if(dist_to_center < LIGHT_RADIUS * LIGHT_RADIUS)
            {
                dist_to_light = D;
            };
        };
        $return(dist_to_light);
    });

    auto out_dir = to_callable<Vec3f>(
        "out_dir", [](const Vec3f &n, Pointer<PCG> pcg)
    {
        Vec3f u(1.0f, 0.0f, 0.0f);
        $if(abs(n[1]) < 1 - EPS)
        {
            u = cross(n, Vec3f(0.0f, 1.0f, 0.0f)).normalize();
        };
        Vec3f v = cross(n, u);
        f32 phi = 2 * PI * pcg.deref().uniform_float();
        f32 ay = sqrt(pcg.deref().uniform_float());
        f32 ax = sqrt(1 - ay * ay);
        $return(ax * (cos(phi) * u + sin(phi) * v) + ay * n);
    });

    auto make_nested = to_callable<f32>(
        "make_nested", [](f32 f)
    {
        f = f * 40;
        i32 i = cast<i32>(f);
        $if(f < 0)
        {
            $if(floor_div_rem(i, 2) == 1)
            {
                f = f - floor(f);
            }
            $else
            {
                f = floor(f) + 1 - f;
            };
        };
        f = (f - 0.2) / 40;
        $return(f);
    });

    auto sdf = to_callable<f32>(
        "sdf", [&](Vec3f o)
    {
        f32 wall = min(o[1] + 0.1f, o[2] + 0.4f);
        f32 sphere = (o - Vec3f(0.0f, 0.35f, 0.0f)).length() - 0.36f;

        Vec3f q = abs(o - Vec3f(0.8f, 0.3f, 0.0f)) - Vec3f(0.3f, 0.3f, 0.3f);
        f32 box = Vec3f(max(f32(0), q[0]), max(f32(0), q[1]), max(f32(0), q[2])).length()
            + min(q.max_elem(), f32(0));

        Vec3f O = o - Vec3f(-0.8f, 0.3f, 0.0f);
        Vec2f d = Vec2f(Vec2f(O[0], O[2]).length() - 0.3f, abs(O[1]) - 0.3f);
        f32 cylinder = min(d.max_elem(), f32(0))
            + Vec2f(max(f32(0), d[0]), max(f32(0), d[1])).length();

        f32 geometry = make_nested(min(sphere, min(box, cylinder)));
        geometry = max(geometry, -(0.32f - (o[1] * 0.6f + o[2] * 0.8f)));
        $return(min(wall, geometry));
    });

    auto ray_march = to_callable<f32>(
        "ray_march", [&](const Vec3f &p, const Vec3f &d)
    {
        i32 j = 0;
        f32 dist = 0.0f;

        $while(true)
        {
            $if(j >= 100 || dist >= INF)
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
        
        $return(min(f32(INF), dist));
    });

    auto sdf_normal = to_callable<Vec3f>(
        "sdf_normal", [&](const Vec3f &p)
    {
        f32 d = 1e-3f;
        Vec3f n(0.0f, 0.0f, 0.0f);
        f32 sdf_center = sdf(p);
        for(int i = 0; i < 3; ++i)
        {
            Vec3f inc = p;
            inc[i] = inc[i] + d;
            n[i] = (1 / d) * (sdf(inc) - sdf_center);
        }
        $return(n.normalize());
    });

    auto next_hit = to_callable<void>("next_hit", [&](
        const Vec3f   &pos,
        const Vec3f   &d, 
        Pointer<f32>   closest,
        Pointer<Vec3f> normal,
        Pointer<Vec3f> c)
    {
        *closest = INF;
        *normal  = Vec3f(0.0f, 0.0f, 0.0f);
        *c       = Vec3f(0.0f, 0.0f, 0.0f);

        f32 ray_march_dist = ray_march(pos, d);
        $if(ray_march_dist < DIST_LIMIT)
        {
            *closest = ray_march_dist;
            *normal = sdf_normal(pos + d * *closest);

            Vec3f hit_pos = pos + d * *closest;
            i32 t = cast<i32>((hit_pos[0] + 10) * 1.1f + 0.5f) % 3;
            *c = Vec3f(
                0.4f + 0.3f * select(t == 0, f32(1), f32(0)),
                0.4f + 0.2f * select(t == 1, f32(1), f32(0)),
                0.4f + 0.3f * select(t == 2, f32(1), f32(0)));
        };
    });

    auto render_pixel = to_callable<void>("render_pixel", [&](
        Pointer<Vec3f> color_buffer, i32 x, i32 y, Pointer<PCG> rng)
    {
        Vec3f pos(CAMERA_POS[0], CAMERA_POS[1], CAMERA_POS[2]);

        Vec3f d;
        d.x = 2 * FOV * (x + rng.deref().uniform_float()) / HEIGHT - FOV * ASPECT - 1e-5f;
        d.y = 2 * FOV * (y + rng.deref().uniform_float()) / HEIGHT - FOV - 1e-5f;
        d.z = -1.0f;
        d = d.normalize();

        Vec3f throughput(1.0f, 1.0f, 1.0f);

        i32 depth = 0;
        boolean hit_light = false;

        $while(depth < MAX_DEPTH)
        {
            depth = depth + 1;

            f32 closest;
            Vec3f normal, c;
            next_hit(pos, d, closest.address(), normal.address(), c.address());
            f32 dist_to_light = intersect_light(pos, d);

            $if(dist_to_light < closest)
            {
                hit_light = true;
                depth = MAX_DEPTH;
            }
            $else
            {
                Vec3f hit_pos = pos + closest * d;
                $if(normal.length_square() != 0.0f)
                {
                    d = out_dir(normal, rng);
                    pos = hit_pos + 1e-3 * d;
                    throughput = c * throughput;
                }
                $else
                {
                    depth = MAX_DEPTH;
                };
            };
        };

        $if(hit_light)
        {
            i32 idx = (HEIGHT - 1 - y) * WIDTH + x;
            Pointer<Vec3f> pixel = color_buffer[idx].address();
            *pixel = *pixel + throughput;
        };
    });

    auto render = to_kernel(
        "render", [&](
            Pointer<Vec3f> color_buffer, i32 x_thread_count, i32 spp, i32 iter)
    {
        i32 x = block_index_x() * block_dim_x() + thread_index_x();
        i32 y = block_index_y() * block_dim_y() + thread_index_y();

        $if(y >= HEIGHT)
        {
            $return();
        };

        $while(x < WIDTH)
        {
            PCG pcg(cast<u64>((iter + 1) * (y * WIDTH + x)));

            i32 i = 0;
            $while(i < spp)
            {
                i = i + 1;
                render_pixel(color_buffer, x, y, pcg.address());
            };

            x = x + x_thread_count;
        };
    });

    return ctx.gen_ptx(gen::OptLevel::O3, true);
}

void run()
{
    const std::string ptx = generate_ptx();
    std::cout << ptx << std::endl;

    CUdevice cuDevice;
    CUcontext context;
    cuInit(0);
    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&context, 0, cuDevice);
    CUJ_SCOPE_GUARD({ cuCtxDestroy(context); });

    CUDAModule cuda_module;
    cuda_module.load_ptx_from_memory(ptx.data(), ptx.size());

    float *device_color_buffer = nullptr;
    check_cuda_error(cudaMalloc(
        &device_color_buffer, sizeof(float) * 3 * WIDTH * HEIGHT));
    CUJ_SCOPE_GUARD({ cudaFree(device_color_buffer); });

    check_cuda_error(cudaMemset(
        device_color_buffer, 0, sizeof(float) * 3 * WIDTH * HEIGHT));

    constexpr int BLOCK_SIZE_X = 8;
    constexpr int BLOCK_SIZE_Y = 8;

    constexpr int BLOCK_COUNT_X = 16;
    constexpr int BLOCK_COUNT_Y = (HEIGHT + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;

    std::cout << "start rendering" << std::endl;
    const auto start_time = std::chrono::steady_clock::now();

    int x_thread_count = BLOCK_SIZE_X * BLOCK_COUNT_X;
    int iters = 4;
    int spp = 10;
    for(int i = 0; i < iters; ++i)
    {
        cuda_module.launch(
            "render",
            { BLOCK_COUNT_X, BLOCK_COUNT_Y, 1 },
            { BLOCK_SIZE_X, BLOCK_SIZE_Y, 1 },
            device_color_buffer, x_thread_count, spp, i);
    }

    cudaDeviceSynchronize();

    const auto end_time = std::chrono::steady_clock::now();
    std::cout << "rendering time: "
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
