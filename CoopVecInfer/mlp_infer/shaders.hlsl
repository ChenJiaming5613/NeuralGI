/* shaders.hlsl */

// ============================================================================
// Compute Shader
// ============================================================================

// float 数据 (RGB)
StructuredBuffer<float> g_InputBuffer : register(t0);
// 40x40 的 UAV Texture
RWTexture2D<float4> g_OutputTexture : register(u0);

// 宏定义：确保与 Python 导出时的维度一致
#define MAX_DIM 64
#define INPUT_DIM 3
#define OUTPUT_DIM 3

// 中间层循环次数计算：
// 你的 hidden_dims = [64, 64, 64, 64, 64, 64] (共6层)
// 第1层 (Index 0) 在输入阶段处理
// 剩下的 (Index 1 到 5) 是全连接的 64->64 变换，共 5 层
#define MIDDLE_LAYER_COUNT 5

float3 RunVoxelMLP(float3 InPos, StructuredBuffer<float> Weights)
{
    // 全局偏移量，用于模拟指针移动
    uint offset = 0;

    // 静态数组缓冲区 (显存堆栈)
    float layer_in[MAX_DIM];
    float temp_out[MAX_DIM];

    // ====================================================
    // 1. 输入层: Input(3) -> Hidden(64) + ReLU
    // ====================================================
    {
        int in_dim = INPUT_DIM;   // 3
        int out_dim = MAX_DIM;    // 64

        // 偏置(Bias)通常存在权重(Weight)之后
        // Bias Offset = 当前Offset + (Weight总数)
        uint bias_offset_base = offset + (out_dim * in_dim);

        for (int o = 0; o < out_dim; o++)
        {
            // 读取 Bias
            float sum = Weights[bias_offset_base + o];

            // 读取 Weight 并进行点积 (展开以优化 float3)
            uint w_idx = offset + (o * in_dim);

            sum += InPos.x * Weights[w_idx + 0];
            sum += InPos.y * Weights[w_idx + 1];
            sum += InPos.z * Weights[w_idx + 2];

            // ReLU 激活
            layer_in[o] = max(0.0f, sum);
        }

        // 更新 Offset: 跳过这一层的 Weights 和 Biases
        offset += (out_dim * in_dim) + out_dim;
    }

    // ====================================================
    // 2. 中间层循环: Hidden(64) -> Hidden(64) + ReLU
    //    合并了之前拆分的逻辑，执行 5 次
    // ====================================================
    for (int L = 0; L < MIDDLE_LAYER_COUNT; L++)
    {
        int dim = MAX_DIM; // 64
        uint bias_offset_base = offset + (dim * dim);

        for (int o = 0; o < dim; o++)
        {
            // 1. 初始化 sum 为 bias
            float sum = Weights[bias_offset_base + o];

            // 权重起始位置
            uint w_start_idx = offset + (o * dim);

            // 2. 密集矩阵乘法 (64 x 64)
            // 编译器通常会自动优化这种简单的 MAC (Multiply-Accumulate) 循环
            for (int i = 0; i < dim; i++)
            {
                sum += layer_in[i] * Weights[w_start_idx + i];
            }

            // 3. ReLU 激活，暂存到 temp_out
            temp_out[o] = max(0.0f, sum);
        }

        // 4. 将结果拷回 layer_in，作为下一层的输入
        // 显卡上这种小规模的连续内存拷贝非常快
        for (int k = 0; k < dim; k++)
        {
            layer_in[k] = temp_out[k];
        }

        // 5. 更新 Offset
        offset += (dim * dim) + dim;
    }

    // ====================================================
    // 3. 输出层: Hidden(64) -> Output(3) (Linear / No Activation)
    // ====================================================
    float3 result;
    {
        int in_dim = MAX_DIM;     // 64
        int out_dim = OUTPUT_DIM; // 3

        uint bias_offset_base = offset + (out_dim * in_dim);

        for (int o = 0; o < out_dim; o++)
        {
            float sum = Weights[bias_offset_base + o];
            uint w_start_idx = offset + (o * in_dim);

            for (int i = 0; i < in_dim; i++)
            {
                sum += layer_in[i] * Weights[w_start_idx + i];
            }

            // 输出层通常是线性的 (没有 ReLU)
            result[o] = sum;
        }
    }

    return result;
}

[numthreads(8, 8, 1)]
void main_cs(uint3 id : SV_DispatchThreadID)
{
    // 纹理尺寸为 40x40
    if (id.x >= 40 || id.y >= 40) return;
    float3 rgb = RunVoxelMLP(float3(
        id.x / 39.0,
        id.y / 39.0,
        20 / 31.0
    ), g_InputBuffer);
    rgb = clamp(rgb * 0.5, 0, 1);
    // 写入 UAV Texture
    g_OutputTexture[id.xy] = float4(rgb, 1.0);
}

// ============================================================================
// Graphics Shader (Blit)
// ============================================================================

Texture2D g_Texture : register(t0);
SamplerState g_Sampler : register(s0);

// 全屏三角形 Vertex Shader
void blit_vs(
    uint id : SV_VertexID,
    out float4 o_pos : SV_Position,
    out float2 o_uv : TEXCOORD0
)
{
    o_uv = float2((id << 1) & 2, id & 2);
    o_pos = float4(o_uv * float2(2, -2) + float2(-1, 1), 0, 1);
}

// 采样 Texture 的 Pixel Shader
void blit_ps(
    float4 i_pos : SV_Position,
    float2 i_uv : TEXCOORD0,
    out float4 o_color : SV_Target0
)
{
    // 使用 Point Sampler 采样
    o_color = g_Texture.Sample(g_Sampler, i_uv);
}