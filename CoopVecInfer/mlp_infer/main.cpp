#include <donut/app/ApplicationBase.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/vfs/VFS.h>
#include <nvrhi/utils.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "DeviceUtils.h"
#include "GraphicsResources.h"

using namespace donut;

static const char* g_WindowTitle = "Donut Example: MLP Infer";

class MLPInferApp : public app::IRenderPass
{
private:
    // Shaders
    nvrhi::ShaderHandle m_ComputeShader;
    nvrhi::ShaderHandle m_BlitVS;
    nvrhi::ShaderHandle m_BlitPS;

    // Pipelines
    nvrhi::ComputePipelineHandle m_ComputePipeline;
    nvrhi::GraphicsPipelineHandle m_BlitPipeline;

    // Resources
    nvrhi::TextureHandle m_UAVTexture;
    nvrhi::BufferHandle m_InputBuffer;
    nvrhi::SamplerHandle m_PointSampler;

    // Bindings
    nvrhi::BindingLayoutHandle m_ComputeBindingLayout;
    nvrhi::BindingSetHandle m_ComputeBindingSet;

    nvrhi::BindingLayoutHandle m_BlitBindingLayout;
    nvrhi::BindingSetHandle m_BlitBindingSet;

    nvrhi::CommandListHandle m_CommandList;

public:
    using IRenderPass::IRenderPass;

    bool Init()
    {
        // 1. 初始化 Shader Factory 和 Command List
        std::filesystem::path appShaderPath = app::GetDirectoryWithExecutable() / "shaders/mlp_infer" / app::GetShaderTypeName(GetDevice()->getGraphicsAPI());
        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();
        engine::ShaderFactory shaderFactory(GetDevice(), nativeFS, appShaderPath);

        m_CommandList = GetDevice()->createCommandList();

        // 2. 加载 Shaders
        m_ComputeShader = shaderFactory.CreateShader("shaders.hlsl", "main_cs", nullptr, nvrhi::ShaderType::Compute);
        m_BlitVS = shaderFactory.CreateShader("shaders.hlsl", "blit_vs", nullptr, nvrhi::ShaderType::Vertex);
        m_BlitPS = shaderFactory.CreateShader("shaders.hlsl", "blit_ps", nullptr, nvrhi::ShaderType::Pixel);

        if (!m_ComputeShader || !m_BlitVS || !m_BlitPS)
        {
            log::error("Failed to load shaders");
            return false;
        }

        // 3. 创建 40x40 UAV Texture (格式与 BackBuffer 一致，这里假设为 RGBA8_UNORM)
        nvrhi::TextureDesc texDesc;
        texDesc.width = 40;
        texDesc.height = 40;
        texDesc.format = nvrhi::Format::RGBA8_UNORM;
        texDesc.isUAV = true;
        texDesc.isShaderResource = true;
        texDesc.debugName = "OutputTexture";
        texDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        texDesc.keepInitialState = true;
        m_UAVTexture = GetDevice()->createTexture(texDesc);

        // 生成测试数据 (颜色渐变)
        std::vector<float> bufferData;
        {
            // 1. 读取二进制文件到内存
            std::ifstream file("model.bin", std::ios::binary | std::ios::ate);
            if (!file) {
                std::cerr << "Error: Cannot open model.bin" << std::endl;
                exit(1);
            }

            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);

            // 确保文件大小是 float (4 bytes) 的倍数
            if (size % 4 != 0) {
                std::cerr << "Error: File size is not a multiple of 4 bytes." << std::endl;
                exit(1);
            }

            bufferData.resize(size / 4);
            if (!file.read(reinterpret_cast<char*>(bufferData.data()), size)) {
                std::cerr << "Error: Failed to read file data." << std::endl;
                exit(1);
            }

            std::cout << "Loaded model weights: " << bufferData.size() << " floats." << std::endl;

            std::cout << "Sample\n Top 10:" << std::endl;
            for (size_t i = 0; i < 10; i++)
            {
                std::cout << bufferData[i] << " ";
            }
            std::cout << std::endl;
        }

        // 4. 创建并上传 float Structured Buffer
        nvrhi::BufferDesc bufDesc;
        bufDesc.byteSize = bufferData.size() * sizeof(float);
        bufDesc.structStride = sizeof(float); // Shader 中是 StructuredBuffer<float>，Stride 为 4
        bufDesc.debugName = "MLPBuffer";
        bufDesc.canHaveUAVs = false; // 只作为 SRV 读取
        bufDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        bufDesc.keepInitialState = true;
        m_InputBuffer = GetDevice()->createBuffer(bufDesc);

        // 上传数据
        m_CommandList->open();
        m_CommandList->writeBuffer(m_InputBuffer, bufferData.data(), bufferData.size() * sizeof(float));
        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);

        // 5. 创建 Sampler (Point Filter)
        nvrhi::SamplerDesc samplerDesc;
        samplerDesc.minFilter = false; // Point
        samplerDesc.magFilter = false; // Point
        samplerDesc.mipFilter = false; // Point
        m_PointSampler = GetDevice()->createSampler(samplerDesc);

        // 6. 创建 Binding Layouts 和 Sets

        // --- Compute Bindings ---
        nvrhi::BindingLayoutDesc computeLayoutDesc;
        computeLayoutDesc.visibility = nvrhi::ShaderType::Compute;
        computeLayoutDesc.addItem(nvrhi::BindingLayoutItem::StructuredBuffer_SRV(0)); // t0
        computeLayoutDesc.addItem(nvrhi::BindingLayoutItem::Texture_UAV(0));          // u0
        m_ComputeBindingLayout = GetDevice()->createBindingLayout(computeLayoutDesc);

        nvrhi::BindingSetDesc computeSetDesc;
        computeSetDesc.addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(0, m_InputBuffer));
        computeSetDesc.addItem(nvrhi::BindingSetItem::Texture_UAV(0, m_UAVTexture));
        m_ComputeBindingSet = GetDevice()->createBindingSet(computeSetDesc, m_ComputeBindingLayout);

        // --- Blit Bindings ---
        nvrhi::BindingLayoutDesc blitLayoutDesc;
        blitLayoutDesc.visibility = nvrhi::ShaderType::Pixel;
        blitLayoutDesc.addItem(nvrhi::BindingLayoutItem::Texture_SRV(0)); // t0
        blitLayoutDesc.addItem(nvrhi::BindingLayoutItem::Sampler(0));     // s0
        m_BlitBindingLayout = GetDevice()->createBindingLayout(blitLayoutDesc);

        nvrhi::BindingSetDesc blitSetDesc;
        blitSetDesc.addItem(nvrhi::BindingSetItem::Texture_SRV(0, m_UAVTexture));
        blitSetDesc.addItem(nvrhi::BindingSetItem::Sampler(0, m_PointSampler));
        m_BlitBindingSet = GetDevice()->createBindingSet(blitSetDesc, m_BlitBindingLayout);

        // 7. 创建 Compute Pipeline
        nvrhi::ComputePipelineDesc computePsoDesc;
        computePsoDesc.CS = m_ComputeShader;
        computePsoDesc.bindingLayouts = { m_ComputeBindingLayout };
        m_ComputePipeline = GetDevice()->createComputePipeline(computePsoDesc);

        return true;
    }

    void BackBufferResizing() override
    {
        // Backbuffer 尺寸改变时重建 Graphics Pipeline
        m_BlitPipeline = nullptr;
    }

    void Animate(float fElapsedTimeSeconds) override
    {
        GetDeviceManager()->SetInformativeWindowTitle(g_WindowTitle);
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        // 延迟创建 Blit Pipeline (需要 Framebuffer 格式信息)
        if (!m_BlitPipeline)
        {
            nvrhi::GraphicsPipelineDesc psoDesc;
            psoDesc.VS = m_BlitVS;
            psoDesc.PS = m_BlitPS;
            psoDesc.primType = nvrhi::PrimitiveType::TriangleList;
            psoDesc.renderState.rasterState.cullMode = nvrhi::RasterCullMode::None;
            psoDesc.renderState.depthStencilState.depthTestEnable = false;
            psoDesc.bindingLayouts = { m_BlitBindingLayout };

            m_BlitPipeline = GetDevice()->createGraphicsPipeline(psoDesc, framebuffer->getFramebufferInfo());
        }

        m_CommandList->open();

        // -------------------------------------------------------
        // Step 1: Compute Pass (Write to 40x40 Texture)
        // -------------------------------------------------------
        nvrhi::ComputeState computeState;
        computeState.pipeline = m_ComputePipeline;
        computeState.bindings = { m_ComputeBindingSet };
        m_CommandList->setComputeState(computeState);
        // 40x40 pixels, group size 8x8 -> 5x5 groups
        m_CommandList->dispatch(5, 5, 1);

        // -------------------------------------------------------
        // Step 2: Blit Pass (Render to Backbuffer)
        // -------------------------------------------------------

        // NVRHI 会根据 BindingSet 自动处理 Resource Transition (UAV -> SRV)

        nvrhi::utils::ClearColorAttachment(m_CommandList, framebuffer, 0, nvrhi::Color(0.f));

        nvrhi::GraphicsState graphicsState;
        graphicsState.pipeline = m_BlitPipeline;
        graphicsState.framebuffer = framebuffer;
        graphicsState.viewport.addViewportAndScissorRect(framebuffer->getFramebufferInfo().getViewport());
        graphicsState.bindings = { m_BlitBindingSet };

        m_CommandList->setGraphicsState(graphicsState);

        // 绘制全屏三角形 (3个顶点)
        nvrhi::DrawArguments args;
        args.vertexCount = 3;
        m_CommandList->draw(args);

        m_CommandList->close();
        GetDevice()->executeCommandList(m_CommandList);
    }
};

#ifdef WIN32
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    nvrhi::GraphicsAPI api = app::GetGraphicsAPIFromCommandLine(__argc, __argv);
    if (api == nvrhi::GraphicsAPI::D3D11)
    {
        log::error("This sample does not support D3D11");
        return 1;
    }

    app::DeviceManager* deviceManager = app::DeviceManager::Create(api);

    app::DeviceCreationParameters deviceParams;
#ifdef _DEBUG
    deviceParams.enableDebugRuntime = true;
    deviceParams.enableGPUValidation = false;
    deviceParams.enableNvrhiValidationLayer = true;
#endif

    SetCoopVectorExtensionParameters(deviceParams, api, true, g_WindowTitle);

    deviceParams.backBufferWidth = 600;
    deviceParams.backBufferHeight = 600;
    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_WindowTitle))
    {
        if (api == nvrhi::GraphicsAPI::VULKAN)
        {
            log::fatal("Cannot initialize a graphics device with the requested parameters. Please try a NVIDIA driver version greater than 570");
        }
        if (api == nvrhi::GraphicsAPI::D3D12)
        {
            log::fatal("Cannot initialize a graphics device with the requested parameters. Please use the Shader Model 6-9-Preview Driver, link in the README");
        }
        return 1;
    }

    auto graphicsResources = std::make_unique<GraphicsResources>(deviceManager->GetDevice());
    if (!graphicsResources->GetCoopVectorFeatures().inferenceSupported && !graphicsResources->GetCoopVectorFeatures().fp16InferencingSupported)
    {
        log::fatal("Not all required Coop Vector features are available");
        return 1;
    }

    {
        MLPInferApp example(deviceManager);
        if (example.Init())
        {
            deviceManager->AddRenderPassToBack(&example);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&example);
        }
    }

    deviceManager->Shutdown();
    delete deviceManager;

    return 0;
}