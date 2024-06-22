using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OnnxRuntimeGenAI;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Zorach
{
    internal class Program
    {
        private const string ZORACH_ONNX_MODEL_PATH = nameof(ZORACH_ONNX_MODEL_PATH);

        static async Task Main(string[] args)
        {
            var modelPath = Environment.GetEnvironmentVariable(nameof(ZORACH_ONNX_MODEL_PATH));

            if (modelPath == null || !Path.Exists(modelPath))
            {
                Console.WriteLine($"(L|S)LM not found in path {modelPath}");
                return;
            }

            var settings = await GetModelSettings(modelPath);
            settings.PastPresentShareBuffer = false;

            // create kernel
            var builder = Kernel.CreateBuilder();
            builder.AddOnnxRuntimeGenAIChatCompletion(modelPath: modelPath);
            var kernel = builder.Build();

            // create chat
            var chat = kernel.GetRequiredService<IChatCompletionService>();
            var history = new ChatHistory();

            // run chat
            while (true)
            {
                Console.Write("Q: ");
                var userQ = Console.ReadLine();
                if (string.IsNullOrEmpty(userQ))
                {
                    break;
                }
                history.AddUserMessage(userQ);

                Console.Write($"Phi3: ");
                var response = "";
                var result = chat.GetStreamingChatMessageContentsAsync(history, executionSettings: settings);
                await foreach (var message in result)
                {
                    Console.Write(message.Content);
                    response += message.Content;
                }
                history.AddAssistantMessage(response);
                Console.WriteLine("");
            }
        }

        static async Task<OnnxRuntimeGenAIPromptExecutionSettings> GetModelSettings(string modelPath)
        {
            var configPath = Path.Combine(modelPath, "genai_config.json");
            using var sr = new FileStream(configPath, FileMode.Open, FileAccess.Read);
            var config = await JsonSerializer.DeserializeAsync<GenAIConfig>(sr);

            if (config != null && config.ExecutionSettings != null)
            {
                return config.ExecutionSettings;
            }
            else
            {
                throw new Exception("Load GenAI config file failed.");
            }
        }
    }

    internal class GenAIConfig
    {
        [JsonPropertyName("search")]
        public OnnxRuntimeGenAIPromptExecutionSettings? ExecutionSettings { get; set; }
    }
}
