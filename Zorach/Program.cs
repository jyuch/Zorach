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

        private const string DEFAULT_SYSTEM_PROMPT = @"
Your name is 'Zorach' and you are an AI assistant.
Answer questions using a direct style.
";

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

            Console.WriteLine($"{DEFAULT_SYSTEM_PROMPT}");
            Console.WriteLine("Type system prompt if you want change.");
            Console.Write("System: ");

            var systemPrompt = Console.ReadLine();
            Console.WriteLine();

            if (string.IsNullOrEmpty(systemPrompt))
            {
                systemPrompt = DEFAULT_SYSTEM_PROMPT;
            }

            // create kernel
            var builder = Kernel.CreateBuilder();
            builder.AddOnnxRuntimeGenAIChatCompletion(modelPath: modelPath);
            var kernel = builder.Build();

            // create chat
            var chat = kernel.GetRequiredService<IChatCompletionService>();
            var history = new ChatHistory();

            history.AddSystemMessage(systemPrompt);

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

                Console.WriteLine();
                Console.Write($"Z: ");
                var response = "";
                var result = chat.GetStreamingChatMessageContentsAsync(history, executionSettings: settings);
                await foreach (var message in result)
                {
                    Console.Write(message.Content);
                    response += message.Content;
                }
                history.AddAssistantMessage(response);
                Console.WriteLine();
                Console.WriteLine();
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
