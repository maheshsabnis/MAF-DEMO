using System.ComponentModel;
using System.ClientModel;
using Azure.AI.OpenAI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;

// Settings come from appsettings.json, which is copied next to the executable on build.
// Use the Azure resource endpoint (https://<resource>.openai.azure.com/)
// and the deployment name configured in Azure, not the underlying model name.
var configuration = new ConfigurationBuilder()
    .SetBasePath(AppContext.BaseDirectory)
    .AddJsonFile("appsettings.json", optional: false, reloadOnChange: false)
    .Build();

var azureOpenAI = configuration.GetSection("AzureOpenAI");
var endpoint = GetRequiredSetting(azureOpenAI, "Endpoint");
var apiKey = GetRequiredSetting(azureOpenAI, "ApiKey");
var deploymentName = GetRequiredSetting(azureOpenAI, "DeploymentName");

if (!Uri.TryCreate(endpoint, UriKind.Absolute, out var endpointUri) ||
    endpointUri.Scheme != Uri.UriSchemeHttps)
{
    throw new InvalidOperationException("AzureOpenAI:Endpoint must be an absolute HTTPS URL.");
}

IChatClient agent = new ChatClientBuilder(
        new AzureOpenAIClient(endpointUri, new ApiKeyCredential(apiKey))
            .GetChatClient(deploymentName).AsIChatClient())
    .UseFunctionInvocation(configure: c => c.MaximumIterationsPerRequest = 20)   // step limit, raised to leave room for ask_user turns
    .Build();

// Treats a leftover "<your-...>" placeholder as missing, so a half-filled
// appsettings.json fails here with a clear message instead of a 401 from Azure.
static string GetRequiredSetting(IConfigurationSection section, string key)
{
    var value = section[key];
    return !string.IsNullOrWhiteSpace(value) && !value.Contains('<')
        ? value
        : throw new InvalidOperationException(
            $"Set AzureOpenAI:{key} in appsettings.json before running this demo.");
}

// ---- Fake cloud state. Nothing real is created. ----
var resourceGroups = new HashSet<string>();
var clusters = new HashSet<string>();
var deployed = new HashSet<string>();

[Description("Check that the Azure subscription is enabled.")]
string CheckSubscription()
{
    Console.WriteLine("ACT: check_subscription");
    return "Subscription demo-subscription is Enabled.";
}

[Description("Create an Azure resource group.")]
string CreateResourceGroup([Description("Resource group name")] string name)
{
    Console.WriteLine($"ACT: create_resource_group({name})");
    resourceGroups.Add(name);
    return $"Resource group {name} created.";
}

[Description("Create an AKS cluster inside an existing resource group.")]
string CreateAksCluster(
    [Description("Existing resource group name")] string resourceGroup,
    [Description("New cluster name")] string clusterName)
{
    Console.WriteLine($"ACT: create_aks_cluster({resourceGroup}, {clusterName})");
    if (!resourceGroups.Contains(resourceGroup))
        return $"Error: resource group {resourceGroup} does not exist. Create it first.";
    clusters.Add(clusterName);
    return $"Cluster {clusterName} created.";
}

[Description("Deploy a container image to an existing AKS cluster.")]
string DeployApp(
    [Description("Existing cluster name")] string clusterName,
    [Description("Container image, for example myapp:1.0")] string image)
{
    Console.WriteLine($"ACT: deploy_app({clusterName}, {image})");
    if (!clusters.Contains(clusterName))
        return $"Error: cluster {clusterName} does not exist. Create it first.";
    deployed.Add(image);
    return $"Deployed {image} to {clusterName}.";
}

[Description("Check that the deployment is running.")]
string VerifyDeployment([Description("Cluster name")] string clusterName)
{
    Console.WriteLine($"ACT: verify_deployment({clusterName})");
    return $"Cluster {clusterName}: {(deployed.Count > 0 ? 3 : 0)} pods running.";
}

[Description("Ask the end user a question. Use this whenever a required detail is missing "
    + "or an action needs approval. Returns the user's answer.")]
string AskUser([Description("The question to put to the user")] string question)
{
    Console.WriteLine($"ASK: {question}");
    Console.Write("You> ");
    var answer = Console.ReadLine();

    // Null means the input stream closed (Ctrl+Z, or piped input running out).
    // Returning a sentinel stops the model from asking the same thing forever.
    return string.IsNullOrWhiteSpace(answer)
        ? "The user did not answer. Make a reasonable assumption and say so in your report."
        : answer;
}

var options = new ChatOptions
{
    Tools =
    [
        AIFunctionFactory.Create(CheckSubscription, "check_subscription"),
        AIFunctionFactory.Create(CreateResourceGroup, "create_resource_group"),
        AIFunctionFactory.Create(CreateAksCluster, "create_aks_cluster"),
        AIFunctionFactory.Create(DeployApp, "deploy_app"),
        AIFunctionFactory.Create(VerifyDeployment, "verify_deployment"),
        AIFunctionFactory.Create(AskUser, "ask_user")
    ]
};

var messages = new List<ChatMessage>
{
    new(ChatRole.System,
        "You are a DevOps agent. Work toward the user's goal step by step using the tools. " +
        "Check the subscription first. " +
        "Never invent resource group names, cluster names or image tags that the user did not give you. " +
        "Call ask_user for any detail you are missing, and ask one thing at a time. " +
        "If a tool returns an error, read it and fix the problem. " +
        "When finished, give a short report of what you did.")
};

Console.WriteLine("DevOps agent demo. Nothing real is created in Azure.");
Console.WriteLine("Describe what you want, or type 'exit' to quit.");
Console.WriteLine();

while (true)
{
    Console.Write("Goal> ");
    var goal = Console.ReadLine();

    if (goal is null || goal.Trim() is "exit" or "quit")
        break;

    if (string.IsNullOrWhiteSpace(goal))
        continue;

    messages.Add(new ChatMessage(ChatRole.User, goal));
    Console.WriteLine();

    try
    {
        var response = await agent.GetResponseAsync(messages, options);

        // Carry the whole exchange forward, including tool calls and their
        // results, so follow-up goals like "now verify it" have context.
        messages.AddRange(response.Messages);

        Console.WriteLine();
        Console.WriteLine(response.Text);
    }
    catch (Exception ex)
    {
        // A dropped connection or a 429 should not kill the session.
        Console.WriteLine($"The model call failed ({ex.GetType().Name}): {ex.Message}");
        Console.WriteLine("Nothing was changed. Try again, or type 'exit'.");
        messages.RemoveAt(messages.Count - 1);
    }

    Console.WriteLine();
}

Console.WriteLine("Done.");