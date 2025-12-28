import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential, AzureCliCredential
except ImportError:
    print("Please install the required packages: pip install azure-ai-projects azure-identity")
    sys.exit(1)

def main():
    # Configuration
    project_endpoint = os.getenv("PROJECT_ENDPOINT")
    model_deployment = os.getenv("MODEL_DEPLOYMENT_NAME")

    if not project_endpoint or not model_deployment:
        print("Error: Please set PROJECT_ENDPOINT and MODEL_DEPLOYMENT_NAME in your .env file.")
        return

    print(f"Connecting to Azure AI Project at {project_endpoint}...")
    print(f"Using model deployment: {model_deployment}")

    # Initialize the AI Project Client
    # We use DefaultAzureCredential which tries multiple authentication methods including Azure CLI.
    # If you are running locally and have Azure CLI installed, ensure you are logged in with `az login`.
    credential = DefaultAzureCredential()
    
    # Fallback to AzureCliCredential if needed (though DefaultAzureCredential includes it)
    # credential = AzureCliCredential()

    project_client = AIProjectClient(
        endpoint=project_endpoint,
        credential=credential,
    )

    # Create an agent
    # Note: In a real application, you might want to reuse an existing agent ID
    print("Creating agent...")
    try:
        agent = project_client.agents.create_agent(
            model=model_deployment,
            name="my-assistant",
            instructions="You are a helpful AI assistant.",
        )
        print(f"Created agent, ID: {agent.id}")

        # Create a thread
        thread = project_client.agents.threads.create()
        print(f"Created thread, ID: {thread.id}")

        # Simple chat loop
        print("\n--- Azure AI Agent (Type 'quit' to exit) ---")
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                break

            # Add message to thread
            project_client.agents.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_input,
            )

            # Run the agent
            run = project_client.agents.runs.create_and_process(
                thread_id=thread.id,
                agent_id=agent.id,
            )

            if run.status == "completed":
                # List messages
                # We fetch messages in descending order to get the latest one first
                messages = project_client.agents.messages.list(
                    thread_id=thread.id,
                    order="desc"
                )
                
                # Find the latest assistant message
                for msg in messages:
                    if msg.role == "assistant":
                        for content in msg.content:
                            if content.type == "text":
                                print(f"Agent: {content.text.value}")
                        break # Only print the latest response
            else:
                print(f"Run failed with status: {run.status}")
                if run.last_error:
                    print(f"Error: {run.last_error}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Cleanup (optional)
        # if 'agent' in locals():
        #     project_client.agents.delete_agent(agent.id)
        #     print("Agent deleted.")
        pass

if __name__ == "__main__":
    main()
