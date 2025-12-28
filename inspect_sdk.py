from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
import inspect

try:
    # We don't need a real connection to inspect the method
    # But we need an instance or the class.
    # AIProjectClient.agents is a property that returns an AgentsClient.
    # Let's try to get the class of the agents client.
    
    print("Inspecting project_client.agents.messages.list...")
    # We can't easily instantiate AIProjectClient without credentials/endpoint usually, 
    # but let's try to see if we can import the underlying client class directly if we knew it.
    # Instead, let's just try to use help() on the method if we can get to it.
    
    # Actually, let's just try to run a script that prints the help.
    # We need to mock the client or just use the installed package structure.
    
    # from azure.ai.projects.operations import AgentsOperations
    pass
    # Wait, I don't know the internal structure.
    
    # Let's try to inspect via the client instance (will fail auth if I try to make calls, but I just want help)
    # I'll use the existing credentials which work.
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    project_endpoint = os.getenv("PROJECT_ENDPOINT")
    credential = DefaultAzureCredential()
    project_client = AIProjectClient(endpoint=project_endpoint, credential=credential)
    
    # print(help(project_client.agents.messages.list))
    print(f"Docstring: {project_client.agents.messages.list.__doc__}")
    print(f"Signature: {inspect.signature(project_client.agents.messages.list)}")

except Exception as e:
    print(f"Error: {e}")
